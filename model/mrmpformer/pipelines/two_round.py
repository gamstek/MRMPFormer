# -*- coding: utf-8 -*-
"""
两轮识别流程：置信度≥0.99 唯一主峰 + XIC(SNR+次峰) 筛选 → 掩蔽主峰 → 第二轮模型 → 合并 newprediction.csv → XIC 图标注

流程：
1. 筛选：prediction.csv 中 score>=0.99 且 XIC 满足 SNR 阈值 + 次峰比例
2. 仅对通过筛选的图像生成主峰掩蔽图
3. 对掩蔽图目录运行第二轮 newtest
4. 合并两轮结果到 newprediction.csv
5. 在原 XIC 平滑图上绘制 1 或 2 个区间框及置信度

用法:
  python run_two_round_detection.py --batch_predictions results/batch_predictions --images_root xic-roi-batch --model checkpoint/checkpoint0029.pth
  python run_two_round_detection.py ... --min_confidence 0.99 --min_snr 3 --min_secondary_ratio 0.05
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from utils.xic_peak_utils import (
    compute_snr_outside_box,
    has_secondary_peak_in_roi,
    one_sided_edge_stop_threshold_stable_tail_mean,
    one_sided_low_noise_baseline,
    roi_full_low_decile_mean_intensity,
)


def _flat_triplet_early_stop_left_rt(
    y: np.ndarray,
    rt: np.ndarray,
    i: int,
    n: int,
    peak_idx: int,
    flat_triplet_step_frac: Optional[float],
) -> Optional[float]:
    """左肩外推：若连续三点相邻差分绝对值均很小，停在靠峰侧第一点 rt[i+2]。"""
    if flat_triplet_step_frac is None or flat_triplet_step_frac <= 0:
        return None
    if peak_idx < 0:
        return None
    if i + 2 >= n or i + 2 > int(peak_idx):
        return None
    ymax = float(np.max(y))
    eps = float(flat_triplet_step_frac) * max(ymax, 1e-12)
    if (
        abs(float(y[i + 2]) - float(y[i + 1])) <= eps
        and abs(float(y[i + 1]) - float(y[i])) <= eps
    ):
        return float(rt[i + 2])
    return None


def _flat_triplet_early_stop_right_rt(
    y: np.ndarray,
    rt: np.ndarray,
    i: int,
    n: int,
    peak_idx: int,
    flat_triplet_step_frac: Optional[float],
) -> Optional[float]:
    """右肩外推：对称地停在 rt[i-2]。"""
    if flat_triplet_step_frac is None or flat_triplet_step_frac <= 0:
        return None
    if peak_idx < 0:
        return None
    if i - 2 < 0 or i - 2 < int(peak_idx):
        return None
    ymax = float(np.max(y))
    eps = float(flat_triplet_step_frac) * max(ymax, 1e-12)
    if (
        abs(float(y[i]) - float(y[i - 1])) <= eps
        and abs(float(y[i - 1]) - float(y[i - 2])) <= eps
    ):
        return float(rt[i - 2])
    return None


def clamp_refined_interval_width_to_pred_and_roi(
    rt_min_adj: float,
    rt_max_adj: float,
    rt_min_pred: float,
    rt_max_pred: float,
    rt_lo: float,
    rt_hi: float,
    max_expand_vs_pred: float = 1.08,
    max_frac_of_roi: float = 0.45,
) -> Tuple[float, float]:
    """
    修正框宽度约束（仅上限）：
    宽度不得超过 min(原始预测宽×expand, ROI跨度×frac)。
    不在此处把框强行扩回「原始预测宽度」：否则会把已收窄的主框重新盖到两侧真实峰上，
    导致 ROI 次峰检测认为「主峰已包住两侧」而找不到 small/small2。
    """
    if not all(np.isfinite(v) for v in (rt_min_adj, rt_max_adj, rt_min_pred, rt_max_pred, rt_lo, rt_hi)):
        return rt_min_adj, rt_max_adj
    w0 = float(rt_max_pred - rt_min_pred)
    span_roi = float(rt_hi - rt_lo)
    if w0 <= 0 or span_roi <= 0 or rt_max_adj <= rt_min_adj:
        return rt_min_adj, rt_max_adj
    w_cap = min(float(max_expand_vs_pred) * w0, float(max_frac_of_roi) * span_roi)
    center = 0.5 * (float(rt_min_adj) + float(rt_max_adj))
    w = float(rt_max_adj - rt_min_adj)
    # 只做「不要太宽」：保留噪声截停得到的较窄宽度；若 w>w_cap 则对称缩至 w_cap
    w_target = float(min(w, w_cap))
    half = 0.5 * w_target
    lo = center - half
    hi = center + half
    if lo < rt_lo:
        lo = float(rt_lo)
        hi = min(float(rt_hi), lo + w_target)
    if hi > rt_hi:
        hi = float(rt_hi)
        lo = max(float(rt_lo), hi - w_target)
    if hi <= lo:
        return rt_min_adj, rt_max_adj
    return float(lo), float(hi)
from generate_masked_roi_for_small_peak_test import (
    load_roi_windows,
    mask_main_peak_and_redraw,
)


def _image_to_compound_index(image_name):
    """N_mz* -> N-1 (0-based)."""
    stem = Path(image_name).stem if "." in str(image_name) else str(image_name)
    m = re.match(r"^(\d+)_mz", stem, re.IGNORECASE)
    return int(m.group(1)) - 1 if m else None


def _posterior_peer_conflict(
    y: np.ndarray,
    rt: np.ndarray,
    ilo: int,
    ihi: int,
    peer_intervals: Sequence[Tuple[float, float]],
    thr: float,
    peak_scale: float = 2.0,
    min_overlap_rt: float = 0.02,
) -> bool:
    """
    后验窗口 [ilo,ihi] 在 RT 上与同伴预测框重叠足够长，且重叠索引上强度仍明显高于噪声阈值时，
    视为外推撞入其他峰，拒绝在该点截停。
    """
    if not peer_intervals:
        return False
    ilo, ihi = int(ilo), int(ihi)
    if ihi < ilo:
        ilo, ihi = ihi, ilo
    rlo, rhi = float(rt[ilo]), float(rt[ihi])
    if rlo > rhi:
        rlo, rhi = rhi, rlo
    thr = float(thr)
    ps = float(max(peak_scale, 1.0))
    mor = float(max(min_overlap_rt, 0.0))
    for plo, phi in peer_intervals:
        plo, phi = float(plo), float(phi)
        if not (np.isfinite(plo) and np.isfinite(phi) and phi > plo):
            continue
        ol, oh = max(rlo, plo), min(rhi, phi)
        if oh - ol < mor:
            continue
        for k in range(ilo, ihi + 1):
            if k < 0 or k >= rt.size:
                continue
            rk = float(rt[k])
            if ol <= rk <= oh and float(y[k]) > thr * ps:
                return True
    return False


def walk_interval_left_to_noise_with_posterior(
    y: np.ndarray,
    rt: np.ndarray,
    i_start: int,
    threshold: float,
    lookahead: int,
    mean_scale: float,
    peer_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    peer_thr_scale: float = 2.0,
    peer_min_overlap_rt: float = 0.02,
    peak_idx: Optional[int] = None,
    flat_triplet_step_frac: Optional[float] = None,
) -> float:
    """
    从 i_start 向左（索引减小）外推：先到 y<=threshold，再要求外向 lookahead 个点均值仍处低水平；
    避免单点噪声下穿过早停。若同伴框重叠且强度仍高则继续外推。
    lookahead<=0 时退化为原逻辑（仅阈值）。
    peak_idx + flat_triplet_step_frac：左肩三连点相邻差分均很小时早停到靠峰侧第一点。
    """
    y = np.asarray(y, dtype=np.float64)
    rt = np.asarray(rt, dtype=np.float64)
    thr = float(threshold)
    i = int(i_start)
    n = int(y.size)
    la = int(lookahead)
    ms = float(max(mean_scale, 1.0))
    peers = list(peer_intervals) if peer_intervals else []
    pk = int(peak_idx) if peak_idx is not None else -1

    if la <= 0:
        while i > 0 and float(y[i]) > thr:
            early = _flat_triplet_early_stop_left_rt(y, rt, i, n, pk, flat_triplet_step_frac)
            if early is not None:
                return early
            i -= 1
        return float(rt[i])

    for _ in range(max(n + 2, 8)):
        while i > 0 and float(y[i]) > thr:
            early = _flat_triplet_early_stop_left_rt(y, rt, i, n, pk, flat_triplet_step_frac)
            if early is not None:
                return early
            i -= 1
        lo = max(0, i - la + 1)
        seg = y[lo : i + 1]
        mean_ok = float(np.mean(seg)) <= thr * ms
        peer_bad = False
        if peers and mean_ok:
            peer_bad = _posterior_peer_conflict(
                y, rt, lo, i, peers, thr, peak_scale=peer_thr_scale, min_overlap_rt=peer_min_overlap_rt
            )
        if mean_ok and not peer_bad:
            return float(rt[i])
        if i <= 0:
            return float(rt[0])
        i -= 1
    return float(rt[i])


def walk_interval_right_to_noise_with_posterior(
    y: np.ndarray,
    rt: np.ndarray,
    i_start: int,
    threshold: float,
    lookahead: int,
    mean_scale: float,
    peer_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    peer_thr_scale: float = 2.0,
    peer_min_overlap_rt: float = 0.02,
    peak_idx: Optional[int] = None,
    flat_triplet_step_frac: Optional[float] = None,
) -> float:
    """从 i_start 向右外推，后验逻辑同左；右肩三连微降早停对称。"""
    y = np.asarray(y, dtype=np.float64)
    rt = np.asarray(rt, dtype=np.float64)
    thr = float(threshold)
    i = int(i_start)
    n = int(y.size)
    la = int(lookahead)
    ms = float(max(mean_scale, 1.0))
    peers = list(peer_intervals) if peer_intervals else []
    pk = int(peak_idx) if peak_idx is not None else -1

    if la <= 0:
        while i < n - 1 and float(y[i]) > thr:
            early = _flat_triplet_early_stop_right_rt(y, rt, i, n, pk, flat_triplet_step_frac)
            if early is not None:
                return early
            i += 1
        return float(rt[i])

    for _ in range(max(n + 2, 8)):
        while i < n - 1 and float(y[i]) > thr:
            early = _flat_triplet_early_stop_right_rt(y, rt, i, n, pk, flat_triplet_step_frac)
            if early is not None:
                return early
            i += 1
        hi = min(n - 1, i + la - 1)
        seg = y[i : hi + 1]
        mean_ok = float(np.mean(seg)) <= thr * ms
        peer_bad = False
        if peers and mean_ok:
            peer_bad = _posterior_peer_conflict(
                y, rt, i, hi, peers, thr, peak_scale=peer_thr_scale, min_overlap_rt=peer_min_overlap_rt
            )
        if mean_ok and not peer_bad:
            return float(rt[i])
        if i >= n - 1:
            return float(rt[n - 1])
        i += 1
    return float(rt[i])


def adjust_first_round_interval(
    rt_array,
    intensity_row,
    rt_min,
    rt_max,
    rt_lo,
    rt_hi,
    min_secondary_ratio=0.05,
    edge_max_span_min=0.50,
    edge_noise_percentile=25.0,
    boundary_posterior_lookahead: int = 5,
    boundary_posterior_mean_scale: float = 1.25,
    peer_rt_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    boundary_peer_thr_scale: float = 2.0,
    boundary_peer_min_overlap_rt: float = 0.02,
    edge_noise_stop_mode: str = "roi_bottom_decile_mean",
    pred_width_anchor: Optional[Tuple[float, float]] = None,
    width_max_expand_vs_pred: float = 1.08,
    width_max_frac_of_roi: float = 0.45,
    flat_triplet_step_frac: Optional[float] = None,
):
    """
    将边界移动停止条件改为：沿移动方向逐点移动，直到首次 <= 单侧噪声阈值即停。
    roi_bottom_decile_mean：全 ROI 强度最低约 10% 的点取均值作为双侧同一阈值；
    stable_tail_mean：外侧 RT 尾区低波动噪声均值。
    edge_noise_stop_mode=low_percentile 时使用原单侧低分位估计。

    后验（默认开启）：在首次满足 y<=阈值 的截停候选处，再要求沿外推方向共 lookahead 个点的
    平均值不超过 threshold * mean_scale，以抑制单点下穿的过早停止；若与同伴预测区间重叠且
    重叠段强度仍明显高于阈值，则继续外推，避免边框吃进相邻预测峰。
    boundary_posterior_lookahead<=0 时关闭后验，行为与旧版一致。
    返回 (rt_min_adj, rt_max_adj)
    """
    rt = np.asarray(rt_array, dtype=np.float64)
    intensity = np.asarray(intensity_row, dtype=np.float64)
    mask_roi = (rt >= rt_lo) & (rt <= rt_hi)
    if np.sum(mask_roi) < 5:
        return rt_min, rt_max

    rt_roi = rt[mask_roi]
    int_roi = np.maximum(np.asarray(intensity[mask_roi], dtype=np.float64), 0.0)
    # 锚定峰顶：优先在模型预测框 [rt_min, rt_max] 内取最高点；否则 ROI 内全局最高。
    # 同一 ROI 内若有相邻过渡更高时，全局 argmax 会把行走锚点移到邻峰，导致主峰区间整体错位。
    mask_pred = (rt_roi >= float(rt_min)) & (rt_roi <= float(rt_max))
    if np.any(mask_pred):
        masked = int_roi.astype(np.float64, copy=True)
        masked[~mask_pred] = -np.inf
        pj = int(np.argmax(masked))
        peak_idx = pj if np.isfinite(masked[pj]) else int(np.argmax(int_roi))
    else:
        peak_idx = int(np.argmax(int_roi))
    peak_rt = float(rt_roi[peak_idx])
    if edge_noise_stop_mode == "low_percentile":
        baseline_left = one_sided_low_noise_baseline(
            rt_roi, int_roi, peak_rt, True,
            max_span_min=edge_max_span_min, noise_percentile=edge_noise_percentile,
        )
        baseline_right = one_sided_low_noise_baseline(
            rt_roi, int_roi, peak_rt, False,
            max_span_min=edge_max_span_min, noise_percentile=edge_noise_percentile,
        )
    elif edge_noise_stop_mode == "roi_bottom_decile_mean":
        thr_g = roi_full_low_decile_mean_intensity(int_roi, bottom_frac=0.10)
        baseline_left = thr_g
        baseline_right = thr_g
    elif edge_noise_stop_mode == "stable_tail_mean":
        baseline_left = one_sided_edge_stop_threshold_stable_tail_mean(
            rt_roi, int_roi, peak_rt, True,
            max_span_min=edge_max_span_min,
        )
        baseline_right = one_sided_edge_stop_threshold_stable_tail_mean(
            rt_roi, int_roi, peak_rt, False,
            max_span_min=edge_max_span_min,
        )
    else:
        baseline_left = one_sided_edge_stop_threshold_stable_tail_mean(
            rt_roi, int_roi, peak_rt, True,
            max_span_min=edge_max_span_min,
        )
        baseline_right = one_sided_edge_stop_threshold_stable_tail_mean(
            rt_roi, int_roi, peak_rt, False,
            max_span_min=edge_max_span_min,
        )
    y_threshold_left = float(baseline_left)
    y_threshold_right = float(baseline_right)

    rt_min_adj, rt_max_adj = rt_min, rt_max

    def _nearest_idx(x: np.ndarray, v: float) -> int:
        return int(np.argmin(np.abs(x - float(v))))

    la = int(boundary_posterior_lookahead) if boundary_posterior_lookahead is not None else 0
    ms = float(boundary_posterior_mean_scale) if boundary_posterior_mean_scale is not None else 1.25
    peers = list(peer_rt_intervals) if peer_rt_intervals else None
    pts = float(boundary_peer_thr_scale)
    pmor = float(boundary_peer_min_overlap_rt)

    y_left = float(np.interp(rt_min, rt_roi, int_roi))
    if y_left > y_threshold_left and rt_min < rt_roi[peak_idx]:
        i_left = _nearest_idx(rt_roi, rt_min)
        rt_min_adj = walk_interval_left_to_noise_with_posterior(
            int_roi,
            rt_roi,
            i_left,
            y_threshold_left,
            la,
            ms,
            peers,
            peer_thr_scale=pts,
            peer_min_overlap_rt=pmor,
            peak_idx=peak_idx,
            flat_triplet_step_frac=flat_triplet_step_frac,
        )

    y_right = float(np.interp(rt_max, rt_roi, int_roi))
    if y_right > y_threshold_right and rt_max > rt_roi[peak_idx]:
        i_right = _nearest_idx(rt_roi, rt_max)
        rt_max_adj = walk_interval_right_to_noise_with_posterior(
            int_roi,
            rt_roi,
            i_right,
            y_threshold_right,
            la,
            ms,
            peers,
            peer_thr_scale=pts,
            peer_min_overlap_rt=pmor,
            peak_idx=peak_idx,
            flat_triplet_step_frac=flat_triplet_step_frac,
        )

    rt_min_adj = max(float(rt_min_adj), rt_lo)
    rt_max_adj = min(float(rt_max_adj), rt_hi)
    if rt_min_adj >= rt_max_adj:
        return rt_min, rt_max
    if pred_width_anchor is not None:
        pl, ph = pred_width_anchor
        if np.isfinite(pl) and np.isfinite(ph) and ph > pl:
            rt_min_adj, rt_max_adj = clamp_refined_interval_width_to_pred_and_roi(
                float(rt_min_adj),
                float(rt_max_adj),
                float(pl),
                float(ph),
                float(rt_lo),
                float(rt_hi),
                max_expand_vs_pred=float(width_max_expand_vs_pred),
                max_frac_of_roi=float(width_max_frac_of_roi),
            )
            rt_min_adj = max(float(rt_min_adj), rt_lo)
            rt_max_adj = min(float(rt_max_adj), rt_hi)
    if rt_min_adj >= rt_max_adj:
        return rt_min, rt_max
    return rt_min_adj, rt_max_adj


def remove_overlap_from_second_interval(rt_min_1, rt_max_1, rt_min_2, rt_max_2, min_width=0.01):
    """
    改进二：若第二次区间与第一次重叠，优先第一次，从第二次区间去除重叠部分。
    取非重叠部分中较宽的一段；若去除后为空或过小，返回 (nan, nan)。
    """
    if pd.isna(rt_min_2) or pd.isna(rt_max_2):
        return np.nan, np.nan
    overlap_lo = max(rt_min_1, rt_min_2)
    overlap_hi = min(rt_max_1, rt_max_2)
    if overlap_lo >= overlap_hi:
        return rt_min_2, rt_max_2

    left_seg = (rt_min_2, rt_min_1) if rt_min_2 < rt_min_1 else (np.nan, np.nan)
    right_seg = (rt_max_1, rt_max_2) if rt_max_2 > rt_max_1 else (np.nan, np.nan)
    left_width = left_seg[1] - left_seg[0] if not np.isnan(left_seg[0]) else 0
    right_width = right_seg[1] - right_seg[0] if not np.isnan(right_seg[0]) else 0
    if left_width >= min_width and left_width >= right_width:
        return left_seg[0], left_seg[1]
    if right_width >= min_width:
        return right_seg[0], right_seg[1]
    return np.nan, np.nan


def images_with_multi_highconf_boxes(pred_df, min_confidence: float) -> Set[str]:
    """
    同一张 ROI 图像若已有 >=2 行 score>=min_confidence，则认为首轮模型已给出多框，
    round2_inference=auto 时可跳过二轮推理，直接从首轮合并两框。
    """
    if pred_df is None or pred_df.empty or "image" not in pred_df.columns:
        return set()
    sc_col = pred_df["score"] if "score" in pred_df.columns else pred_df.get("Score")
    if sc_col is None:
        return set()
    img = pred_df["image"].astype(str).str.strip()
    out = set()
    for name, grp in pred_df.groupby(img, sort=False):
        scg = pd.to_numeric(grp["score"] if "score" in grp.columns else grp["Score"], errors="coerce")
        if int((scg >= float(min_confidence)).sum()) >= 2:
            out.add(str(name).strip())
    return out


def _pick_second_box_row_from_round1(df1: pd.DataFrame, orig_image: str, r1: dict, min_confidence: float):
    """从首轮 prediction 中为同一 image 选另一高置信行作为第二框。"""
    im = str(orig_image).strip()
    sub = df1[df1["image"].astype(str).str.strip() == im].copy()
    if sub.empty:
        return None
    sc = pd.to_numeric(sub["score"] if "score" in sub.columns else sub["Score"], errors="coerce")
    sub = sub.loc[sc >= float(min_confidence)]
    if len(sub) < 2:
        return None
    t1_lo = float(r1.get("rt_min", np.nan))
    t1_hi = float(r1.get("rt_max", np.nan))

    def _near_same_interval(row):
        a, b = float(row.get("rt_min", np.nan)), float(row.get("rt_max", np.nan))
        if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(t1_lo) or not np.isfinite(t1_hi):
            return False
        mid1 = 0.5 * (t1_lo + t1_hi)
        mid2 = 0.5 * (a + b)
        if abs(mid1 - mid2) < 1e-6 and abs((b - a) - (t1_hi - t1_lo)) < 1e-6:
            return True
        return False

    others = sub[~sub.apply(_near_same_interval, axis=1)]
    if others.empty:
        return None
    sc2 = pd.to_numeric(others["score"] if "score" in others.columns else others["Score"], errors="coerce")
    return others.loc[sc2.idxmax()]


def filter_candidates_for_second_round(
    pred_df,
    rt_array,
    intensity_matrix,
    roi_windows,
    min_confidence=0.99,
    min_snr=3.0,
    min_secondary_ratio=0.05,
    noise_barrier_ratio=0.5,
):
    """
    筛选进入第二轮的候选：score>=min_confidence，SNR>=min_snr，存在次峰>=min_secondary_ratio。
    返回 [(row, xic_idx), ...]
    """
    candidates = []
    for _, row in pred_df.iterrows():
        score = row.get("score", row.get("Score", 0))
        if pd.isna(score) or float(score) < min_confidence:
            continue
        image_name = str(row.get("image", "")).strip()
        if not image_name:
            continue
        rt_min = row.get("rt_min")
        rt_max = row.get("rt_max")
        if pd.isna(rt_min) or pd.isna(rt_max):
            continue
        rt_min, rt_max = float(rt_min), float(rt_max)
        if rt_min >= rt_max:
            continue

        idx = _image_to_compound_index(image_name)
        if idx is None:
            idx = int(row.get("compound_name", 1)) - 1
        if idx < 0 or idx >= intensity_matrix.shape[0]:
            continue

        intensity_row = intensity_matrix[idx, :].astype(np.float64)
        snr = compute_snr_outside_box(rt_array, intensity_row, rt_min, rt_max)
        if np.isnan(snr) or snr < min_snr:
            continue

        rt_lo, rt_hi = roi_windows.get(image_name, (float(np.min(rt_array)), float(np.max(rt_array))))
        if isinstance(rt_lo, (list, tuple)):
            rt_lo, rt_hi = rt_lo[0], rt_hi[1]
        has_sec, _, _ = has_secondary_peak_in_roi(
            rt_array, intensity_row, rt_min, rt_max, float(rt_lo), float(rt_hi),
            min_secondary_ratio=min_secondary_ratio,
            noise_barrier_ratio=noise_barrier_ratio,
        )
        if not has_sec:
            continue

        candidates.append((row, idx))
    return candidates


def build_masked_subdir_for_candidates(
    subdir,
    images_root,
    candidates,
    xic_full,
    roi_windows,
    out_dir,
    mask_method="random_noise",
    smooth_sigma=1.0,
    seed=42,
    min_secondary_ratio=0.05,
):
    """
    仅为候选图像生成掩蔽图。candidates 每项为 (row, xic_idx) 或 (row, xic_idx, rt_min_adj, rt_max_adj)。
    若有 rt_min_adj/rt_max_adj（改进三、四调整后），则用其掩蔽。
    """
    np.random.seed(seed)
    rt_row = xic_full[0, :].astype(np.float64)
    if np.nanmax(rt_row) > 200:
        rt_row = rt_row / 60.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_path = subdir / "feature.csv"
    if not feature_path.exists():
        alt = Path(images_root) / "feature.csv"
        if alt.exists():
            feature_path = alt
    feature_df = pd.read_csv(feature_path) if feature_path.exists() else pd.DataFrame()

    new_feature_rows = []
    new_xic_rows = [rt_row]
    new_roi_rows = []
    orig_to_new = {}

    for new_idx, item in enumerate(candidates):
        if len(item) >= 4:
            row, xic_idx, rt_min, rt_max = item[0], item[1], item[2], item[3]
        else:
            row, xic_idx = item[0], item[1]
            rt_min = float(row["rt_min"])
            rt_max = float(row["rt_max"])
        image_name = str(row.get("image", "")).strip()
        compound_name = row.get("compound_name", xic_idx + 1)

        mz = (feature_df.iloc[xic_idx]["mz"] if xic_idx < len(feature_df) and "mz" in feature_df.columns
              else row.get("mz", np.nan))
        q3 = (feature_df.iloc[xic_idx]["q3"] if xic_idx < len(feature_df) and "q3" in feature_df.columns
              else row.get("q3", np.nan))
        try:
            mz_val = float(mz) if pd.notna(mz) else np.nan
            new_name = f"{new_idx + 1}_mz{mz_val:.4f}.jpeg" if pd.notna(mz_val) else f"{new_idx + 1}_mznan.jpeg"
        except (TypeError, ValueError):
            new_name = f"{new_idx + 1}_mznan.jpeg"

        orig_to_new[image_name] = new_name

        intensity = xic_full[xic_idx + 1, :].astype(np.float64).copy()
        rt_lo, rt_hi = roi_windows.get(image_name, (rt_row.min(), rt_row.max()))
        if isinstance(rt_lo, (list, tuple)):
            rt_lo, rt_hi = rt_lo[0], rt_hi[1]
        rt_lo, rt_hi = float(rt_lo), float(rt_hi)

        out_path = out_dir / new_name
        intensity_masked = mask_main_peak_and_redraw(
            rt_row, intensity, rt_min, rt_max, rt_lo, rt_hi, str(out_path),
            mask_method=mask_method,
            smooth_sigma=smooth_sigma,
            use_last_25pct_noise=True,
        )
        new_xic_rows.append(intensity_masked)

        new_feature_rows.append({
            "Compound Name": new_idx + 1,
            "mz": mz,
            "q3": q3,
            "RT": (rt_min + rt_max) / 2,
        })
        new_roi_rows.append({"image": new_name, "rt_lo": rt_lo, "rt_hi": rt_hi})

    if not new_feature_rows:
        return 0, {}

    pd.DataFrame(new_feature_rows).to_csv(out_dir / "feature.csv", index=False)
    xic_new = np.vstack([new_xic_rows[0], np.array(new_xic_rows[1:])])
    np.save(out_dir / "xic_matrix.npy", xic_new)
    pd.DataFrame(new_roi_rows).to_csv(out_dir / "roi_windows.csv", index=False)

    return len(candidates), orig_to_new


def run_newtest_on_dir(images_path, model_path, output_dir, threshold=0.99):
    """对目录运行 newtest，输出 prediction.csv 到 output_dir。"""
    pred_out = Path(output_dir) / "prediction.csv"
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "newtest.py"),
        "--images_path", str(images_path),
        "--model", str(model_path),
        "--prediction_output", str(pred_out),
        "--threshold", str(threshold),
    ]
    r = subprocess.run(cmd, cwd=str(ROOT_DIR))
    return r.returncode == 0, pred_out


def merge_to_newprediction(
    pred_round1_path,
    pred_round2_path,
    candidates_info,
    orig_to_new_map,
    output_path,
    *,
    min_confidence: float = 0.99,
    round2_skipped_images: Optional[Set[str]] = None,
):
    """
    合并两轮结果。candidates_info: [(orig_image, row_round1), ...]
    orig_to_new_map: {orig_image: new_image} 用于从 round2 找对应行。
    round2_skipped_images: auto 模式下跳过二轮推理的图，第二框从首轮另一行填充。
    """
    df1 = pd.read_csv(pred_round1_path)
    df2 = pd.read_csv(pred_round2_path) if Path(pred_round2_path).exists() else pd.DataFrame()

    new_to_orig = {v: k for k, v in orig_to_new_map.items()}
    skip = round2_skipped_images or set()

    rows = []
    for orig_image, r1 in candidates_info:
        rec = {
            "image": orig_image,
            "compound_name": r1.get("compound_name"),
            "mz": r1.get("mz"),
            "q3": r1.get("q3"),
            "rt_min_1": r1.get("rt_min"),
            "rt_max_1": r1.get("rt_max"),
            "score_1": r1.get("score"),
            "rt_min_2": np.nan,
            "rt_max_2": np.nan,
            "score_2": np.nan,
        }
        new_name = orig_to_new_map.get(orig_image)
        filled = False
        if new_name and not df2.empty and "image" in df2.columns:
            match = df2[df2["image"] == new_name]
            if not match.empty:
                r2 = match.iloc[0]
                rt_min_2 = r2.get("rt_min")
                rt_max_2 = r2.get("rt_max")
                rt_min_1 = rec["rt_min_1"]
                rt_max_1 = rec["rt_max_1"]
                if pd.notna(rt_min_1) and pd.notna(rt_max_1):
                    rt_min_2, rt_max_2 = remove_overlap_from_second_interval(
                        float(rt_min_1), float(rt_max_1),
                        float(rt_min_2) if pd.notna(rt_min_2) else np.nan,
                        float(rt_max_2) if pd.notna(rt_max_2) else np.nan,
                    )
                rec["rt_min_2"] = rt_min_2
                rec["rt_max_2"] = rt_max_2
                rec["score_2"] = r2.get("score")
                filled = True
        if not filled and str(orig_image).strip() in skip:
            r2r = _pick_second_box_row_from_round1(df1, orig_image, dict(r1), min_confidence)
            if r2r is not None:
                rt_min_2 = float(r2r.get("rt_min", np.nan))
                rt_max_2 = float(r2r.get("rt_max", np.nan))
                rt_min_1 = rec["rt_min_1"]
                rt_max_1 = rec["rt_max_1"]
                if pd.notna(rt_min_1) and pd.notna(rt_max_1) and np.isfinite(rt_min_2) and np.isfinite(rt_max_2):
                    rt_min_2, rt_max_2 = remove_overlap_from_second_interval(
                        float(rt_min_1), float(rt_max_1), rt_min_2, rt_max_2,
                    )
                rec["rt_min_2"] = rt_min_2
                rec["rt_max_2"] = rt_max_2
                rec["score_2"] = r2r.get("score")
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    return df_out


def plot_newprediction_on_xic(
    newpred_path,
    xic_path,
    roi_windows_path,
    output_plot_dir,
    smooth_sigma=1.0,
):
    """在原 XIC 平滑图上绘制 1 或 2 个区间框及置信度。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(newpred_path)
    xic_full = np.load(xic_path)
    rt_row = xic_full[0, :].astype(np.float64)
    if np.nanmax(rt_row) > 200:
        rt_row = rt_row / 60.0
    intensity_matrix = xic_full[1:, :]

    roi_df = pd.read_csv(roi_windows_path) if Path(roi_windows_path).exists() else pd.DataFrame()
    roi_map = {}
    if not roi_df.empty and "image" in roi_df.columns:
        for _, r in roi_df.iterrows():
            roi_map[str(r["image"]).strip()] = (float(r["rt_lo"]), float(r["rt_hi"]))

    Path(output_plot_dir).mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        image_name = str(row.get("image", "")).strip()
        idx = _image_to_compound_index(image_name)
        if idx is None:
            idx = int(row.get("compound_name", 1)) - 1
        if idx < 0 or idx >= intensity_matrix.shape[0]:
            continue

        intensity = intensity_matrix[idx, :].astype(np.float64)
        if smooth_sigma > 0 and intensity.size >= 10:
            intensity = gaussian_filter1d(intensity, sigma=smooth_sigma, mode="nearest")

        rt_lo, rt_hi = roi_map.get(image_name, (float(rt_row.min()), float(rt_row.max())))
        mask = (rt_row >= rt_lo) & (rt_row <= rt_hi)
        plot_rt = rt_row[mask]
        plot_int = intensity[mask]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(plot_rt, plot_int, "b-", linewidth=1.5, label="XIC (smoothed)")

        boxes = []
        rt_min_1 = row.get("rt_min_1")
        rt_max_1 = row.get("rt_max_1")
        score_1 = row.get("score_1")
        if pd.notna(rt_min_1) and pd.notna(rt_max_1):
            boxes.append((float(rt_min_1), float(rt_max_1), float(score_1) if pd.notna(score_1) else 0, "round1"))
        rt_min_2 = row.get("rt_min_2")
        rt_max_2 = row.get("rt_max_2")
        score_2 = row.get("score_2")
        if pd.notna(rt_min_2) and pd.notna(rt_max_2) and float(rt_max_2) > float(rt_min_2):
            boxes.append((float(rt_min_2), float(rt_max_2), float(score_2) if pd.notna(score_2) else 0, "round2"))

        y_max = float(np.max(plot_int)) if plot_int.size > 0 else 1.0
        colors = ["red", "green"]
        for i, (rmin, rmax, sc, lbl) in enumerate(boxes):
            ax.axvspan(rmin, rmax, alpha=0.2, color=colors[i % 2])
            ax.axvline(rmin, color=colors[i % 2], linestyle="--", linewidth=1)
            ax.axvline(rmax, color=colors[i % 2], linestyle="--", linewidth=1)
            ax.text(rmin, y_max * 0.98, f"{lbl} {sc:.2f}", fontsize=9, color=colors[i % 2])

        ax.set_xlim(rt_lo, rt_hi)
        ax.set_xlabel("Retention Time (min)")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_png = Path(output_plot_dir) / image_name.replace(".jpeg", "_newpred.png").replace(".jpg", "_newpred.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="两轮识别：筛选→掩蔽→第二轮→合并→可视化")
    parser.add_argument("--batch_predictions", type=str, default="results/batch_predictions")
    parser.add_argument("--images_root", type=str, default="xic-roi-batch")
    parser.add_argument("--model", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output_base", type=str, default="results/two_round",
                        help="输出根目录：masked/ round2/ newprediction/ plots/")
    parser.add_argument("--min_confidence", type=float, default=0.99)
    parser.add_argument("--min_snr", type=float, default=3.0)
    parser.add_argument("--min_secondary_ratio", type=float, default=0.05)
    parser.add_argument("--noise_barrier_ratio", type=float, default=0.5,
                        help="噪声阻碍：后25%%平均噪声×此系数加入次峰阈值，噪声越大所需峰高越高")
    parser.add_argument("--mask_method", type=str, default="random_noise",
                        choices=["random_noise", "linear_interp", "baseline_interp"])
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plot", action="store_true", help="不生成 XIC 标注图")
    parser.add_argument(
        "--round2_inference",
        type=str,
        default="always",
        choices=["always", "auto"],
        help="auto: 若首轮对同一 ROI 图已有>=2条高置信框，则跳过二轮 newtest，第二框从首轮合并",
    )
    args = parser.parse_args()

    batch_path = Path(args.batch_predictions).resolve()
    images_root = Path(args.images_root).resolve()
    output_base = Path(args.output_base).resolve()
    masked_dir = output_base / "masked"
    round2_dir = output_base / "round2"
    newpred_dir = output_base / "newprediction"
    plot_dir = output_base / "plots"

    if not batch_path.is_dir():
        print(f"[ERROR] batch_predictions 不存在: {batch_path}")
        return

    subdirs = sorted([d for d in batch_path.iterdir() if d.is_dir()])
    total_candidates = 0

    for subdir in subdirs:
        conc_name = subdir.name
        pred_path = subdir / "prediction.csv"
        xic_path = subdir / "xic_matrix.npy"
        if not xic_path.exists():
            xic_path = images_root / conc_name / "xic_matrix.npy"
        if not pred_path.exists() or not xic_path.exists():
            print(f"[WARN] 跳过 {conc_name}: 缺少 prediction.csv 或 xic_matrix.npy")
            continue

        roi_windows = load_roi_windows(subdir)
        if not roi_windows:
            roi_windows = load_roi_windows(images_root / conc_name)
        if not roi_windows:
            roi_windows = load_roi_windows(subdir)

        xic_full = np.load(str(xic_path))
        rt_array = xic_full[0, :].astype(np.float64)
        if np.nanmax(rt_array) > 200:
            rt_array = rt_array / 60.0
        intensity_matrix = xic_full[1:, :]

        pred_df = pd.read_csv(pred_path)
        candidates = filter_candidates_for_second_round(
            pred_df, rt_array, intensity_matrix, roi_windows,
            min_confidence=args.min_confidence,
            min_snr=args.min_snr,
            min_secondary_ratio=args.min_secondary_ratio,
            noise_barrier_ratio=args.noise_barrier_ratio,
        )

        if not candidates:
            print(f"[INFO] {conc_name}: 无候选（需 score>={args.min_confidence}, SNR>={args.min_snr}, 次峰>={args.min_secondary_ratio}）")
            continue

        adapted_candidates = []
        for row, idx in candidates:
            rt_lo, rt_hi = roi_windows.get(str(row["image"]).strip(), (float(np.min(rt_array)), float(np.max(rt_array))))
            if isinstance(rt_lo, (list, tuple)):
                rt_lo, rt_hi = rt_lo[0], rt_hi[1]
            rt_min_adj, rt_max_adj = adjust_first_round_interval(
                rt_array, intensity_matrix[idx, :], float(row["rt_min"]), float(row["rt_max"]),
                float(rt_lo), float(rt_hi), min_secondary_ratio=args.min_secondary_ratio,
            )
            adapted_candidates.append((row, idx, rt_min_adj, rt_max_adj))

        print(f"[INFO] {conc_name}: {len(adapted_candidates)} 个候选进入第二轮")

        multi_img = images_with_multi_highconf_boxes(pred_df, args.min_confidence)
        round2_skipped: Set[str] = set()
        for_mask = adapted_candidates
        if getattr(args, "round2_inference", "always") == "auto":
            for_mask = []
            for item in adapted_candidates:
                row0 = item[0]
                imn = str(row0.get("image", "")).strip()
                if imn in multi_img:
                    round2_skipped.add(imn)
                else:
                    for_mask.append(item)
            if round2_skipped:
                print(f"[INFO] {conc_name}: round2_inference=auto 跳过 {len(round2_skipped)} 张（首轮已多框）")
            if not for_mask and adapted_candidates:
                print(f"[INFO] {conc_name}: 全部候选跳过二轮推理，仅合并首轮双框")

        out_subdir = masked_dir / conc_name
        n_built, orig_to_new = build_masked_subdir_for_candidates(
            subdir, images_root / conc_name if (images_root / conc_name).is_dir() else subdir,
            for_mask, xic_full, roi_windows, out_subdir,
            mask_method=args.mask_method,
            smooth_sigma=args.smooth_sigma,
            seed=args.seed,
            min_secondary_ratio=args.min_secondary_ratio,
        )
        total_candidates += n_built

        round2_subdir = round2_dir / conc_name
        pred2_path = None
        if for_mask:
            ok, pred2_path = run_newtest_on_dir(
                str(out_subdir), args.model, str(round2_subdir), threshold=args.min_confidence
            )
            if not ok:
                print(f"[WARN] {conc_name}: 第二轮 newtest 失败")
                pred2_path = None
        else:
            print(f"[INFO] {conc_name}: 无掩蔽任务，跳过第二轮 newtest")

        candidates_info = []
        for row, idx, rt_min_adj, rt_max_adj in adapted_candidates:
            r1 = dict(row)
            r1["rt_min"] = rt_min_adj
            r1["rt_max"] = rt_max_adj
            candidates_info.append((str(row["image"]).strip(), r1))
        newpred_path = newpred_dir / conc_name / "newprediction.csv"
        merge_to_newprediction(
            pred_path, pred2_path if pred2_path else "",
            candidates_info, orig_to_new, str(newpred_path),
            min_confidence=float(args.min_confidence),
            round2_skipped_images=round2_skipped if round2_skipped else None,
        )
        print(f"[OK] {conc_name}: newprediction.csv -> {newpred_path}")

        if not args.no_plot:
            plot_subdir = plot_dir / conc_name
            xic_src = subdir / "xic_matrix.npy"
            if not xic_src.exists():
                xic_src = images_root / conc_name / "xic_matrix.npy"
            roi_src = subdir / "roi_windows.csv"
            if not roi_src.exists():
                roi_src = images_root / conc_name / "roi_windows.csv"
            if xic_src.exists():
                plot_newprediction_on_xic(
                    newpred_path, xic_src, roi_src, str(plot_subdir),
                    smooth_sigma=args.smooth_sigma,
                )
                print(f"[OK] {conc_name}: 标注图 -> {plot_subdir}")

    print(f"[DONE] 共 {total_candidates} 个候选完成两轮识别，结果见 {output_base}")


if __name__ == "__main__":
    main()
