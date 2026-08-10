# -*- coding: utf-8 -*-
"""
Provide valley-split utilities for `run_valley_split_from_predictions.py`.

双峰/谷拆分策略（对齐 ROI 内 XIC 段，减少误拆与漏拆）：

1. **默认** `use_scipy_prominence=True`：`scipy.signal.find_peaks` + **prominence**
   相对动态范围，抑制单峰上升/下降沿上的噪声小鼓包（避免「顶点附近误拆成两框」）。
2. **只考虑沿 RT 排序后的相邻峰对**（不再用「全段最高两个局部极大」），避免主峰 + 远处噪声
   被配成一对而漏掉中间真双峰。
3. **谷点居中**：argmin 谷在两峰索引间距中的相对位置须在 (min_valley_central_frac, 1-...)，
   避免谷落在某一峰脚（单峰被硬切成两半的常见情形）。
4. 仍保留与 analyze_double_peaks 类似的对比度、谷深、valley_ratio、次峰/主峰比等 gate。
5. `use_scipy_prominence=False` 时回退旧版「局部极大 + 最高两峰」，便于对照。

可通过 `ValleySplitParams` / 命令行 `--vs_*` 调整。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from utils.roi_rt_mapping import box_to_rt_range, rt_to_pixel_x, rt_window_bounds_minutes


# ---------------------------------------------------------------------------
# 参数集：与 analyze_double_peaks.py 顶部常量对应关系见 docstring / print_valley_split_param_help
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValleySplitParams:
    """峰谷拆分 gate：全部不通过则保持单框。"""

    smooth_sigma: float = 1.0
    # dynamic / max_val：低对比度不单拆（同 MIN_CONTRAST_RATIO）
    min_contrast_ratio: float = 0.08
    # 峰高阈值 = baseline + min_peak_ratio * dynamic（同 MIN_PEAK_RATIO）
    min_peak_ratio: float = 0.06
    # 谷须低于 min(两峰) * valley_ratio（同 VALLEY_RATIO；越小越严）
    valley_ratio: float = 0.90
    # 谷深 = min(两峰)-谷，须 >= min_valley_depth_ratio * dynamic（同 MIN_VALLEY_DEPTH_RATIO）
    min_valley_depth_ratio: float = 0.06
    min_peak_gap: int = 3
    max_peak_gap_ratio: float = 0.92
    # 较小峰 / 较大峰 >= 该值；0 表示不启用（analyze 无此项，更严时可 0.12~0.25）
    min_secondary_to_primary_ratio: float = 0.08
    min_segment_points: int = 12
    box_split_margin_px: float = 0.5
    # --- prominence + 相邻峰对（默认开启，抗误拆/漏拆）---
    use_scipy_prominence: bool = True
    # 最小 prominence ≈ 该比例 * dynamic；略大则更少误拆单峰
    prominence_ratio: float = 0.055
    # find_peaks distance ≈ max(min_peak_gap, ratio * N)
    peak_min_distance_ratio: float = 0.028
    # 参与配对的峰数量上限（按 prominence 排序后取前 K，再按 RT 排序做相邻对）
    max_peaks_for_pairing: int = 8
    # 谷点相对位置 v/(i_hi-i_lo) 须落在 (c, 1-c)；过小易在单峰上误切
    min_valley_central_frac: float = 0.12


# 与 analyze_double_peaks 默认（偏宽松、易检出双峰）一致
RELAXED_VALLEY_SPLIT_PARAMS = ValleySplitParams(
    smooth_sigma=1.2,
    min_contrast_ratio=0.05,
    min_peak_ratio=0.04,
    valley_ratio=0.92,
    min_valley_depth_ratio=0.04,
    min_peak_gap=2,
    max_peak_gap_ratio=0.92,
    min_secondary_to_primary_ratio=0.0,
    min_segment_points=12,
    box_split_margin_px=0.5,
    use_scipy_prominence=True,
    prominence_ratio=0.04,
    peak_min_distance_ratio=0.022,
    max_peaks_for_pairing=10,
    min_valley_central_frac=0.10,
)

# 推荐默认：prominence + 相邻峰 + 谷居中
DEFAULT_VALLEY_SPLIT_PARAMS = ValleySplitParams()

# 更严：弱次峰、浅谷不拆
STRICT_VALLEY_SPLIT_PARAMS = ValleySplitParams(
    smooth_sigma=1.35,
    min_contrast_ratio=0.12,
    min_peak_ratio=0.09,
    valley_ratio=0.84,
    min_valley_depth_ratio=0.11,
    min_peak_gap=5,
    max_peak_gap_ratio=0.88,
    min_secondary_to_primary_ratio=0.18,
    min_segment_points=16,
    box_split_margin_px=0.5,
    use_scipy_prominence=True,
    prominence_ratio=0.08,
    peak_min_distance_ratio=0.035,
    max_peaks_for_pairing=6,
    min_valley_central_frac=0.15,
)


def valley_params_from_preset(name: str) -> ValleySplitParams:
    n = (name or "default").strip().lower()
    if n in ("relaxed", "analyze", "analyze_default"):
        return RELAXED_VALLEY_SPLIT_PARAMS
    if n in ("strict",):
        return STRICT_VALLEY_SPLIT_PARAMS
    return DEFAULT_VALLEY_SPLIT_PARAMS


def print_valley_split_param_help() -> None:
    """与 analyze_double_peaks.py --list_params 说明风格一致。"""
    print("=" * 60)
    print("Valley-split / double-peak gate（prominence + 相邻峰对 + 谷居中 + analyze 类 gate）")
    print("=" * 60)
    for label, p in [
        ("default（较严，推荐）", DEFAULT_VALLEY_SPLIT_PARAMS),
        ("relaxed（同 analyze_double_peaks 脚本默认）", RELAXED_VALLEY_SPLIT_PARAMS),
        ("strict（更严）", STRICT_VALLEY_SPLIT_PARAMS),
    ]:
        print(f"\n--- {label} ---")
        print(p)
    print("-" * 60)
    print("漏拆（该拆没拆）-> valley_ratio↑、min_peak_ratio↓、min_valley_depth_ratio↓、smooth_sigma↓")
    print("误拆（不该拆拆了）-> prominence_ratio↑、min_valley_central_frac↑、valley_ratio↓、")
    print("                min_valley_depth_ratio↑、min_secondary_to_primary_ratio↑")
    print("漏拆（该拆没拆）-> prominence_ratio↓、smooth_sigma↓、min_valley_depth_ratio↓、")
    print("                valley_ratio↑、min_secondary_to_primary_ratio↓")
    print("=" * 60)


def _find_valley_between_two_peaks(x_seg, y_seg, peak_i_1, peak_i_2):
    """
    Find valley (argmin) between two peak indices within a segment.

    Returns:
      valley_rt (float), valley_idx (int)
    """
    if y_seg is None:
        return None, None
    peak_i_1 = int(peak_i_1)
    peak_i_2 = int(peak_i_2)
    if peak_i_1 == peak_i_2:
        return None, None
    lo = min(peak_i_1, peak_i_2)
    hi = max(peak_i_1, peak_i_2)
    if hi - lo < 3:
        return None, None
    valley_idx_rel = int(np.argmin(y_seg[lo : hi + 1])) + lo
    valley_rt = float(x_seg[valley_idx_rel])
    return valley_rt, valley_idx_rel


def _local_maxima_indices(y, min_height=None) -> List[int]:
    """Simple local maxima detector with optional minimum height."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < 3:
        return []
    peaks = []
    for i in range(1, y.size - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            if min_height is None or y[i] >= min_height:
                peaks.append(i)
    return peaks


def _smooth_segment(arr: np.ndarray, sigma: float) -> np.ndarray:
    """高斯光滑；与 analyze_double_peaks._smooth 一致。"""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < 10 or sigma <= 0:
        return arr
    s = min(float(sigma), arr.size / 25.0)
    return gaussian_filter1d(np.maximum(arr, 0.0), sigma=s, mode="nearest")


def _pick_two_dominant_peaks(
    arr: np.ndarray, baseline: float, dynamic: float, params: ValleySplitParams
) -> Optional[Tuple[int, int]]:
    """
    旧版：取强度最高的两个局部极大（沿 RT 为 i_lo < i_hi）。
    易与远处噪声峰配对或把单峰沿上两噪声当两峰，仅作 legacy 回退。
    """
    if dynamic <= 0:
        return None
    min_height = baseline + params.min_peak_ratio * dynamic
    peaks = _local_maxima_indices(arr, min_height=min_height)
    if len(peaks) < 2:
        return None
    top2 = sorted(peaks, key=lambda i: float(arr[i]), reverse=True)[:2]
    i_lo, i_hi = int(min(top2)), int(max(top2))
    return i_lo, i_hi


def _valley_central_ok(
    arr: np.ndarray, i_lo: int, i_hi: int, min_central_frac: float
) -> bool:
    """
    谷（argmin）在两峰之间的相对位置须居中，避免「谷」落在某一峰边缘（单峰误拆）。
    frac = (valley_idx - i_lo) / (i_hi - i_lo).
    """
    if i_hi <= i_lo or min_central_frac <= 0:
        return True
    span = i_hi - i_lo
    if span < 1:
        return False
    voff = int(np.argmin(arr[i_lo : i_hi + 1]))
    frac = voff / float(span)
    return min_central_frac <= frac <= 1.0 - min_central_frac


def _find_best_split_pair_prominence(
    arr: np.ndarray, baseline: float, dynamic: float, params: ValleySplitParams
) -> Optional[Tuple[int, int]]:
    """
    scipy find_peaks(prominence) → 按 prominence 取前 K 个峰 → 按 RT 排序 →
    仅尝试**相邻**峰对，选谷深最大且通过 gate 的一对。
    """
    n = len(arr)
    if n < 6 or dynamic <= 0:
        return None
    dist = max(int(params.min_peak_gap), int(round(n * float(params.peak_min_distance_ratio))))
    dist = max(1, dist)
    prom_thresh = max(1e-12, float(params.prominence_ratio) * float(dynamic))
    peaks, properties = find_peaks(arr, prominence=prom_thresh, distance=dist)
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 2:
        return None

    min_h = baseline + params.min_peak_ratio * dynamic
    cand: List[Tuple[int, float]] = []
    proms = properties.get("prominences")
    if proms is None or len(proms) != len(peaks):
        return None
    for j, p in enumerate(peaks):
        if arr[p] >= min_h:
            cand.append((int(p), float(proms[j])))
    if len(cand) < 2:
        return None

    cand.sort(key=lambda t: -t[1])
    cap = max(2, int(params.max_peaks_for_pairing))
    cand = cand[:cap]
    idx_sorted = sorted(p for p, _ in cand)

    best_pair: Optional[Tuple[int, int]] = None
    best_depth = -1.0
    for k in range(len(idx_sorted) - 1):
        i_lo, i_hi = idx_sorted[k], idx_sorted[k + 1]
        gap = i_hi - i_lo
        if gap < params.min_peak_gap:
            continue
        if gap > int(n * params.max_peak_gap_ratio):
            continue
        if not _valley_central_ok(arr, i_lo, i_hi, params.min_valley_central_frac):
            continue
        if not _pair_passes_double_peak_gate(arr, i_lo, i_hi, params):
            continue
        voff = int(np.argmin(arr[i_lo : i_hi + 1]))
        valley_idx = i_lo + voff
        valley = float(arr[valley_idx])
        depth = min(float(arr[i_lo]), float(arr[i_hi])) - valley
        if depth > best_depth:
            best_depth = depth
            best_pair = (i_lo, i_hi)
    return best_pair


def _find_best_split_pair_legacy(
    arr: np.ndarray, baseline: float, dynamic: float, params: ValleySplitParams
) -> Optional[Tuple[int, int]]:
    """旧版两峰 + 谷居中 + gate。"""
    picked = _pick_two_dominant_peaks(arr, baseline, dynamic, params)
    if picked is None:
        return None
    i_lo, i_hi = picked
    if not _valley_central_ok(arr, i_lo, i_hi, params.min_valley_central_frac):
        return None
    if not _pair_passes_double_peak_gate(arr, i_lo, i_hi, params):
        return None
    return i_lo, i_hi


def _pair_passes_double_peak_gate(
    arr: np.ndarray, i_lo: int, i_hi: int, params: ValleySplitParams
) -> bool:
    """
    对已定两峰下标执行 analyze_double_peaks 中的间距、谷深、谷位检验。
    arr 已为「去底 + 光滑」后的工作数组。
    """
    max_val = float(np.max(arr))
    baseline = float(np.percentile(arr, 25))
    dynamic = max_val - baseline
    if dynamic <= 0:
        return False
    if dynamic / (max_val + 1e-12) < params.min_contrast_ratio:
        return False

    h1, h2 = float(arr[i_lo]), float(arr[i_hi])
    valley = float(np.min(arr[i_lo : i_hi + 1]))
    gap = i_hi - i_lo
    if gap < params.min_peak_gap:
        return False
    if gap > int(arr.size * params.max_peak_gap_ratio):
        return False
    if min(h1, h2) < params.min_peak_ratio * dynamic:
        return False
    if valley >= params.valley_ratio * min(h1, h2):
        return False
    valley_depth = min(h1, h2) - valley
    if valley_depth < params.min_valley_depth_ratio * dynamic:
        return False
    larger = max(h1, h2)
    smaller = min(h1, h2)
    if larger <= 0:
        return False
    if params.min_secondary_to_primary_ratio > 0:
        if smaller < params.min_secondary_to_primary_ratio * larger:
            return False
    return True


def _split_one_box_by_valley(
    img_path,
    box,
    score,
    rt_array,
    intensity,
    true_rt,
    rt_window,
    params: Optional[ValleySplitParams] = None,
):
    """
    Split a single predicted pixel box into up to two boxes at the valley between two peaks.
    仅当通过双峰 gate 时才拆分。
    """
    params = params or DEFAULT_VALLEY_SPLIT_PARAMS

    x1, y1, x2, y2 = map(float, box)
    if x2 < x1:
        x1, x2 = x2, x1

    left, right, rt_lo, rt_hi = box_to_rt_range(x1, y1, x2, y2, true_rt, rt_array, rt_window=rt_window)
    if right <= left:
        return [box], [score], 0

    mask = (rt_array >= left) & (rt_array <= right)
    if int(np.sum(mask)) < max(6, params.min_segment_points):
        return [box], [score], 0
    x_seg = rt_array[mask].astype(np.float64)
    y_seg = intensity[mask].astype(np.float64)
    y_seg = np.maximum(y_seg, 0.0)

    # 与 analyze_double_peaks.is_double_peak：先去最小值再光滑
    y_work = np.maximum(y_seg - np.min(y_seg), 0.0)
    if np.max(y_work) <= 0:
        return [box], [score], 0

    arr = _smooth_segment(y_work, params.smooth_sigma)
    max_val = float(np.max(arr))
    baseline = float(np.percentile(arr, 25))
    dynamic = max_val - baseline
    if dynamic <= 0:
        return [box], [score], 0

    if params.use_scipy_prominence:
        pair = _find_best_split_pair_prominence(arr, baseline, dynamic, params)
    else:
        pair = _find_best_split_pair_legacy(arr, baseline, dynamic, params)
    if pair is None:
        return [box], [score], 0
    i_lo, i_hi = pair

    valley_rt, _ = _find_valley_between_two_peaks(x_seg, arr, i_lo, i_hi)
    if valley_rt is None:
        return [box], [score], 0

    x_valley = rt_to_pixel_x(valley_rt, rt_lo, rt_hi)
    eps = params.box_split_margin_px
    if x_valley <= x1 + eps or x_valley >= x2 - eps:
        return [box], [score], 0

    box_left = np.array([x1, y1, x_valley, y2], dtype=np.float32)
    box_right = np.array([x_valley, y1, x2, y2], dtype=np.float32)
    score_val = float(score)
    return (
        [box_left, box_right],
        [
            np.array([score_val], dtype=np.float32),
            np.array([score_val], dtype=np.float32),
        ],
        1,
    )


def _split_prediction_by_valley(
    prediction_for_quantify,
    xic_list,
    xic_info,
    roi_windows,
    use_peak_performance=True,
    valley_method="auto",
    valley_params: Optional[ValleySplitParams] = None,
):
    """
    Expand each predicted box by splitting at valley between two peaks.

    Returns:
      prediction_expanded (list aligned to xic indices),
      n_split (int count of split events)

    use_peak_performance / valley_method 保留签名兼容；当前实现仅使用 valley_params 与局部 XIC gate。
    """
    _ = use_peak_performance, valley_method
    params = valley_params or DEFAULT_VALLEY_SPLIT_PARAMS
    n_split = 0
    prediction_expanded = []

    for i, (img_path, scores, boxes) in enumerate(prediction_for_quantify):
        if not img_path or (boxes is None) or np.asarray(boxes).size == 0:
            prediction_expanded.append((img_path, scores, boxes))
            continue

        rt_axis = xic_list[i][0].astype(np.float64)
        intensity_row = xic_list[i][1].astype(np.float64)
        true_rt = float(xic_info.loc[i, "RT"])

        image_name = os.path.basename(img_path)
        rt_window = roi_windows.get(image_name) if roi_windows is not None else None

        new_boxes = []
        new_scores = []

        for j in range(min(len(scores), len(boxes))):
            s = scores[j]
            b = boxes[j]
            score_val = float(np.asarray(s).reshape(-1)[0])
            split_boxes, split_scores, did_split = _split_one_box_by_valley(
                img_path=img_path,
                box=b,
                score=score_val,
                rt_array=rt_axis,
                intensity=intensity_row,
                true_rt=true_rt,
                rt_window=rt_window,
                params=params,
            )
            if did_split:
                n_split += did_split
            for sb, ss in zip(split_boxes, split_scores):
                new_boxes.append(np.asarray(sb, dtype=np.float32))
                new_scores.append(np.asarray([float(np.asarray(ss).reshape(-1)[0])], dtype=np.float32))

        if len(new_boxes) == 0:
            prediction_expanded.append((img_path, scores, boxes))
        else:
            boxes_arr = np.vstack(new_boxes).astype(np.float32)
            scores_arr = np.vstack(new_scores).astype(np.float32)
            prediction_expanded.append((img_path, scores_arr, boxes_arr))

    return prediction_expanded, n_split


def _prediction_to_results(prediction_expanded):
    """
    Convert expanded prediction structure into `results` format usable by
    `newtest._plot_predictions_with_baseline`.
    """
    results = []
    for img_path, scores, boxes in prediction_expanded:
        if not img_path:
            continue
        boxes = np.asarray(boxes, dtype=np.float32) if boxes is not None else np.empty((0, 4), dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32) if scores is not None else np.empty((0, 1), dtype=np.float32)
        if boxes.size == 0 or len(scores) == 0:
            continue
        scores = scores.reshape(-1, 1)
        results.append({"image_path": img_path, "boxes": boxes, "scores": scores.squeeze(-1)})
    return results


def _plot_xic_smoothed_with_valley(xic_list, xic_info, df_all, plot_dir, sigma=2.0):
    """
    Best-effort plot: smooth XIC trace and mark integrated intervals by rt_min/rt_max.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = df_all.copy() if isinstance(df_all, pd.DataFrame) else pd.DataFrame()

    for i in range(len(xic_list)):
        rt = xic_list[i][0].astype(np.float64)
        intensity = np.maximum(xic_list[i][1].astype(np.float64), 0.0)
        if sigma and intensity.size >= 10:
            sig = min(float(sigma), intensity.size / 25.0)
            intensity_s = gaussian_filter1d(intensity, sigma=sig, mode="nearest")
        else:
            intensity_s = intensity

        true_rt = float(xic_info.loc[i, "RT"])
        rt_lo, rt_hi = rt_window_bounds_minutes(true_rt, rt)
        mask = (rt >= rt_lo) & (rt <= rt_hi)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rt[mask], intensity_s[mask], color="blue", linewidth=1.4, label="XIC (smoothed)")

        if not df.empty:
            if "compound_name" in df.columns:
                comp_name = xic_info.loc[i, "Compound Name"] if "Compound Name" in xic_info.columns else i + 1
                df_i = df[df["compound_name"] == comp_name]
            else:
                df_i = df[df.get("mz", df.iloc[:, 0]) == xic_info.loc[i, "mz"]]

            for _, row in df_i.iterrows():
                if "rt_min" in row and "rt_max" in row:
                    ax.axvline(float(row["rt_min"]), color="red", linestyle="--", linewidth=1, alpha=0.6)
                    ax.axvline(float(row["rt_max"]), color="red", linestyle="--", linewidth=1, alpha=0.6)

        ax.set_xlim(rt_lo, rt_hi)
        ax.set_xlabel("RT (min)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Valley split trace (mz={float(xic_info.loc[i,'mz']):g})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

        out_path = plot_dir / f"xic_valley_split_{i:04d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
