# -*- coding: utf-8 -*-
"""
fullscan — 整谱 XIC 全峰识别推理模式（Phase 0-4）

与 docs/superpowers/specs/2026-08-26-full-xic-peak-scan-design.md 对齐：
  Phase0  extract_full_xics     读取 mzML 全部 transition 时序数据（uid/化合物名/Q1/RT/强度）
  Phase1  enumerate_peaks       find_peaks + prominence 候选枚举（双门槛、RT 尺度 distance、
                                前沿区排除、相邻谷不显著合并、单通道上限）
  Phase2  refine_all_boundaries 两遍边界精修（复用 adjust_first_round_interval，peer 防撞）
  Phase3a gate_peaks            compute_local_snr / 峰高 / 点数 / 面积门控
  Phase3b validate_with_model   每峰切 ±1min 窗 → 批量 build_predictor → 框映射（apex 落框配对）
  Phase4  write_outputs         full_scan_peaks.csv + scan_summary.csv + 整谱标注图

输入严格限定为 mzML 色谱时序数据：{uid(native_id), 化合物名称, 母离子 m/z(Q1),
强度数组(可选高斯平滑), RT 数组(分钟)}；不读取 q3/子离子、MS1/MS2 谱图。

说明：峰检测使用各通道**原生 RT/强度**（不做公共轴插值），避免插值伪影影响峰形；
公共 RT 轴对齐仅在需要 xic_matrix 类输出时才必要，本模式无此输出。

用法（须在 model/ 目录下运行）:
  python -m inference.fullscan --mzml ../data/test/mzml/sample.mzML
  python -m inference.fullscan --mzml ../data/test/mzml --model checkpoint/quanformer.pth
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT_DIR = Path(__file__).resolve().parent.parent  # model/ 目录

from utils.mzml_load import load_ms_experiment
from utils.mzml_chromatogram_ids import filesystem_slug_for_native_id, resolve_native_ids_for_chromatograms
from utils.quantify import AREA_TIME_UNIT_SCALE
from utils.roi_rt_mapping import box_to_rt_range, rt_window_bounds_minutes
from utils.xic_peak_utils import compute_local_snr, roi_full_low_decile_mean_intensity
from preprocessing.xic_extraction import render_roi_jpeg
from inference.two_round_detection import adjust_first_round_interval


# ---------------------------------------------------------------------------
# Phase 0 — 全谱 XIC 组装
# ---------------------------------------------------------------------------


def _native_id_to_str(native_id):
    if native_id is None:
        return ""
    if isinstance(native_id, bytes):
        return native_id.decode("utf-8", errors="replace")
    return str(native_id)


def _parse_q1_from_text(text):
    """从 native_id 文本解析母离子 m/z（Q1）；与 xic_extraction 的 Q1 规则一致，不解析 Q3。"""
    for pat in (r"Q1=([\d\.]+)", r"q1=([\d\.]+)", r"precursor[=:_ ]([\d\.]+)"):
        m = re.search(pat, str(text))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _q1_from_chrom_metadata(chrom):
    try:
        mz_pre = float(chrom.getPrecursor().getMZ())
        if np.isfinite(mz_pre) and mz_pre > 0:
            return mz_pre
    except Exception:
        pass
    return None


def _extract_q1(chrom, native_id_text):
    q1 = _parse_q1_from_text(native_id_text)
    if q1 is None:
        q1 = _q1_from_chrom_metadata(chrom)
    return q1


def _rt_to_minutes(rt_raw):
    """RT 单位判定（D15）：优先中位时间步长，回退 >200 启发式。返回分钟数组。"""
    rt = np.asarray(rt_raw, dtype=np.float64)
    if rt.size < 2:
        return rt
    steps = np.diff(np.sort(rt))
    steps = steps[steps > 0]
    median_step = float(np.median(steps)) if steps.size else 0.0
    if median_step >= 0.1 or float(np.nanmax(rt)) > 200:
        return rt / 60.0
    return rt


def extract_full_xics(mzml_path, smooth_sigma=0.8, min_chrom_points=0, min_max_intensity=0.0,
                      verbose=True):
    """
    Phase0：读取 mzML 全部 transition 时序数据。

    返回 (features, qc_excluded)：
      features: [{"chrom_index", "uid", "compound_name", "q1", "rt"(分钟), "intensity"}]
      qc_excluded: [{"chrom_index", "uid", "q1", "reason", "n_points", "max_intensity"}]
    """
    if verbose:
        print(f"[INFO] fullscan Phase0 读取 mzML: {mzml_path}")
    exp = load_ms_experiment(mzml_path, verbose=False)
    chromatograms = exp.getChromatograms()
    native_ids = resolve_native_ids_for_chromatograms(mzml_path, chromatograms, _native_id_to_str)

    features: List[dict] = []
    qc_excluded: List[dict] = []
    seen_axes: Dict[Tuple[float, str], list] = {}  # (q1, uid) -> [rt 数组...]，D14 真重复判定

    for i, chrom in enumerate(chromatograms):
        rt_raw = np.array([p.getRT() for p in chrom])
        intensity_raw = np.array([p.getIntensity() for p in chrom])
        uid = native_ids[i] if i < len(native_ids) else _native_id_to_str(chrom.getNativeID())
        q1 = _extract_q1(chrom, uid)

        if len(rt_raw) == 0:
            qc_excluded.append({"chrom_index": i, "uid": uid, "q1": q1, "reason": "empty",
                                "n_points": 0, "max_intensity": 0.0})
            continue
        if q1 is None:
            qc_excluded.append({"chrom_index": i, "uid": uid, "q1": q1, "reason": "tic_excluded",
                                "n_points": int(len(rt_raw)),
                                "max_intensity": float(np.max(intensity_raw)) if len(intensity_raw) else 0.0})
            continue

        rt = _rt_to_minutes(rt_raw)
        n_pts = int(rt.size)
        imax_raw = float(np.max(intensity_raw)) if n_pts else 0.0

        # 去重：仅 (Q1, uid, RT 轴完全一致) 判真重复（D14）
        k = (round(float(q1), 4), uid)
        dup = any(ax.size == rt.size and np.allclose(ax, rt, atol=1e-5) for ax in seen_axes.get(k, ()))
        if dup:
            qc_excluded.append({"chrom_index": i, "uid": uid, "q1": q1, "reason": "duplicate",
                                "n_points": n_pts, "max_intensity": imax_raw})
            continue
        seen_axes.setdefault(k, []).append(rt)

        # 高斯平滑（可选，参数控制）：smooth_sigma=0 时不平滑
        if smooth_sigma > 0:
            intensity = gaussian_filter1d(intensity_raw.astype(np.float64), sigma=float(smooth_sigma))
        else:
            intensity = intensity_raw.astype(np.float64)
        imax = float(np.max(intensity)) if n_pts else 0.0

        if min_chrom_points > 0 and n_pts < int(min_chrom_points):
            qc_excluded.append({"chrom_index": i, "uid": uid, "q1": q1, "reason": "too_few_points",
                                "n_points": n_pts, "max_intensity": imax})
            continue
        if min_max_intensity > 0.0 and imax < float(min_max_intensity):
            qc_excluded.append({"chrom_index": i, "uid": uid, "q1": q1, "reason": "low_max_intensity",
                                "n_points": n_pts, "max_intensity": imax})
            continue

        features.append({
            "chrom_index": i,
            "uid": uid or "",
            "compound_name": uid or "",
            "q1": float(q1),
            "rt": rt,
            "intensity": intensity,
        })

    if not features:
        raise ValueError("mzML 无有效 transition（全部被剔除）: %s" % mzml_path)
    if verbose:
        print(f"[INFO] fullscan Phase0: {len(chromatograms)} 条色谱 → {len(features)} 条有效通道"
              f"（剔除 {len(qc_excluded)}）")
    return features, qc_excluded


# ---------------------------------------------------------------------------
# Phase 1 — 候选枚举
# ---------------------------------------------------------------------------


def _merge_insignificant_valleys(y, peaks, valley_ratio=0.90, min_central_frac=0.12):
    """
    相邻候选峰之间谷不显著（谷不够深或不在两峰间居中）→ 合并保留较高者。
    与 valley_split._pair_passes_double_peak_gate 语义一致（D6：先合并再精修）。
    """
    if len(peaks) < 2:
        return list(peaks)
    out: List[int] = []
    for pk in sorted(int(p) for p in peaks):
        if not out:
            out.append(pk)
            continue
        i_lo, i_hi = out[-1], pk
        if i_hi <= i_lo:
            continue
        if i_hi - i_lo < 3:
            out[-1] = i_lo if y[i_lo] >= y[i_hi] else i_hi
            continue
        v_idx = int(np.argmin(y[i_lo:i_hi + 1])) + i_lo
        frac = (v_idx - i_lo) / float(i_hi - i_lo)
        h1, h2 = float(y[i_lo]), float(y[i_hi])
        valley = float(y[v_idx])
        keep_both = (
            min_central_frac <= frac <= 1.0 - min_central_frac
            and valley < float(valley_ratio) * min(h1, h2)
        )
        if keep_both:
            out.append(pk)
        else:
            out[-1] = i_lo if h1 >= h2 else i_hi
    return out


def enumerate_peaks(rt, intensity, baseline_percentile=25.0, baseline_mode="global_percentile",
                    min_peak_ratio=0.04, prominence_ratio=0.055, min_prominence_abs=0.0,
                    min_peak_gap_points=3, min_peak_width_min=0.10, void_time_min=0.5,
                    max_peaks_per_channel=50, valley_ratio=0.90, min_valley_central_frac=0.12):
    """
    Phase1：整谱候选枚举。

    返回按 RT 排序的 apex 索引列表。双门槛 prominence（相对 dynamic + 绝对下限）防 D2；
    distance 按通道中位步长换算为 RT 尺度防 D3；前置区排除防 D5；单通道上限防 D11。
    """
    rt = np.asarray(rt, dtype=np.float64)
    y = np.maximum(np.asarray(intensity, dtype=np.float64), 0.0)
    n = y.size
    if n < 10:
        return []

    body_mask = rt >= float(void_time_min)
    if int(np.count_nonzero(body_mask)) < 5:
        body_mask = np.ones(n, dtype=bool)

    if baseline_mode == "local_valley":
        # 局部谷基线：取动态范围内较低一档的分位，抗全局漂移（D1）
        baseline = float(np.percentile(y[body_mask], float(baseline_percentile)))
    else:
        baseline = float(np.percentile(y[body_mask], float(baseline_percentile)))
    dynamic = float(np.max(y[body_mask])) - baseline
    if dynamic <= 0:
        return []

    steps = np.diff(np.sort(rt))
    steps = steps[steps > 0]
    median_step = float(np.median(steps)) if steps.size else 0.01
    dist = max(int(min_peak_gap_points), int(np.ceil(float(min_peak_width_min) / max(median_step, 1e-9))))
    dist = max(1, dist)
    prom = max(float(prominence_ratio) * dynamic, float(min_prominence_abs))
    height = baseline + float(min_peak_ratio) * dynamic

    peaks, props = find_peaks(y, prominence=prom, height=height, distance=dist)
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return []

    # plateau → 平台中点
    plats = props.get("plateau_sizes")
    if plats is not None:
        plefts = props.get("left_edges")
        peaks = [int(pl) + int(ps) // 2 for pl, ps in zip(plefts, plats)]
    peaks = sorted(int(p) for p in peaks if body_mask[p])
    if not peaks:
        return []

    # 相邻谷不显著合并
    peaks = _merge_insignificant_valleys(y, peaks, valley_ratio=float(valley_ratio),
                                         min_central_frac=float(min_valley_central_frac))

    # 单通道上限（D11）
    if max_peaks_per_channel and len(peaks) > int(max_peaks_per_channel):
        peaks = sorted(
            sorted(peaks, key=lambda p: float(y[p]), reverse=True)[:int(max_peaks_per_channel)])
    return peaks


# ---------------------------------------------------------------------------
# Phase 2 — 边界精修（两遍，复用 adjust_first_round_interval）
# ---------------------------------------------------------------------------


def refine_all_boundaries(rt, intensity, apexes, init_half_width_min=0.05,
                          boundary_posterior_lookahead=5, boundary_posterior_mean_scale=1.25,
                          edge_noise_stop_mode="stable_tail_mean", edge_max_span_min=1.0):
    """
    Phase2：两遍精修（D13）。
    首遍全部候选无 peer 粗走 → 以首遍结果为 peer 区间走第二遍，防相邻峰互相截停。

    截停阈值用峰侧局部稳定尾噪声（edge_noise_stop_mode="stable_tail_mean"）：全通道
    低十分位均值在整谱场景过低（基线漂移/持续信号），会把峰一路扩到数据尽头（冒烟实测）；
    局部尾噪声只取峰顶两侧有限跨度内的低波动区，贴近该峰的真实基线。
    返回与 apexes 等长的 [(rt_min, rt_max), ...]（无效为 (nan, nan)）。
    """
    rt = np.asarray(rt, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    rt_lo = float(np.min(rt))
    rt_hi = float(np.max(rt))
    half = max(float(init_half_width_min), 3.0 * (rt_hi - rt_lo) / max(y.size - 1, 1))

    def _walk(ap_idx, peers):
        apex = float(rt[ap_idx])
        rt_min_ini = max(rt_lo, apex - half)
        rt_max_ini = min(rt_hi, apex + half)
        if rt_max_ini <= rt_min_ini:
            return np.nan, np.nan
        return adjust_first_round_interval(
            rt, y, rt_min_ini, rt_max_ini, rt_lo, rt_hi,
            min_secondary_ratio=0.05,
            edge_noise_stop_mode=str(edge_noise_stop_mode),
            edge_max_span_min=float(edge_max_span_min),
            boundary_posterior_lookahead=int(boundary_posterior_lookahead),
            boundary_posterior_mean_scale=float(boundary_posterior_mean_scale),
            peer_rt_intervals=peers,
            boundary_peer_thr_scale=2.0,
        )

    # 第一遍：无 peer
    pass1 = [_walk(ap, None) for ap in apexes]
    # 第二遍：以第一遍结果为 peer
    pass2 = []
    for j, ap in enumerate(apexes):
        peers = [(a, b) for i2, (a, b) in enumerate(pass1)
                 if i2 != j and np.isfinite(a) and np.isfinite(b) and b > a]
        pass2.append(_walk(ap, peers or None))
    return pass2


# ---------------------------------------------------------------------------
# Phase 3a — 门控
# ---------------------------------------------------------------------------


def _max_consec_above(arr, thr):
    a = np.asarray(arr) > float(thr)
    if not np.any(a):
        return 0
    d = np.diff(a.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if a[0]:
        starts = np.insert(starts, 0, 0)
    if a[-1]:
        ends = np.append(ends, len(a))
    if starts.size == 0 or ends.size == 0:
        return 0
    return int(np.max(ends - starts))


def gate_peaks(rt, intensity, apexes, intervals, min_snr=3.0, min_peak_span_points=5, min_area=0.0):
    """
    Phase3a：对精修后的候选峰做质量门控。
    返回 [{"apex_idx", "rt_min", "rt_max", "snr", "n_points", "area"}, ...]（RT 排序）。
    """
    rt = np.asarray(rt, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    baseline_g = roi_full_low_decile_mean_intensity(y, bottom_frac=0.10)
    out = []
    for k, (rt_min, rt_max) in enumerate(intervals):
        if not (np.isfinite(rt_min) and np.isfinite(rt_max) and rt_max > rt_min):
            continue
        peers = [(a, b) for j, (a, b) in enumerate(intervals)
                 if j != k and np.isfinite(a) and np.isfinite(b) and b > a]
        snr = compute_local_snr(rt, y, rt_min, rt_max, neighbor_intervals=peers)
        if snr is None or np.isnan(snr):
            continue
        mask = (rt >= rt_min) & (rt <= rt_max)
        n_pts_box = int(np.count_nonzero(mask))
        if n_pts_box < 3:
            continue
        int_seg = np.maximum(y[mask], 0.0)
        n_points = _max_consec_above(int_seg, baseline_g)
        area = float(np.trapz(y[mask], rt[mask]) * float(AREA_TIME_UNIT_SCALE)) if n_pts_box >= 2 else 0.0
        if float(snr) < float(min_snr):
            continue
        if int(min_peak_span_points) > 0 and n_points < int(min_peak_span_points):
            continue
        if float(min_area) > 0 and area < float(min_area):
            continue
        out.append({
            "apex_idx": int(apexes[k]),
            "rt_min": float(rt_min),
            "rt_max": float(rt_max),
            "snr": float(snr),
            "n_points": int(n_points),
            "area": float(area),
        })
    out.sort(key=lambda p: p["rt_min"])
    return out


# ---------------------------------------------------------------------------
# Phase 3b — 模型验证（--model 提供时）
# ---------------------------------------------------------------------------


def validate_with_model(peaks_by_channel, rts, intensities, model_path, threshold=0.99,
                        window_half_min=1.0, keep_windows=False, verbose=True):
    """
    Phase3b（每 mzML 一次批量验证）：将全部通道的全部峰一次性渲染到同一临时目录，
    单轮 build_predictor 推理，再按 apex 落框贪心配对（D9）回填各峰。

    peaks_by_channel: {ch_idx: [peak dicts]}；rts / intensities: {ch_idx: rt/y 数组}
    就地更新每峰：model_score / validated / boundary_source / rt_min / rt_max
    （validated → 模型框 RT 优先；否则保持信号外推边界）。
    返回渲染窗口目录（keep_windows=True 时保留，否则清理并返回 None）。
    """
    from utils.predict_utils import build_predictor  # 惰性导入，避免无模型时加载 torch

    tmp_dir = tempfile.mkdtemp(prefix="fullscan_windows_")
    file_map: Dict[str, Tuple[int, int, float, float]] = {}  # win_name -> (ch_idx, p_idx, rt_lo, rt_hi)

    # 1) 渲染全部通道的全部峰窗口
    name_idx = 0
    for ch_idx in sorted(peaks_by_channel.keys()):
        rt = np.asarray(rts[ch_idx], dtype=np.float64)
        y = np.asarray(intensities[ch_idx], dtype=np.float64)
        for p_idx, p in enumerate(peaks_by_channel[ch_idx]):
            apex = float(p["rt_peak"])
            rt_lo, rt_hi = rt_window_bounds_minutes(apex, rt)
            if window_half_min > 0:
                rt_lo = max(float(rt_lo), apex - float(window_half_min))
                rt_hi = min(float(rt_hi), apex + float(window_half_min))
                if rt_hi <= rt_lo:
                    rt_lo, rt_hi = rt_window_bounds_minutes(apex, rt)
            mask = (rt >= rt_lo) & (rt <= rt_hi)
            win_rt = rt[mask]
            win_int = y[mask]
            if win_rt.size < 2:
                continue
            name = f"win_{name_idx:07d}.jpeg"
            render_roi_jpeg(win_rt, win_int, rt_lo, rt_hi, os.path.join(tmp_dir, name))
            file_map[name] = (ch_idx, p_idx, rt_lo, rt_hi)
            name_idx += 1

    if not file_map:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # 2) 批量推理（单轮，模型只加载一次）
    results = build_predictor(model_path=model_path, images_path=tmp_dir, threshold=threshold,
                              plot=False, verbose=verbose)
    img_results = {os.path.basename(str(r["image_path"])): r for r in results}

    # 3) 预计算候选配对（apex 落框）
    candidates: List[Tuple[int, int, str, int, float, float, float, float]] = []  # (ch, p, img, box_i, score, dist, left, right)
    for name, (ch_idx, p_idx, rt_lo, rt_hi) in file_map.items():
        p = peaks_by_channel[ch_idx][p_idx]
        res = img_results.get(name)
        if not res or len(res.get("boxes", [])) == 0:
            continue
        boxes = np.asarray(res["boxes"]).reshape(-1, 4)
        scores = np.asarray(res["scores"]).reshape(-1)
        apex = float(p["rt_peak"])
        rt = np.asarray(rts[ch_idx], dtype=np.float64)
        for b_i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[b_i]
            left, right, _, _ = box_to_rt_range(x1, y1, x2, y2, apex, rt, rt_window=(rt_lo, rt_hi))
            if left <= apex <= right:
                center = 0.5 * (left + right)
                candidates.append((ch_idx, p_idx, name, b_i, float(scores[b_i]),
                                   abs(center - apex), left, right))

    # 4) 贪心分配：分数高优先，分数相同中心距离近优先；一个框至多验证一个峰
    assigned: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    used = set()
    for ch_idx, p_idx, name, b_i, score, dist, left, right in sorted(
            candidates, key=lambda t: (-t[4], t[5])):
        key_p = (ch_idx, p_idx)
        if key_p in assigned or (name, b_i) in used:
            continue
        used.add((name, b_i))
        assigned[key_p] = (score, left, right)

    # 5) 回填结果
    for ch_idx, peaks in peaks_by_channel.items():
        for p_idx, p in enumerate(peaks):
            if (ch_idx, p_idx) in assigned:
                score, left, right = assigned[(ch_idx, p_idx)]
                p["model_score"] = score
                p["validated"] = bool(score >= float(threshold))
                if p["validated"]:
                    p["rt_min"] = float(left)
                    p["rt_max"] = float(right)
                    p["boundary_source"] = "model"
            else:
                p["model_score"] = np.nan
                p["validated"] = False

    if not keep_windows:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    return tmp_dir


# ---------------------------------------------------------------------------
# Phase 4 — 输出
# ---------------------------------------------------------------------------


def plot_full_scan(rt, intensity, peaks, out_path, uid, q1):
    """Phase4a：整谱标注图（宽按 RT 跨度缩放，D12）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rt = np.asarray(rt, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    span = float(np.max(rt) - np.min(rt))
    fig_w = max(6.0, 1.2 * span)  # 1.2 inch/min → 120 px/min @100dpi
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    ax.plot(rt, y, "b-", linewidth=1.2)
    y_top = float(np.max(y)) if y.size else 1.0
    for p in peaks:
        color = "red" if p.get("validated") else "gray"
        ax.axvspan(p["rt_min"], p["rt_max"], alpha=0.22, color=color)
        ax.axvline(p["rt_min"], color=color, linestyle="--", linewidth=0.8)
        ax.axvline(p["rt_max"], color=color, linestyle="--", linewidth=0.8)
        sc = p.get("model_score")
        sc_txt = "" if sc is None or not np.isfinite(float(sc)) else " %.2f" % float(sc)
        ax.text(p["rt_peak"], y_top * 0.98, "#%d%s" % (p["peak_no"], sc_txt),
                fontsize=8, color=color, rotation=45, ha="right", va="top")
    ax.set_xlim(float(np.min(rt)), float(np.max(rt)))
    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("Intensity")
    ax.set_title("fullscan %s | Q1=%.4f | %d peaks" % (uid, float(q1), len(peaks)))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def write_outputs(mzml_stem, features, peaks_by_channel, qc_excluded, sample_dir,
                  no_plots=False, keep_windows=None):
    """
    Phase4：写 full_scan_peaks.csv / scan_summary.csv / scan_qc_excluded.csv + 整谱标注图。
    peaks_by_channel: {chrom_index: [peak dicts]}；keep_windows: {chrom_index: window_dir}
    """
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    peak_rows = []
    summary_rows = []
    for f in features:
        ci = f["chrom_index"]
        peaks = peaks_by_channel.get(ci, [])
        summary_rows.append({
            "chrom_index": ci, "uid": f["uid"], "compound_name": f["compound_name"],
            "q1": f["q1"], "n_peaks": len(peaks),
            "rt_range_min": round(float(np.min(f["rt"])), 4),
            "rt_range_max": round(float(np.max(f["rt"])), 4),
            "max_intensity": round(float(np.max(f["intensity"])), 2),
        })
        for p in peaks:
            peak_rows.append({
                "mzml_stem": mzml_stem,
                "chrom_index": ci,
                "uid": f["uid"],
                "compound_name": f["compound_name"],
                "q1": f["q1"],
                "peak_no": p["peak_no"],
                "rt_min": round(p["rt_min"], 4),
                "rt_peak": round(p["rt_peak"], 4),
                "rt_max": round(p["rt_max"], 4),
                "apex_intensity": round(p["apex_intensity"], 2),
                "area": round(p["area"], 4),
                "snr": round(p["snr"], 3),
                "n_points": p["n_points"],
                "validated": bool(p.get("validated", False)),
                "model_score": p.get("model_score", np.nan),
                "boundary_source": p.get("boundary_source", "signal"),
            })

    pd.DataFrame(peak_rows).to_csv(sample_dir / "full_scan_peaks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(sample_dir / "scan_summary.csv", index=False, encoding="utf-8-sig")
    if qc_excluded:
        pd.DataFrame(qc_excluded).to_csv(sample_dir / "scan_qc_excluded.csv",
                                         index=False, encoding="utf-8-sig")

    if not no_plots:
        plots_dir = sample_dir / "scan_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for f in features:
            ci = f["chrom_index"]
            peaks = peaks_by_channel.get(ci, [])
            slug = filesystem_slug_for_native_id(f["uid"] or "", max_len=48) or "empty"
            out_png = plots_dir / ("chrom%03d_%s.png" % (ci, slug))
            try:
                plot_full_scan(f["rt"], f["intensity"], peaks, str(out_png), f["uid"], f["q1"])
            except Exception as e:
                print(f"[WARN] fullscan 绘图失败 {out_png}: {e}")

    if keep_windows:
        win_src = keep_windows.get("all") if isinstance(keep_windows, dict) else None
        if win_src:
            target = sample_dir / "scan_windows"
            target.mkdir(parents=True, exist_ok=True)
            for fn in Path(win_src).iterdir():
                try:
                    shutil.copy2(fn, target / fn.name)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _collect_mzml_inputs(mzml_arg, batch_dir_arg):
    """惰性复用 cli._collect_mzml_inputs，避免 cli ↔ fullscan 循环导入。"""
    from .cli import _collect_mzml_inputs as _cli_collect
    return _cli_collect(mzml_arg, batch_dir_arg)


def _scan_params_from_args(args) -> dict:
    def _g(name, default):
        return getattr(args, name, default)
    return {
        "baseline_percentile": float(_g("scan_baseline_percentile", 25.0)),
        "baseline_mode": str(_g("scan_baseline_mode", "global_percentile")),
        "min_peak_ratio": float(_g("scan_min_peak_ratio", 0.04)),
        "prominence_ratio": float(_g("scan_prominence_ratio", 0.055)),
        "min_prominence_abs": float(_g("scan_min_prominence_abs", 0.0)),
        "min_peak_gap_points": int(_g("scan_min_peak_gap_points", 3)),
        "min_peak_width_min": float(_g("scan_min_peak_width_min", 0.10)),
        "void_time_min": float(_g("scan_void_time_min", 0.5)),
        "max_peaks_per_channel": int(_g("scan_max_peaks_per_channel", 50)),
        "valley_ratio": 0.90,
        "min_valley_central_frac": 0.12,
        "init_half_width_min": float(_g("scan_init_half_width_min", 0.05)),
        "boundary_posterior_lookahead": int(_g("scan_boundary_posterior_lookahead", 5)),
        "boundary_posterior_mean_scale": float(_g("scan_boundary_posterior_mean_scale", 1.25)),
        "edge_noise_stop_mode": str(_g("scan_edge_noise_stop_mode", "stable_tail_mean")),
        "edge_max_span_min": float(_g("scan_edge_max_span_min", 1.0)),
        "min_snr": float(_g("scan_min_snr", 3.0)),
        "min_peak_span_points": int(_g("scan_min_peak_span_points", 5)),
        "min_area": float(_g("scan_min_area", 0.0)),
        "window_half_min": float(_g("scan_window_half_min", 1.0)),
    }


def run_fullscan_on_mzml(mzml_path, key, args, out_root):
    """对单个 mzML 执行整谱全峰识别，返回 (峰行数, 通道数)。"""
    smooth_sigma = float(getattr(args, "smooth_sigma", 0.8) or 0.0)
    min_max_intensity = float(getattr(args, "pipeline_min_max_intensity", 1000.0) or 0.0)
    min_chrom_points = int(getattr(args, "pipeline_min_chrom_points", 10) or 0)
    model_path = getattr(args, "model", None)
    threshold = float(getattr(args, "threshold", 0.99))
    keep_windows = bool(getattr(args, "keep_windows", False))
    no_plots = bool(getattr(args, "no_plots", False))
    sp = _scan_params_from_args(args)

    sample_dir = Path(out_root) / key
    sample_dir.mkdir(parents=True, exist_ok=True)

    features, qc_excluded = extract_full_xics(
        mzml_path, smooth_sigma=smooth_sigma,
        min_chrom_points=min_chrom_points, min_max_intensity=min_max_intensity,
    )

    peaks_by_channel: Dict[int, list] = {}
    keep_win_dirs: Dict[int, str] = {}
    n_peaks_total = 0
    rts: Dict[int, np.ndarray] = {}
    intensities: Dict[int, np.ndarray] = {}
    for f in features:
        ci = f["chrom_index"]
        rt = f["rt"]
        y = f["intensity"]
        rts[ci] = rt
        intensities[ci] = y

        apexes = enumerate_peaks(rt, y, **{k2: v2 for k2, v2 in sp.items()
                                           if k2 in ("baseline_percentile", "baseline_mode",
                                                     "min_peak_ratio", "prominence_ratio",
                                                     "min_prominence_abs", "min_peak_gap_points",
                                                     "min_peak_width_min", "void_time_min",
                                                     "max_peaks_per_channel", "valley_ratio",
                                                     "min_valley_central_frac")})
        intervals = refine_all_boundaries(
            rt, y, apexes,
            init_half_width_min=sp["init_half_width_min"],
            boundary_posterior_lookahead=sp["boundary_posterior_lookahead"],
            boundary_posterior_mean_scale=sp["boundary_posterior_mean_scale"],
            edge_noise_stop_mode=sp["edge_noise_stop_mode"],
            edge_max_span_min=sp["edge_max_span_min"],
        )
        gated = gate_peaks(rt, y, apexes, intervals,
                           min_snr=sp["min_snr"],
                           min_peak_span_points=sp["min_peak_span_points"],
                           min_area=sp["min_area"])

        peaks = []
        for pk_no, g in enumerate(gated, start=1):
            ap = int(g["apex_idx"])
            peaks.append({
                "peak_no": pk_no,
                "apex_idx": ap,
                "rt_peak": float(rt[ap]),
                "rt_min": float(g["rt_min"]),
                "rt_max": float(g["rt_max"]),
                "apex_intensity": float(y[ap]),
                "area": float(g["area"]),
                "snr": float(g["snr"]),
                "n_points": int(g["n_points"]),
                "model_score": np.nan,
                "validated": False,
                "boundary_source": "signal",
            })

        peaks_by_channel[ci] = peaks
        n_peaks_total += len(peaks)
        if getattr(args, "verbose", False):
            print(f"[fullscan] 通道#{ci} {f['uid']}: 枚举 {len(apexes)} 候选 → 门控 {len(peaks)} 峰")

    # 模型验证：每 mzML 一次批量推理（全部通道窗口一次性渲染，模型只加载一次）
    if model_path:
        win_dir = validate_with_model(
            peaks_by_channel, rts, intensities, model_path, threshold=threshold,
            window_half_min=sp["window_half_min"],
            keep_windows=keep_windows,
            verbose=bool(getattr(args, "verbose", False)),
        )
        if win_dir:
            keep_win_dirs = {"all": win_dir}

    mzml_stem = Path(mzml_path).stem
    write_outputs(mzml_stem, features, peaks_by_channel, qc_excluded, sample_dir,
                  no_plots=no_plots, keep_windows=keep_win_dirs if keep_windows else None)
    print(f"[OK] fullscan {mzml_stem}: {len(features)} 通道 / {n_peaks_total} 峰 → {sample_dir}")
    return n_peaks_total, len(features)


def main(args):
    """统一入口（cli --mode fullscan 与 python -m inference.fullscan 共用）。"""
    mzml_inputs = _collect_mzml_inputs(getattr(args, "mzml", None), getattr(args, "batch_dir", None))
    out_root = Path(getattr(args, "output_dir", None) or "../output/inference/full_scan").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    total_peaks = 0
    total_ch = 0
    for mzml_path, key in mzml_inputs:
        try:
            np_, nch = run_fullscan_on_mzml(str(mzml_path), key, args, out_root)
            total_peaks += np_
            total_ch += nch
        except Exception as e:
            print(f"[ERROR] fullscan 失败 {mzml_path}: {e}")
    print(f"[DONE] fullscan 完成: {len(mzml_inputs)} 个 mzML, 合计 {total_ch} 通道 / {total_peaks} 峰 → {out_root}")


def build_parser():
    ap = argparse.ArgumentParser(description="整谱 XIC 全峰识别（fullscan）")
    ap.add_argument("--mzml", type=str, default=None, help="单个 mzML 或包含 mzML 的目录（递归）")
    ap.add_argument("--batch_dir", type=str, default=None, help="mzML 目录（递归）")
    ap.add_argument("--model", type=str, default=None, help="模型路径（提供则开启模型验证）")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--smooth_sigma", type=float, default=0.8, help="高斯平滑 sigma；0=关闭")
    ap.add_argument("--output_dir", type=str, default="../output/inference/full_scan")
    ap.add_argument("--pipeline_min_max_intensity", type=float, default=1000.0)
    ap.add_argument("--pipeline_min_chrom_points", type=int, default=10)
    ap.add_argument("--scan_baseline_percentile", type=float, default=25.0)
    ap.add_argument("--scan_baseline_mode", type=str, default="global_percentile",
                    choices=["global_percentile", "local_valley"])
    ap.add_argument("--scan_min_peak_ratio", type=float, default=0.04)
    ap.add_argument("--scan_prominence_ratio", type=float, default=0.055)
    ap.add_argument("--scan_min_prominence_abs", type=float, default=0.0)
    ap.add_argument("--scan_min_peak_gap_points", type=int, default=3)
    ap.add_argument("--scan_min_peak_width_min", type=float, default=0.10)
    ap.add_argument("--scan_void_time_min", type=float, default=0.5)
    ap.add_argument("--scan_max_peaks_per_channel", type=int, default=50)
    ap.add_argument("--scan_init_half_width_min", type=float, default=0.05)
    ap.add_argument("--scan_boundary_posterior_lookahead", type=int, default=5)
    ap.add_argument("--scan_boundary_posterior_mean_scale", type=float, default=1.25)
    ap.add_argument("--scan_edge_noise_stop_mode", type=str, default="stable_tail_mean",
                    choices=["stable_tail_mean", "roi_bottom_decile_mean", "low_percentile"],
                    help="边界截停阈值：stable_tail_mean=峰侧局部稳定尾噪声（默认，整谱场景稳健）")
    ap.add_argument("--scan_edge_max_span_min", type=float, default=1.0,
                    help="边界截停阈值估计的最大单侧跨度（min）")
    ap.add_argument("--scan_min_snr", type=float, default=3.0)
    ap.add_argument("--scan_min_peak_span_points", type=int, default=5)
    ap.add_argument("--scan_min_area", type=float, default=0.0)
    ap.add_argument("--scan_window_half_min", type=float, default=1.0)
    ap.add_argument("--keep_windows", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())
