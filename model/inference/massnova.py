# -*- coding: utf-8 -*-
"""
massnova — 整谱 XIC 全峰识别推理模式（Phase 0-4）
适用场景：MassNova 集成 / 仅提供时序数据（不依赖标注）。

与 docs/superpowers/specs/2026-08-26-full-xic-peak-scan-design.md 对齐（模型前置、精修兜底架构）：
  Phase0  extract_full_xics     读取 mzML 全部 transition 时序数据（uid/化合物名/Q1/RT/强度）
  Phase1  enumerate_peaks       find_peaks + prominence 候选枚举（双门槛、RT 尺度 distance、
                                前沿区排除、相邻谷不显著合并、单通道上限）；候选仅带预估 rt
  Phase3b validate_with_model   模型前置：全部候选按预估 rt 切 ±1min 窗 → 批量 build_predictor
                                → apex 落框配对；命中者边界直接取模型框（boundary_source=model）
  Phase2/3a 兜底                仅对模型未命中残差做两遍边界精修（复用 adjust_first_round_interval，
                                以模型框为 peer 防撞）+ SNR/峰高/点数/面积门控（boundary_source=signal）
  Phase4  write_outputs         实验布局输出（对齐 pipeline）：
                                <实验根>/prediction_refined/<样本>/  最终 CSV + 2min 窗口图
                                <实验根>/prediction_signal/<样本>/   2min 窗口图（信号兜底边界）
                                <实验根>/prediction_model/<样本>/    2min 窗口图（模型框边界）
                                <实验根>/scan_plots|model_plots/<样本>/  长 XIC 图
  Phase5  write_massnova_report 跨样本推理报告 inference_report_<实验名>.md + all.csv

输入严格限定为 mzML 色谱时序数据：{uid(native_id), 化合物名称, 母离子 m/z(Q1),
强度数组(可选高斯平滑), RT 数组(分钟)}；不读取 q3/子离子、MS1/MS2 谱图。

说明：峰检测使用各通道**原生 RT/强度**（不做公共轴插值），避免插值伪影影响峰形；
公共 RT 轴对齐仅在需要 xic_matrix 类输出时才必要，本模式无此输出。

用法（须在 model/ 目录下运行）:
  python -m inference.massnova --mzml ../data/test/mzml/sample.mzML
  python -m inference.massnova --mzml ../data/test/mzml --model checkpoint/quanformer.pth
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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT_DIR = Path(__file__).resolve().parent.parent  # model/ 目录

from utils.mzml_chromatogram_ids import filesystem_slug_for_native_id, resolve_native_ids_for_chromatograms
from utils.quantify import AREA_TIME_UNIT_SCALE
from utils.roi_rt_mapping import box_to_rt_range, rt_window_bounds_minutes
from utils.xic_peak_utils import compute_local_snr, roi_full_low_decile_mean_intensity
from utils.peak_scores import signal_peak_score, unified_peak_score
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


def render_roi_jpeg(rt_min, intensity, rt_lo, rt_hi, out_path,
                    color="blue", linewidth=1.5):
    """Render the exact 400x300 axis-free ROI used for model inference.

    This lightweight copy keeps the array/DLL runtime independent of
    ``preprocessing.xic_extraction``, whose mzML front end requires pyopenms.
    """
    fig = Figure(figsize=(4, 3), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(
        np.asarray(rt_min, dtype=np.float64),
        np.asarray(intensity, dtype=np.float64),
        color=color,
        linewidth=linewidth,
    )
    if float(rt_hi) > float(rt_lo) and len(rt_min) > 0:
        ax.set_xlim(float(rt_lo), float(rt_hi))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas.print_jpeg(out_path)


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
    from utils.mzml_load import load_ms_experiment  # mzML-only optional dependency: pyopenms

    if verbose:
        print(f"[INFO] massnova Phase0 读取 mzML: {mzml_path}")
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
        print(f"[INFO] massnova Phase0: {len(chromatograms)} 条色谱 → {len(features)} 条有效通道"
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
                          edge_noise_stop_mode="stable_tail_mean", edge_max_span_min=1.0,
                          peer_rt_intervals=None):
    """
    Phase2（兜底路径）：对给定候选做两遍精修（D13）。
    首遍全部候选无内部 peer 粗走 → 以首遍结果为 peer 区间走第二遍，防相邻峰互相截停。
    peer_rt_intervals: 外部 peer（如已定的模型框边界），两遍走查均纳入防撞，
    防止残差候选的边界扩进已验证峰的区域。

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
    external = [(float(a), float(b)) for a, b in (peer_rt_intervals or [])
                if np.isfinite(float(a)) and np.isfinite(float(b)) and float(b) > float(a)]

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

    # 第一遍：无内部 peer（外部 peer 仍防撞）
    pass1 = [_walk(ap, external or None) for ap in apexes]
    # 第二遍：以第一遍结果 + 外部 peer 为 peer
    pass2 = []
    for j, ap in enumerate(apexes):
        peers = external + [(a, b) for i2, (a, b) in enumerate(pass1)
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


def gate_peaks(rt, intensity, apexes, intervals, min_snr=10.0, min_peak_span_points=5, min_area=0.0,
               neighbor_intervals=None):
    """
    Phase3a（兜底路径）：对精修后的候选峰做质量门控。
    neighbor_intervals: 额外邻居边界（如已定的模型框），SNR 噪声区估计时纳入。
    返回 [{"apex_idx", "rt_min", "rt_max", "snr", "n_points", "area"}, ...]（RT 排序）。
    """
    rt = np.asarray(rt, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    baseline_g = roi_full_low_decile_mean_intensity(y, bottom_frac=0.10)
    extra = [(float(a), float(b)) for a, b in (neighbor_intervals or [])
             if np.isfinite(float(a)) and np.isfinite(float(b)) and float(b) > float(a)]
    out = []
    for k, (rt_min, rt_max) in enumerate(intervals):
        if not (np.isfinite(rt_min) and np.isfinite(rt_max) and rt_max > rt_min):
            continue
        peers = extra + [(a, b) for j, (a, b) in enumerate(intervals)
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


def validate_with_model(peaks_by_channel, rts, intensities, model_path, threshold=0.5,
                        window_half_min=1.0, keep_windows=False, verbose=True,
                        predictor=None, onnx_use_gpu=0, onnx_batch_size=128):
    """
    Phase3b（模型前置，每 mzML 一次批量验证）：全部枚举候选（未经信号精修/门控，
    仅带预估 rt）一次性渲染到同一临时目录，单轮 build_predictor 推理，
    再按 apex 落框贪心配对（D9）回填各候选。

    peaks_by_channel: {ch_idx: [candidate dicts]}（每项至少含 rt_peak）；
    rts / intensities: {ch_idx: rt/y 数组}
    就地更新每候选：model_score / validated / boundary_source / rt_min / rt_max
    （命中 → 模型框 RT 即最终边界，boundary_source="model"；未命中候选保持原样，
    交由 Phase2/3a 兜底）。
    返回渲染窗口目录（keep_windows=True 时保留，否则清理并返回 None）。
    """
    if predictor is None and str(model_path).lower().endswith(".onnx"):
        from inference.onnx_window_predictor import OnnxWindowPredictor
        predictor = OnnxWindowPredictor(
            model_path, use_gpu=onnx_use_gpu, batch_size=onnx_batch_size)
    elif predictor is None:
        from utils.predict_utils import build_predictor  # 惰性导入，避免无模型时加载 torch

    tmp_dir = tempfile.mkdtemp(prefix="massnova_windows_")
    file_map = {}  # (ch_idx, p_idx) -> (shared window name, rt_lo, rt_hi)
    window_names = {}  # Same channel and exact bounds imply identical input arrays.

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
            window_key = (ch_idx, float(rt_lo), float(rt_hi))
            name = window_names.get(window_key)
            if name is None:
                name = f"win_{name_idx:07d}.jpeg"
                render_roi_jpeg(win_rt, win_int, rt_lo, rt_hi, os.path.join(tmp_dir, name))
                window_names[window_key] = name
                name_idx += 1
            file_map[(ch_idx, p_idx)] = (name, rt_lo, rt_hi)

    if not file_map:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # 2) 批量推理（单轮，模型只加载一次）
    try:
        if predictor is None:
            results = build_predictor(model_path=model_path, images_path=tmp_dir, threshold=threshold,
                                      plot=False, verbose=verbose)
        else:
            results = predictor(images_path=tmp_dir, threshold=threshold, verbose=verbose)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    img_results = {os.path.basename(str(r["image_path"])): r for r in results}

    # 3) 预计算候选配对（apex 落框）
    candidates: List[Tuple[int, int, str, int, float, float, float, float]] = []  # (ch, p, img, box_i, score, dist, left, right)
    for (ch_idx, p_idx), (name, rt_lo, rt_hi) in file_map.items():
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
            if right > left and left <= apex <= right:
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
                p["validated"] = bool(score > float(threshold))
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


def _finalize_peak_metrics(rt, intensity, peaks):
    """在最终边界上统一计算 snr / n_points / area（模型框与信号框同口径）。

    模型前置后，命中峰不再经过信号精修/门控，其指标需在模型框边界上补算；
    信号兜底峰也用同一口径重算（邻居集合含全部最终边界），保证 CSV 各行可比。
    """
    rt = np.asarray(rt, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    baseline_g = roi_full_low_decile_mean_intensity(y, bottom_frac=0.10)
    intervals = [(float(p["rt_min"]), float(p["rt_max"])) for p in peaks]
    for k, p in enumerate(peaks):
        lo, hi = intervals[k]
        mask = (rt >= lo) & (rt <= hi)
        n_pts_box = int(np.count_nonzero(mask))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo and n_pts_box >= 3:
            int_seg = np.maximum(y[mask], 0.0)
            p["n_points"] = _max_consec_above(int_seg, baseline_g)
            p["area"] = float(np.trapz(y[mask], rt[mask]) * float(AREA_TIME_UNIT_SCALE)) \
                if n_pts_box >= 2 else 0.0
        else:
            p["n_points"] = 0
            p["area"] = 0.0
        peers = [(a, b) for j, (a, b) in enumerate(intervals)
                 if j != k and np.isfinite(a) and np.isfinite(b) and b > a]
        snr = compute_local_snr(rt, y, lo, hi, neighbor_intervals=peers)
        p["snr"] = float(snr) if snr is not None and np.isfinite(float(snr)) else 0.0


def _finalize_peak_scores(peaks, *, snr_pivot=10.0, points_good=10.0,
                          snr_weight=0.8, points_weight=0.2):
    """给每个保留峰补齐来源分数和统一峰分；信号分不是模型概率。"""
    for p in peaks:
        if p.get("boundary_source") == "signal":
            score_signal, conf_snr, conf_points = signal_peak_score(
                p.get("snr", np.nan), p.get("n_points", np.nan),
                snr_pivot=snr_pivot, points_good=points_good,
                snr_weight=snr_weight, points_weight=points_weight,
            )
        else:
            score_signal = conf_snr = conf_points = np.nan
        score_peak, score_source = unified_peak_score(
            p.get("boundary_source", ""), p.get("model_score", np.nan), score_signal,
        )
        p["signal_score"] = score_signal
        p["signal_score_snr_component"] = conf_snr
        p["signal_score_points_component"] = conf_points
        p["peak_score"] = score_peak
        p["score_source"] = score_source


def plot_massnova_stage_windows(rt, intensity, peaks, stage_dir, chrom_index, uid, q1,
                                window_half_min=1.0, sigma=1.0, title_prefix="",
                                boundary_source=None):
    """pipeline 式 2min 窗口图：每峰 apex±window_half 切窗，窗内相交峰全部标注（多 query）。

    与 predictor._plot_model_xic 同用共享绘图核心 plot_xic_with_queries，样式一致；
    boundary_source=None 标注全部峰，否则只标注该来源（"signal"/"model"）的峰。
    返回本阶段实际生成的图数。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.plot_xic_peaks import plot_xic_with_queries

    rt = np.asarray(rt, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if rt.size < 2 or not peaks:
        return 0
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    slug = filesystem_slug_for_native_id(uid or "", max_len=48) or "empty"
    rt_lo_all = float(np.min(rt))
    rt_hi_all = float(np.max(rt))
    half = float(window_half_min) if window_half_min > 0 else 0.0

    def _match(pk):
        return boundary_source is None or pk.get("boundary_source") == boundary_source

    n_plots = 0
    for seed in peaks:
        if not _match(seed):
            continue
        apex = float(seed["rt_peak"])
        lo_w, hi_w = rt_lo_all, rt_hi_all
        if half > 0:
            lo_w = max(lo_w, apex - half)
            hi_w = min(hi_w, apex + half)
            if hi_w <= lo_w:
                continue
        queries = []
        for pk in peaks:
            if not _match(pk):
                continue
            if float(pk["rt_max"]) < lo_w or float(pk["rt_min"]) > hi_w:
                continue
            sc = pk.get("peak_score")
            queries.append({
                "rt_lo": float(pk["rt_min"]),
                "rt_hi": float(pk["rt_max"]),
                "rt_peak": float(pk["rt_peak"]),
                "height": float(pk["apex_intensity"]),
                "snr": float(pk["snr"]),
                "n_points": int(pk["n_points"]),
                "score": float(sc) if sc is not None and np.isfinite(float(sc)) else None,
                "score_source": pk.get("score_source", "missing"),
            })
        if not queries:
            continue
        mask = (rt >= lo_w) & (rt <= hi_w)
        if int(np.count_nonzero(mask)) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
        plot_xic_with_queries(
            ax, rt, intensity, queries, q1=q1, sigma=float(sigma),
            title="%s - %s | peak#%d" % (title_prefix, uid or "chrom", int(seed["peak_no"])),
            roi_window=(lo_w, hi_w),
        )
        out_png = stage_dir / ("chrom%03d_peak%03d_%s.png" % (int(chrom_index), int(seed["peak_no"]), slug))
        try:
            fig.savefig(out_png, bbox_inches="tight", dpi=150)
            n_plots += 1
        except Exception as e:
            print(f"[WARN] massnova 窗口图保存失败 {out_png}: {e}")
        finally:
            plt.close(fig)
    return n_plots


_MASSNOVA_WINDOW_STAGES = (
    # (文件夹名, boundary_source 过滤（None=全部）, 图标题前缀)
    ("prediction_signal", "signal", "Signal prediction"),
    ("prediction_model", "model", "Model prediction"),
    ("prediction_refined", None, "Refined prediction"),
)


def plot_massnova_scan(rt, intensity, peaks, out_path, uid, q1):
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
        sc = p.get("peak_score")
        sc_txt = "" if sc is None or not np.isfinite(float(sc)) else " %.2f" % float(sc)
        source_txt = " M" if p.get("score_source") == "model" else " S"
        ax.text(p["rt_peak"], y_top * 0.98, "#%d%s%s" % (p["peak_no"], sc_txt, source_txt),
                fontsize=8, color=color, rotation=45, ha="right", va="top")
    ax.set_xlim(float(np.min(rt)), float(np.max(rt)))
    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("Intensity")
    ax.set_title("massnova %s | Q1=%.4f | %d peaks" % (uid, float(q1), len(peaks)))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_massnova_model_xic(rt, intensity, peaks, out_path, uid, q1, sigma=1.0):
    """Phase4a'：model_plots（--plot）——与其他模式（pipeline/roi2inference）一致的可视化。

    复用共享绘图核心 plot_xic_with_queries：XIC 曲线 + 每峰彩色阴影（query）+ 峰顶竖线
    + 图例（含 score）+ 左上角信息框（RT/Q1/峰高/SNR/区间点数/置信度），样式与
    predictor._plot_model_xic 完全一致；每通道一张图。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.plot_xic_peaks import plot_xic_with_queries

    queries = []
    for p in peaks:
        lo, hi = float(p["rt_min"]), float(p["rt_max"])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            continue
        sc = p.get("peak_score")
        sc = float(sc) if sc is not None and np.isfinite(float(sc)) else None
        queries.append({
            "rt_lo": lo,
            "rt_hi": hi,
            "rt_peak": float(p["rt_peak"]),
            "height": float(p["apex_intensity"]),
            "snr": float(p["snr"]),
            "n_points": int(p["n_points"]),
            "score": sc,
            "score_source": p.get("score_source", "missing"),
        })
    if not queries:
        return False

    rt = np.asarray(rt, dtype=np.float64)
    span = float(np.max(rt) - np.min(rt)) if rt.size > 1 else 0.0
    fig_w = max(9.0, 1.2 * span)  # 宽度按 RT 跨度缩放（同 scan_plots），不小于其他模式的 9 inch
    fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=120)
    plot_xic_with_queries(
        ax, rt, intensity, queries, q1=q1, sigma=float(sigma),
        title="Model prediction - %s" % (uid or "chrom"),
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return True


def write_outputs(mzml_stem, features, peaks_by_channel, qc_excluded, out_root, key,
                  no_plots=False, keep_windows=None, plot=False, plot_sigma=1.0,
                  window_half_min=1.0):
    """
    Phase4：按实验布局写输出，返回峰明细行（供跨样本推理报告汇总）。

    实验布局（<实验根> = output_dir，缺省 ../output/inference/massnova_<实验名>）：
      <实验根>/prediction_refined/<样本>/   massnova_peaks.csv / scan_summary.csv /
                                            scan_qc_excluded.csv + 2min 窗口图（最终结果）
      <实验根>/prediction_signal/<样本>/    2min 窗口图（仅信号兜底峰边界）
      <实验根>/prediction_model/<样本>/     2min 窗口图（仅模型框峰边界）
      <实验根>/scan_plots/<样本>/           长 XIC 整谱标注图（保留）
      <实验根>/model_plots/<样本>/          长 XIC 标注图（与其他模式一致，保留）
      <实验根>/scan_windows/<样本>/         模型验证窗口 JPEG（--keep_windows）
    """
    out_root = Path(out_root)
    refined_dir = out_root / "prediction_refined" / key
    refined_dir.mkdir(parents=True, exist_ok=True)

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
                "signal_score": p.get("signal_score", np.nan),
                "signal_score_snr_component": p.get("signal_score_snr_component", np.nan),
                "signal_score_points_component": p.get("signal_score_points_component", np.nan),
                "peak_score": p.get("peak_score", np.nan),
                "score_source": p.get("score_source", "missing"),
                "boundary_source": p.get("boundary_source", "signal"),
            })

    pd.DataFrame(peak_rows).to_csv(refined_dir / "massnova_peaks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(refined_dir / "scan_summary.csv", index=False, encoding="utf-8-sig")
    if qc_excluded:
        pd.DataFrame(qc_excluded).to_csv(refined_dir / "scan_qc_excluded.csv",
                                         index=False, encoding="utf-8-sig")

    if not no_plots:
        plots_dir = out_root / "scan_plots" / key
        plots_dir.mkdir(parents=True, exist_ok=True)
        for f in features:
            ci = f["chrom_index"]
            peaks = peaks_by_channel.get(ci, [])
            slug = filesystem_slug_for_native_id(f["uid"] or "", max_len=48) or "empty"
            out_png = plots_dir / ("chrom%03d_%s.png" % (ci, slug))
            try:
                plot_massnova_scan(f["rt"], f["intensity"], peaks, str(out_png), f["uid"], f["q1"])
            except Exception as e:
                print(f"[WARN] massnova 绘图失败 {out_png}: {e}")

    # model_plots：与其他模式（pipeline/roi2inference）一致的可视化（--plot 开启）
    if plot:
        model_plots_dir = out_root / "model_plots" / key
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        n_model_plots = 0
        for f in features:
            ci = f["chrom_index"]
            peaks = peaks_by_channel.get(ci, [])
            if not peaks:
                continue
            slug = filesystem_slug_for_native_id(f["uid"] or "", max_len=48) or "empty"
            out_png = model_plots_dir / ("chrom%03d_%s_model.png" % (ci, slug))
            try:
                if plot_massnova_model_xic(f["rt"], f["intensity"], peaks, str(out_png),
                                           f["uid"], f["q1"], sigma=plot_sigma):
                    n_model_plots += 1
            except Exception as e:
                print(f"[WARN] massnova model_plots 绘图失败 {out_png}: {e}")
        if n_model_plots:
            print(f"[INFO] massnova model_plots: {n_model_plots} 张 → {model_plots_dir}")

        # pipeline 式 2min 窗口图（三阶段：signal / model / refined，实验根级按样本分子夹）
        for stage_name, bsrc, prefix in _MASSNOVA_WINDOW_STAGES:
            stage_dir = out_root / stage_name / key
            n_stage = 0
            for f in features:
                peaks = peaks_by_channel.get(f["chrom_index"], [])
                if not peaks:
                    continue
                try:
                    n_stage += plot_massnova_stage_windows(
                        f["rt"], f["intensity"], peaks, stage_dir, f["chrom_index"],
                        f["uid"], f["q1"], window_half_min=window_half_min,
                        sigma=plot_sigma, title_prefix=prefix, boundary_source=bsrc)
                except Exception as e:
                    print(f"[WARN] massnova {stage_name} 绘图失败 chrom{f['chrom_index']:03d}: {e}")
            if n_stage:
                print(f"[INFO] massnova {stage_name}: {n_stage} 张 → {stage_dir}")

    if keep_windows:
        win_src = keep_windows.get("all") if isinstance(keep_windows, dict) else None
        if win_src:
            target = out_root / "scan_windows" / key
            target.mkdir(parents=True, exist_ok=True)
            for fn in Path(win_src).iterdir():
                try:
                    shutil.copy2(fn, target / fn.name)
                except OSError:
                    pass

    return peak_rows
def _compound_of(uid):
    """uid 形如 '阿维菌素-1'（化合物名-离子通道）→ 化合物名（矩阵按化合物合并离子通道）。"""
    m = re.match(r"^(.+)-[12]$", str(uid or "").strip())
    return m.group(1) if m else str(uid or "").strip()


def write_massnova_report(out_root, exp_name, sample_infos, args, total_seconds=None):
    """Phase5：跨样本推理报告（对齐 pipeline 的 inference_report_<实验名>.md + all.csv）。

    sample_infos: run_massnova_on_mzml 返回的 info dict 列表。
    返回 {"report_md", "all_csv", "n_samples", "n_rows"}。
    """
    import datetime
    out_root = Path(out_root)
    rows = [r for info in sample_infos for r in info["peak_rows"]]
    all_csv = out_root / "all.csv"
    adf = pd.DataFrame(rows)
    adf.to_csv(all_csv, index=False, encoding="utf-8-sig")

    threshold = float(getattr(args, "threshold", 0.5))
    model_path = getattr(args, "model", None)
    samples = [info["key"] for info in sample_infos]

    L = []
    L.append("# 推理报告（massnova）")
    L.append("")
    L.append(f"- 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 推理模式: `massnova`（整谱 XIC 全峰识别，模型前置、精修兜底） | 输出目录: `{out_root}`")
    L.append(f"- 模型: `{model_path or '未启用（纯信号路径）'}` | 置信度阈值: {threshold}")
    L.append("- Phase1 候选检测器: SciPy find_peaks + prominence")
    if total_seconds is not None:
        L.append(f"- 样本数: {len(samples)} | 总耗时: {total_seconds:.1f}s")
    L.append("")

    # 1. 样本摘要
    L.append("## 1. 样本摘要")
    L.append("")
    L.append("| 样品 | 通道 | 候选 | 峰数 | 模型框 | 信号兜底 | 模型验证率 | 平均置信度 | 平均 SNR |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for info in sample_infos:
        df = pd.DataFrame(info["peak_rows"])
        n = len(df)
        n_model = int((df["boundary_source"] == "model").sum()) if n else 0
        sc = df["model_score"].dropna() if n else pd.Series(dtype=float)
        mean_sc = "%.3f" % float(sc.mean()) if len(sc) else "—"
        mean_snr = "%.3g" % float(df["snr"].mean()) if n else "—"
        val_rate = ("%.1f%%" % (100.0 * n_model / n)) if n else "—"
        L.append(f"| {info['key']} | {info['n_channels']} | {info['n_candidates']} | {n} | "
                 f"{n_model} | {n - n_model} | {val_rate} | {mean_sc} | {mean_snr} |")
    L.append("")

    # 2. 化合物 × 样品 检出面积矩阵
    L.append("## 2. 化合物 × 样品 检出面积矩阵（主峰, intensity·min）")
    L.append("")
    L.append("> 单元格 = 该样品该化合物各离子通道最大峰面积（—=未检出）；离子通道（uid 后缀 -1/-2）已合并。")
    L.append("")
    if not adf.empty:
        adf = adf.assign(_compound=adf["uid"].map(_compound_of))
        compounds = sorted(adf["_compound"].unique())
        L.append("| 化合物 | 检出/样品 |" + "".join(f" {s} |" for s in samples))
        L.append("|---|---|" + "|".join(["---"] * len(samples)))
        for c in compounds:
            g_all = adf[adf["_compound"] == c]
            hit = [s for s in samples
                   if len(g_all[(g_all["mzml_stem"] == s) & (g_all["area"] > 0)])]
            cells = []
            for s in samples:
                sub = g_all[(g_all["mzml_stem"] == s) & (g_all["area"] > 0)]
                cells.append("%.1f" % float(sub["area"].max()) if len(sub) else "—")
            L.append(f"| {c} | {len(hit)}/{len(samples)} | " + " | ".join(cells) + " |")
    else:
        L.append("（无数据）")
    L.append("")

    # 3. 输出布局与列说明
    L.append("## 3. 输出布局")
    L.append("")
    L.append("```text")
    L.append(f"{out_root.name}/")
    L.append("  prediction_signal/<样品>/    2min 窗口图（仅信号兜底峰边界，无模型分）")
    L.append("  prediction_model/<样品>/     2min 窗口图（仅模型框峰边界，含 score）")
    L.append("  prediction_refined/<样品>/   最终结果：massnova_peaks.csv / scan_summary.csv /")
    L.append("                               scan_qc_excluded.csv + 2min 窗口图（全部峰）")
    L.append("  scan_plots/<样品>/           长 XIC 整谱标注图（红=模型验证，灰=信号兜底）")
    L.append("  model_plots/<样品>/          长 XIC 标注图（与其他模式同款，全峰）")
    L.append("  all.csv                      跨样本合并明细（每峰一行）")
    L.append("  inference_report_<实验名>.md  本报告")
    L.append("```")
    L.append("")
    L.append("## 4. massnova_peaks.csv 关键列")
    L.append("")
    L.append("| 列 | 含义 |")
    L.append("|---|---|")
    L.append("| rt_min / rt_peak / rt_max | 峰边界与峰顶（min）；模型框峰取模型框边界 |")
    L.append("| area / snr / n_points | 最终边界上的面积 / 本地信噪比 / baseline 以上连续点数 |")
    L.append("| validated | 模型验证通过（model_score > 阈值） |")
    L.append("| model_score | 模型置信度；未命中模型为空 |")
    L.append("| boundary_source | model=模型框边界；signal=信号兜底精修边界（建议人工复核） |")
    L.append("")

    # 5. 已知问题与解决方案（实验记录）
    L.append("## 5. 已知问题与解决方案（实验记录）")
    L.append("")
    L.append("开发/实验过程中遇到的问题抽象化归档，含根因层与对策状态：")
    L.append("")
    L.append("| 编号 | 抽象问题 | 根因层 | 典型实例 | 状态与对策 |")
    L.append("|---|---|---|---|---|")
    L.append("| P1 | 多候选架构下同一色谱簇被多个候选独立切窗验证，输出重叠重复框（跨窗口无协调机制；"
             "pipeline 有单 ROI 天然防线 + (mz,q3) 面积去重，massnova 缺等价环节） | 架构层 | "
             "恶虫威-1 5.0–5.6 三头簇曾输出 3 个 validated 峰，面积重复计约 3 遍 | "
             "✅ 已修：仅在 apex 接近、区间明显重叠且峰间无深谷时去重；"
             "模型框优先保留代表框） |")
    L.append("| P2 | 边界走查截停阈值（stable_tail_mean）在宽峰/双驼峰缓降尾上不收敛，"
             "兜底框远大于真实峰跨度 | 信号层 | 莠去津-1 兜底框 1.59min vs 真实 0.71min；"
             "恶虫威-2 兜底框 4.58–6.18 | ✅ 已修：兜底宽度保险丝（单侧跨度 > "
             "scan_width_fuse_ratio×半高跨度即回缩，且先回缩再门控） |")
    L.append("| P3 | SNR 邻居区段噪声估计把本峰边界拖尾的信号衰减计入 p-p 噪声 → SNR 被抬高数十倍低估 "
             "→ 真峰被 SNR 门误杀 | 后处理层 | 甲羧除草醚-2 16.97min 峰 SNR=3.6（修复后 >800） | "
             "✅ 已修：compute_local_snr 安静点过滤统一作用于邻居区段与回退扇区 |")
    L.append("| P4 | 模型对非「单峰居中」形态（多头簇/双驼峰/偏心峰）框回归偏窄、偏移甚至缺失 | "
             "模型层（训练分布：ROI 均为 2min 窗单峰居中） | 恶虫威-1 去重后保留框只罩主头；"
             "莠去津-1 漏检、-2 只框半个峰 | ⚠️ 未根治：训练端需补多峰/偏心样本重训；"
             "推理端可做「模型框+信号延拓」混合边界校正（待做） |")
    L.append("| P5 | GPU 推理在阈值边缘（score≈阈值）存在运行间数值抖动，validated/兜底归属可能翻转 | "
             "工程层 | 联苯三唑醇-1 相邻两次运行模型命中翻转 | "
             "ℹ️ 已知现象：阈值附近峰建议结合 boundary_source 列人工复核 |")
    L.append("")
    L.append("P1/P2 对应参数：`scan_dup_apex_tol`（默认 0.2min，0=关闭）、"
             "`scan_dup_min_overlap_fraction`（默认 0.25）、"
             "`scan_dup_shallow_valley_min_ratio`（默认 0.70）、"
             "`scan_width_fuse_ratio`（默认 1.5，0=关闭）。")
    L.append("")
    # 空白样本假阳性预警（对齐 pipeline 报告行为）
    blanks = [s for s in samples if s.startswith(("空白", "BLANK", "blank"))]
    if blanks and not adf.empty:
        fp = adf[(adf["mzml_stem"].isin(blanks)) & (adf["validated"])]
        if len(fp):
            L.append(f"> [WARN] 空白样本检出 {len(fp)} 个模型验证峰（疑似假阳性，详见上方矩阵）。")
            L.append("")

    L.append("---")
    L.append("> 由 `model/inference/massnova.py` 生成。")
    report_md = out_root / ("inference_report_%s.md" % exp_name)
    report_md.write_text("\n".join(L), encoding="utf-8")

    print(f"[INFO] 合并明细: {all_csv} ({len(adf)} 行)")
    print(f"[INFO] 推理报告: {report_md}")
    print("===== 样本摘要 =====")
    for info in sample_infos:
        df = pd.DataFrame(info["peak_rows"])
        n_model = int((df["boundary_source"] == "model").sum()) if len(df) else 0
        print(f"  {info['key']}: 通道={info['n_channels']} 候选={info['n_candidates']} "
              f"峰={len(df)} 模型框={n_model} 信号兜底={len(df) - n_model}")
    return {"report_md": str(report_md), "all_csv": str(all_csv),
            "n_samples": len(samples), "n_rows": len(adf)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _dedup_identical_peak_bounds(peaks, tolerance=1e-6):
    """Keep one final box per near-identical interval, independent of apex valleys."""
    def rank(peak):
        score = float(peak.get("peak_score", float("-inf")))
        return (peak.get("boundary_source") == "model",
                score if np.isfinite(score) else float("-inf"))

    kept = []
    for peak in sorted(peaks, key=rank, reverse=True):
        if any(abs(float(peak["rt_min"]) - float(other["rt_min"])) <= tolerance
               and abs(float(peak["rt_max"]) - float(other["rt_max"])) <= tolerance
               for other in kept):
            continue
        kept.append(peak)
    return sorted(kept, key=lambda peak: float(peak["rt_min"]))


def _dedup_overlapping_peaks(
        peaks, apex_tol=0.2, rt=None, intensity=None,
        min_overlap_fraction=0.25, shallow_valley_min_ratio=0.70):
    """仅在区间明显重叠且峰间没有深谷时删除重复候选框。

    判为重复必须同时满足：峰顶距离不超过 apex_tol、重叠宽度至少占较窄区间
    min_overlap_fraction、基线校正后的谷底至少达到较小峰顶的
    shallow_valley_min_ratio。这样可避免仅因边界轻微接触而误删相邻真实峰。
    代表峰优先级仍为模型框 > 信号框；同来源保留峰顶更高者。
    """
    if apex_tol <= 0 or len(peaks) <= 1:
        return peaks

    rt_arr = np.asarray(rt, dtype=np.float64) if rt is not None else None
    y_arr = np.asarray(intensity, dtype=np.float64) if intensity is not None else None
    has_signal = (rt_arr is not None and y_arr is not None and rt_arr.size == y_arr.size
                  and rt_arr.size > 0)
    baseline = (roi_full_low_decile_mean_intensity(y_arr, bottom_frac=0.10)
                if has_signal else np.nan)

    def _apex_index(p):
        idx = p.get("apex_idx")
        if has_signal and idx is not None:
            try:
                idx = int(idx)
                if 0 <= idx < y_arr.size:
                    return idx
            except (TypeError, ValueError):
                pass
        if has_signal:
            return int(np.argmin(np.abs(rt_arr - float(p["rt_peak"]))))
        return None

    def _has_no_deep_valley(a, b):
        if not has_signal:
            # Without the XIC, conservatively retain both candidates because
            # the absence of a separating valley cannot be established.
            return False
        ia, ib = _apex_index(a), _apex_index(b)
        if ia is None or ib is None:
            return False
        lo, hi = sorted((ia, ib))
        if lo == hi:
            return True
        valley = float(np.min(y_arr[lo:hi + 1]))
        smaller_apex = min(float(y_arr[ia]), float(y_arr[ib]))
        denominator = smaller_apex - float(baseline)
        if denominator <= 0:
            return False
        valley_ratio = (valley - float(baseline)) / denominator
        return valley_ratio >= float(shallow_valley_min_ratio)

    def _rank(p):
        return (1 if p.get("boundary_source") == "model" else 0,
                float(p.get("apex_intensity") or 0.0))

    out = []
    for p in sorted(peaks, key=lambda q: float(q["rt_min"])):
        if not out:
            out.append(p)
            continue
        kept = out[-1]
        overlap_width = (min(float(p["rt_max"]), float(kept["rt_max"]))
                         - max(float(p["rt_min"]), float(kept["rt_min"])))
        narrower_width = min(float(p["rt_max"]) - float(p["rt_min"]),
                             float(kept["rt_max"]) - float(kept["rt_min"]))
        overlap_fraction = (overlap_width / max(narrower_width, 1e-12)
                            if overlap_width > 0 and narrower_width > 0 else 0.0)
        substantial_overlap = overlap_fraction >= float(min_overlap_fraction)
        near = abs(float(p["rt_peak"]) - float(kept["rt_peak"])) <= float(apex_tol)
        no_deep_valley = _has_no_deep_valley(kept, p)
        if substantial_overlap and near and no_deep_valley:
            if _rank(p) > _rank(kept):
                out[-1] = p
        else:
            out.append(p)
    return out


def _half_width_spans(rt, y, apex_idx, lo, hi, baseline):
    """从峰顶向两侧找强度跌破 baseline+0.5×峰高 的位置，返回 (左跨, 右跨)（min）。"""
    apex_val = float(y[apex_idx])
    thr = baseline + 0.5 * (apex_val - baseline)
    i = int(apex_idx)
    while i > 0 and float(y[i]) > thr and float(rt[i]) > float(lo):
        i -= 1
    j = int(apex_idx)
    n = int(y.size)
    while j < n - 1 and float(y[j]) > thr and float(rt[j]) < float(hi):
        j += 1
    return float(rt[apex_idx] - rt[i]), float(rt[j] - rt[apex_idx])


def fuse_fallback_boundaries(rt, y, apexes, intervals, ratio=1.5, baseline=None):
    """兜底宽度保险丝：单侧跨度超过 ratio×该侧半高跨度时回缩（只收缩不扩张）。

    stable_tail_mean 截停在宽峰/双驼峰的缓降尾上找不到稳定平台，边界会一路走到
    安静区（如莠去津-1 兜底框 1.59min vs 真实 0.71min）。半高跨度是从峰顶到
    "半峰高强度点"的距离，对峰形自适应；ratio 默认 1.5。ratio<=0 关闭。
    """
    if ratio <= 0 or not apexes:
        return list(intervals)
    rt = np.asarray(rt, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if baseline is None:
        baseline = roi_full_low_decile_mean_intensity(y, bottom_frac=0.10)
    out = []
    for ap, (lo, hi) in zip(apexes, intervals):
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            out.append((lo, hi))
            continue
        apex_rt = float(rt[int(ap)])
        left_w, right_w = _half_width_spans(rt, y, int(ap), lo, hi, baseline)
        new_lo = max(lo, apex_rt - ratio * max(left_w, 1e-3))
        new_hi = min(hi, apex_rt + ratio * max(right_w, 1e-3))
        out.append((new_lo, new_hi) if new_hi > new_lo else (lo, hi))
    return out


def _collect_mzml_inputs(mzml_arg, batch_dir_arg):
    """惰性复用 cli._collect_mzml_inputs，避免 cli ↔ massnova 循环导入。"""
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
        "min_snr": float(_g("scan_min_snr", 10.0)),
        "min_peak_span_points": int(_g("scan_min_peak_span_points", 5)),
        "min_area": float(_g("scan_min_area", 0.0)),
        "window_half_min": float(_g("scan_window_half_min", 1.0)),
        "dup_apex_tol": float(_g("scan_dup_apex_tol", 0.2)),
        "dup_min_overlap_fraction": float(_g("scan_dup_min_overlap_fraction", 0.25)),
        "dup_shallow_valley_min_ratio": float(_g("scan_dup_shallow_valley_min_ratio", 0.70)),
        "signal_score_snr_pivot": float(_g("signal_score_snr_pivot", 10.0)),
        "signal_score_points_good": float(_g("signal_score_points_good", 10.0)),
        "signal_score_snr_weight": float(_g("signal_score_snr_weight", 0.8)),
        "signal_score_points_weight": float(_g("signal_score_points_weight", 0.2)),
        "width_fuse_ratio": float(_g("scan_width_fuse_ratio", 1.5)),
    }


def finalize_channel_peaks(rt, intensity, candidates, scan_params, *, threshold=0.5, verbose=False,
                           channel_label=""):
    """Build final model + signal peaks for one already-smoothed XIC channel.

    This is the shared post-processing core used by both mzML inference and the
    embedded DLL runtime.  Keeping it here prevents the C bridge from growing a
    second, subtly different implementation of refinement, gating, scoring and
    duplicate removal.
    """
    rt = np.asarray(rt, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    sp = scan_params
    peaks = []
    residual_cands = []
    for candidate in candidates:
        if candidate.get("validated"):
            peaks.append({
                "peak_no": 0,
                "apex_idx": int(candidate["apex_idx"]),
                "rt_peak": float(candidate["rt_peak"]),
                "rt_min": float(candidate["rt_min"]),
                "rt_max": float(candidate["rt_max"]),
                "apex_intensity": float(candidate["apex_intensity"]),
                "area": 0.0,
                "snr": 0.0,
                "n_points": 0,
                "model_score": float(candidate["model_score"]),
                "validated": True,
                "boundary_source": "model",
            })
        else:
            residual_cands.append(candidate)

    if residual_cands:
        model_peers = [(peak["rt_min"], peak["rt_max"]) for peak in peaks]
        residual_apexes = [int(candidate["apex_idx"]) for candidate in residual_cands]
        intervals = refine_all_boundaries(
            rt, y, residual_apexes,
            init_half_width_min=sp["init_half_width_min"],
            boundary_posterior_lookahead=sp["boundary_posterior_lookahead"],
            boundary_posterior_mean_scale=sp["boundary_posterior_mean_scale"],
            edge_noise_stop_mode=sp["edge_noise_stop_mode"],
            edge_max_span_min=sp["edge_max_span_min"],
            peer_rt_intervals=model_peers or None,
        )
        intervals = fuse_fallback_boundaries(
            rt, y, residual_apexes, intervals, ratio=sp["width_fuse_ratio"])
        gated = gate_peaks(
            rt, y, residual_apexes, intervals,
            min_snr=sp["min_snr"],
            min_peak_span_points=sp["min_peak_span_points"],
            min_area=sp["min_area"],
            neighbor_intervals=model_peers,
        )
        for gated_peak in gated:
            apex_index = int(gated_peak["apex_idx"])
            peaks.append({
                "peak_no": 0,
                "apex_idx": apex_index,
                "rt_peak": float(rt[apex_index]),
                "rt_min": float(gated_peak["rt_min"]),
                "rt_max": float(gated_peak["rt_max"]),
                "apex_intensity": float(y[apex_index]),
                "area": float(gated_peak["area"]),
                "snr": float(gated_peak["snr"]),
                "n_points": int(gated_peak["n_points"]),
                "model_score": np.nan,
                "validated": False,
                "boundary_source": "signal",
            })

    count_before_dedup = len(peaks)
    peaks = _dedup_overlapping_peaks(
        peaks,
        apex_tol=sp["dup_apex_tol"],
        rt=rt,
        intensity=y,
        min_overlap_fraction=sp["dup_min_overlap_fraction"],
        shallow_valley_min_ratio=sp["dup_shallow_valley_min_ratio"],
    )
    if verbose and len(peaks) < count_before_dedup:
        prefix = f" {channel_label}" if channel_label else ""
        print(f"[massnova]{prefix}: 去重 {count_before_dedup - len(peaks)} 个重叠候选框")
    peaks.sort(key=lambda peak: peak["rt_min"])
    for peak_no, peak in enumerate(peaks, start=1):
        peak["peak_no"] = peak_no
    _finalize_peak_metrics(rt, y, peaks)
    _finalize_peak_scores(
        peaks,
        snr_pivot=sp["signal_score_snr_pivot"],
        points_good=sp["signal_score_points_good"],
        snr_weight=sp["signal_score_snr_weight"],
        points_weight=sp["signal_score_points_weight"],
    )
    # Do not let the deep-valley safeguard preserve duplicate output intervals.
    # Scores are now available for deterministic selection within each source.
    peaks = [peak for peak in peaks
             if np.isfinite(peak["peak_score"]) and peak["peak_score"] > float(threshold)]
    peaks = _dedup_identical_peak_bounds(peaks)
    for peak_no, peak in enumerate(peaks, start=1):
        peak["peak_no"] = peak_no
    return peaks


def run_massnova_on_mzml(mzml_path, key, args, out_root):
    """对单个 mzML 执行整谱全峰识别（模型前置、精修兜底），返回 info dict（含峰明细行）。

    流程：Phase1 枚举候选（仅预估 rt）→ Phase3b 模型前置（全部候选切窗推理，
    命中者边界直接取模型框）→ Phase2/3a 兜底（仅对未命中残差做边界精修 + 门控）
    → 最终边界上统一补算指标 → 按实验布局写输出。
    """
    smooth_sigma = float(getattr(args, "smooth_sigma", 0.8) or 0.0)
    min_max_intensity = float(getattr(args, "pipeline_min_max_intensity", 1000.0) or 0.0)
    min_chrom_points = int(getattr(args, "pipeline_min_chrom_points", 10) or 0)
    model_path = getattr(args, "model", None)
    threshold = float(getattr(args, "threshold", 0.5))
    keep_windows = bool(getattr(args, "keep_windows", False))
    no_plots = bool(getattr(args, "no_plots", False))
    plot = bool(getattr(args, "plot", False))
    plot_sigma = float(getattr(args, "plot_smooth_sigma", 1.0) or 1.0)
    sp = _scan_params_from_args(args)

    features, qc_excluded = extract_full_xics(
        mzml_path, smooth_sigma=smooth_sigma,
        min_chrom_points=min_chrom_points, min_max_intensity=min_max_intensity,
    )

    phase1_keys = ("baseline_percentile", "baseline_mode", "min_peak_ratio",
                   "prominence_ratio", "min_prominence_abs", "min_peak_gap_points",
                   "min_peak_width_min", "void_time_min", "max_peaks_per_channel",
                   "valley_ratio", "min_valley_central_frac")

    # Phase1：候选枚举（预估 rt，不做信号精修）
    candidates_by_channel: Dict[int, list] = {}
    rts: Dict[int, np.ndarray] = {}
    intensities: Dict[int, np.ndarray] = {}
    n_candidates = 0
    for f in features:
        ci = f["chrom_index"]
        rt = f["rt"]
        y = f["intensity"]
        rts[ci] = rt
        intensities[ci] = y
        apexes = enumerate_peaks(rt, y, **{k: v for k, v in sp.items() if k in phase1_keys})
        cands = [{
            "apex_idx": int(ap),
            "rt_peak": float(rt[ap]),
            "apex_intensity": float(y[ap]),
            "model_score": np.nan,
            "validated": False,
        } for ap in apexes]
        candidates_by_channel[ci] = cands
        n_candidates += len(cands)

    # Phase3b（前置）：模型验证全部候选；命中者边界直接取模型框
    keep_win_dirs: Dict[int, str] = {}
    if model_path:
        win_dir = validate_with_model(
            candidates_by_channel, rts, intensities, model_path, threshold=threshold,
            window_half_min=sp["window_half_min"],
            keep_windows=keep_windows,
            verbose=bool(getattr(args, "verbose", False)),
            onnx_use_gpu=int(getattr(args, "use_gpu", 0)),
            onnx_batch_size=int(getattr(args, "batch_size", 128)),
        )
        if win_dir:
            keep_win_dirs = {"all": win_dir}

    # Phase2/3a（兜底）+ Phase4：使用与嵌入式数组入口相同的最终峰处理核心。
    peaks_by_channel: Dict[int, list] = {}
    n_peaks_total = 0
    for f in features:
        ci = f["chrom_index"]
        rt = f["rt"]
        y = f["intensity"]
        peaks = finalize_channel_peaks(
            rt, y, candidates_by_channel.get(ci, []), sp,
            threshold=threshold,
            verbose=bool(getattr(args, "verbose", False)),
            channel_label=f"通道#{ci} {f['uid']}",
        )
        peaks_by_channel[ci] = peaks
        n_peaks_total += len(peaks)
        if getattr(args, "verbose", False):
            n_model = sum(p["boundary_source"] == "model" for p in peaks)
            print(f"[massnova] 通道#{ci} {f['uid']}: 候选 {len(candidates_by_channel.get(ci, []))}"
                  f" → 模型命中 {n_model} + 信号兜底 {len(peaks) - n_model} 峰")
    mzml_stem = Path(mzml_path).stem
    peak_rows = write_outputs(mzml_stem, features, peaks_by_channel, qc_excluded,
                              out_root, key,
                              no_plots=no_plots,
                              keep_windows=keep_win_dirs if keep_windows else None,
                              plot=plot, plot_sigma=plot_sigma,
                              window_half_min=sp["window_half_min"])
    print(f"[OK] massnova {mzml_stem}: {len(features)} 通道 / {n_candidates} 候选"
          f" / {n_peaks_total} 峰 → {Path(out_root) / 'prediction_refined' / key}")
    return {
        "key": key,
        "mzml_stem": mzml_stem,
        "n_channels": len(features),
        "n_candidates": n_candidates,
        "n_peaks": n_peaks_total,
        "peak_rows": peak_rows,
    }


def main(args):
    """统一入口（cli --mode massnova 与 python -m inference.massnova 共用）。

    实验布局（对齐 pipeline）：输出根缺省 ../output/inference/massnova_<实验名>，
    内含 prediction_signal / prediction_model / prediction_refined（各按样本分子夹）、
    scan_plots / model_plots、all.csv 与 inference_report_<实验名>.md。
    """
    import time
    from .cli import _resolve_exp_name

    t0 = time.perf_counter()
    mzml_inputs = _collect_mzml_inputs(getattr(args, "mzml", None), getattr(args, "batch_dir", None))
    exp_name = _resolve_exp_name(args)
    if getattr(args, "output_dir", None):
        out_root = Path(args.output_dir)
    else:
        out_root = ROOT_DIR.parent / "output" / "inference" / ("massnova_%s" % exp_name)
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sample_infos = []
    for mzml_path, key in mzml_inputs:
        try:
            sample_infos.append(run_massnova_on_mzml(str(mzml_path), key, args, out_root))
        except Exception as e:
            print(f"[ERROR] massnova 失败 {mzml_path}: {e}")
    total_peaks = sum(info["n_peaks"] for info in sample_infos)
    total_ch = sum(info["n_channels"] for info in sample_infos)
    total_seconds = time.perf_counter() - t0

    rep = {}
    if sample_infos:
        try:
            rep = write_massnova_report(out_root, exp_name, sample_infos, args,
                                        total_seconds=total_seconds)
        except Exception as e:
            print(f"[WARN] massnova 推理报告生成失败（不影响主流程）: {e}")
    print(f"[DONE] massnova 完成: {len(mzml_inputs)} 个 mzML, 合计 {total_ch} 通道 / "
          f"{total_peaks} 峰 → {out_root}"
          + (f" | 报告: {rep.get('report_md')}" if rep else ""))


def build_parser():
    ap = argparse.ArgumentParser(description="整谱 XIC 全峰识别（massnova）")
    ap.add_argument("--mzml", type=str, default=None, help="单个 mzML 或包含 mzML 的目录（递归）")
    ap.add_argument("--batch_dir", type=str, default=None, help="mzML 目录（递归）")
    ap.add_argument("--model", type=str, default=None, help="模型路径（提供则开启模型验证）")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--use_gpu", type=int, choices=[-1, 0, 1], default=0,
                    help="ONNX: -1=CPU，0=优先GPU失败回退CPU，1=必须GPU")
    ap.add_argument("--batch_size", type=int, default=128, help="ONNX 候选窗口批大小")
    ap.add_argument("--smooth_sigma", type=float, default=0.8, help="高斯平滑 sigma；0=关闭")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="实验输出根目录；null=按 pipeline 规则 ../output/inference/massnova_<实验名>")
    ap.add_argument("--exp_name", type=str, default=None,
                    help="实验名（用于输出目录 massnova_<实验名> 与推理报告 inference_report_<实验名>.md；"
                         "缺省回退：单 mzML→文件名，目录输入→目录名，再兜底 UTC 时间戳）")
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
    ap.add_argument("--scan_min_snr", type=float, default=10.0)
    ap.add_argument("--scan_min_peak_span_points", type=int, default=5)
    ap.add_argument("--scan_min_area", type=float, default=0.0)
    ap.add_argument("--scan_window_half_min", type=float, default=1.0)
    ap.add_argument("--scan_dup_apex_tol", type=float, default=0.2,
                    help="跨候选去重：峰顶间距上限；还需明显区间重叠且没有深谷；0=关闭")
    ap.add_argument("--scan_dup_min_overlap_fraction", type=float, default=0.25,
                    help="跨候选去重：重叠宽度至少占较窄区间的比例")
    ap.add_argument("--scan_dup_shallow_valley_min_ratio", type=float, default=0.70,
                    help="跨候选去重：基线校正后谷底/较小峰顶不低于该比例才视为无深谷")
    ap.add_argument("--scan_width_fuse_ratio", type=float, default=1.5,
                    help="兜底宽度保险丝：单侧跨度超过 ratio×半高跨度时回缩；0=关闭")
    ap.add_argument("--signal_score_snr_pivot", type=float, default=10.0,
                    help="信号峰规则分：SNR分量达到0.5时的SNR")
    ap.add_argument("--signal_score_points_good", type=float, default=10.0,
                    help="信号峰规则分：点数分量达到1的有效峰点数")
    ap.add_argument("--signal_score_snr_weight", type=float, default=0.8)
    ap.add_argument("--signal_score_points_weight", type=float, default=0.2)
    ap.add_argument("--keep_windows", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="生成 model_plots/（与其他模式一致的 XIC 曲线+多 query 阴影+置信度信息框）")
    ap.add_argument("--plot_smooth_sigma", type=float, default=1.0,
                    help="model_plots 绘图平滑 sigma")
    ap.add_argument("--verbose", action="store_true")
    return ap


if __name__ == "__main__":
    main(build_parser().parse_args())
