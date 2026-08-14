# -*- coding: utf-8 -*-
"""
Unified workflow for:
1) post-newtest peak refinement (interval correction + small-peak + valley fallback),
2) standard (calibration) mode selection/repair,
3) sample mode filtering + final composite confidence.

This script is designed to reuse existing project logic as much as possible:
- adjust_first_round_interval（含阈值后验 lookahead 与同伴框防撞峰，run_two_round_detection.py）
- compute_snr_outside_box / has_secondary_peak_in_roi (utils/xic_peak_utils.py)
- valley split gate (_split_one_box_by_valley from valley_split.py)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..inference.two_round_detection import (
    adjust_first_round_interval,
    remove_overlap_from_second_interval,
    walk_interval_left_to_noise_with_posterior,
    walk_interval_right_to_noise_with_posterior,
)
from utils.xic_peak_utils import (
    compute_snr_outside_box,
    get_last_25pct_avg_noise,
    has_secondary_peak_in_roi,
    one_sided_edge_stop_threshold_stable_tail_mean,
    one_sided_low_noise_baseline,
    roi_full_low_decile_mean_intensity,
)
from .valley_split import (
    _split_one_box_by_valley,
    _find_best_split_pair_prominence,
    DEFAULT_VALLEY_SPLIT_PARAMS,
)
from utils.roi_rt_mapping import box_to_rt_range


def _safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _max_consecutive_positive_points(arr: np.ndarray) -> int:
    """Count longest consecutive run where intensity > 0."""
    y = np.asarray(arr, dtype=np.float64)
    if y.size == 0:
        return 0
    gt0 = y > 0
    if not np.any(gt0):
        return 0
    d = np.diff(gt0.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if gt0[0]:
        starts = np.insert(starts, 0, 0)
    if gt0[-1]:
        ends = np.append(ends, y.size)
    if starts.size == 0 or ends.size == 0:
        return 0
    return int(np.max(ends - starts))


def _scan_points_in_interval(
    rt_full: np.ndarray,
    intensity_full: np.ndarray,
    rt_min: float,
    rt_max: float,
) -> float:
    """
    Recompute scan-point metric inside an RT interval using the same
    definition as prediction stage: longest consecutive run of intensity > 0.
    """
    if not (np.isfinite(rt_min) and np.isfinite(rt_max) and rt_max > rt_min):
        return float("nan")
    rt = np.asarray(rt_full, dtype=np.float64)
    yy = np.asarray(intensity_full, dtype=np.float64)
    if rt.size == 0 or yy.size == 0 or rt.size != yy.size:
        return float("nan")
    m = (rt >= float(rt_min)) & (rt <= float(rt_max))
    if not np.any(m):
        return 0.0
    return float(_max_consecutive_positive_points(yy[m]))


def _compound_key_from_prediction_row(r: pd.Series) -> str:
    """与 testXIC 通道一致：mz+q3，若有 native_id 则追加 slug（多 transition 同 mz/q3 可区分）。"""
    mz = _safe_float(r.get("mz"), np.nan)
    q3 = _safe_float(r.get("q3"), np.nan)
    key = "mz%.4f_q3%.4f" % (mz, q3)
    nid = r.get("native_id")
    if nid is not None and not (isinstance(nid, float) and np.isnan(nid)):
        s = str(nid).strip()
        if s:
            from utils.mzml_chromatogram_ids import filesystem_slug_for_native_id

            key = "%s_nid%s" % (key, filesystem_slug_for_native_id(s, max_len=40))
    return key


def _safe_folder_name(stem: str) -> str:
    """Windows 非法路径字符替换为下划线（与 extract_json.py 一致）。"""
    bad = '<>:"/\\|?*'
    return "".join(c if c not in bad else "_" for c in stem)


def _matplotlib_cjk_font():
    """Best-effort Chinese labels for plot annotations (Windows / common fonts)."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if path is None or not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _prediction_row_for_image(pred_df: Optional[pd.DataFrame], image_key: str) -> Optional[pd.Series]:
    if pred_df is None or pred_df.empty or not image_key:
        return None
    key = str(image_key).strip()
    if "image" not in pred_df.columns:
        return None
    m = pred_df["image"].astype(str).str.strip() == key
    if not m.any():
        stem = Path(key).stem
        m = pred_df["image"].astype(str).apply(lambda s: Path(str(s).strip()).stem == stem)
    if not m.any():
        return None
    return pred_df.loc[m].iloc[0]


def _load_feature_table_for_mapping(
    result_dir: Path,
    xic_path: Path,
    xic_dir_opt: Optional[str],
) -> Optional[pd.DataFrame]:
    """Load feature.csv for row alignment (SNR dir first, then xic matrix dir, then --xic_dir)."""
    candidates: List[Path] = [result_dir / "feature.csv", xic_path.parent / "feature.csv"]
    if xic_dir_opt:
        candidates.append(Path(xic_dir_opt).resolve() / "feature.csv")
    for p in candidates:
        df = _load_optional_csv(p)
        if df is not None and not df.empty:
            return df
    return None


def _parse_roi_index_from_image_name(image_name: str) -> Optional[int]:
    """Parse 1-based ROI index from stems like ``27_mz706.5000_q3318.2000`` (strip ``_snr...`` suffix)."""
    stem = Path(str(image_name).strip()).stem
    if "_snr" in stem.lower():
        stem = re.split(r"_snr", stem, maxsplit=1, flags=re.IGNORECASE)[0]
    m = re.match(r"^(\d+)_mz", stem, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    return n if n > 0 else None


def _parse_mz_from_image_stem(image_name: str) -> Optional[float]:
    stem = Path(str(image_name).strip()).stem
    if "_snr" in stem.lower():
        stem = re.split(r"_snr", stem, maxsplit=1, flags=re.IGNORECASE)[0]
    m = re.search(r"_mz([\d.]+)", stem, flags=re.IGNORECASE)
    if not m:
        return None
    return _safe_float(m.group(1), np.nan)


def _find_feature_row_index(
    ft: Optional[pd.DataFrame],
    mz,
    q3,
    image_name: str,
    compound_name=None,
) -> Optional[int]:
    """
    Resolve 0-based feature / xic_matrix row index.

    Priority: (mz, q3) → ROI filename prefix ``N_mz...`` → Compound Name column.
    """
    if ft is None or ft.empty:
        return None

    mz_f = _safe_float(mz, np.nan)
    q3_f = _safe_float(q3, np.nan)
    if np.isfinite(mz_f) and np.isfinite(q3_f) and "mz" in ft.columns and "q3" in ft.columns:
        mz_r = ft["mz"].apply(lambda v: round(_safe_float(v, np.nan), 4))
        q3_r = ft["q3"].apply(lambda v: round(_safe_float(v, np.nan), 2))
        m = (mz_r == round(mz_f, 4)) & (q3_r == round(q3_f, 2))
        if m.any():
            return int(np.flatnonzero(m.to_numpy())[0])

    roi_n = _parse_roi_index_from_image_name(image_name)
    if roi_n is not None:
        idx = int(roi_n) - 1
        if 0 <= idx < len(ft):
            stem_mz = _parse_mz_from_image_stem(image_name)
            row_mz = _safe_float(ft.iloc[idx].get("mz"), np.nan)
            if stem_mz is not None and np.isfinite(row_mz) and np.isfinite(stem_mz):
                if abs(row_mz - stem_mz) <= 0.05:
                    return idx
            else:
                return idx

    cn = _safe_float(compound_name, np.nan)
    if np.isfinite(cn) and cn > 0:
        for col in ("Compound Name", "compound_name", "Compound"):
            if col not in ft.columns:
                continue
            m = pd.to_numeric(ft[col], errors="coerce") == int(cn)
            if m.any():
                return int(np.flatnonzero(m.to_numpy())[0])
    return None


def _feature_nominal_rt_label(
    ft: Optional[pd.DataFrame],
    mz,
    q3,
    image_name: str,
    compound_name=None,
) -> Tuple[float, str]:
    """Return (nominal_rt_min, compound_label) from feature.csv via mz/q3 or image prefix."""
    if ft is None or ft.empty:
        return float("nan"), ""
    idx = _find_feature_row_index(ft, mz, q3, image_name, compound_name)
    if idx is None or idx < 0 or idx >= len(ft):
        return float("nan"), ""
    row = ft.iloc[idx]
    rt_col = "RT" if "RT" in ft.columns else None
    nom = _safe_float(row.get(rt_col), np.nan) if rt_col else float("nan")
    name_col = None
    for c in ("Compound Name", "compound_name", "Compound"):
        if c in ft.columns:
            name_col = c
            break
    lab_parts = []
    if name_col:
        lab_parts.append(str(row.get(name_col, "")).strip())
    mz_v = _safe_float(row.get("mz"), np.nan)
    q3_v = _safe_float(row.get("q3"), np.nan)
    if np.isfinite(mz_v):
        lab_parts.append("mz=%.4f" % mz_v)
    if np.isfinite(q3_v):
        lab_parts.append("q3=%.2f" % q3_v)
    if not lab_parts:
        cn = _safe_float(compound_name, np.nan)
        lab_parts.append("#%s" % (int(cn) if np.isfinite(cn) else "?"))
    return float(nom), " ".join(lab_parts)


def _df_index_as_pred_row(ix) -> Optional[int]:
    """Integer row label from a DataFrame index (e.g. prediction.csv row)."""
    try:
        if isinstance(ix, (int, np.integer)):
            return int(ix)
        v = float(ix)
        if np.isfinite(v):
            return int(v)
    except Exception:
        pass
    return None


def _resolve_xic_matrix_row(
    image_name: str,
    compound,
    *,
    mz=None,
    q3=None,
    feature_df: Optional[pd.DataFrame] = None,
    pred_csv_row_idx: Optional[int],
    n_rows: int,
) -> Optional[int]:
    """
    Map one prediction row to an XIC ``intensity_mat`` row (0-based).

    Priority:
      1) ``feature_df`` row via (mz, q3) match (same rounding as testXIC).
      2) ROI image stem prefix ``N_mz...`` → row ``N - 1`` (optional mz check on stem).
      3) ``feature_df`` row via Compound Name column == compound_name.
      4) ``prediction.csv`` row index (last resort).
    """
    if feature_df is not None and not feature_df.empty:
        idx = _find_feature_row_index(feature_df, mz, q3, image_name, compound)
        if idx is not None and 0 <= idx < n_rows:
            return idx

    roi_n = _parse_roi_index_from_image_name(image_name)
    if roi_n is not None:
        i = int(roi_n) - 1
        if 0 <= i < n_rows:
            return i

    if pred_csv_row_idx is not None:
        j = int(pred_csv_row_idx)
        if 0 <= j < n_rows:
            return j
    return None


def _peer_intervals_from_group(g: pd.DataFrame, skip_row_index: Optional[int] = None) -> Optional[List[Tuple[float, float]]]:
    """同图其他预测行的 (rt_min, rt_max)，用于边界外推时避免吃进相邻预测峰。"""
    out: List[Tuple[float, float]] = []
    for j, rr in g.iterrows():
        if skip_row_index is not None and int(j) == int(skip_row_index):
            continue
        lo = _safe_float(rr.get("rt_min"), np.nan)
        hi = _safe_float(rr.get("rt_max"), np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out.append((float(lo), float(hi)))
    return out if out else None


def _boundary_posterior_kwargs(args) -> dict:
    return {
        "boundary_posterior_lookahead": int(getattr(args, "boundary_posterior_lookahead", 5)),
        "boundary_posterior_mean_scale": float(getattr(args, "boundary_posterior_mean_scale", 1.25)),
        "boundary_peer_thr_scale": float(getattr(args, "boundary_peer_thr_scale", 2.0)),
        "boundary_peer_min_overlap_rt": float(getattr(args, "boundary_peer_min_overlap_rt", 0.02)),
    }


def _adjust_first_round_interval_kwargs(args, pred_anchor=None) -> dict:
    d = _boundary_posterior_kwargs(args)
    ff = getattr(args, "edge_flat_triplet_step_frac", 0.012)
    ff_use = None if ff is None or float(ff) <= 0 else float(ff)
    d.update(
        edge_noise_stop_mode=str(getattr(args, "edge_noise_stop_mode", "roi_bottom_decile_mean")),
        pred_width_anchor=pred_anchor,
        width_max_expand_vs_pred=float(getattr(args, "refine_width_max_expand_vs_pred", 1.08)),
        width_max_frac_of_roi=float(getattr(args, "refine_width_max_frac_of_roi", 0.45)),
        flat_triplet_step_frac=ff_use,
    )
    return d


def _effective_small_peak_rt_tol(args) -> float:
    if not bool(getattr(args, "enable_small_peak_rt_gate", False)):
        return 1e9
    return float(getattr(args, "small_peak_rt_tol", 0.3))


def _effective_valley_small_peak_rt_tol(args) -> float:
    if not bool(getattr(args, "enable_small_peak_rt_gate", False)):
        return 1e9
    return float(getattr(args, "valley_small_peak_rt_tol", 0.8))


def _load_roi_windows(path: Path) -> Dict[str, Tuple[float, float]]:
    rw = path / "roi_windows.csv"
    if not rw.exists():
        return {}
    df = pd.read_csv(rw)
    if not {"image", "rt_lo", "rt_hi"}.issubset(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        out[str(r["image"]).strip()] = (float(r["rt_lo"]), float(r["rt_hi"]))
    return out


def _normalize_roi_map_keys(m: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """同时注册完整路径键与 basename，便于 SNR 子目录 roi_windows 带文件夹前缀时仍能命中。"""
    if not m:
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for k, v in m.items():
        ks = str(k).strip()
        out[ks] = v
        base = Path(ks).name
        out.setdefault(base, v)
    return out


def _merge_roi_window_maps(
    *maps: Optional[Dict[str, Tuple[float, float]]],
) -> Dict[str, Tuple[float, float]]:
    """后传入的表在同一 basename 上覆盖先传入的（通常 xic 目录的键更贴近 prediction.csv）。"""
    merged: Dict[str, Tuple[float, float]] = {}
    for mm in maps:
        if not mm:
            continue
        merged.update(_normalize_roi_map_keys(mm))
    return merged


def _roi_lookup_keys_for_prediction(image_name: str) -> List[str]:
    """prediction 图名常带 _snr… 后缀，而 ROI 表可能用无 SNR 后缀的 jpeg 名。"""
    s = str(image_name).strip()
    keys = [s, Path(s).name]
    stem = Path(s).stem
    if "_snr" in stem:
        short = stem.split("_snr")[0]
        for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG"):
            keys.append(short + ext)
    return keys


def _resolve_roi_rt_window(
    roi_map: Dict[str, Tuple[float, float]],
    image_name: str,
    rt_fallback_lo: float,
    rt_fallback_hi: float,
) -> Tuple[float, float]:
    if not roi_map:
        return float(rt_fallback_lo), float(rt_fallback_hi)
    for k in _roi_lookup_keys_for_prediction(image_name):
        if k in roi_map:
            lo, hi = roi_map[k]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return float(lo), float(hi)
    pred_base = Path(str(image_name).strip()).name
    pred_stem = Path(pred_base).stem
    for rk, rv in roi_map.items():
        if Path(str(rk)).name == pred_base:
            lo, hi = rv
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return float(lo), float(hi)
    if "_snr" in pred_stem:
        short_stem = pred_stem.split("_snr")[0]
        for rk, rv in roi_map.items():
            rstem = Path(str(rk)).stem
            if rstem == short_stem:
                lo, hi = rv
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    return float(lo), float(hi)
    return float(rt_fallback_lo), float(rt_fallback_hi)


def _peak_rt_height(rt: np.ndarray, y: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    mask = (rt >= lo) & (rt <= hi)
    if not np.any(mask):
        return np.nan, 0.0
    ys = y[mask]
    rs = rt[mask]
    i = int(np.argmax(ys))
    return float(rs[i]), float(ys[i])


def _interval_area(rt: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = (rt >= lo) & (rt <= hi)
    if np.sum(mask) < 2:
        return 0.0
    rs = rt[mask]
    ys = np.maximum(y[mask], 0.0)
    return float(np.trapz(ys, rs))


def _segment_skew(rt: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = (rt >= lo) & (rt <= hi)
    if np.sum(mask) < 5:
        return np.nan
    rs = rt[mask]
    ys = np.maximum(y[mask], 0.0)
    w = ys + 1e-12
    mu = float(np.sum(rs * w) / np.sum(w))
    var = float(np.sum(((rs - mu) ** 2) * w) / np.sum(w))
    if var <= 1e-15:
        return 0.0
    sigma = np.sqrt(var)
    m3 = float(np.sum(((rs - mu) ** 3) * w) / np.sum(w))
    return m3 / (sigma ** 3 + 1e-12)


def _secondary_min_height(
    rt: np.ndarray,
    y: np.ndarray,
    rt_lo: float,
    rt_hi: float,
    main_height: float,
    min_secondary_ratio: float,
    noise_barrier_ratio: float,
) -> float:
    baseline = float(np.percentile(np.maximum(y, 0.0), 25))
    dynamic = max(float(main_height) - baseline, 0.0)
    noise_avg = get_last_25pct_avg_noise(rt, y, float(rt_lo), float(rt_hi), frac=0.25)
    return baseline + float(min_secondary_ratio) * dynamic + float(noise_barrier_ratio) * max(noise_avg, 1e-9)


def _secondary_min_height_local(
    rt: np.ndarray,
    y: np.ndarray,
    cand_rt: float,
    main_height: float,
    min_secondary_ratio: float,
    noise_barrier_ratio: float,
    window_half: float = 0.30,
) -> float:
    """
    Candidate-local threshold:
    use a local window around small-peak candidate and compute noise from its trailing 25%.
    """
    if not np.isfinite(cand_rt):
        return np.inf
    lo = max(float(np.min(rt)), float(cand_rt) - float(window_half))
    hi = min(float(np.max(rt)), float(cand_rt) + float(window_half))
    if hi <= lo:
        return np.inf
    baseline = float(np.percentile(np.maximum(y[(rt >= lo) & (rt <= hi)], 0.0), 25))
    dynamic = max(float(main_height) - baseline, 0.0)
    noise_avg_local = get_last_25pct_avg_noise(rt, y, lo, hi, frac=0.25)
    return baseline + float(min_secondary_ratio) * dynamic + float(noise_barrier_ratio) * max(noise_avg_local, 1e-9)


def _adaptive_small_boundary_alpha(
    main_height: float,
    small_height: float,
    alpha_default: float = 0.5,
    alpha_min: float = 0.30,
    strong_ratio_start: float = 0.55,
) -> float:
    """
    Adaptive alpha for small-peak boundary baseline:
    - weak/normal small peaks keep default alpha (behavior close to current)
    - stronger small peaks get smaller alpha, allowing slightly wider boundaries
    """
    mh = _safe_float(main_height, np.nan)
    sh = _safe_float(small_height, np.nan)
    if (not np.isfinite(mh)) or mh <= 0 or (not np.isfinite(sh)) or sh <= 0:
        return float(alpha_default)
    ratio = max(0.0, min(1.0, float(sh) / float(mh)))
    if ratio <= float(strong_ratio_start):
        return float(alpha_default)
    # Linear decay: ratio in [strong_ratio_start, 1.0] -> alpha in [alpha_default, alpha_min]
    t = (ratio - float(strong_ratio_start)) / max(1e-9, (1.0 - float(strong_ratio_start)))
    alpha = float(alpha_default) - t * (float(alpha_default) - float(alpha_min))
    return float(max(alpha_min, min(alpha_default, alpha)))


def _small_boundary_baseline_level(
    y: np.ndarray,
    noise_avg: float,
    main_height: float,
    small_height: float,
) -> float:
    base = float(np.percentile(np.maximum(y, 0.0), 25))
    alpha = _adaptive_small_boundary_alpha(main_height=main_height, small_height=small_height)
    return float(base + alpha * max(float(noise_avg), 1e-9))


def _boundary_intensity_at_rt(rt: np.ndarray, y: np.ndarray, boundary_rt: float) -> float:
    if (not np.isfinite(boundary_rt)) or rt is None or y is None or len(rt) == 0:
        return float("nan")
    i = int(np.argmin(np.abs(np.asarray(rt, dtype=np.float64) - float(boundary_rt))))
    yy = np.asarray(y, dtype=np.float64)
    if i < 0 or i >= yy.size:
        return float("nan")
    return float(max(0.0, yy[i]))


def _enforce_low_noise_boundary_with_mid_reverse(
    rt: np.ndarray,
    y: np.ndarray,
    boundary_rt: float,
    other_boundary_rt: float,
    search_lo: float,
    search_hi: float,
    baseline_level: float,
    side: str,
) -> float:
    """
    If boundary intensity is above low-noise baseline:
    1) move toward center by one current-width distance and search low-noise point;
    2) if still not found, reverse direction to search.
    """
    if side not in ("left", "right"):
        return float(boundary_rt)
    vals = (boundary_rt, other_boundary_rt, search_lo, search_hi, baseline_level)
    if not all(np.isfinite(v) for v in vals):
        return float(boundary_rt)
    if search_hi <= search_lo:
        return float(boundary_rt)

    rt_a = np.asarray(rt, dtype=np.float64)
    y_a = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    if rt_a.size == 0 or y_a.size == 0:
        return float(boundary_rt)

    # Restrict to allowed interval.
    m = (rt_a >= float(search_lo)) & (rt_a <= float(search_hi))
    if np.count_nonzero(m) < 2:
        return float(boundary_rt)
    ridx = np.where(m)[0]
    b_idx = int(ridx[int(np.argmin(np.abs(rt_a[ridx] - float(boundary_rt))))])
    if y_a[b_idx] <= float(baseline_level):
        return float(rt_a[b_idx])

    width = max(1e-9, abs(float(other_boundary_rt) - float(boundary_rt)))
    if side == "left":
        mid_rt = min(float(search_hi), float(boundary_rt) + width)
    else:
        mid_rt = max(float(search_lo), float(boundary_rt) - width)
    mid_idx = int(ridx[int(np.argmin(np.abs(rt_a[ridx] - float(mid_rt))))])

    def _scan(order):
        for i in order:
            if y_a[int(i)] <= float(baseline_level):
                return float(rt_a[int(i)])
        return np.nan

    # First pass: toward center
    if side == "left":
        lo_i, hi_i = min(b_idx, mid_idx), max(b_idx, mid_idx)
        ans = _scan(range(lo_i, hi_i + 1))
    else:
        lo_i, hi_i = min(mid_idx, b_idx), max(mid_idx, b_idx)
        ans = _scan(range(hi_i, lo_i - 1, -1))
    if np.isfinite(ans):
        return float(ans)

    # Second pass: reverse direction
    if side == "left":
        ans2 = _scan(range(b_idx, int(ridx[0]) - 1, -1))
    else:
        ans2 = _scan(range(b_idx, int(ridx[-1]) + 1))
    if np.isfinite(ans2):
        return float(ans2)
    return float(boundary_rt)


def _cap_left_shrink_for_strong_roi_secondary(
    overlap_left: float,
    overlap_right: float,
    final_left: float,
    main_height: float,
    small_height: float,
    strong_ratio_threshold: float = 0.60,
    max_left_shrink_abs: float = 0.10,
    max_left_shrink_ratio: float = 0.30,
) -> float:
    """
    Limit excessive right-shift on left boundary for strong roi_secondary small peaks.
    Only applies when small/main height ratio is sufficiently high.
    """
    vals = (overlap_left, overlap_right, final_left, main_height, small_height)
    if not all(np.isfinite(v) for v in vals):
        return float(final_left)
    if overlap_right <= overlap_left:
        return float(final_left)
    if main_height <= 0 or small_height <= 0:
        return float(final_left)
    ratio = float(small_height) / float(main_height)
    if ratio < float(strong_ratio_threshold):
        return float(final_left)
    overlap_w = float(overlap_right - overlap_left)
    allowed_shift = min(float(max_left_shrink_abs), float(max_left_shrink_ratio) * overlap_w)
    cap_left = float(overlap_left) + max(0.0, allowed_shift)
    return float(min(float(final_left), cap_left))


def _cap_right_shrink_for_strong_roi_secondary(
    overlap_left: float,
    overlap_right: float,
    final_right: float,
    main_height: float,
    small_height: float,
    strong_ratio_threshold: float = 0.60,
    max_right_shrink_abs: float = 0.10,
    max_right_shrink_ratio: float = 0.30,
) -> float:
    """
    Limit excessive left-shift on right boundary for strong roi_secondary small peaks.
    Only applies when small/main height ratio is sufficiently high.
    """
    vals = (overlap_left, overlap_right, final_right, main_height, small_height)
    if not all(np.isfinite(v) for v in vals):
        return float(final_right)
    if overlap_right <= overlap_left:
        return float(final_right)
    if main_height <= 0 or small_height <= 0:
        return float(final_right)
    ratio = float(small_height) / float(main_height)
    if ratio < float(strong_ratio_threshold):
        return float(final_right)
    overlap_w = float(overlap_right - overlap_left)
    allowed_shift = min(float(max_right_shrink_abs), float(max_right_shrink_ratio) * overlap_w)
    cap_right = float(overlap_right) - max(0.0, allowed_shift)
    return float(max(float(final_right), cap_right))


def _local_noise_baseline_around(
    rt: np.ndarray,
    y: np.ndarray,
    center_rt: float,
    window_half: float = 0.25,
    noise_percentile: float = 25.0,
) -> float:
    if not np.isfinite(center_rt):
        return float(np.percentile(np.maximum(y, 0.0), float(noise_percentile)))
    lo = max(float(np.min(rt)), center_rt - window_half)
    hi = min(float(np.max(rt)), center_rt + window_half)
    m = (rt >= lo) & (rt <= hi)
    if np.sum(m) < 3:
        return float(np.percentile(np.maximum(y, 0.0), float(noise_percentile)))
    return float(np.percentile(np.maximum(y[m], 0.0), float(noise_percentile)))


def _interval_overlap_len(lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    if not all(np.isfinite(v) for v in (lo1, hi1, lo2, hi2)) or hi1 <= lo1 or hi2 <= lo2:
        return 0.0
    return max(0.0, min(hi1, hi2) - max(lo1, lo2))


def _small_peak_is_weak(
    main_lo: float,
    main_hi: float,
    small_lo: float,
    small_hi: float,
    max_width: float,
    min_overlap_frac: float,
) -> bool:
    if not np.isfinite(small_lo) or not np.isfinite(small_hi) or small_hi <= small_lo:
        return False
    w = float(small_hi - small_lo)
    if w < float(max_width):
        return True
    if not (np.isfinite(main_lo) and np.isfinite(main_hi) and main_hi > main_lo):
        return False
    ov = _interval_overlap_len(main_lo, main_hi, small_lo, small_hi)
    return (ov / max(w, 1e-9)) >= float(min_overlap_frac)


def _constrain_adjusted_interval(
    rt: np.ndarray,
    y: np.ndarray,
    rt_min_adj: float,
    rt_max_adj: float,
    main_peak_rt: float,
    small_min_h: float,
    noise_percentile: float = 25.0,
    edge_max_span_min: float = 0.50,
    edge_noise_stop_mode: str = "roi_bottom_decile_mean",
) -> Tuple[float, float]:
    """
    Add reverse-move safeguard:
    if boundary after toward-peak adjustment reaches > 2*small_min_h,
    move boundary in reverse direction until near local noise baseline.
    roi_bottom_decile_mean：第一轮已在 ROI 内完成截停与宽度上限；此处若再按阈值反向外推会把框扩到接近整段 ROI，
    故跳过（与 adjust_first_round_interval 一致）。
    """
    if str(edge_noise_stop_mode) == "roi_bottom_decile_mean":
        return rt_min_adj, rt_max_adj

    if not (np.isfinite(rt_min_adj) and np.isfinite(rt_max_adj) and rt_max_adj > rt_min_adj and np.isfinite(main_peak_rt)):
        return rt_min_adj, rt_max_adj

    yy = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    xx = np.asarray(rt, dtype=np.float64)
    new_lo, new_hi = float(rt_min_adj), float(rt_max_adj)

    def _side_threshold(toward_left: bool) -> float:
        if edge_noise_stop_mode == "low_percentile":
            return one_sided_low_noise_baseline(
                xx,
                yy,
                float(main_peak_rt),
                toward_left,
                max_span_min=float(edge_max_span_min),
                noise_percentile=float(noise_percentile),
            )
        return one_sided_edge_stop_threshold_stable_tail_mean(
            xx,
            yy,
            float(main_peak_rt),
            toward_left,
            max_span_min=float(edge_max_span_min),
        )

    def _walk_reverse(boundary_rt: float, toward_left: bool) -> float:
        # toward_left=True means move to smaller RT, else bigger RT.
        base = float(_side_threshold(toward_left))
        idx = int(np.argmin(np.abs(xx - boundary_rt)))
        if toward_left:
            i = idx
            while i > 0 and yy[i] > base:
                i -= 1
            return float(xx[i])
        i = idx
        while i < len(xx) - 1 and yy[i] > base:
            i += 1
        return float(xx[i])

    y_lo = float(np.interp(new_lo, xx, yy))
    y_hi = float(np.interp(new_hi, xx, yy))
    base_lo = float(_side_threshold(True))
    base_hi = float(_side_threshold(False))

    # left boundary was moved toward peak => reverse to left until one-sided low-noise baseline
    if y_lo > base_lo and new_lo < main_peak_rt:
        new_lo = _walk_reverse(new_lo, toward_left=True)
    # right boundary was moved toward peak => reverse to right until one-sided low-noise baseline
    if y_hi > base_hi and new_hi > main_peak_rt:
        new_hi = _walk_reverse(new_hi, toward_left=False)

    if new_hi <= new_lo:
        return rt_min_adj, rt_max_adj
    return new_lo, new_hi


def _refine_main_interval_near_reference(
    rt: np.ndarray,
    y: np.ndarray,
    main_lo: float,
    main_hi: float,
    ref_rt: float,
    search_tol: float = 0.5,
    max_width: float = 0.35,
    boundary_noise_percentile: float = 25.0,
    boundary_posterior_lookahead: int = 5,
    boundary_posterior_mean_scale: float = 1.25,
    peer_rt_intervals: Optional[List[Tuple[float, float]]] = None,
    boundary_peer_thr_scale: float = 2.0,
    boundary_peer_min_overlap_rt: float = 0.02,
) -> Tuple[float, float, float, float]:
    """
    Re-locate main interval around the strongest peak near reference RT (if available),
    then rebuild boundaries by descending to local noise baseline.
    与 adjust_first_round_interval 一致：阈值截停 + 外向 lookahead 后验均值 + 可选同伴框防撞。
    """
    if rt.size < 5 or y.size != rt.size:
        return main_lo, main_hi, np.nan, 0.0
    yy = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    xx = np.asarray(rt, dtype=np.float64)

    cur_pk_rt, cur_pk_h = _peak_rt_height(xx, yy, main_lo, main_hi)
    if not np.isfinite(cur_pk_rt):
        cur_pk_rt = float(xx[int(np.argmax(yy))])
        cur_pk_h = float(np.max(yy))

    if np.isfinite(ref_rt):
        lo = max(float(np.min(xx)), float(ref_rt) - float(search_tol))
        hi = min(float(np.max(xx)), float(ref_rt) + float(search_tol))
    else:
        lo = max(float(np.min(xx)), float(cur_pk_rt) - 0.5)
        hi = min(float(np.max(xx)), float(cur_pk_rt) + 0.5)
    m = (xx >= lo) & (xx <= hi)
    if np.sum(m) < 3:
        return main_lo, main_hi, cur_pk_rt, cur_pk_h

    rs = xx[m]
    ys = yy[m]
    i_local = int(np.argmax(ys))
    cand_rt = float(rs[i_local])
    cand_h = float(ys[i_local])

    # avoid switching to a much weaker candidate
    if np.isfinite(cur_pk_h) and cand_h < 0.75 * cur_pk_h:
        return main_lo, main_hi, cur_pk_rt, cur_pk_h

    base = _local_noise_baseline_around(
        xx, yy, cand_rt, window_half=0.25, noise_percentile=float(boundary_noise_percentile)
    )
    i = int(np.argmin(np.abs(xx - cand_rt)))
    la = int(boundary_posterior_lookahead) if boundary_posterior_lookahead is not None else 0
    if la <= 0:
        li = i
        while li > 0 and yy[li] > base:
            li -= 1
        ri = i
        while ri < len(xx) - 1 and yy[ri] > base:
            ri += 1
        new_lo = float(xx[li])
        new_hi = float(xx[ri])
    else:
        new_lo = float(
            walk_interval_left_to_noise_with_posterior(
                yy,
                xx,
                i,
                float(base),
                la,
                float(boundary_posterior_mean_scale),
                peer_rt_intervals,
                peer_thr_scale=float(boundary_peer_thr_scale),
                peer_min_overlap_rt=float(boundary_peer_min_overlap_rt),
            )
        )
        new_hi = float(
            walk_interval_right_to_noise_with_posterior(
                yy,
                xx,
                i,
                float(base),
                la,
                float(boundary_posterior_mean_scale),
                peer_rt_intervals,
                peer_thr_scale=float(boundary_peer_thr_scale),
                peer_min_overlap_rt=float(boundary_peer_min_overlap_rt),
            )
        )
    if new_hi <= new_lo:
        new_lo, new_hi = main_lo, main_hi

    if (new_hi - new_lo) > float(max_width):
        half = float(max_width) / 2.0
        new_lo = max(float(np.min(xx)), cand_rt - half)
        new_hi = min(float(np.max(xx)), cand_rt + half)
    if new_hi <= new_lo:
        return main_lo, main_hi, cur_pk_rt, cur_pk_h
    return new_lo, new_hi, cand_rt, cand_h


def _shrink_interval_around_peak(
    rt: np.ndarray,
    y: np.ndarray,
    lo: float,
    hi: float,
    peak_rt: float,
    baseline_level: float,
    max_width: float = 0.35,
    min_width: float = 0.0,
) -> Tuple[float, float]:
    """Shrink a candidate interval by walking from peak to baseline crossing."""
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        return np.nan, np.nan
    mask = (rt >= lo) & (rt <= hi)
    if np.sum(mask) < 3:
        return lo, hi
    rs = rt[mask]
    ys = y[mask]
    if not np.isfinite(peak_rt):
        peak_rt = float(rs[int(np.argmax(ys))])
    pi = int(np.argmin(np.abs(rs - peak_rt)))

    li = pi
    while li > 0 and ys[li] > baseline_level:
        li -= 1
    ri = pi
    while ri < len(rs) - 1 and ys[ri] > baseline_level:
        ri += 1

    new_lo = float(rs[li])
    new_hi = float(rs[ri])
    if new_hi <= new_lo:
        return lo, hi
    # Safety cap for extremely wide tails.
    if (new_hi - new_lo) > max_width:
        half = max_width / 2.0
        new_lo = max(float(np.min(rt)), float(peak_rt) - half)
        new_hi = min(float(np.max(rt)), float(peak_rt) + half)
    # Root-level width guard: avoid over-narrow intervals by detection criterion, not post-plot expansion.
    if float(min_width) > 0 and (new_hi - new_lo) < float(min_width):
        half = float(min_width) / 2.0
        new_lo = max(float(np.min(rt)), float(peak_rt) - half)
        new_hi = min(float(np.max(rt)), float(peak_rt) + half)
    return new_lo, new_hi


def _find_secondary_peak_by_roi_rules(
    rt: np.ndarray,
    y: np.ndarray,
    main_lo: float,
    main_hi: float,
    rt_lo: float,
    rt_hi: float,
    min_secondary_ratio: float,
    noise_barrier_ratio: float,
    smooth_sigma: float = 1.0,
    extra_exclude: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    Find secondary peak in ROI using the same threshold idea as has_secondary_peak_in_roi:
      min_height = baseline + min_secondary_ratio * dynamic + noise_barrier_ratio * noise_avg
    extra_exclude: 额外禁止出现峰顶的 RT 区间（用于第三峰等，与第二峰同一套规则）。
    Returns (peak_rt, peak_height) or (nan, nan).
    """
    m_roi = (rt >= rt_lo) & (rt <= rt_hi)
    if np.sum(m_roi) < 10:
        return np.nan, np.nan
    rr = rt[m_roi]
    yy = np.maximum(y[m_roi], 0.0).astype(np.float64)
    if smooth_sigma > 0 and yy.size >= 10:
        yy = gaussian_filter1d(yy, sigma=min(float(smooth_sigma), yy.size / 25.0), mode="nearest")
        yy = np.maximum(yy, 0.0)

    m_main = (rr >= main_lo) & (rr <= main_hi)
    if np.any(m_main):
        main_height = float(np.max(yy[m_main]))
    else:
        main_height = float(np.max(yy))
    baseline = float(np.percentile(yy, 25))
    dynamic = main_height - baseline
    if dynamic <= 0:
        return np.nan, np.nan
    noise_avg = get_last_25pct_avg_noise(rt, y, float(rt_lo), float(rt_hi), frac=0.25)
    min_h = baseline + float(min_secondary_ratio) * dynamic + float(noise_barrier_ratio) * max(noise_avg, 1e-9)

    exclude_boxes: List[Tuple[float, float]] = [(float(main_lo), float(main_hi))]
    if extra_exclude:
        for a, b in extra_exclude:
            if np.isfinite(a) and np.isfinite(b) and float(b) > float(a):
                exclude_boxes.append((float(a), float(b)))

    peaks = []
    for i in range(1, yy.size - 1):
        if yy[i] >= yy[i - 1] and yy[i] >= yy[i + 1] and yy[i] >= min_h:
            pr = float(rr[i])
            if any(float(lo) <= pr <= float(hi) for lo, hi in exclude_boxes):
                continue
            peaks.append(i)
    if not peaks:
        return np.nan, np.nan
    best = max(peaks, key=lambda i: float(yy[i]))
    return float(rr[best]), float(yy[best])


def _build_interval_around_peak_in_segment(
    rt: np.ndarray,
    y: np.ndarray,
    peak_rt: float,
    seg_lo: float,
    seg_hi: float,
    max_width: float,
    min_width: float,
    baseline_level: float,
) -> Tuple[float, float]:
    if not np.isfinite(peak_rt) or not np.isfinite(seg_lo) or not np.isfinite(seg_hi) or seg_hi <= seg_lo:
        return np.nan, np.nan
    half0 = max(0.05, float(max_width) / 2.0)
    lo0 = max(float(seg_lo), float(peak_rt) - half0)
    hi0 = min(float(seg_hi), float(peak_rt) + half0)
    if hi0 <= lo0:
        return np.nan, np.nan
    lo1, hi1 = _shrink_interval_around_peak(
        rt, y, lo0, hi0, float(peak_rt), baseline_level=float(baseline_level),
        max_width=float(max_width), min_width=float(min_width)
    )
    lo1 = max(float(seg_lo), float(lo1))
    hi1 = min(float(seg_hi), float(hi1))
    if hi1 <= lo1:
        return np.nan, np.nan
    return float(lo1), float(hi1)


def _rt_offset_gate_with_width(
    main_rt: float,
    cand_lo: float,
    cand_hi: float,
    rt_tol: float,
    max_width: float,
    boundary_pad: float = 0.08,
) -> bool:
    """
    RT gate with center/boundary constraints:
    - center near main peak
    - both boundaries not excessively far from main peak
    """
    if not all(np.isfinite(v) for v in (main_rt, cand_lo, cand_hi)) or cand_hi <= cand_lo:
        return False
    w = float(cand_hi - cand_lo)
    c = 0.5 * (float(cand_lo) + float(cand_hi))
    if abs(c - float(main_rt)) > float(rt_tol):
        return False
    btol = float(rt_tol) + max(float(boundary_pad), 0.25 * w)
    if abs(float(cand_lo) - float(main_rt)) > btol:
        return False
    if abs(float(cand_hi) - float(main_rt)) > btol:
        return False
    return True


def _chord_height_at_rt(p1_rt: float, p1_h: float, p2_rt: float, p2_h: float, x: float) -> float:
    """两峰顶点连线上在 x 处（RT）的线性插值强度。"""
    if not all(np.isfinite(v) for v in (p1_rt, p1_h, p2_rt, p2_h, x)):
        return float("nan")
    if abs(float(p2_rt) - float(p1_rt)) < 1e-12:
        return float(0.5 * (p1_h + p2_h))
    t = (float(x) - float(p1_rt)) / (float(p2_rt) - float(p1_rt))
    t = float(np.clip(t, 0.0, 1.0))
    return float(p1_h + t * (p2_h - p1_h))


def _detect_double_peak_with_valley_in_interval(
    rt: np.ndarray,
    y: np.ndarray,
    lo: float,
    hi: float,
    min_gap_rt: float = 0.12,
    max_valley_ratio: float = 0.78,
    min_valley_drop_ratio: float = 0.0,
    min_peak_above_valley_ratio: float = 0.0,
    min_peak_sep_ratio_of_span: float = 0.0,
):
    """
    Detect two prominent peaks and an in-between valley inside [lo, hi].
    Return (p1_rt, p1_h, p2_rt, p2_h, valley_rt) or None.

    Optional anti-spike gates (比例均相对 max(p1_h,p2_h)；0 表示不启用):
    - min_valley_drop_ratio: (弦上谷位强度 − 谷强度) / H_max 下限
    - min_peak_above_valley_ratio: 每个峰顶 (峰高 − 谷高) / H_max 下限
    - min_peak_sep_ratio_of_span: |p2_rt−p1_rt| / (hi−lo) 下限
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    m = (rt >= float(lo)) & (rt <= float(hi))
    if np.sum(m) < 12:
        return None
    rr = rt[m].astype(np.float64)
    yy = np.maximum(y[m].astype(np.float64), 0.0)
    yy_s = gaussian_filter1d(yy, sigma=min(1.0, yy.size / 30.0), mode="nearest")
    yy_s = np.maximum(yy_s, 0.0)
    baseline = float(np.percentile(yy_s, 25))
    dynamic = float(np.max(yy_s)) - baseline
    if dynamic <= 1e-12:
        return None
    pair = _find_best_split_pair_prominence(yy_s, baseline, dynamic, DEFAULT_VALLEY_SPLIT_PARAMS)
    if pair is None:
        return None
    i_lo, i_hi = int(pair[0]), int(pair[1])
    if i_hi <= i_lo:
        return None
    p1_rt, p1_h = float(rr[i_lo]), float(yy_s[i_lo])
    p2_rt, p2_h = float(rr[i_hi]), float(yy_s[i_hi])
    if abs(p2_rt - p1_rt) < float(min_gap_rt):
        return None
    i_mid = i_lo + int(np.argmin(yy_s[i_lo : i_hi + 1]))
    valley_rt = float(rr[i_mid])
    valley_h = float(yy_s[i_mid])
    top_h = max(p1_h, p2_h, 1e-12)
    if (valley_h / top_h) > float(max_valley_ratio):
        return None

    span_rt = float(hi) - float(lo)
    mvr = float(max(float(min_valley_drop_ratio), 0.0))
    mpar = float(max(float(min_peak_above_valley_ratio), 0.0))
    mspr = float(max(float(min_peak_sep_ratio_of_span), 0.0))

    if mspr > 0.0 and span_rt > 1e-12:
        if abs(p2_rt - p1_rt) < mspr * span_rt:
            return None

    if mpar > 0.0:
        if (p1_h - valley_h) < mpar * top_h or (p2_h - valley_h) < mpar * top_h:
            return None

    if mvr > 0.0:
        h_chord = _chord_height_at_rt(p1_rt, p1_h, p2_rt, p2_h, valley_rt)
        if not np.isfinite(h_chord):
            return None
        valley_drop = float(h_chord) - float(valley_h)
        if valley_drop < mvr * top_h:
            return None

    return (p1_rt, p1_h, p2_rt, p2_h, valley_rt)


def _composite_conf(ai_score, snr, rt_shift, skew_diff, rt_tol=0.2, skew_tol=1.0) -> float:
    ai = max(0.0, min(1.0, _safe_float(ai_score, 0.0)))
    snr_norm = max(0.0, min(1.0, _safe_float(snr, 0.0) / 10.0))
    rt_term = float(np.exp(-abs(_safe_float(rt_shift, 999.0)) / max(rt_tol, 1e-6)))
    skew_term = float(np.exp(-abs(_safe_float(skew_diff, 999.0)) / max(skew_tol, 1e-6)))
    # Main-weighted confidence with explainable factors.
    conf = 0.45 * ai + 0.25 * snr_norm + 0.20 * rt_term + 0.10 * skew_term
    return float(max(0.0, min(1.0, conf)))


def _rt_linear_conf(rt_shift: float, max_shift: float = 0.5) -> float:
    """
    From the figure: RT shift is the primary factor.
      - rt_shift = 0   -> conf = 1
      - rt_shift = 0.5 -> conf = 0
    Linear mapping, clamped to [0, 1].
    """
    s = abs(_safe_float(rt_shift, 999.0))
    m = float(max(max_shift, 1e-9))
    return float(max(0.0, min(1.0, 1.0 - (s / m))))


def _rt_nonlinear_conf(rt_shift: float, max_shift: float = 0.5, power: float = 2.0, at_max_value: float = 0.01) -> float:
    """
    Non-linear RT confidence decay:
      - rt_shift = 0 -> 1
      - rt_shift = max_shift -> at_max_value (default 0.01)
    Larger RT deviation drops confidence faster (non-linear).
    """
    s = abs(_safe_float(rt_shift, 999.0))
    m = float(max(max_shift, 1e-9))
    p = float(max(power, 0.5))
    v = float(min(max(at_max_value, 1e-6), 0.5))
    # exp(-k*(s/m)^p) with k chosen so that exp(-k)=v when s=m
    k = float(-np.log(v))
    return float(max(0.0, min(1.0, np.exp(-k * (s / m) ** p))))


def _snr_factor_conf(snr: float, snr_ref: float = 10.0) -> float:
    """Lower SNR lowers confidence (normalized to [0,1])."""
    v = _safe_float(snr, 0.0)
    r = float(max(snr_ref, 1e-9))
    return float(max(0.0, min(1.0, v / r)))


def _snr_box_over_noise_mean(
    rt: np.ndarray,
    y: np.ndarray,
    main_lo: float,
    main_hi: float,
    small_lo: float = np.nan,
    small_hi: float = np.nan,
    *,
    min_noise_pts: int = 5,
) -> Tuple[float, float, float]:
    """
    SNR definition requested:
      - noise = mean intensity outside ALL predicted boxes (union of main+small intervals)
      - signal = max intensity inside each box
      - snr = signal / max(noise_mean, eps)

    Returns (noise_mean_outside_all_boxes, snr_main, snr_small).
    snr_small is NaN if small interval is not finite/valid.
    """
    rr = np.asarray(rt, dtype=np.float64)
    yy = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    if rr.size != yy.size or rr.size < 3:
        return float("nan"), float("nan"), float("nan")

    inside = np.zeros(rr.shape, dtype=bool)
    if np.isfinite(main_lo) and np.isfinite(main_hi) and main_hi > main_lo:
        inside |= (rr >= float(main_lo)) & (rr <= float(main_hi))
    if np.isfinite(small_lo) and np.isfinite(small_hi) and small_hi > small_lo:
        inside |= (rr >= float(small_lo)) & (rr <= float(small_hi))

    noise = yy[~inside]
    if noise.size < int(min_noise_pts):
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(noise))
    mu = max(mu, 1e-12)

    snr_main = float("nan")
    if np.isfinite(main_lo) and np.isfinite(main_hi) and main_hi > main_lo:
        sig_main = float(np.max(yy[(rr >= float(main_lo)) & (rr <= float(main_hi))]))
        snr_main = float(sig_main / mu)

    snr_small = float("nan")
    if np.isfinite(small_lo) and np.isfinite(small_hi) and small_hi > small_lo:
        sig_small = float(np.max(yy[(rr >= float(small_lo)) & (rr <= float(small_hi))]))
        snr_small = float(sig_small / mu)

    return float(mu), float(snr_main), float(snr_small)


def _skew_direction_boost(
    ref_skew: float,
    main_rt: float,
    small_rt: float,
    *,
    skew_high: float = 1.0,
    skew_scale: float = 1.0,
    boost_max: float = 1.15,
    penalty_min: float = 0.90,
) -> float:
    """
    From the figure: if the standard skew is high, boost the small-peak confidence
    in the corresponding direction.

    Implementation (continuous strength):
      - if |ref_skew| <= skew_high -> no boost (1.0)
      - else: strength = tanh((|ref_skew| - skew_high) / skew_scale) in [0,1)
        direction match -> factor = 1 + strength * (boost_max - 1)
        mismatch        -> factor = 1 - strength * (1 - penalty_min)
    """
    sk = _safe_float(ref_skew, np.nan)
    mr = _safe_float(main_rt, np.nan)
    sr = _safe_float(small_rt, np.nan)
    if (not np.isfinite(sk)) or (not np.isfinite(mr)) or (not np.isfinite(sr)):
        return 1.0
    sh = float(max(skew_high, 0.0))
    if abs(sk) <= sh:
        return 1.0
    direction_match = (sk > 0 and sr > mr) or (sk < 0 and sr < mr)
    sc = float(max(skew_scale, 1e-6))
    strength = float(np.tanh((abs(sk) - sh) / sc))
    bmax = float(max(boost_max, 1.0))
    pmin = float(min(max(penalty_min, 0.0), 1.0))
    if direction_match:
        return float(1.0 + strength * (bmax - 1.0))
    return float(1.0 - strength * (1.0 - pmin))


def _final_conf_from_figure(
    ai_score: float,
    snr: float,
    rt_shift: float,
    skew_diff: float,
    *,
    rt_max_shift: float = 0.5,
    rt_power: float = 2.0,
    snr_ref: float = 10.0,
    skew_tol: float = 1.0,
    snr_weight: float = 0.30,
    skew_weight: float = 0.10,
) -> float:
    """
    Final confidence logic (as requested):
      - Base on original model confidence (ai_score)
      - Apply non-linear RT shift decay (dominant): bigger shift -> faster drop
      - Apply SNR penalty (secondary)
      - Apply skew (low weight; mainly for small-peak compensation outside via direction boost)
    """
    ai = float(max(0.0, min(1.0, _safe_float(ai_score, 0.0))))
    rt_c = _rt_nonlinear_conf(rt_shift, max_shift=rt_max_shift, power=rt_power, at_max_value=0.01)
    snr_c = _snr_factor_conf(snr, snr_ref=snr_ref)
    skew_c = float(np.exp(-abs(_safe_float(skew_diff, 999.0)) / max(float(skew_tol), 1e-6)))

    sw = float(max(0.0, min(1.0, snr_weight)))
    kw = float(max(0.0, min(1.0, skew_weight)))
    # Multiplicative model: base AI score is primary, RT dominates by direct multiplication.
    conf = ai * rt_c
    conf *= (1.0 - sw) + sw * snr_c
    conf *= (1.0 - kw) + kw * skew_c
    return float(max(0.0, min(1.0, conf)))


def _locate_xic_matrix_from_sample_csv(sample_csv: Path) -> Optional[Path]:
    """
    Try to locate xic_matrix.npy near sample_refined_csv path.
    Common layouts:
      results/batch_predictions/<sample>/prediction_refined.csv
      results/batch_predictions/<sample>/predicted_plots/prediction_refined.csv
    """
    cands = [
        sample_csv.parent / "xic_matrix.npy",
        sample_csv.parent.parent / "xic_matrix.npy",
        sample_csv.parent.parent.parent / "xic_matrix.npy",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def _plot_sample_final(
    df_final: pd.DataFrame,
    xic_path: Path,
    out_dir: Path,
    sigma: float = 1.0,
):
    xic = np.load(str(xic_path))
    rt = xic[0, :].astype(np.float64)
    if np.nanmax(rt) > 200:
        rt = rt / 60.0
    inten_mat = xic[1:, :].astype(np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: use the original ROI window used during prediction/integration
    # (testXIC -> roi_windows.csv). This is "预测时切割的范围".
    roi_map: Dict[str, Tuple[float, float]] = {}
    try:
        roi_map = _load_roi_windows(Path(xic_path).parent)
    except Exception:
        roi_map = {}
    feature_df = _load_optional_csv(Path(xic_path).parent / "feature.csv")

    for ix, r in df_final.iterrows():
        image_name = str(r.get("image", "")).strip()
        idx = _resolve_xic_matrix_row(
            image_name,
            r.get("compound_name"),
            mz=r.get("mz"),
            q3=r.get("q3"),
            feature_df=feature_df,
            pred_csv_row_idx=_df_index_as_pred_row(ix),
            n_rows=int(inten_mat.shape[0]),
        )
        if idx is None or idx < 0 or idx >= inten_mat.shape[0]:
            continue
        y = np.maximum(inten_mat[idx, :], 0.0)
        if sigma > 0 and y.size >= 10:
            y = gaussian_filter1d(y, sigma=min(float(sigma), y.size / 25.0), mode="nearest")
            y = np.maximum(y, 0.0)

        expected_rt = _safe_float(r.get("ref_rt_peak"), np.nan)
        target_rt_used = _safe_float(r.get("target_rt_used"), np.nan)

        roi_window = roi_map.get(image_name)
        roi_lo = np.nan
        roi_hi = np.nan
        if roi_window is not None:
            roi_lo = _safe_float(roi_window[0], np.nan)
            roi_hi = _safe_float(roi_window[1], np.nan)

        peak_rt = _safe_float(r.get("main_rt_peak"), np.nan)
        if np.isfinite(peak_rt):
            xlo = max(float(np.min(rt)), peak_rt - 1.0)
            xhi = min(float(np.max(rt)), peak_rt + 1.0)
        else:
            xlo, xhi = float(np.min(rt)), float(np.max(rt))

        # Ensure the prediction ROI window and expected/target RT are visible in the plot.
        if np.isfinite(roi_lo) and np.isfinite(roi_hi) and roi_hi > roi_lo:
            xlo = min(float(xlo), float(roi_lo))
            xhi = max(float(xhi), float(roi_hi))
        if np.isfinite(expected_rt):
            xlo = min(float(xlo), float(expected_rt))
            xhi = max(float(xhi), float(expected_rt))
        if np.isfinite(target_rt_used):
            xlo = min(float(xlo), float(target_rt_used))
            xhi = max(float(xhi), float(target_rt_used))

        if xhi <= xlo:
            continue
        m = (rt >= xlo) & (rt <= xhi)
        xr = rt[m]
        yr = y[m]
        if xr.size < 2:
            continue
        # Skip "blank" candidates (often caused by xic_matrix rows being all-zeros after filtering).
        # This keeps plot output aligned with the "淘汰图不用画出" expectation.
        if float(np.max(yr)) <= 1e-12:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xr, yr, color="blue", linewidth=1.5)

        mlo = _safe_float(r.get("main_rt_min"), np.nan)
        mhi = _safe_float(r.get("main_rt_max"), np.nan)
        if np.isfinite(mlo) and np.isfinite(mhi) and mhi > mlo:
            ax.axvspan(mlo, mhi, color="#2ecc71", alpha=0.24, label="Main interval")

        # Expected / target RT markers.
        # Note: expected_rt comes from standard refs; target_rt_used is the effective RT used in this sample.
        if np.isfinite(expected_rt):
            ax.axvline(
                expected_rt,
                color="#3498db",
                linestyle="--",
                linewidth=1.6,
                label="Expected RT (ref_rt_peak)",
            )
        if np.isfinite(target_rt_used) and (not np.isfinite(expected_rt) or abs(target_rt_used - expected_rt) > 1e-6):
            ax.axvline(
                target_rt_used,
                color="#8e44ad",
                linestyle=":",
                linewidth=1.6,
                label="Target RT used",
            )

        # Prediction/integration cut range (from roi_windows.csv).
        if np.isfinite(roi_lo) and np.isfinite(roi_hi) and roi_hi > roi_lo:
            ax.axvspan(
                roi_lo,
                roi_hi,
                color="#2980b9",
                alpha=0.10,
                label="Prediction ROI window",
            )

        # Keep/candidate small intervals:
        # - kept: passes standard-based filter
        # - candidate-only: recovered but not kept, still shown for debugging
        keep_small = int(_safe_float(r.get("keep_small_by_standard"), 0)) == 1
        slo = _safe_float(r.get("small_rt_min"), np.nan)
        shi = _safe_float(r.get("small_rt_max"), np.nan)
        if keep_small and np.isfinite(slo) and np.isfinite(shi) and shi > slo:
            ax.axvspan(slo, shi, color="#f39c12", alpha=0.22, label="Small interval (kept)")
        elif (not keep_small) and np.isfinite(slo) and np.isfinite(shi) and shi > slo:
            ax.axvspan(slo, shi, color="#95a5a6", alpha=0.20, label="Small candidate (filtered)")

        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("Retention Time (min)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Sample final (sigma={sigma}) - {image_name}")
        ax.grid(True, alpha=0.25)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=9)

        main_conf = _safe_float(r.get("main_conf_final", r.get("main_conf_composite")), np.nan)
        small_conf = _safe_float(r.get("small_conf_final", r.get("small_conf_composite")), np.nan)
        best_conf = _safe_float(r.get("final_conf_best"), np.nan)
        y_top = float(np.max(yr)) if yr.size else 1.0
        y_off = max((float(np.max(yr)) - float(np.min(yr))) * 0.04, 1e-9)
        # Confidence labels above peaks (final conf preferred).
        if np.isfinite(mlo) and np.isfinite(mhi) and mhi > mlo:
            m_rt, m_h = _peak_rt_height(rt, y, mlo, mhi)
            if np.isfinite(m_rt) and np.isfinite(m_h) and np.isfinite(main_conf):
                ax.text(
                    m_rt,
                    m_h + y_off,
                    f"main {main_conf:.3f}",
                    color="#27ae60",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
        if np.isfinite(slo) and np.isfinite(shi) and shi > slo:
            s_rt, s_h = _peak_rt_height(rt, y, slo, shi)
            if np.isfinite(s_rt) and np.isfinite(s_h) and np.isfinite(small_conf):
                ax.text(
                    s_rt,
                    s_h + y_off,
                    f"small {small_conf:.3f}",
                    color="#d35400",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        if np.isfinite(best_conf):
            # Top-left summary
            ax.text(
                0.01,
                0.98,
                f"final_best {best_conf:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
            )

        out_name = Path(image_name).stem if image_name else f"compound_{idx+1}"
        fig.savefig(out_dir / f"{out_name}_sample_final.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _recover_small_for_sample(
    rt: np.ndarray,
    y: np.ndarray,
    main_rt: float,
    main_lo: float,
    main_hi: float,
    main_height: float,
    main_score_ai: float,
    target_rt: float,
    roi_half_window: float,
    sample_rt_tolerance: float,
    small_peak_rt_tol: float,
    min_secondary_ratio: float,
    noise_barrier_ratio: float,
    max_small_width: float,
    small_noise_window_half: float,
    min_small_width: float,
):
    """
    Recover small peak for sample mode:
      1) ROI secondary search with noise-aware threshold
      2) valley split fallback (prominence pair)
    """
    if not np.isfinite(main_rt):
        return None
    # Sample stage target RT should come from standards (ref_rt_peak), not current apex.
    # Fallback to main_rt only when ref is unavailable.
    center_rt = float(target_rt) if np.isfinite(target_rt) else float(main_rt)
    rt_lo = max(float(np.min(rt)), center_rt - float(roi_half_window))
    rt_hi = min(float(np.max(rt)), center_rt + float(roi_half_window))
    if rt_hi <= rt_lo:
        return None

    sec_min_h = _secondary_min_height(
        rt, y, rt_lo, rt_hi, main_height,
        min_secondary_ratio=min_secondary_ratio,
        noise_barrier_ratio=noise_barrier_ratio,
    )
    sec_rt, sec_h = _find_secondary_peak_by_roi_rules(
        rt=rt,
        y=y,
        main_lo=main_lo,
        main_hi=main_hi,
        rt_lo=rt_lo,
        rt_hi=rt_hi,
        min_secondary_ratio=min_secondary_ratio,
        noise_barrier_ratio=noise_barrier_ratio,
        smooth_sigma=1.0,
    )
    sec_min_h_local = _secondary_min_height_local(
        rt, y, cand_rt=sec_rt, main_height=main_height,
        min_secondary_ratio=min_secondary_ratio,
        noise_barrier_ratio=noise_barrier_ratio,
        window_half=small_noise_window_half,
    )
    if np.isfinite(sec_rt) and np.isfinite(sec_h) and sec_h >= sec_min_h_local and abs(sec_rt - main_rt) <= small_peak_rt_tol:
        half0 = max(0.05, float(max_small_width) / 2.0)
        lo2n, hi2n = remove_overlap_from_second_interval(main_lo, main_hi, sec_rt - half0, sec_rt + half0, min_width=0.01)
        if np.isfinite(lo2n) and np.isfinite(hi2n) and hi2n > lo2n:
            baseline_small = float(np.percentile(y, 25)) + 0.5 * max(get_last_25pct_avg_noise(rt, y, rt_lo, rt_hi, frac=0.25), 1e-9)
            lo2s, hi2s = _shrink_interval_around_peak(
                rt, y, lo2n, hi2n, sec_rt, baseline_level=baseline_small,
                max_width=float(max_small_width), min_width=float(min_small_width)
            )
            ratio = float(sec_h / max(main_height, 1e-12))
            sc = float(main_score_ai) * max(0.0, min(1.0, ratio))
            return (lo2s, hi2s, sec_rt, sec_h, sc, "sample_recover_roi")

    # valley split fallback
    mask = (rt >= rt_lo) & (rt <= rt_hi)
    rr = rt[mask]
    yy = np.maximum(y[mask], 0.0)
    if rr.size >= 10:
        yy_s = gaussian_filter1d(yy, sigma=min(1.0, yy.size / 25.0), mode="nearest")
        yy_s = np.maximum(yy_s - np.min(yy_s), 0.0)
        baseline = float(np.percentile(yy_s, 25))
        dynamic = float(np.max(yy_s)) - baseline
        if dynamic > 0:
            pair = _find_best_split_pair_prominence(yy_s, baseline, dynamic, DEFAULT_VALLEY_SPLIT_PARAMS)
            if pair is not None:
                i_lo, i_hi = int(pair[0]), int(pair[1])
                p1_rt, p1_h = float(rr[i_lo]), float(yy_s[i_lo])
                p2_rt, p2_h = float(rr[i_hi]), float(yy_s[i_hi])
                # choose non-main-like as secondary
                if abs(p1_rt - main_rt) > abs(p2_rt - main_rt):
                    sec_rt2, sec_h2 = p1_rt, p1_h
                    valley_rt = float(rr[i_hi - (i_hi - i_lo) // 2]) if i_hi > i_lo else p2_rt
                    lo2n, hi2n = remove_overlap_from_second_interval(main_lo, main_hi, rt_lo, valley_rt, min_width=0.01)
                else:
                    sec_rt2, sec_h2 = p2_rt, p2_h
                    valley_rt = float(rr[i_lo + (i_hi - i_lo) // 2]) if i_hi > i_lo else p1_rt
                    lo2n, hi2n = remove_overlap_from_second_interval(main_lo, main_hi, valley_rt, rt_hi, min_width=0.01)
                sec_min_h_local2 = _secondary_min_height_local(
                    rt, y, cand_rt=sec_rt2, main_height=main_height,
                    min_secondary_ratio=min_secondary_ratio,
                    noise_barrier_ratio=noise_barrier_ratio,
                    window_half=small_noise_window_half,
                )
                if (
                    np.isfinite(sec_rt2)
                    and np.isfinite(sec_h2)
                    and sec_h2 >= sec_min_h_local2
                    and abs(sec_rt2 - main_rt) <= small_peak_rt_tol
                    and np.isfinite(lo2n)
                    and np.isfinite(hi2n)
                    and hi2n > lo2n
                ):
                    baseline_small = float(np.percentile(y, 25)) + 0.5 * max(get_last_25pct_avg_noise(rt, y, rt_lo, rt_hi, frac=0.25), 1e-9)
                    lo2s, hi2s = _shrink_interval_around_peak(
                        rt, y, lo2n, hi2n, sec_rt2, baseline_level=baseline_small,
                        max_width=float(max_small_width), min_width=float(min_small_width)
                    )
                    ratio = float(sec_h2 / max(main_height, 1e-12))
                    sc = float(main_score_ai) * max(0.0, min(1.0, ratio))
                    return (lo2s, hi2s, sec_rt2, sec_h2, sc, "sample_recover_valley")
    return None


def _split_overlaps_by_lowest_intensity(
    rt: np.ndarray,
    y: np.ndarray,
    peaks: List[Dict],
    min_split_gap: float = 0.01,
) -> List[Dict]:
    """
    Split overlapping predicted peak intervals.

    Strategy:
    - Sort by rt_min
    - For each overlapping pair, find the lowest intensity point within the overlap window
      and split at that RT (like "取重叠部分中最低强度点进行分割").
    - Keep a small guard gap to avoid zero-width intervals.

    peaks: list of dicts, each must include rt_min/rt_max.
    Returns a new list of peaks with adjusted boundaries.
    """
    if not peaks:
        return peaks
    out = [dict(p) for p in peaks]
    out.sort(key=lambda d: float(d.get("rt_min", np.inf)))
    rt = np.asarray(rt, dtype=np.float64)
    y = np.maximum(np.asarray(y, dtype=np.float64), 0.0)

    def _lowest_rt_in(lo: float, hi: float) -> float:
        m = (rt >= float(lo)) & (rt <= float(hi))
        if np.sum(m) < 2:
            return float(0.5 * (float(lo) + float(hi)))
        rr = rt[m]
        yy = y[m]
        i = int(np.argmin(yy))
        return float(rr[i])

    for i in range(len(out) - 1):
        a = out[i]
        b = out[i + 1]
        alo = _safe_float(a.get("rt_min"), np.nan)
        ahi = _safe_float(a.get("rt_max"), np.nan)
        blo = _safe_float(b.get("rt_min"), np.nan)
        bhi = _safe_float(b.get("rt_max"), np.nan)
        if not all(np.isfinite(v) for v in (alo, ahi, blo, bhi)):
            continue
        if ahi <= alo or bhi <= blo:
            continue
        if ahi <= blo:
            continue  # no overlap
        ov_lo = max(alo, blo)
        ov_hi = min(ahi, bhi)
        if ov_hi <= ov_lo:
            continue
        split_rt = _lowest_rt_in(ov_lo, ov_hi)
        # guard gap
        gap = float(max(min_split_gap, 0.0))
        new_ahi = min(float(ahi), float(split_rt) - gap)
        new_blo = max(float(blo), float(split_rt) + gap)
        # only apply if still valid widths
        if new_ahi > alo:
            a["rt_max"] = float(new_ahi)
            a["source"] = str(a.get("source", "model")) + "|ov_split"
        if bhi > new_blo:
            b["rt_min"] = float(new_blo)
            b["source"] = str(b.get("source", "model")) + "|ov_split"
    return out


def _refine_standard_row_boundaries_at_overlap_valley(
    row_dict: dict,
    rt: np.ndarray,
    y: np.ndarray,
    min_gap: float,
    max_pass: int = 24,
) -> dict:
    """
    同一 XIC 行上 main / small / small2 若有 RT 重叠，在重叠区间内取强度最低点的 RT 作为新分界，
    左侧框右边界、右侧框左边界各让出 min_gap（分钟）。
    """
    r = dict(row_dict)
    tag_cols = [
        ("main", "main_rt_min", "main_rt_max", "main_rt_peak", "main_height"),
        ("small", "small_rt_min", "small_rt_max", "small_rt_peak", "small_height"),
        ("small2", "small2_rt_min", "small2_rt_max", "small2_rt_peak", "small2_height"),
        ("small3", "small3_rt_min", "small3_rt_max", "small3_rt_peak", "small3_height"),
    ]
    rt_a = np.asarray(rt, dtype=np.float64)
    y_a = np.maximum(np.asarray(y, dtype=np.float64), 0.0)

    def collect():
        s = []
        for tag, lk, hk, _, _ in tag_cols:
            lo = _safe_float(r.get(lk), np.nan)
            hi = _safe_float(r.get(hk), np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                s.append((tag, lk, hk, lo, hi))
        s.sort(key=lambda x: x[3])
        return s

    def low_rt(lo: float, hi: float) -> float:
        m = (rt_a >= float(lo)) & (rt_a <= float(hi))
        if np.count_nonzero(m) < 2:
            return float(0.5 * (float(lo) + float(hi)))
        rr = rt_a[m]
        yy = y_a[m]
        return float(rr[int(np.argmin(yy))])

    g = max(float(min_gap), 0.0)
    for _ in range(int(max_pass)):
        segs = collect()
        if len(segs) < 2:
            break
        changed = False
        for i in range(len(segs) - 1):
            _, lk_a, hk_a, alo, ahi = segs[i]
            _, lk_b, hk_b, blo, bhi = segs[i + 1]
            if ahi <= blo + 1e-12:
                continue
            ov_lo = max(alo, blo)
            ov_hi = min(ahi, bhi)
            if ov_hi <= ov_lo:
                continue
            split_rt = low_rt(ov_lo, ov_hi)
            new_ahi = min(ahi, float(split_rt) - g)
            new_blo = max(blo, float(split_rt) + g)
            hit = False
            if new_ahi > alo + 1e-9 and new_ahi < ahi - 1e-9:
                r[hk_a] = float(new_ahi)
                hit = True
            if new_blo < bhi - 1e-9 and new_blo > blo + 1e-9:
                r[lk_b] = float(new_blo)
                hit = True
            if hit:
                changed = True
                break
        if not changed:
            break

    for _, lok, hik, pk_k, ht_k in tag_cols:
        lo = _safe_float(r.get(lok), np.nan)
        hi = _safe_float(r.get(hik), np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pk, h = _peak_rt_height(rt_a, y_a, lo, hi)
            if np.isfinite(pk):
                r[pk_k] = float(pk)
            if np.isfinite(h):
                r[ht_k] = float(h)
    return r


def _apply_overlap_valley_split_standard_df(
    out: pd.DataFrame,
    intensity_mat: np.ndarray,
    rt: np.ndarray,
    min_gap: float,
    feature_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if out is None or out.empty:
        return out
    rt = np.asarray(rt, dtype=np.float64)
    rows = []
    for ix, row in out.iterrows():
        image_name = str(row.get("image", "")).strip()
        idx = _resolve_xic_matrix_row(
            image_name,
            row.get("compound_name"),
            mz=row.get("mz"),
            q3=row.get("q3"),
            feature_df=feature_df,
            pred_csv_row_idx=_df_index_as_pred_row(ix),
            n_rows=int(intensity_mat.shape[0]),
        )
        if idx is None or idx < 0 or idx >= int(intensity_mat.shape[0]):
            rows.append(row.to_dict())
            continue
        y = np.maximum(intensity_mat[int(idx), :].astype(np.float64), 0.0)
        rows.append(_refine_standard_row_boundaries_at_overlap_valley(row.to_dict(), rt, y, min_gap))
    return pd.DataFrame(rows, columns=out.columns)


def _apply_overlap_valley_split_keep_all_df(
    out: pd.DataFrame,
    intensity_mat: np.ndarray,
    rt: np.ndarray,
    min_gap: float,
    feature_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """对 keep_all 多行结果按 image 分组，在重叠区间最低强度点切分 rt_min/rt_max。"""
    if out is None or out.empty or "image" not in out.columns:
        return out
    rt = np.asarray(rt, dtype=np.float64)
    result = out.copy()
    for image_name, sub in result.groupby(result["image"].astype(str), dropna=False):
        if len(sub) < 2:
            continue
        r0 = sub.iloc[0]
        xic_idx = _resolve_xic_matrix_row(
            str(image_name).strip(),
            r0.get("compound_name"),
            mz=r0.get("mz"),
            q3=r0.get("q3"),
            feature_df=feature_df,
            pred_csv_row_idx=_df_index_as_pred_row(sub.index[0]),
            n_rows=int(intensity_mat.shape[0]),
        )
        if xic_idx is None or xic_idx < 0 or xic_idx >= int(intensity_mat.shape[0]):
            continue
        y = np.maximum(intensity_mat[int(xic_idx), :].astype(np.float64), 0.0)
        peaks_sorted = sorted(
            [
                {
                    "rt_min": float(sub.loc[i, "rt_min"]),
                    "rt_max": float(sub.loc[i, "rt_max"]),
                    "_idx": i,
                }
                for i in sub.index
            ],
            key=lambda d: d["rt_min"],
        )
        body = [{"rt_min": p["rt_min"], "rt_max": p["rt_max"], "source": ""} for p in peaks_sorted]
        fixed = _split_overlaps_by_lowest_intensity(rt, y, body, float(min_gap))
        for p, f in zip(peaks_sorted, fixed):
            idx = p["_idx"]
            result.loc[idx, "rt_min"] = float(f["rt_min"])
            result.loc[idx, "rt_max"] = float(f["rt_max"])
            if "rt_peak" in result.columns:
                pk, h = _peak_rt_height(rt, y, float(f["rt_min"]), float(f["rt_max"]))
                if np.isfinite(pk):
                    result.loc[idx, "rt_peak"] = float(pk)
                if "height" in result.columns and np.isfinite(h):
                    result.loc[idx, "height"] = float(h)
    return result


def _annotate_refined_plot_axes(
    ax,
    r: pd.Series,
    rt_full: np.ndarray,
    intensity_full: np.ndarray,
    pred_row: Optional[pd.Series],
    ft: Optional[pd.DataFrame],
) -> None:
    """Overlay nominal RT line + Chinese metric box (RT / compound / response / SNR / scan points)."""
    _matplotlib_cjk_font()
    main_rt = _safe_float(r.get("main_rt_peak"), np.nan)
    main_h = _safe_float(r.get("main_height"), np.nan)
    snr = _safe_float(r.get("main_snr"), np.nan)
    cn = r.get("compound_name")
    nom_rt, ft_lab = _feature_nominal_rt_label(
        ft,
        r.get("mz"),
        r.get("q3"),
        str(r.get("image", "")).strip(),
        cn,
    )

    rt_line = nom_rt
    if not np.isfinite(rt_line):
        if pred_row is not None:
            rt_line = _safe_float(pred_row.get("retention_time"), np.nan)
            if not np.isfinite(rt_line):
                rt_line = _safe_float(pred_row.get("old_rt"), np.nan)

    mlo = _safe_float(r.get("main_rt_min"), np.nan)
    mhi = _safe_float(r.get("main_rt_max"), np.nan)
    # Prefer scan points recomputed from refined (corrected) main interval.
    pt = _scan_points_in_interval(rt_full, intensity_full, mlo, mhi)
    if not np.isfinite(pt) and pred_row is not None:
        pt = _safe_float(pred_row.get("point_counts"), np.nan)

    main_rt_fmt = ("%.4f" % main_rt) if np.isfinite(main_rt) else "—"
    nom_rt_fmt = ("%.4f" % nom_rt) if np.isfinite(nom_rt) else "—"
    h_fmt = ("%.6g" % main_h) if np.isfinite(main_h) else "—"
    snr_fmt = ("%.4g" % snr) if np.isfinite(snr) else "—"
    pt_fmt = ("%d" % int(pt)) if np.isfinite(pt) else "—"

    if ft_lab:
        comp_disp = ft_lab
    else:
        comp_disp = "#%s mz=%.4f q3=%.2f" % (
            cn if cn is not None else "—",
            _safe_float(r.get("mz"), np.nan),
            _safe_float(r.get("q3"), np.nan),
        )

    xlo, xhi = ax.get_xlim()
    if np.isfinite(rt_line) and xlo <= rt_line <= xhi:
        ax.axvline(
            float(rt_line),
            color="#c0392b",
            linestyle="--",
            linewidth=1.8,
            zorder=5,
            label="标称RT",
        )
    if np.isfinite(main_rt) and xlo <= main_rt <= xhi:
        if not np.isfinite(rt_line) or abs(main_rt - rt_line) > 1e-4:
            ax.axvline(
                float(main_rt),
                color="#27ae60",
                linestyle=":",
                linewidth=1.2,
                zorder=4,
                alpha=0.9,
                label="主峰RT",
            )

    txt = (
        "主峰RT: %s min\n标称RT(feature): %s min\n初步匹配化合物: %s\n响应(峰高): %s\n信噪比: %s\n扫描点数: %s"
        % (main_rt_fmt, nom_rt_fmt, comp_disp, h_fmt, snr_fmt, pt_fmt)
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.92),
        zorder=10,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)


def _plot_refined_predictions(
    result_dir: Path,
    refined_df: pd.DataFrame,
    rt: np.ndarray,
    intensity_mat: np.ndarray,
    roi_map: Dict[str, Tuple[float, float]],
    sigma: float = 1.0,
    plot_dir_name: str = "refined_plots",
    plot_output_parent: Optional[Path] = None,
    plot_file_prefix: Optional[str] = None,
    pred_lookup_csv: Optional[Path] = None,
    feature_csv: Optional[Path] = None,
):
    if plot_output_parent is not None:
        out_dir = Path(plot_output_parent).resolve() / plot_dir_name
    else:
        out_dir = result_dir / plot_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = _load_optional_csv(pred_lookup_csv) if pred_lookup_csv else _load_optional_csv(result_dir / "prediction.csv")
    ft_df = _load_optional_csv(feature_csv) if feature_csv else None
    if ft_df is None:
        alt = result_dir / "feature.csv"
        ft_df = _load_optional_csv(alt)
    for ix, r in refined_df.iterrows():
        image_name = str(r.get("image", "")).strip()
        idx = _resolve_xic_matrix_row(
            image_name,
            r.get("compound_name"),
            mz=r.get("mz"),
            q3=r.get("q3"),
            feature_df=ft_df,
            pred_csv_row_idx=_df_index_as_pred_row(ix),
            n_rows=int(intensity_mat.shape[0]),
        )
        if idx is None or idx < 0 or idx >= intensity_mat.shape[0]:
            continue
        y = intensity_mat[idx, :].astype(np.float64)
        y = np.maximum(y, 0.0)
        y_metric = y.copy()
        if sigma > 0 and y.size >= 10:
            y = gaussian_filter1d(y, sigma=min(float(sigma), y.size / 25.0), mode="nearest")
            y = np.maximum(y, 0.0)

        rt_lo, rt_hi = _resolve_roi_rt_window(
            roi_map, image_name, float(np.min(rt)), float(np.max(rt))
        )
        peak_rt = _safe_float(r.get("main_rt_peak"), np.nan)
        if np.isfinite(peak_rt):
            win_lo = max(float(np.min(rt)), peak_rt - 1.0)
            win_hi = min(float(np.max(rt)), peak_rt + 1.0)
            if win_hi <= win_lo:
                win_lo, win_hi = float(rt_lo), float(rt_hi)
        else:
            win_lo, win_hi = float(rt_lo), float(rt_hi)
        mask = (rt >= win_lo) & (rt <= win_hi)
        x_plot = rt[mask]
        y_plot = y[mask]
        if x_plot.size < 2:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_plot, y_plot, color="blue", linewidth=1.5)

        mlo = _safe_float(r.get("main_rt_min"), np.nan)
        mhi = _safe_float(r.get("main_rt_max"), np.nan)
        if np.isfinite(mlo) and np.isfinite(mhi) and mhi > mlo:
            ax.axvspan(mlo, mhi, color="#2ecc71", alpha=0.24, label="Main interval")

        slo = _safe_float(r.get("small_rt_min"), np.nan)
        shi = _safe_float(r.get("small_rt_max"), np.nan)
        if np.isfinite(slo) and np.isfinite(shi) and shi > slo:
            ax.axvspan(slo, shi, color="#f39c12", alpha=0.22, label="Small interval")
        s2lo = _safe_float(r.get("small2_rt_min"), np.nan)
        s2hi = _safe_float(r.get("small2_rt_max"), np.nan)
        if np.isfinite(s2lo) and np.isfinite(s2hi) and s2hi > s2lo:
            ax.axvspan(s2lo, s2hi, color="#8e44ad", alpha=0.18, label="Small interval #2")
        s3lo = _safe_float(r.get("small3_rt_min"), np.nan)
        s3hi = _safe_float(r.get("small3_rt_max"), np.nan)
        if np.isfinite(s3lo) and np.isfinite(s3hi) and s3hi > s3lo:
            ax.axvspan(s3lo, s3hi, color="#16a085", alpha=0.16, label="Small interval #3")

        ax.set_xlim(float(win_lo), float(win_hi))
        ax.set_xlabel("Retention Time (min)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Refined prediction (sigma={sigma}) - {image_name}")
        ax.grid(True, alpha=0.25)
        pr = _prediction_row_for_image(pred_df, image_name)
        _annotate_refined_plot_axes(ax, r, rt, y_metric, pr, ft_df)

        out_name = Path(image_name).stem if image_name else f"compound_{idx+1}"
        pfx = (plot_file_prefix or "").strip()
        for c in '\\/:*?"<>|':
            pfx = pfx.replace(c, "_")
        png_base = "%s_%s_refined.png" % (pfx, out_name) if pfx else "%s_refined.png" % out_name
        out_png = out_dir / png_base
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_post_newtest(args):
    root = Path(args.results_dir).resolve()
    pred_path = root / "prediction.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Need prediction.csv in {root}")

    # newtest.py 批量输出通常只写 prediction.csv；xic_matrix.npy 仍在 testXIC 生成的 ROI 目录。
    xic_path = root / "xic_matrix.npy"
    xic_dir_opt = getattr(args, "xic_dir", None)
    if not xic_path.exists():
        if xic_dir_opt:
            alt = Path(xic_dir_opt).resolve() / "xic_matrix.npy"
            if alt.exists():
                xic_path = alt
        if not xic_path.exists():
            hint = (
                f"Need xic_matrix.npy in {root}, or pass --xic_dir to the ROI folder from preprocessing.xic_extraction "
                f"(same name as this sample, e.g. ...\\sample_xic_rois\\{root.name})."
            )
            raise FileNotFoundError(hint)

    if str(xic_path.resolve()) != str((root / "xic_matrix.npy").resolve()):
        print(f"[INFO] Using xic_matrix.npy from: {xic_path}")

    feature_df = _load_feature_table_for_mapping(root, xic_path, xic_dir_opt)

    df = pd.read_csv(pred_path)
    xic = np.load(str(xic_path))
    rt = xic[0, :].astype(np.float64)
    if np.nanmax(rt) > 200:
        rt = rt / 60.0
    intensity_mat = xic[1:, :].astype(np.float64)
    roi_map = _merge_roi_window_maps(
        _load_roi_windows(root),
        _load_roi_windows(Path(xic_dir_opt).resolve()) if xic_dir_opt else None,
    )

    rows = []
    grouped = df.groupby(df["image"].astype(str), dropna=False)
    for image_name, g_raw in grouped:
        pred_indices = np.asarray(g_raw.index)
        g = g_raw.copy().reset_index(drop=True)
        if g.empty:
            continue
        # Keep only images that have at least one valid predicted box in input prediction.csv
        has_valid_box = False
        for _, rr_box in g.iterrows():
            bx1 = _safe_float(rr_box.get("box_x1"), np.nan)
            by1 = _safe_float(rr_box.get("box_y1"), np.nan)
            bx2 = _safe_float(rr_box.get("box_x2"), np.nan)
            by2 = _safe_float(rr_box.get("box_y2"), np.nan)
            if all(np.isfinite(v) for v in (bx1, by1, bx2, by2)) and (bx2 > bx1) and (by2 > by1):
                has_valid_box = True
                break
        if not has_valid_box:
            continue
        # XIC row: mz+q3 / image stem prefix via feature.csv; else prediction.csv row index.
        xic_idx = None
        for i in range(len(g)):
            pri = int(pred_indices[i])
            gi = g.iloc[i]
            xic_idx = _resolve_xic_matrix_row(
                str(image_name),
                gi.get("compound_name"),
                mz=gi.get("mz"),
                q3=gi.get("q3"),
                feature_df=feature_df,
                pred_csv_row_idx=pri,
                n_rows=int(intensity_mat.shape[0]),
            )
            if xic_idx is not None:
                break
        if xic_idx is None or xic_idx < 0 or xic_idx >= intensity_mat.shape[0]:
            continue

        y = intensity_mat[xic_idx, :]
        rt_lo, rt_hi = _resolve_roi_rt_window(
            roi_map, str(image_name), float(np.min(rt)), float(np.max(rt))
        )

        # Main-row selection fix:
        # do NOT rely on score only (often equal 1.00). Choose the row with strongest
        # true signal in interval (peak height), then area, then score.
        candidates = []
        for irow, rr in g.iterrows():
            lo = _safe_float(rr.get("rt_min"), np.nan)
            hi = _safe_float(rr.get("rt_max"), np.nan)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            pk_rt_i, pk_h_i = _peak_rt_height(rt, y, lo, hi)
            area_i = _interval_area(rt, y, lo, hi)
            sc_i = _safe_float(rr.get("score"), 0.0)
            candidates.append((irow, pk_h_i, area_i, sc_i))
        if not candidates:
            continue
        hs = np.array([c[1] for c in candidates], dtype=float)
        ars = np.array([c[2] for c in candidates], dtype=float)
        scs = np.array([c[3] for c in candidates], dtype=float)
        i_h = int(np.nanargmax(hs))
        i_a = int(np.nanargmax(ars))
        if i_h == i_a:
            main_idx = int(candidates[i_h][0])
        else:
            ord_h = np.argsort(np.argsort(-hs))
            ord_a = np.argsort(np.argsort(-ars))
            comb = ord_h + ord_a
            bests = np.where(comb == np.min(comb))[0]
            if bests.size == 1:
                main_idx = int(candidates[int(bests[0])][0])
            else:
                b2 = int(bests[np.argmax(scs[bests])])
                main_idx = int(candidates[b2][0])
        r_main = g.iloc[main_idx].copy()
        rt_min = _safe_float(r_main.get("rt_min"), np.nan)
        rt_max = _safe_float(r_main.get("rt_max"), np.nan)

        main_score = _safe_float(r_main.get("score"), 0.0)

        # Follow run_two_round_detection gate first: confidence + SNR + secondary-peak condition
        main_snr_raw = compute_snr_outside_box(rt, y, rt_min, rt_max)
        has_sec_gate_raw, _, _ = has_secondary_peak_in_roi(
            rt, y, rt_min, rt_max, float(rt_lo), float(rt_hi),
            min_secondary_ratio=args.min_secondary_ratio,
            noise_barrier_ratio=args.noise_barrier_ratio,
        )
        gate_ok = bool(
            (main_score >= float(args.min_confidence))
            and (np.isfinite(main_snr_raw) and main_snr_raw >= float(args.min_snr))
            and has_sec_gate_raw
        )

        # ========= New mode: keep all predicted boxes and refine/split =========
        if bool(getattr(args, "keep_all_pred_boxes", False)):
            # Build refined peaks for every valid predicted box row in this image group.
            refined: List[Dict] = []
            for irow, rr in g.iterrows():
                rt_min0 = _safe_float(rr.get("rt_min"), np.nan)
                rt_max0 = _safe_float(rr.get("rt_max"), np.nan)
                sc0 = _safe_float(rr.get("score"), 0.0)
                if not np.isfinite(rt_min0) or not np.isfinite(rt_max0) or rt_max0 <= rt_min0:
                    continue

                # 1) interval correction for this box (independent)
                peers_i = _peer_intervals_from_group(g, skip_row_index=irow)
                rt_min_adj, rt_max_adj = adjust_first_round_interval(
                    rt,
                    y,
                    rt_min0,
                    rt_max0,
                    float(rt_lo),
                    float(rt_hi),
                    min_secondary_ratio=args.min_secondary_ratio,
                    edge_max_span_min=float(args.edge_max_span_min),
                    edge_noise_percentile=float(args.edge_noise_percentile),
                    peer_rt_intervals=peers_i,
                    **_adjust_first_round_interval_kwargs(args, (rt_min0, rt_max0)),
                )

                pk_rt, pk_h = _peak_rt_height(rt, y, rt_min_adj, rt_max_adj)
                if not np.isfinite(pk_rt):
                    continue

                peak = {
                    "image": image_name,
                    "compound_name": rr.get("compound_name"),
                    "mz": _safe_float(rr.get("mz"), np.nan),
                    "q3": _safe_float(rr.get("q3"), np.nan),
                    "rt_min": float(rt_min_adj),
                    "rt_max": float(rt_max_adj),
                    "rt_peak": float(pk_rt),
                    "height": float(pk_h),
                    "score_ai": float(sc0),
                    "source": "model_refined",
                    "box_x1": _safe_float(rr.get("box_x1"), np.nan),
                    "box_y1": _safe_float(rr.get("box_y1"), np.nan),
                    "box_x2": _safe_float(rr.get("box_x2"), np.nan),
                    "box_y2": _safe_float(rr.get("box_y2"), np.nan),
                }
                refined.append(peak)

                # 2) second-pass small-peak recognition: if this refined box has a strong secondary
                # peak in its ROI, trigger "left/right re-predict" style search around THIS box.
                if bool(getattr(args, "enable_lr_repredict_on_small_fail", True)):
                    sec_gate, _, _ = has_secondary_peak_in_roi(
                        rt, y, float(rt_min_adj), float(rt_max_adj), float(rt_lo), float(rt_hi),
                        min_secondary_ratio=args.min_secondary_ratio,
                        noise_barrier_ratio=args.noise_barrier_ratio,
                    )
                    if bool(sec_gate):
                        seg_l_lo, seg_l_hi = float(rt_lo), min(float(rt_hi), float(rt_min_adj))
                        seg_r_lo, seg_r_hi = max(float(rt_lo), float(rt_max_adj)), float(rt_hi)
                        side_peaks = []
                        for side_name, slo, shi in (("left", seg_l_lo, seg_l_hi), ("right", seg_r_lo, seg_r_hi)):
                            if not np.isfinite(slo) or not np.isfinite(shi) or shi - slo < 0.02:
                                continue
                            pk_rt_s, pk_h_s = _find_secondary_peak_by_roi_rules(
                                rt=rt,
                                y=y,
                                main_lo=float(rt_min_adj),
                                main_hi=float(rt_max_adj),
                                rt_lo=float(slo),
                                rt_hi=float(shi),
                                min_secondary_ratio=float(args.min_secondary_ratio),
                                noise_barrier_ratio=float(args.noise_barrier_ratio),
                                smooth_sigma=1.0,
                            )
                            if not np.isfinite(pk_rt_s) or not np.isfinite(pk_h_s):
                                continue
                            noise_avg_s = get_last_25pct_avg_noise(rt, y, float(slo), float(shi), frac=0.25)
                            baseline_s = float(np.percentile(y, 25)) + 0.5 * max(noise_avg_s, 1e-9)
                            lo_s, hi_s = _build_interval_around_peak_in_segment(
                                rt=rt,
                                y=y,
                                peak_rt=float(pk_rt_s),
                                seg_lo=float(slo),
                                seg_hi=float(shi),
                                max_width=float(args.max_small_width),
                                min_width=float(args.min_small_width),
                                baseline_level=float(baseline_s),
                            )
                            if np.isfinite(lo_s) and np.isfinite(hi_s) and hi_s > lo_s:
                                side_peaks.append((side_name, float(lo_s), float(hi_s), float(pk_rt_s), float(pk_h_s)))
                        # add side peaks as additional refined peaks
                        for side_name, lo_s, hi_s, pk_rt_s, pk_h_s in side_peaks:
                            refined.append(
                                {
                                    "image": image_name,
                                    "compound_name": rr.get("compound_name"),
                                    "mz": _safe_float(rr.get("mz"), np.nan),
                                    "q3": _safe_float(rr.get("q3"), np.nan),
                                    "rt_min": float(lo_s),
                                    "rt_max": float(hi_s),
                                    "rt_peak": float(pk_rt_s),
                                    "height": float(pk_h_s),
                                    "score_ai": float(sc0) * 0.95,
                                    "source": f"lr_repredict({side_name})",
                                    "box_x1": np.nan,
                                    "box_y1": np.nan,
                                    "box_x2": np.nan,
                                    "box_y2": np.nan,
                                }
                            )

            # 3) Final peak splitting: split overlaps at lowest intensity in overlap.
            if bool(getattr(args, "split_overlaps", True)):
                refined = _split_overlaps_by_lowest_intensity(
                    rt=rt,
                    y=y,
                    peaks=refined,
                    min_split_gap=float(getattr(args, "overlap_split_min_gap", 0.01)),
                )

            # Deduplicate near-identical peaks by rt_peak proximity, keep highest height.
            dedup_tol = float(getattr(args, "all_boxes_dedup_rt_tol", 0.02))
            refined = [p for p in refined if np.isfinite(_safe_float(p.get("rt_min"), np.nan)) and np.isfinite(_safe_float(p.get("rt_max"), np.nan))]
            refined.sort(key=lambda d: float(d.get("height", 0.0)), reverse=True)
            uniq = []
            for p in refined:
                rp = _safe_float(p.get("rt_peak"), np.nan)
                if not np.isfinite(rp):
                    continue
                if any(abs(rp - _safe_float(q.get("rt_peak"), np.nan)) <= dedup_tol for q in uniq):
                    continue
                uniq.append(p)

            # Write per-peak rows
            for rank, ppk in enumerate(sorted(uniq, key=lambda d: float(d.get("rt_min", np.inf))), start=1):
                ppk2 = dict(ppk)
                ppk2["peak_rank"] = int(rank)
                ppk2["roi_rt_lo"] = float(rt_lo)
                ppk2["roi_rt_hi"] = float(rt_hi)
                rows.append(ppk2)
            continue
        # ========= end new mode =========

        # 1) Main interval correction: always apply.
        # Stop condition is fixed to one-sided low-noise 20th percentile as requested.
        peers_main = _peer_intervals_from_group(g, skip_row_index=main_idx)
        rt_min_adj_try, rt_max_adj_try = adjust_first_round_interval(
            rt,
            y,
            rt_min,
            rt_max,
            float(rt_lo),
            float(rt_hi),
            min_secondary_ratio=args.min_secondary_ratio,
            edge_max_span_min=float(args.edge_max_span_min),
            edge_noise_percentile=float(args.edge_noise_percentile),
            peer_rt_intervals=peers_main,
            **_adjust_first_round_interval_kwargs(args, (rt_min, rt_max)),
        )

        # Sanity guard: if corrected interval drifts away from original main apex, revert.
        orig_peak_rt, orig_peak_h = _peak_rt_height(rt, y, rt_min, rt_max)
        adj_peak_rt, adj_peak_h = _peak_rt_height(rt, y, rt_min_adj_try, rt_max_adj_try)
        drift_bad = False
        if np.isfinite(orig_peak_rt) and np.isfinite(adj_peak_rt):
            if abs(adj_peak_rt - orig_peak_rt) > float(args.max_main_peak_drift):
                drift_bad = True
        if np.isfinite(orig_peak_h) and orig_peak_h > 0 and np.isfinite(adj_peak_h):
            if adj_peak_h < float(args.min_main_height_keep_ratio) * orig_peak_h:
                drift_bad = True
        if drift_bad:
            rt_min_adj, rt_max_adj = rt_min, rt_max
        else:
            rt_min_adj, rt_max_adj = rt_min_adj_try, rt_max_adj_try

        main_peak_rt, main_height = _peak_rt_height(rt, y, rt_min_adj, rt_max_adj)
        main_snr = compute_snr_outside_box(rt, y, rt_min_adj, rt_max_adj)
        main_skew = _segment_skew(rt, y, rt_min_adj, rt_max_adj)
        main_skew_local = _segment_skew(
            rt, y, main_peak_rt - float(args.std_skew_window), main_peak_rt + float(args.std_skew_window)
        )
        sec_min_h = _secondary_min_height(
            rt, y, float(rt_lo), float(rt_hi), main_height,
            min_secondary_ratio=float(args.min_secondary_ratio),
            noise_barrier_ratio=float(args.noise_barrier_ratio),
        )
        # Additional safeguard requested: if toward-peak moved boundary too high (>2x small threshold),
        # reverse it toward local noise baseline.
        rt_min_adj, rt_max_adj = _constrain_adjusted_interval(
            rt,
            y,
            rt_min_adj,
            rt_max_adj,
            main_peak_rt=main_peak_rt,
            small_min_h=sec_min_h,
            noise_percentile=float(args.edge_noise_percentile),
            edge_max_span_min=float(args.edge_max_span_min),
            edge_noise_stop_mode=str(getattr(args, "edge_noise_stop_mode", "roi_bottom_decile_mean")),
        )
        main_peak_rt, main_height = _peak_rt_height(rt, y, rt_min_adj, rt_max_adj)

        # 1b) Explicit double-peak split in main interval (for clearly separated non-shoulder peaks).
        small_from_main_split = None
        if bool(getattr(args, "enable_main_double_split", True)):
            ds = _detect_double_peak_with_valley_in_interval(
                rt=rt,
                y=y,
                lo=rt_min_adj,
                hi=rt_max_adj,
                min_gap_rt=float(args.main_double_split_min_gap),
                max_valley_ratio=float(args.main_double_split_max_valley_ratio),
                min_valley_drop_ratio=float(
                    getattr(args, "main_double_split_min_valley_drop_ratio", 0.0)
                ),
                min_peak_above_valley_ratio=float(
                    getattr(args, "main_double_split_min_peak_above_valley_ratio", 0.0)
                ),
                min_peak_sep_ratio_of_span=float(
                    getattr(args, "main_double_split_min_peak_sep_ratio_of_span", 0.0)
                ),
            )
            if ds is not None:
                p1_rt, p1_h, p2_rt, p2_h, valley_rt = ds
                if p1_h >= p2_h:
                    main_rt_new, main_h_new = p1_rt, p1_h
                    sec_rt_new, sec_h_new = p2_rt, p2_h
                else:
                    main_rt_new, main_h_new = p2_rt, p2_h
                    sec_rt_new, sec_h_new = p1_rt, p1_h

                lo_l, hi_l = float(rt_min_adj), float(valley_rt)
                lo_r, hi_r = float(valley_rt), float(rt_max_adj)
                if main_rt_new <= valley_rt:
                    mlo0, mhi0 = lo_l, hi_l
                    slo0, shi0 = lo_r, hi_r
                else:
                    mlo0, mhi0 = lo_r, hi_r
                    slo0, shi0 = lo_l, hi_l

                noise_avg = get_last_25pct_avg_noise(rt, y, float(rt_lo), float(rt_hi), frac=0.10)
                baseline_small = _small_boundary_baseline_level(
                    y=y,
                    noise_avg=noise_avg,
                    main_height=main_height,
                    small_height=sec_h_new,
                )
                mlo, mhi = _shrink_interval_around_peak(
                    rt, y, mlo0, mhi0, main_rt_new, baseline_level=baseline_small,
                    max_width=float(args.max_small_width), min_width=float(args.min_small_width),
                )
                slo, shi = _shrink_interval_around_peak(
                    rt, y, slo0, shi0, sec_rt_new, baseline_level=baseline_small,
                    max_width=float(args.max_small_width), min_width=float(args.min_small_width),
                )

                # Replace main interval by split-main side when valid.
                if np.isfinite(mlo) and np.isfinite(mhi) and mhi > mlo:
                    rt_min_adj, rt_max_adj = float(mlo), float(mhi)
                    main_peak_rt, main_height = _peak_rt_height(rt, y, rt_min_adj, rt_max_adj)
                    main_snr = compute_snr_outside_box(rt, y, rt_min_adj, rt_max_adj)
                    main_skew = _segment_skew(rt, y, rt_min_adj, rt_max_adj)
                    main_skew_local = _segment_skew(
                        rt, y, main_peak_rt - float(args.std_skew_window), main_peak_rt + float(args.std_skew_window)
                    )

                if (
                    np.isfinite(slo)
                    and np.isfinite(shi)
                    and shi > slo
                    and _rt_offset_gate_with_width(
                        main_rt=main_peak_rt,
                        cand_lo=float(slo),
                        cand_hi=float(shi),
                        rt_tol=_effective_valley_small_peak_rt_tol(args),
                        max_width=float(args.max_small_width),
                        boundary_pad=float(args.small_boundary_pad),
                    )
                ):
                    ratio = float(sec_h_new / max(main_height, 1e-12))
                    sec_min_h_local = _secondary_min_height_local(
                        rt, y, cand_rt=float(sec_rt_new), main_height=main_height,
                        min_secondary_ratio=float(args.min_secondary_ratio),
                        noise_barrier_ratio=float(args.noise_barrier_ratio),
                        window_half=float(args.small_noise_window_half),
                    )
                    if np.isfinite(sec_h_new) and sec_h_new >= sec_min_h_local:
                        adj_sc = main_score * max(0.0, min(1.0, ratio))
                        small_from_main_split = (
                            float(slo), float(shi), float(sec_rt_new), float(sec_h_new), main_score, adj_sc, ratio
                        )

        # 2) Small-peak from model rows near main (rt tolerance), discounted by height ratio.
        small_from_model = None
        for j, rr in g.iterrows():
            if j == main_idx:
                continue
            lo2 = _safe_float(rr.get("rt_min"), np.nan)
            hi2 = _safe_float(rr.get("rt_max"), np.nan)
            if not np.isfinite(lo2) or not np.isfinite(hi2) or lo2 >= hi2:
                continue
            if not _rt_offset_gate_with_width(
                main_rt=main_peak_rt,
                cand_lo=float(lo2),
                cand_hi=float(hi2),
                rt_tol=_effective_small_peak_rt_tol(args),
                max_width=float(args.max_small_width),
                boundary_pad=float(args.small_boundary_pad),
            ):
                continue
            pk_rt, pk_h = _peak_rt_height(rt, y, lo2, hi2)
            if not np.isfinite(pk_rt) or not np.isfinite(main_peak_rt):
                continue
            sec_min_h_local = _secondary_min_height_local(
                rt, y, cand_rt=pk_rt, main_height=main_height,
                min_secondary_ratio=float(args.min_secondary_ratio),
                noise_barrier_ratio=float(args.noise_barrier_ratio),
                window_half=float(args.small_noise_window_half),
            )
            if not np.isfinite(pk_h) or pk_h < sec_min_h_local:
                continue
            raw_sc = _safe_float(rr.get("score"), 0.0)
            ratio = float(pk_h / max(main_height, 1e-12))
            adj_sc = raw_sc * min(1.0, max(0.0, ratio))
            # Reuse original pipeline behavior: remove overlap with main interval first.
            lo2n, hi2n = remove_overlap_from_second_interval(rt_min_adj, rt_max_adj, lo2, hi2, min_width=0.01)
            if not np.isfinite(lo2n) or not np.isfinite(hi2n) or hi2n <= lo2n:
                continue
            # Then shrink interval around its local peak to avoid overly wide small-peak boxes.
            noise_avg = get_last_25pct_avg_noise(rt, y, float(rt_lo), float(rt_hi), frac=0.10)
            baseline_small = _small_boundary_baseline_level(
                y=y,
                noise_avg=noise_avg,
                main_height=main_height,
                small_height=pk_h,
            )
            lo2s, hi2s = _shrink_interval_around_peak(
                rt, y, lo2n, hi2n, pk_rt, baseline_level=baseline_small,
                max_width=float(args.max_small_width), min_width=float(args.min_small_width)
            )
            cand = (lo2s, hi2s, pk_rt, pk_h, raw_sc, adj_sc, ratio)
            if small_from_model is None or cand[5] > small_from_model[5]:
                small_from_model = cand

        # 3) Search extra peaks in ROI (same rule as 2nd peak; up to 3).
        # 若已有模型小框，仍必须在 ROI 上继续找其余峰（此前错误地用 if small_from_model is None 整体跳过，导致第三峰丢失）。
        small_from_roi = None
        small_from_roi_extra: List[Tuple] = []
        exclude_extra: List[Tuple[float, float]] = []
        if small_from_main_split is not None:
            exclude_extra.append(
                (float(small_from_main_split[0]), float(small_from_main_split[1]))
            )
        if small_from_model is not None:
            exclude_extra.append((float(small_from_model[0]), float(small_from_model[1])))
        _relax = float(getattr(args, "secondary_roi_global_gate_relax_frac", 0.0))
        _relax = max(0.0, min(_relax, 0.35))
        _thr_global_base = float(sec_min_h) * (1.0 - _relax)

        for _roi_k in range(3):
            sec_rt, sec_h = _find_secondary_peak_by_roi_rules(
                rt=rt,
                y=y,
                main_lo=rt_min_adj,
                main_hi=rt_max_adj,
                rt_lo=float(rt_lo),
                rt_hi=float(rt_hi),
                min_secondary_ratio=float(args.min_secondary_ratio),
                noise_barrier_ratio=float(args.noise_barrier_ratio),
                smooth_sigma=1.0,
                extra_exclude=exclude_extra,
            )
            if not (
                np.isfinite(sec_rt)
                and np.isfinite(main_peak_rt)
                and np.isfinite(sec_h)
                and sec_h >= _thr_global_base
            ):
                break
            half0 = max(0.05, float(args.max_small_width) / 2.0)
            lo2n, hi2n = remove_overlap_from_second_interval(
                rt_min_adj, rt_max_adj, sec_rt - half0, sec_rt + half0, min_width=0.01
            )
            cand_tuple = None
            if np.isfinite(lo2n) and np.isfinite(hi2n) and hi2n > lo2n:
                noise_avg = get_last_25pct_avg_noise(rt, y, float(rt_lo), float(rt_hi), frac=0.10)
                baseline_small = _small_boundary_baseline_level(
                    y=y,
                    noise_avg=noise_avg,
                    main_height=main_height,
                    small_height=sec_h,
                )
                lo2s, hi2s = _shrink_interval_around_peak(
                    rt, y, lo2n, hi2n, sec_rt, baseline_level=baseline_small,
                    max_width=float(args.max_small_width), min_width=float(args.min_small_width)
                )
                lo2s = _cap_left_shrink_for_strong_roi_secondary(
                    overlap_left=float(lo2n),
                    overlap_right=float(hi2n),
                    final_left=float(lo2s),
                    main_height=float(main_height),
                    small_height=float(sec_h),
                )
                hi2s = _cap_right_shrink_for_strong_roi_secondary(
                    overlap_left=float(lo2n),
                    overlap_right=float(hi2n),
                    final_right=float(hi2s),
                    main_height=float(main_height),
                    small_height=float(sec_h),
                )
                lo2s = _enforce_low_noise_boundary_with_mid_reverse(
                    rt=rt,
                    y=y,
                    boundary_rt=float(lo2s),
                    other_boundary_rt=float(hi2s),
                    search_lo=float(lo2n),
                    search_hi=float(hi2n),
                    baseline_level=float(baseline_small),
                    side="left",
                )
                hi2s = _enforce_low_noise_boundary_with_mid_reverse(
                    rt=rt,
                    y=y,
                    boundary_rt=float(hi2s),
                    other_boundary_rt=float(lo2s),
                    search_lo=float(lo2n),
                    search_hi=float(hi2n),
                    baseline_level=float(baseline_small),
                    side="right",
                )
                ratio = float(sec_h / max(main_height, 1e-12))
                adj_sc = main_score * max(0.0, min(1.0, ratio))
                sec_min_h_local = _secondary_min_height_local(
                    rt, y, cand_rt=sec_rt, main_height=main_height,
                    min_secondary_ratio=float(args.min_secondary_ratio),
                    noise_barrier_ratio=float(args.noise_barrier_ratio),
                    window_half=float(args.small_noise_window_half),
                )
                if sec_h >= sec_min_h_local and _rt_offset_gate_with_width(
                    main_rt=main_peak_rt,
                    cand_lo=float(lo2s),
                    cand_hi=float(hi2s),
                    rt_tol=_effective_small_peak_rt_tol(args),
                    max_width=float(args.max_small_width),
                    boundary_pad=float(args.small_boundary_pad),
                ):
                    cand_tuple = (lo2s, hi2s, sec_rt, sec_h, main_score, adj_sc, ratio)
                    exclude_extra.append((float(lo2s), float(hi2s)))
            if cand_tuple is None:
                break
            if _roi_k == 0:
                small_from_roi = cand_tuple
            else:
                small_from_roi_extra.append(cand_tuple)

        # 4) Valley split fallback (can coexist with other small-peak sources).
        small_from_valley = None
        if bool(args.enable_valley_fallback):
            bx = [
                _safe_float(r_main.get("box_x1"), np.nan),
                _safe_float(r_main.get("box_y1"), np.nan),
                _safe_float(r_main.get("box_x2"), np.nan),
                _safe_float(r_main.get("box_y2"), np.nan),
            ]
            if all(np.isfinite(v) for v in bx):
                split_boxes, split_scores, did = _split_one_box_by_valley(
                    img_path=str(image_name),
                    box=np.asarray(bx, dtype=np.float32),
                    score=main_score,
                    rt_array=rt,
                    intensity=y,
                    true_rt=float(r_main.get("old_rt", main_peak_rt if np.isfinite(main_peak_rt) else rt.mean())),
                    rt_window=(float(rt_lo), float(rt_hi)),
                    params=DEFAULT_VALLEY_SPLIT_PARAMS,
                )
                if did and len(split_boxes) == 2:
                    parts = []
                    for sb in split_boxes:
                        l, h, _, _ = box_to_rt_range(sb[0], sb[1], sb[2], sb[3], float(main_peak_rt), rt, (float(rt_lo), float(rt_hi)))
                        pk_rt, pk_h = _peak_rt_height(rt, y, l, h)
                        parts.append((l, h, pk_rt, pk_h))
                    if np.isfinite(main_peak_rt):
                        parts = sorted(parts, key=lambda t: abs(t[2] - main_peak_rt))
                    # nearest is main-like, farther is small-like
                    if len(parts) >= 2:
                        p_small = parts[1]
                        valley_rt_tol = _effective_valley_small_peak_rt_tol(args)
                        if np.isfinite(p_small[2]) and abs(p_small[2] - main_peak_rt) <= valley_rt_tol:
                            lo2n, hi2n = remove_overlap_from_second_interval(
                                rt_min_adj, rt_max_adj, float(p_small[0]), float(p_small[1]), min_width=0.01
                            )
                            if not np.isfinite(lo2n) or not np.isfinite(hi2n) or hi2n <= lo2n:
                                continue
                            noise_avg = get_last_25pct_avg_noise(rt, y, float(rt_lo), float(rt_hi), frac=0.10)
                            baseline_small = _small_boundary_baseline_level(
                                y=y,
                                noise_avg=noise_avg,
                                main_height=main_height,
                                small_height=float(p_small[3]),
                            )
                            lo2s, hi2s = _shrink_interval_around_peak(
                                rt, y, lo2n, hi2n, float(p_small[2]), baseline_level=baseline_small,
                                max_width=float(args.max_small_width), min_width=float(args.min_small_width)
                            )
                            sec_min_h_local = _secondary_min_height_local(
                                rt, y, cand_rt=float(p_small[2]), main_height=main_height,
                                min_secondary_ratio=float(args.min_secondary_ratio),
                                noise_barrier_ratio=float(args.noise_barrier_ratio),
                                window_half=float(args.small_noise_window_half),
                            )
                            if (not np.isfinite(p_small[3])) or p_small[3] < sec_min_h_local:
                                continue
                            ratio = float(p_small[3] / max(main_height, 1e-12))
                            adj_sc = main_score * max(0.0, min(1.0, ratio))
                            small_from_valley = (lo2s, hi2s, p_small[2], p_small[3], main_score, adj_sc, ratio)

        # Merge small-peak candidates from multiple sources; keep up to three extras (max 4 boxes incl. main).
        small_candidates = []
        if small_from_main_split is not None:
            small_candidates.append(("main_double_split", small_from_main_split))
        if small_from_model is not None:
            small_candidates.append(("model", small_from_model))
        if small_from_roi is not None:
            small_candidates.append(("roi_secondary", small_from_roi))
        for ix, ent in enumerate(small_from_roi_extra):
            small_candidates.append((f"roi_secondary_{ix + 2}", ent))
        if small_from_valley is not None:
            small_candidates.append(("valley_split", small_from_valley))

        # Only trigger left/right re-predict when the original predicted main-box peak
        # itself fails the small-peak style height criterion (user-requested strict scope).
        orig_main_pk_rt, orig_main_pk_h = _peak_rt_height(rt, y, rt_min, rt_max)
        orig_main_sec_min_h_local = _secondary_min_height_local(
            rt, y, cand_rt=orig_main_pk_rt, main_height=orig_main_pk_h,
            min_secondary_ratio=float(args.min_secondary_ratio),
            noise_barrier_ratio=float(args.noise_barrier_ratio),
            window_half=float(args.small_noise_window_half),
        )
        orig_main_peak_fail_small_rule = (
            (not np.isfinite(orig_main_pk_rt))
            or (not np.isfinite(orig_main_pk_h))
            or (not np.isfinite(orig_main_sec_min_h_local))
            or (orig_main_pk_h < orig_main_sec_min_h_local)
        )

        lr_repredict_applied = 0
        # New rule:
        # if no small peak satisfies criteria, cancel current predicted main box and
        # re-predict from left/right regions around that box (on XIC signal), then
        # map the two side detections into main/small outputs.
        if (
            (len(small_candidates) == 0)
            and bool(getattr(args, "enable_lr_repredict_on_small_fail", True))
            and bool(orig_main_peak_fail_small_rule)
        ):
            seg_l_lo, seg_l_hi = float(rt_lo), min(float(rt_hi), float(rt_min_adj))
            seg_r_lo, seg_r_hi = max(float(rt_lo), float(rt_max_adj)), float(rt_hi)
            side_peaks = []
            for side_name, slo, shi in (("left", seg_l_lo, seg_l_hi), ("right", seg_r_lo, seg_r_hi)):
                if not np.isfinite(slo) or not np.isfinite(shi) or shi - slo < 0.02:
                    continue
                pk_rt_s, pk_h_s = _find_secondary_peak_by_roi_rules(
                    rt=rt,
                    y=y,
                    main_lo=rt_min_adj,
                    main_hi=rt_max_adj,
                    rt_lo=float(slo),
                    rt_hi=float(shi),
                    min_secondary_ratio=float(args.min_secondary_ratio),
                    noise_barrier_ratio=float(args.noise_barrier_ratio),
                    smooth_sigma=1.0,
                )
                if not np.isfinite(pk_rt_s) or not np.isfinite(pk_h_s):
                    continue
                noise_avg_s = get_last_25pct_avg_noise(rt, y, float(slo), float(shi), frac=0.25)
                baseline_s = float(np.percentile(y, 25)) + 0.5 * max(noise_avg_s, 1e-9)
                lo_s, hi_s = _build_interval_around_peak_in_segment(
                    rt=rt, y=y, peak_rt=pk_rt_s, seg_lo=float(slo), seg_hi=float(shi),
                    max_width=float(args.max_small_width), min_width=float(args.min_small_width),
                    baseline_level=float(baseline_s),
                )
                if np.isfinite(lo_s) and np.isfinite(hi_s) and hi_s > lo_s:
                    side_peaks.append((side_name, lo_s, hi_s, float(pk_rt_s), float(pk_h_s)))

            if side_peaks:
                side_peaks = sorted(side_peaks, key=lambda t: t[4], reverse=True)
                main_side = side_peaks[0]
                rt_min_adj, rt_max_adj = float(main_side[1]), float(main_side[2])
                main_peak_rt, main_height = float(main_side[3]), float(main_side[4])
                main_snr = compute_snr_outside_box(rt, y, rt_min_adj, rt_max_adj)
                main_skew = _segment_skew(rt, y, rt_min_adj, rt_max_adj)
                main_skew_local = _segment_skew(
                    rt, y, main_peak_rt - float(args.std_skew_window), main_peak_rt + float(args.std_skew_window)
                )
                main_score = main_score * 0.95  # slight penalty for fallback replacement
                if len(side_peaks) >= 2:
                    sp = side_peaks[1]
                    sp_lo, sp_hi, sp_rt, sp_h = float(sp[1]), float(sp[2]), float(sp[3]), float(sp[4])
                    if _rt_offset_gate_with_width(
                        main_rt=main_peak_rt,
                        cand_lo=sp_lo,
                        cand_hi=sp_hi,
                        rt_tol=_effective_small_peak_rt_tol(args),
                        max_width=float(args.max_small_width),
                        boundary_pad=float(args.small_boundary_pad),
                    ):
                        ratio = float(sp_h / max(main_height, 1e-12))
                        lr_small = (
                            sp_lo,
                            sp_hi,
                            sp_rt,
                            sp_h,
                            main_score,
                            main_score * max(0.0, min(1.0, ratio)),
                            ratio,
                        )
                        small_candidates.append(("lr_repredict", lr_small))
                else:
                    small_candidates = []
                lr_repredict_applied = 1

        # Deduplicate by RT proximity and keep top-2 by adjusted score.
        dedup_tol = float(getattr(args, "small_dedup_rt_tol", 0.03))
        small_candidates = sorted(small_candidates, key=lambda t: float(t[1][5]), reverse=True)
        uniq = []
        for src, cand in small_candidates:
            rt_pk = float(cand[2])
            if any(abs(rt_pk - float(c2[1][2])) <= dedup_tol for c2 in uniq):
                continue
            uniq.append((src, cand))
        small_candidates = uniq[:3]

        if len(small_candidates) >= 1:
            small_src = small_candidates[0][0]
            small = small_candidates[0][1]
        else:
            small_src = "none"
            small = None
        if len(small_candidates) >= 2:
            small2_src = small_candidates[1][0]
            small2 = small_candidates[1][1]
        else:
            small2_src = "none"
            small2 = None
        if len(small_candidates) >= 3:
            small3_src = small_candidates[2][0]
            small3 = small_candidates[2][1]
        else:
            small3_src = "none"
            small3 = None

        has_sec_gate, _, _ = has_secondary_peak_in_roi(
            rt, y, rt_min_adj, rt_max_adj, float(rt_lo), float(rt_hi),
            min_secondary_ratio=args.min_secondary_ratio,
            noise_barrier_ratio=args.noise_barrier_ratio,
        )
        noise_avg = get_last_25pct_avg_noise(rt, y, float(rt_lo), float(rt_hi), frac=0.25)

        rows.append(
            {
                "image": image_name,
                "compound_name": r_main.get("compound_name"),
                "mz": _safe_float(r_main.get("mz"), np.nan),
                "q3": _safe_float(r_main.get("q3"), np.nan),
                "main_rt_min": rt_min_adj,
                "main_rt_max": rt_max_adj,
                "main_rt_peak": main_peak_rt,
                "main_height": main_height,
                "main_score_ai": main_score,
                "main_snr": main_snr,
                "main_skew": main_skew,
                "main_skew_local": main_skew_local,
                "small_rt_min": small[0] if small else np.nan,
                "small_rt_max": small[1] if small else np.nan,
                "small_rt_peak": small[2] if small else np.nan,
                "small_height": small[3] if small else np.nan,
                "small_score_ai_raw": small[4] if small else np.nan,
                "small_score_ai_discounted": small[5] if small else np.nan,
                "small_height_ratio": small[6] if small else np.nan,
                "small_source": small_src,
                "small2_rt_min": small2[0] if small2 else np.nan,
                "small2_rt_max": small2[1] if small2 else np.nan,
                "small2_rt_peak": small2[2] if small2 else np.nan,
                "small2_height": small2[3] if small2 else np.nan,
                "small2_score_ai_raw": small2[4] if small2 else np.nan,
                "small2_score_ai_discounted": small2[5] if small2 else np.nan,
                "small2_height_ratio": small2[6] if small2 else np.nan,
                "small2_source": small2_src,
                "small3_rt_min": small3[0] if small3 else np.nan,
                "small3_rt_max": small3[1] if small3 else np.nan,
                "small3_rt_peak": small3[2] if small3 else np.nan,
                "small3_height": small3[3] if small3 else np.nan,
                "small3_score_ai_raw": small3[4] if small3 else np.nan,
                "small3_score_ai_discounted": small3[5] if small3 else np.nan,
                "small3_height_ratio": small3[6] if small3 else np.nan,
                "small3_source": small3_src,
                "has_secondary_gate": int(bool(has_sec_gate)),
                "gate_ok_for_adjustment": int(bool(gate_ok)),
                "lr_repredict_applied": int(lr_repredict_applied),
                "noise_avg_last25pct": noise_avg,
                "small_min_height_required": sec_min_h,
            }
        )

    out = pd.DataFrame(rows)
    if (
        len(rows) > 0
        and (not bool(getattr(args, "disable_overlap_valley_split", False)))
    ):
        min_gap_ov = float(getattr(args, "overlap_split_min_gap", 0.01))
        if bool(getattr(args, "keep_all_pred_boxes", False)):
            out = _apply_overlap_valley_split_keep_all_df(
                out, intensity_mat, rt, min_gap_ov, feature_df=feature_df
            )
        else:
            out = _apply_overlap_valley_split_standard_df(
                out, intensity_mat, rt, min_gap_ov, feature_df=feature_df
            )
    out_path = root / args.output_name
    if bool(getattr(args, "keep_all_pred_boxes", False)):
        out_path = root / str(getattr(args, "all_boxes_output_name", "prediction_refined_all.csv"))
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved refined predictions: {out_path}")
    if bool(args.plot):
        pop = getattr(args, "plot_output_parent", None)
        plot_parent = Path(str(pop).strip()).resolve() if pop and str(pop).strip() else None
        pfx = getattr(args, "plot_file_prefix", None)
        if pfx is not None:
            pfx = str(pfx).strip() or None
        if plot_parent is not None and not pfx:
            pfx = root.parent.name
        _feat_csv = None
        if (root / "feature.csv").is_file():
            _feat_csv = root / "feature.csv"
        elif xic_dir_opt:
            _fc = Path(xic_dir_opt).resolve() / "feature.csv"
            if _fc.is_file():
                _feat_csv = _fc
        elif (xic_path.parent / "feature.csv").is_file():
            _feat_csv = xic_path.parent / "feature.csv"
        _plot_refined_predictions(
            result_dir=root,
            refined_df=out,
            rt=rt,
            intensity_mat=intensity_mat,
            roi_map=roi_map,
            sigma=float(args.plot_sigma),
            plot_dir_name=str(args.plot_dir_name),
            plot_output_parent=plot_parent,
            plot_file_prefix=pfx,
            pred_lookup_csv=root / "prediction.csv",
            feature_csv=_feat_csv,
        )
        dest_msg = (
            (plot_parent / str(args.plot_dir_name)) if plot_parent is not None else (root / args.plot_dir_name)
        )
        print(f"[OK] Saved refined plots: {dest_msg}")


def _fit_line_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    k, b = np.polyfit(x, y, 1)
    yhat = k * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot <= 0 else (1 - ss_res / ss_tot)
    return float(k), float(b), float(r2)


def _infer_conc_from_name(text: str):
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*(ppb|ppm)", str(text), flags=re.IGNORECASE)
    if not m:
        return np.nan
    v = float(m.group(1))
    u = m.group(2).lower()
    return v if u == "ppb" else v * 1000.0


def run_standard_mode(args):
    root = Path(args.standards_root).resolve()
    files = sorted(root.rglob(args.input_name))
    if not files:
        raise FileNotFoundError(f"No {args.input_name} under {root}")

    all_rows = []
    for f in files:
        df = pd.read_csv(f)
        conc = _infer_conc_from_name(str(f.parent))
        if not np.isfinite(conc):
            continue
        df["concentration_ppb"] = conc
        df["source_file"] = str(f)
        all_rows.append(df)
    if not all_rows:
        raise RuntimeError("No valid standard rows parsed.")
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df["compound_key"] = all_df.apply(_compound_key_from_prediction_row, axis=1)

    selected_rows = []
    ref_rows = []
    for key, g in all_df.groupby("compound_key"):
        singles = g[g["small_source"] == "none"]
        ref_rt = float(np.nanmedian(singles["main_rt_peak"])) if len(singles) else float(np.nanmedian(g["main_rt_peak"]))
        skew_col = "main_skew_local" if "main_skew_local" in g.columns else "main_skew"
        ref_skew = float(np.nanmedian(singles[skew_col])) if len(singles) else float(np.nanmedian(g[skew_col]))

        pick = []
        for conc, gg in g.groupby("concentration_ppb"):
            # Stage-1 (standard mode): select true main only.
            # If first prediction has two intervals, do not use small candidate yet.
            cands = []
            for _, r in gg.iterrows():
                cands.append(("main", _safe_float(r["main_rt_peak"]), _safe_float(r["main_height"]), _safe_float(r["main_score_ai"]), _safe_float(r.get("main_skew_local", r.get("main_skew", np.nan))), r))
            if not cands:
                continue
            # keep the one closest to standard single-peak RT
            best = min(cands, key=lambda t: abs(t[1] - ref_rt) if np.isfinite(t[1]) else 999)
            tag, rtp, h, sc, skew, raw = best
            pick.append(
                {
                    "compound_key": key,
                    "mz": _safe_float(raw.get("mz"), np.nan),
                    "q3": _safe_float(raw.get("q3"), np.nan),
                    "concentration_ppb": conc,
                    "selected_peak_tag": tag,
                    "selected_rt_peak": rtp,
                    "selected_height": h,
                    "selected_score_ai": sc,
                    "selected_skew": skew,
                    "area_proxy": h,  # proxy if no area columns
                    "source_file": raw.get("source_file"),
                    "source_image": raw.get("image", ""),
                    "small_rt_peak_raw": _safe_float(raw.get("small_rt_peak"), np.nan),
                    "small_height_raw": _safe_float(raw.get("small_height"), np.nan),
                    "small_score_ai_raw": _safe_float(raw.get("small_score_ai_discounted"), np.nan),
                    "small_source_raw": raw.get("small_source", "none"),
                }
            )

        if len(pick) < 2:
            continue
        p = pd.DataFrame(pick).sort_values("concentration_ppb")
        k, b, r2 = _fit_line_r2(p["concentration_ppb"], p["area_proxy"])
        r2_before = r2
        if np.isfinite(r2) and r2 < args.r2_threshold:
            # Try repair: replace worst residual point with alt peak (within rt tolerance) if improves R2.
            yhat = k * p["concentration_ppb"].to_numpy() + b
            res = np.abs(p["area_proxy"].to_numpy() - yhat)
            widx = int(np.argmax(res))
            conc_bad = float(p.iloc[widx]["concentration_ppb"])
            g_bad = g[g["concentration_ppb"] == conc_bad]
            alt_candidates = []
            for _, r in g_bad.iterrows():
                for tag, rtp, h, sc, skew in [
                    ("main", _safe_float(r["main_rt_peak"]), _safe_float(r["main_height"]), _safe_float(r["main_score_ai"]), _safe_float(r.get("main_skew_local", r.get("main_skew", np.nan)))),
                    ("small", _safe_float(r.get("small_rt_peak"), np.nan), _safe_float(r.get("small_height"), np.nan), _safe_float(r.get("small_score_ai_discounted"), np.nan), _safe_float(r.get("main_skew_local", r.get("main_skew", np.nan)))),
                ]:
                    if np.isfinite(rtp) and abs(rtp - float(p.iloc[widx]["selected_rt_peak"])) <= args.rt_tolerance_std:
                        alt_candidates.append((tag, rtp, h, sc, skew))
            best_r2 = r2
            best_alt = None
            for tag, rtp, h, sc, skew in alt_candidates:
                p2 = p.copy()
                p2.loc[p2.index[widx], ["selected_peak_tag", "selected_rt_peak", "selected_height", "selected_score_ai", "selected_skew", "area_proxy"]] = [
                    tag, rtp, h, sc, skew, h
                ]
                _, _, r2_new = _fit_line_r2(p2["concentration_ppb"], p2["area_proxy"])
                if np.isfinite(r2_new) and r2_new > best_r2:
                    best_r2 = r2_new
                    best_alt = p2
            if best_alt is not None:
                p = best_alt
                r2 = best_r2

        p["r2_compound"] = r2
        p["r2_before_compound"] = r2_before
        p["r2_after_compound"] = r2
        p["slope"] = k
        p["intercept"] = b

        # Stage-2: after true main is chosen, recover nearby small peak only if
        # secondary apex is within tolerance around selected main.
        small_keep = []
        for _, rr in p.iterrows():
            srt = _safe_float(rr.get("small_rt_peak_raw"), np.nan)
            mrt = _safe_float(rr.get("selected_rt_peak"), np.nan)
            if np.isfinite(srt) and np.isfinite(mrt) and abs(srt - mrt) <= float(args.small_near_main_tol_std):
                small_keep.append(1)
            else:
                small_keep.append(0)
        p["small_keep_after_main"] = small_keep
        p["small_rt_peak_selected"] = np.where(p["small_keep_after_main"] == 1, p["small_rt_peak_raw"], np.nan)
        p["small_height_selected"] = np.where(p["small_keep_after_main"] == 1, p["small_height_raw"], np.nan)
        p["small_score_selected"] = np.where(p["small_keep_after_main"] == 1, p["small_score_ai_raw"], np.nan)

        selected_rows.append(p)
        ref_rows.append({"compound_key": key, "ref_rt_peak": float(np.nanmedian(p["selected_rt_peak"])), "ref_skew": float(np.nanmedian(p["selected_skew"])), "r2": r2})

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    refs = pd.DataFrame(ref_rows)
    # === mz-level expected RT from multi-channel (same precursor mz, multiple q3) ===
    # For each channel (compound_key), compute signal strength + RT stability, then
    # use top-K channels to compute weighted mean RT as mz-level expected RT.
    if (not sel.empty) and (not refs.empty) and ("mz" in sel.columns):
        try:
            gk = sel.groupby("compound_key", dropna=False)
            key_stats = gk.agg(
                mz=("mz", "first"),
                q3=("q3", "first"),
                rt_med=("selected_rt_peak", "median"),
                rt_std=("selected_rt_peak", "std"),
                height_med=("selected_height", "median"),
                score_med=("selected_score_ai", "median"),
            ).reset_index()
            key_stats["rt_std"] = pd.to_numeric(key_stats["rt_std"], errors="coerce").fillna(0.0)
            key_stats["height_med"] = pd.to_numeric(key_stats["height_med"], errors="coerce").fillna(0.0)
            # Merge per-channel r2 from refs
            key_stats = key_stats.merge(refs[["compound_key", "r2"]], on="compound_key", how="left")
            key_stats["r2"] = pd.to_numeric(key_stats["r2"], errors="coerce").fillna(0.0)

            # Weight: strong signal (height) * stable RT (1/(std+eps)) * r2 gate (>=0)
            eps = 1e-6
            key_stats["stability_w"] = 1.0 / (key_stats["rt_std"].astype(float) + eps)
            key_stats["weight"] = key_stats["height_med"].astype(float) * key_stats["stability_w"].astype(float) * key_stats["r2"].clip(lower=0.0, upper=1.0).astype(float)

            top_k = int(max(1, getattr(args, "mz_rt_top_k", 2)))
            mz_rows = []
            for mz, gm in key_stats.groupby("mz", dropna=False):
                mzv = _safe_float(mz, np.nan)
                if not np.isfinite(mzv):
                    continue
                gm = gm.copy()
                gm = gm[np.isfinite(pd.to_numeric(gm["rt_med"], errors="coerce"))]
                if gm.empty:
                    continue
                gm = gm.sort_values("weight", ascending=False)
                use = gm.head(top_k)
                w = pd.to_numeric(use["weight"], errors="coerce").to_numpy(dtype=float)
                rt = pd.to_numeric(use["rt_med"], errors="coerce").to_numpy(dtype=float)
                if not np.any(np.isfinite(w)) or float(np.nansum(w)) <= 0:
                    rt_w = float(np.nanmedian(rt))
                else:
                    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
                    rt_w = float(np.nansum(w * rt) / max(float(np.nansum(w)), 1e-12))
                mz_rows.append(
                    {
                        "mz": float(mzv),
                        "mz_expected_rt": rt_w,
                        "n_channels_total": int(gm.shape[0]),
                        "n_channels_used": int(use.shape[0]),
                        "used_compound_keys": "|".join(use["compound_key"].astype(str).tolist()),
                    }
                )
            mz_refs = pd.DataFrame(mz_rows)
            if not mz_refs.empty:
                mz_refs.to_csv(out_dir / "standard_mz_expected_rt.csv", index=False, encoding="utf-8-sig")
                mz_map = {round(float(r["mz"]), 4): float(r["mz_expected_rt"]) for _, r in mz_refs.iterrows() if np.isfinite(_safe_float(r.get("mz_expected_rt"), np.nan))}
                refs["mz_round4"] = refs["compound_key"].astype(str).str.extract(r"^mz([\d\.]+)_q3", expand=False)
                refs["mz_round4"] = pd.to_numeric(refs["mz_round4"], errors="coerce").round(4)
                refs["mz_expected_rt"] = refs["mz_round4"].map(mz_map)
                refs.drop(columns=["mz_round4"], inplace=True, errors="ignore")
        except Exception:
            # Keep standard_mode robust even if mz-level aggregation fails for some inputs.
            pass
    sel.to_csv(out_dir / "standard_selected_peaks.csv", index=False, encoding="utf-8-sig")
    refs.to_csv(out_dir / "standard_refs.csv", index=False, encoding="utf-8-sig")

    # R2 comparison (compound level): before vs after correction.
    if not sel.empty:
        cmp_compound = (
            sel.groupby("compound_key", dropna=False)[["r2_before_compound", "r2_after_compound"]]
            .first()
            .reset_index()
            .rename(columns={"r2_before_compound": "r2_before", "r2_after_compound": "r2_after"})
        )
        cmp_compound["improved"] = cmp_compound["r2_after"] - cmp_compound["r2_before"]
        cmp_compound.to_csv(out_dir / "r2_compare_compound.csv", index=False, encoding="utf-8-sig")

        # mz-level quantitative table:
        # for one precursor with multiple product ions, keep max area per (mz, concentration).
        mz_quant = (
            sel.sort_values("area_proxy", ascending=False)
            .groupby(["mz", "concentration_ppb"], dropna=False, as_index=False)
            .first()
        )
        mz_quant.to_csv(out_dir / "standard_mz_quant_max_area.csv", index=False, encoding="utf-8-sig")

        # mz-level R2 before/after summary:
        # before: all points; after: if needed remove one worst residual point (robust correction summary).
        r2_mz_rows = []
        for mz, gm in mz_quant.groupby("mz", dropna=False):
            gm = gm.sort_values("concentration_ppb")
            x = gm["concentration_ppb"].to_numpy(dtype=float)
            y = gm["area_proxy"].to_numpy(dtype=float)
            k0, b0, r2_0 = _fit_line_r2(x, y)
            r2_1 = r2_0
            k1, b1 = k0, b0
            removed_idx = np.nan
            if np.isfinite(r2_0) and r2_0 < args.r2_threshold and len(gm) >= 5 and np.isfinite(k0):
                yhat = k0 * x + b0
                res = np.abs(y - yhat)
                worst = int(np.argmax(res))
                keep = np.ones(len(x), dtype=bool)
                keep[worst] = False
                k_tmp, b_tmp, r2_tmp = _fit_line_r2(x[keep], y[keep])
                if np.isfinite(r2_tmp) and r2_tmp > r2_1:
                    r2_1, k1, b1 = r2_tmp, k_tmp, b_tmp
                    removed_idx = worst
            r2_mz_rows.append(
                {
                    "mz": mz,
                    "n_points": len(gm),
                    "r2_before": r2_0,
                    "r2_after": r2_1,
                    "slope_before": k0,
                    "intercept_before": b0,
                    "slope_after": k1,
                    "intercept_after": b1,
                    "removed_point_index_for_after": removed_idx,
                }
            )
        pd.DataFrame(r2_mz_rows).to_csv(out_dir / "r2_compare_mz_max_area.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Standard outputs: {out_dir}")


def run_sample_mode(args):
    sample_csv_path = Path(args.sample_refined_csv).resolve()
    pred = pd.read_csv(sample_csv_path)
    refs = pd.read_csv(Path(args.standard_refs_csv).resolve())
    ref_map = {r["compound_key"]: (float(r["ref_rt_peak"]), float(r["ref_skew"])) for _, r in refs.iterrows()}
    # Avoid double main-box correction by default when input is already refined.
    looks_refined = (
        ("prediction_refined" in sample_csv_path.name.lower())
        or {"main_snr", "main_skew_local", "small_source"}.issubset(set(pred.columns))
    )
    refine_main_enabled = bool(args.refine_main_in_sample)
    if bool(getattr(args, "auto_skip_refine_on_refined_input", True)) and looks_refined and (not bool(getattr(args, "force_refine_main_in_sample", False))):
        refine_main_enabled = False
    print(f"[INFO] sample_mode refine_main_in_sample effective={int(refine_main_enabled)} (requested={int(bool(args.refine_main_in_sample))}, refined_input={int(looks_refined)})")

    # Optional: connect to best-adjusted standard result CSV (mz-level or selected-peaks table).
    mz_ref_map = {}
    if getattr(args, "standard_best_csv", None):
        best_path = Path(args.standard_best_csv).resolve()
        if best_path.exists():
            best = pd.read_csv(best_path)
            if "mz" in best.columns:
                # Build mz-level reference RT/skew by median from available columns.
                rt_col = "selected_rt_peak" if "selected_rt_peak" in best.columns else None
                if rt_col is None and "ref_rt_peak" in best.columns:
                    rt_col = "ref_rt_peak"
                skew_col = "selected_skew" if "selected_skew" in best.columns else None
                if skew_col is None and "ref_skew" in best.columns:
                    skew_col = "ref_skew"
                if rt_col is not None:
                    g = best.groupby("mz", dropna=False)
                    for mz, gg in g:
                        mzv = _safe_float(mz, np.nan)
                        if not np.isfinite(mzv):
                            continue
                        rt_ref = float(np.nanmedian(pd.to_numeric(gg[rt_col], errors="coerce")))
                        sk_ref = float(np.nanmedian(pd.to_numeric(gg[skew_col], errors="coerce"))) if skew_col is not None else np.nan
                        mz_ref_map[round(mzv, 4)] = (rt_ref, sk_ref)

    # XIC trace is required for plotting and for the final-confidence SNR
    # definition (signal in box / noise mean outside all boxes).
    xic_path = None
    rt = None
    inten_mat = None
    xic_path = _locate_xic_matrix_from_sample_csv(sample_csv_path)
    if xic_path is not None and xic_path.exists():
        xic = np.load(str(xic_path))
        rt = xic[0, :].astype(np.float64)
        if np.nanmax(rt) > 200:
            rt = rt / 60.0
        inten_mat = xic[1:, :].astype(np.float64)

    feature_df_sample = (
        _load_optional_csv(xic_path.parent / "feature.csv") if xic_path is not None else None
    )

    out_rows = []
    for ix, r in pred.iterrows():
        key = _compound_key_from_prediction_row(r)
        ref = ref_map.get(key, (np.nan, np.nan))
        if (not np.isfinite(ref[0])) and mz_ref_map:
            mz_key = round(_safe_float(r.get("mz"), np.nan), 4)
            ref = mz_ref_map.get(mz_key, ref)
        ref_rt, ref_sk = ref
        main_rt = _safe_float(r.get("main_rt_peak"), np.nan)
        main_sk = _safe_float(r.get("main_skew_local", r.get("main_skew", np.nan)), np.nan)
        main_snr = _safe_float(r.get("main_snr"), np.nan)
        main_ai = _safe_float(r.get("main_score_ai"), 0.0)

        # Per-row XIC trace (if available) for SNR/noise computations.
        y_row = None
        idx = None
        if (rt is not None) and (inten_mat is not None):
            idx = _resolve_xic_matrix_row(
                str(r.get("image", "")).strip(),
                r.get("compound_name"),
                mz=r.get("mz"),
                q3=r.get("q3"),
                feature_df=feature_df_sample,
                pred_csv_row_idx=_df_index_as_pred_row(ix),
                n_rows=int(inten_mat.shape[0]),
            )
            if idx is not None and 0 <= idx < inten_mat.shape[0]:
                y_row = np.maximum(inten_mat[idx, :].astype(np.float64), 0.0)

        # Optional main-interval refinement in sample mode:
        # re-locate main peak near standard reference to avoid inherited wrong boxes.
        main_refine_applied = 0
        if refine_main_enabled and (rt is not None) and (inten_mat is not None):
            if y_row is not None:
                y = y_row
                mlo0 = _safe_float(r.get("main_rt_min"), np.nan)
                mhi0 = _safe_float(r.get("main_rt_max"), np.nan)
                if np.isfinite(mlo0) and np.isfinite(mhi0) and mhi0 > mlo0:
                    mlo1, mhi1, mrt1, mh1 = _refine_main_interval_near_reference(
                        rt=rt,
                        y=y,
                        main_lo=mlo0,
                        main_hi=mhi0,
                        ref_rt=ref_rt,
                        search_tol=float(args.sample_main_search_tol),
                        max_width=float(args.sample_main_max_width),
                        boundary_noise_percentile=float(args.sample_main_boundary_noise_percentile),
                        peer_rt_intervals=None,
                        **_boundary_posterior_kwargs(args),
                    )
                    r["main_rt_min"] = mlo1
                    r["main_rt_max"] = mhi1
                    r["main_rt_peak"] = mrt1
                    r["main_height"] = mh1
                    main_rt = mrt1
                    main_snr = compute_snr_outside_box(rt, y, mlo1, mhi1)
                    main_sk = _segment_skew(rt, y, mlo1, mhi1)
                    r["main_snr"] = main_snr
                    r["main_skew_local"] = main_sk
                    main_refine_applied = 1

        keep_small = False
        small_rt = _safe_float(r.get("small_rt_peak"), np.nan)
        small_ai = _safe_float(r.get("small_score_ai_discounted"), np.nan)
        small_recover_trigger = "none"

        # Recover small peak when missing, or (optional) when existing small interval looks truncated / swallowed by main.
        if bool(args.recover_small) and (rt is not None) and (inten_mat is not None):
            idx = _resolve_xic_matrix_row(
                str(r.get("image", "")).strip(),
                r.get("compound_name"),
                mz=r.get("mz"),
                q3=r.get("q3"),
                feature_df=feature_df_sample,
                pred_csv_row_idx=_df_index_as_pred_row(ix),
                n_rows=int(inten_mat.shape[0]),
            )
            if idx is not None and 0 <= idx < inten_mat.shape[0]:
                missing_small = not np.isfinite(small_rt)
                weak_small = False
                mlo_w = _safe_float(r.get("main_rt_min"), np.nan)
                mhi_w = _safe_float(r.get("main_rt_max"), np.nan)
                slo_w = _safe_float(r.get("small_rt_min"), np.nan)
                shi_w = _safe_float(r.get("small_rt_max"), np.nan)
                if (
                    bool(getattr(args, "sample_recover_weak_small", False))
                    and (not missing_small)
                ):
                    weak_small = _small_peak_is_weak(
                        mlo_w, mhi_w, slo_w, shi_w,
                        max_width=float(args.sample_weak_small_max_width),
                        min_overlap_frac=float(args.sample_weak_small_overlap_frac),
                    )
                if missing_small or weak_small:
                    small_recover_trigger = "missing" if missing_small else "weak_retry"
                    backup_small = {
                        "small_rt_min": r.get("small_rt_min"),
                        "small_rt_max": r.get("small_rt_max"),
                        "small_rt_peak": r.get("small_rt_peak"),
                        "small_height": r.get("small_height"),
                        "small_score_ai_discounted": r.get("small_score_ai_discounted"),
                        "small_source": r.get("small_source"),
                    }
                    y = np.maximum(inten_mat[idx, :].astype(np.float64), 0.0)
                    main_lo = _safe_float(r.get("main_rt_min"), np.nan)
                    main_hi = _safe_float(r.get("main_rt_max"), np.nan)
                    main_h = _safe_float(r.get("main_height"), np.nan)
                    rec_small = _recover_small_for_sample(
                        rt=rt,
                        y=y,
                        main_rt=main_rt,
                        main_lo=main_lo,
                        main_hi=main_hi,
                        main_height=main_h if np.isfinite(main_h) else np.max(y),
                        main_score_ai=main_ai,
                        target_rt=ref_rt,
                        roi_half_window=float(args.sample_roi_half_window),
                        sample_rt_tolerance=float(args.sample_rt_tolerance),
                        small_peak_rt_tol=float(args.sample_small_peak_rt_tol),
                        min_secondary_ratio=float(args.sample_min_secondary_ratio),
                        noise_barrier_ratio=float(args.sample_noise_barrier_ratio),
                        max_small_width=float(args.sample_max_small_width),
                        small_noise_window_half=float(args.sample_small_noise_window_half),
                        min_small_width=float(args.sample_min_small_width),
                    )
                    if rec_small is not None:
                        r["small_rt_min"] = rec_small[0]
                        r["small_rt_max"] = rec_small[1]
                        r["small_rt_peak"] = rec_small[2]
                        r["small_height"] = rec_small[3]
                        r["small_score_ai_discounted"] = rec_small[4]
                        r["small_source"] = rec_small[5]
                        small_rt = rec_small[2]
                        small_ai = rec_small[4]
                    elif weak_small:
                        for k, v in backup_small.items():
                            r[k] = v
                        small_rt = _safe_float(r.get("small_rt_peak"), np.nan)
                        small_ai = _safe_float(r.get("small_score_ai_discounted"), np.nan)
                        small_recover_trigger = "weak_retry_failed"
        if np.isfinite(small_rt) and np.isfinite(ref_rt):
            rt_ok = abs(small_rt - ref_rt) <= args.sample_rt_tolerance
            skew_ok = (not np.isfinite(ref_sk)) or (not np.isfinite(main_sk)) or (abs(main_sk - ref_sk) <= args.skew_tolerance)
            ai_ok = np.isfinite(small_ai) and (float(small_ai) >= float(args.sample_keep_small_min_ai))
            keep_small = bool(rt_ok and (skew_ok or ai_ok))

        # RT shift should be measured against the effective target RT:
        # - if standard ref exists: use ref_rt
        # - else: fall back to main_rt (shift=0, do not penalize due to missing reference)
        target_rt_used = ref_rt if np.isfinite(ref_rt) else main_rt
        main_rt_shift = abs(main_rt - target_rt_used) if np.isfinite(main_rt) and np.isfinite(target_rt_used) else 0.0
        main_sk_diff = abs(main_sk - ref_sk) if np.isfinite(main_sk) and np.isfinite(ref_sk) else np.nan
        main_conf = _composite_conf(main_ai, main_snr, main_rt_shift, main_sk_diff, rt_tol=args.rt_score_tolerance, skew_tol=args.skew_tolerance)
        main_lo = _safe_float(r.get("main_rt_min"), np.nan)
        main_hi = _safe_float(r.get("main_rt_max"), np.nan)
        small_lo_tmp = _safe_float(r.get("small_rt_min"), np.nan)
        small_hi_tmp = _safe_float(r.get("small_rt_max"), np.nan)
        if y_row is not None and (rt is not None):
            noise_mu, snr_main_nm, snr_small_nm = _snr_box_over_noise_mean(
                rt,
                y_row,
                main_lo,
                main_hi,
                small_lo_tmp,
                small_hi_tmp,
                min_noise_pts=5,
            )
        else:
            noise_mu, snr_main_nm, snr_small_nm = float("nan"), float("nan"), float("nan")
            # Fallback to existing main_snr when xic trace is unavailable.
            if np.isfinite(main_snr):
                snr_main_nm = float(main_snr)
        main_conf_final = _final_conf_from_figure(
            main_ai,
            snr_main_nm,
            main_rt_shift,
            main_sk_diff,
            rt_max_shift=float(args.final_rt_max_shift),
            rt_power=float(args.final_rt_power),
            snr_ref=float(args.final_snr_ref),
            skew_tol=float(args.skew_tolerance),
            snr_weight=float(args.final_snr_weight),
            skew_weight=float(args.final_skew_weight),
        )

        # Compute small-peak confidence for any detected small peak (finite AI), not only when
        # keep_small_by_standard is True — so plots/CSV can show small conf for filtered candidates.
        small_conf = np.nan
        small_conf_final = np.nan
        if np.isfinite(small_rt) and np.isfinite(small_ai):
            small_rt_shift = abs(small_rt - target_rt_used) if np.isfinite(small_rt) and np.isfinite(target_rt_used) else 0.0
            small_conf = _composite_conf(small_ai, main_snr, small_rt_shift, main_sk_diff, rt_tol=args.rt_score_tolerance, skew_tol=args.skew_tolerance)
            small_conf_final = _final_conf_from_figure(
                small_ai,
                snr_small_nm,
                small_rt_shift,
                main_sk_diff,
                rt_max_shift=float(args.final_rt_max_shift),
                rt_power=float(args.final_rt_power),
                snr_ref=float(args.final_snr_ref),
                skew_tol=float(args.skew_tolerance),
                snr_weight=float(args.final_snr_weight),
                skew_weight=float(args.final_skew_weight),
            )
            small_conf_final *= _skew_direction_boost(
                ref_sk,
                main_rt,
                small_rt,
                skew_high=float(args.final_skew_high),
                skew_scale=float(args.final_skew_scale),
                boost_max=float(args.final_skew_boost_max),
                penalty_min=float(args.final_skew_penalty_min),
            )
            small_conf_final = float(max(0.0, min(1.0, small_conf_final)))

        rec = dict(r)
        rec["compound_key"] = key
        rec["ref_rt_peak"] = ref_rt
        rec["ref_skew"] = ref_sk
        rec["target_rt_used"] = target_rt_used
        rec["target_rt_source"] = "standard_ref" if np.isfinite(ref_rt) else "main_rt_fallback"
        rec["main_refine_applied"] = int(main_refine_applied)
        rec["main_interval_width"] = _safe_float(r.get("main_rt_max"), np.nan) - _safe_float(r.get("main_rt_min"), np.nan)
        rec["small_interval_width"] = _safe_float(r.get("small_rt_max"), np.nan) - _safe_float(r.get("small_rt_min"), np.nan)
        rec["keep_small_by_standard"] = int(keep_small)
        rec["main_conf_composite"] = main_conf
        rec["small_conf_composite"] = small_conf
        rec["main_conf_final"] = main_conf_final
        rec["small_conf_final"] = small_conf_final
        rec["noise_mean_outside_all_boxes"] = noise_mu
        rec["snr_main_over_noise_mean"] = snr_main_nm
        rec["snr_small_over_noise_mean"] = snr_small_nm
        # Only promoted small peaks affect final_conf_best / need_manual_review (unchanged gate).
        best_final = main_conf_final
        if keep_small and np.isfinite(small_conf_final):
            best_final = max(float(best_final), float(small_conf_final))
        rec["final_conf_best"] = float(best_final)
        rec["need_manual_review"] = int(float(best_final) < float(args.final_conf_threshold))
        rec["small_recover_trigger"] = small_recover_trigger
        if np.isfinite(small_rt) and np.isfinite(ref_rt):
            rec["small_rt_gate_pass"] = int(abs(small_rt - ref_rt) <= args.sample_rt_tolerance)
            rec["small_skew_gate_pass"] = int(
                (not np.isfinite(ref_sk)) or (not np.isfinite(main_sk)) or (abs(main_sk - ref_sk) <= args.skew_tolerance)
            )
            rec["small_ai_gate_pass"] = int(np.isfinite(small_ai) and (float(small_ai) >= float(args.sample_keep_small_min_ai)))
        else:
            rec["small_rt_gate_pass"] = 0
            rec["small_skew_gate_pass"] = 0
            rec["small_ai_gate_pass"] = 0
        out_rows.append(rec)

    out = pd.DataFrame(out_rows)
    out_csv_arg = Path(args.output_csv).resolve()
    if bool(getattr(args, "flat_sample_output", False)):
        out_path = out_csv_arg
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        run_name = (sample_csv_path.parent.name or "").strip()
        if not run_name or run_name in (".", ".."):
            run_name = "sample_output"
        run_name = _safe_folder_name(run_name)
        run_dir = out_csv_arg.parent / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / out_csv_arg.name
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Sample-mode output: {out_path}")
    if "small_recover_trigger" in out.columns:
        trig = out["small_recover_trigger"].astype(str)
        print(
            "[INFO] small_recover_trigger counts:\n"
            + trig.value_counts().to_string()
        )
    if "keep_small_by_standard" in out.columns:
        print(f"[INFO] keep_small_by_standard=1 count: {int(out['keep_small_by_standard'].fillna(0).astype(int).sum())}")
        print(f"[INFO] main_interval_width mean: {pd.to_numeric(out.get('main_interval_width'), errors='coerce').mean():.4f} min")
    if bool(args.plot):
        sample_csv = Path(args.sample_refined_csv).resolve()
        xic_path = _locate_xic_matrix_from_sample_csv(sample_csv)
        if xic_path is None:
            print(f"[WARN] plot requested but xic_matrix.npy not found near: {sample_csv}")
        else:
            if args.plot_dir:
                plot_dir = Path(args.plot_dir).expanduser()
                if not plot_dir.is_absolute():
                    plot_dir = (out_path.parent / plot_dir).resolve()
                else:
                    plot_dir = plot_dir.resolve()
            else:
                plot_dir = (out_path.parent / "sample_final_plots").resolve()
            plot_dir.mkdir(parents=True, exist_ok=True)
            _plot_sample_final(out, xic_path, plot_dir, sigma=float(args.plot_sigma))
            print(f"[OK] Sample plots saved: {plot_dir}")


def run_predict_from_standard_rt(args):
    """
    从样品目录的 xic_matrix 按各通道 XIC（可选平滑后）最高峰 RT 裁 ROI，运行 MRMPFormer 输出 prediction.csv。
    """
    sample_dir = Path(args.sample_xic_dir).resolve()
    feat_path = sample_dir / "feature.csv"
    xic_path = sample_dir / "xic_matrix.npy"
    if not feat_path.exists() or not xic_path.exists():
        raise FileNotFoundError(f"Need feature.csv + xic_matrix.npy in {sample_dir}")

    if getattr(args, "standard_refs_csv", None):
        print("[INFO] --standard_refs_csv 已弃用；ROI 以各通道强度最高点 RT 为中心。")

    feat = pd.read_csv(feat_path)
    xic = np.load(str(xic_path))
    rt = xic[0, :].astype(np.float64)
    if np.nanmax(rt) > 200:
        rt = rt / 60.0
    inten = xic[1:, :].astype(np.float64)

    n = min(len(feat), inten.shape[0])
    if n <= 0:
        raise ValueError("No valid feature/xic rows for ROI generation.")
    if len(feat) != inten.shape[0]:
        print(f"[WARN] feature/xic row mismatch: feature={len(feat)}, xic={inten.shape[0]}, using first {n}.")

    compounds = []
    for i in range(n):
        r = feat.iloc[i]
        mz = _safe_float(r.get("mz"), np.nan)
        q3 = _safe_float(r.get("q3"), np.nan)
        compounds.append(
            {
                "mz_name": float(mz) if np.isfinite(mz) else np.nan,
                "q3": float(q3) if np.isfinite(q3) else np.nan,
                "rt": rt,
                "intensity": inten[i, :],
            }
        )

    out_base = Path(args.output_dir).resolve()
    if bool(getattr(args, "flat_output", False)):
        out_dir = out_base
    else:
        out_dir = out_base / _safe_folder_name(sample_dir.name or "sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] sample rows={n}, ROI 窗口以各通道 XIC 最高峰 RT 为中心（roi_smooth_sigma 平滑后）")
    print(f"[INFO] Generating ROIs to: {out_dir}")

    from ..preprocessing.xic_extraction import extract_xic_from_arrays
    extract_xic_from_arrays(str(out_dir), compounds, smooth_sigma=float(args.roi_smooth_sigma))

    from ..inference.predictor import run_single as run_newtest_single
    qargs = argparse.Namespace(
        feature=None,
        model=str(Path(args.model).resolve()),
        threshold=float(args.threshold),
        verbose=bool(args.verbose),
        predict_smooth_sigma=float(args.predict_smooth_sigma),
        baseline_correction=False,
        integration_method="linear",
        baseline_json=None,
        plot=bool(args.plot),
        keep_smoothed_inputs=False,
    )
    pred_out = out_dir / "prediction.csv"
    plot_dir = out_dir / "predicted_plots"
    ok = run_newtest_single(qargs, str(out_dir), str(pred_out), str(plot_dir))
    if not ok:
        raise RuntimeError("MRMPFormer prediction failed while generating prediction.csv")
    print(f"[OK] prediction.csv generated: {pred_out}")


def build_parser():
    p = argparse.ArgumentParser(description="Unified peak workflow (post-newtest / standard / sample).")
    sp = p.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("post_newtest", help="Refine prediction.csv into main+small peaks with valley fallback.")
    p1.add_argument(
        "--results_dir",
        required=True,
        help="Directory with prediction.csv (newtest output). xic_matrix.npy may live here or under --xic_dir.",
    )
    p1.add_argument(
        "--xic_dir",
        default=None,
        help=(
            "ROI 目录（testXIC 输出）：含 xic_matrix.npy、roi_windows.csv。"
            "当 results_dir 下没有 xic_matrix.npy 时必须指定，通常为与样品同名的子目录。"
        ),
    )
    p1.add_argument("--output_name", default="prediction_refined.csv")
    p1.add_argument("--small_peak_rt_tol", type=float, default=0.3)
    p1.add_argument(
        "--enable_small_peak_rt_gate",
        action="store_true",
        help="启用小峰相对主峰的 RT 距离门控；默认关闭（第三峰/重复寻峰不按 RT 限制）",
    )
    p1.add_argument(
        "--edge_noise_stop_mode",
        type=str,
        default="roi_bottom_decile_mean",
        choices=["roi_bottom_decile_mean", "stable_tail_mean", "low_percentile"],
        help="边框阈值：roi_bottom_decile_mean=全ROI最低10%强度均值；stable_tail_mean=外侧尾区；low_percentile=单侧分位",
    )
    p1.add_argument(
        "--edge_flat_triplet_step_frac",
        type=float,
        default=0.010,
        help="三连点相邻强度差相对峰最大高度的比例上限，低于则早停到靠峰侧第一点；0 关闭",
    )
    p1.add_argument(
        "--refine_width_max_expand_vs_pred",
        type=float,
        default=1.08,
        help="上限：修正框宽度不得超过 原始预测宽×该倍数（可与 ROI 比例一起收紧；不会强行扩框）",
    )
    p1.add_argument(
        "--refine_width_max_frac_of_roi",
        type=float,
        default=0.45,
        help="上限：修正框宽度不得超过 ROI 时间窗×该比例（噪声截停得到的较窄主峰框会保留）",
    )
    p1.add_argument("--min_confidence", type=float, default=0.99, help="主峰修正门控：最小置信度")
    p1.add_argument("--min_snr", type=float, default=3.0, help="主峰修正门控：最小 SNR")
    p1.add_argument(
        "--min_secondary_ratio",
        type=float,
        default=0.04,
        help="次峰相对主峰动态的最小比例；略低于 0.05 以便保留贴近 5% 门槛的弱次峰",
    )
    p1.add_argument(
        "--noise_barrier_ratio",
        type=float,
        default=0.45,
        help="噪声项系数（×ROI 后段平均噪声）；略降可减轻弱次峰被挡",
    )
    p1.add_argument(
        "--secondary_roi_global_gate_relax_frac",
        type=float,
        default=0.055,
        help=(
            "ROI 次峰第一档：允许 sec_h ≥ sec_min_h×(1−该值)，接住≈5%%动态附近的边际次峰；"
            "与缩小主峰框无关，默认略大于 0；设为 0 恢复严格 sec_min_h"
        ),
    )
    p1.add_argument("--small_noise_window_half", type=float, default=0.30, help="小峰局部噪声窗口半宽(min)，噪声取该窗口后25%")
    p1.add_argument("--main_boundary_noise_percentile", type=float, default=20.0, help="主峰边界切割噪声分位(越小区间越宽)")
    p1.add_argument(
        "--edge_max_span_min",
        type=float,
        default=0.24,
        help="峰顶单侧估计基线/截停时沿该侧的最大RT跨度(min)，略收紧默认",
    )
    p1.add_argument(
        "--edge_noise_percentile",
        type=float,
        default=25.0,
        help="单侧低噪声分位数，用于 adjust_first_round_interval 与边界反向截停",
    )
    p1.add_argument(
        "--boundary_posterior_lookahead",
        type=int,
        default=5,
        help="阈值截停后验：沿外推方向取该点数求均值，须仍≤阈值×mean_scale；0 关闭后验（仅首点阈值）",
    )
    p1.add_argument(
        "--boundary_posterior_mean_scale",
        type=float,
        default=1.25,
        help="后验窗口均值相对单侧噪声阈值的上限倍数（略>1 以容忍基线抖动）",
    )
    p1.add_argument(
        "--boundary_peer_thr_scale",
        type=float,
        default=2.0,
        help="与同伴预测框 RT 重叠且 y>阈值×该值时视为撞入邻峰，继续外推",
    )
    p1.add_argument(
        "--boundary_peer_min_overlap_rt",
        type=float,
        default=0.02,
        help="触发同伴撞峰判据所需的后验窗口与同伴框的最小 RT 重叠(min)",
    )
    p1.add_argument("--small_boundary_pad", type=float, default=0.08, help="小峰边界RT门控附加容差(min)，用于约束框边界不过远")
    p1.add_argument("--enable_valley_fallback", action="store_true")
    p1.add_argument(
        "--valley_small_peak_rt_tol",
        type=float,
        default=0.8,
        help="谷分裂分支使用的小峰-主峰峰顶距离容差(min)；建议大于 --small_peak_rt_tol 以捕获分离较大的双峰。",
    )
    p1.add_argument("--enable_main_double_split", action="store_true", help="在主框内检测明显双峰并直接劈裂为主峰+次峰")
    p1.add_argument("--disable_main_double_split", dest="enable_main_double_split", action="store_false", help="关闭主框明显双峰劈裂")
    p1.set_defaults(enable_main_double_split=True)
    p1.add_argument("--main_double_split_min_gap", type=float, default=0.12, help="主框双峰劈裂最小峰顶间隔(min)")
    p1.add_argument("--main_double_split_max_valley_ratio", type=float, default=0.78, help="主框双峰劈裂最大谷底/峰顶比，越小越严格")
    p1.add_argument(
        "--main_double_split_min_valley_drop_ratio",
        type=float,
        default=0.06,
        help=(
            "双峰劈裂：谷在两峰顶点连线之下的垂深 (弦高−谷强) / max(峰高) 须≥此值；0 关闭。防浅谷毛刺"
        ),
    )
    p1.add_argument(
        "--main_double_split_min_peak_above_valley_ratio",
        type=float,
        default=0.08,
        help="双峰劈裂：大、小峰顶各自 (峰高−谷强)/max(峰高) 须≥此值；0 关闭",
    )
    p1.add_argument(
        "--main_double_split_min_peak_sep_ratio_of_span",
        type=float,
        default=0.12,
        help="双峰劈裂：|峰2_RT−峰1_RT| / (主框hi−lo) 须≥此值；0 关闭。峰距占框宽比例不足则不当成双峰",
    )
    p1.add_argument("--max_small_width", type=float, default=0.35, help="小峰区间最大宽度(min)，防止阴影过宽")
    p1.add_argument("--min_small_width", type=float, default=0.03, help="小峰区间最小宽度(min)，防止过窄")
    p1.add_argument("--small_dedup_rt_tol", type=float, default=0.03, help="多个小峰来源去重时的RT容差(min)")
    p1.add_argument("--max_main_peak_drift", type=float, default=0.12, help="主峰修正后允许的峰顶漂移(min)，超过则回退")
    p1.add_argument("--min_main_height_keep_ratio", type=float, default=0.7, help="修正后峰高低于原峰该比例则回退")
    p1.add_argument("--std_skew_window", type=float, default=0.3, help="偏度计算窗口：围绕峰顶 ±window(min)，默认0.3")
    p1.add_argument("--plot", action="store_true", help="输出平滑后区间阴影图")
    p1.add_argument("--plot_sigma", type=float, default=1.0, help="绘图高斯平滑 sigma（默认1.0）")
    p1.add_argument("--plot_dir_name", type=str, default="refined_plots", help="绘图子目录名（位于 results_dir 或 --plot_output_parent 下）")
    p1.add_argument(
        "--plot_output_parent",
        default=None,
        help="若指定：框修正图写入 <此目录>/<plot_dir_name>/，便于多样品汇总； PNG 文件名默认带 results_dir 父目录名前缀",
    )
    p1.add_argument(
        "--plot_file_prefix",
        default=None,
        help="框修正 PNG 文件名前缀；与 --plot_output_parent 联用时默认取 SNR 结果上一级目录名（多为样品名）",
    )
    p1.add_argument(
        "--enable_lr_repredict_on_small_fail",
        action="store_true",
        help="若无小峰判据通过，则取消当前框并在其左右区间分别重检峰（默认开启）",
    )
    p1.add_argument(
        "--disable_lr_repredict_on_small_fail",
        dest="enable_lr_repredict_on_small_fail",
        action="store_false",
        help="关闭“无小峰时左右重检峰并替换主框”逻辑",
    )
    p1.set_defaults(enable_lr_repredict_on_small_fail=True)

    # New optional mode: keep all predicted boxes and output per-peak refined intervals.
    p1.add_argument(
        "--keep_all_pred_boxes",
        action="store_true",
        help="保留每张图的所有预测框并分别进行框修正/二次识别/重叠分割，输出为多行（不再是 main+small 宽表）。",
    )
    p1.add_argument(
        "--all_boxes_output_name",
        type=str,
        default="prediction_refined_all.csv",
        help="[keep_all_pred_boxes] 输出CSV文件名（多行，每行一个峰）。",
    )
    p1.add_argument(
        "--split_overlaps",
        action="store_true",
        help="[keep_all_pred_boxes] 若多个峰区间重叠，则在重叠区最低强度点切分。",
    )
    p1.set_defaults(split_overlaps=True)
    p1.add_argument(
        "--overlap_split_min_gap",
        type=float,
        default=0.01,
        help="重叠切分点两侧保留的最小间隔(min)；用于 keep_all 模式及最终同图多框谷值切分。",
    )
    p1.add_argument(
        "--disable_overlap_valley_split",
        action="store_true",
        help="关闭：写出前对同一张图上的多个框，若 RT 重叠则在重叠区取强度最低点作为新分界（默认开启）",
    )
    p1.add_argument(
        "--all_boxes_dedup_rt_tol",
        type=float,
        default=0.02,
        help="[keep_all_pred_boxes] 多来源峰去重的RT容差(min)，保留峰高更大的那条。",
    )

    p2 = sp.add_parser("standard_mode", help="Select standard peaks across concentrations and repair low-R2 points.")
    p2.add_argument("--standards_root", required=True, help="Root directory containing many refined csv files")
    p2.add_argument("--input_name", default="prediction_refined.csv")
    p2.add_argument("--output_dir", required=True)
    p2.add_argument("--r2_threshold", type=float, default=0.995)
    p2.add_argument("--rt_tolerance_std", type=float, default=0.2)
    p2.add_argument("--small_near_main_tol_std", type=float, default=0.3, help="标品阶段：确定主峰后，小峰需距主峰<=该值(min)")
    p2.add_argument("--mz_rt_top_k", type=int, default=2, help="同一 mz 多通道时，用于 mz 级预期 RT 的最强/最稳通道数量（加权平均）")

    p3 = sp.add_parser("sample_mode", help="Filter small peaks by standard refs and compute composite confidences.")
    p3.add_argument("--sample_refined_csv", required=True)
    p3.add_argument("--standard_refs_csv", required=True)
    p3.add_argument("--standard_best_csv", default=None, help="可选：标品最佳修正结果CSV（如 standard_mz_quant_max_area.csv）")
    p3.add_argument(
        "--output_csv",
        required=True,
        help=(
            "目标 CSV 路径：默认同名文件写在 <该路径父目录>/<sample_refined_csv 所在文件夹名>/ 下；"
            "加 --flat_sample_output 则严格写入本参数所指路径。"
        ),
    )
    p3.add_argument(
        "--flat_sample_output",
        action="store_true",
        help="不创建子文件夹，直接将结果写入 --output_csv 给定路径（恢复旧行为）。",
    )
    p3.add_argument("--sample_rt_tolerance", type=float, default=1.0)
    p3.add_argument("--rt_score_tolerance", type=float, default=0.2)
    p3.add_argument("--skew_tolerance", type=float, default=1.0)
    p3.add_argument("--refine_main_in_sample", dest="refine_main_in_sample", action="store_true", help="样品阶段在参考RT附近重定位主峰并重建区间")
    p3.add_argument("--no_refine_main_in_sample", dest="refine_main_in_sample", action="store_false", help="关闭样品阶段主峰重定位")
    p3.set_defaults(refine_main_in_sample=True)
    p3.add_argument("--auto_skip_refine_on_refined_input", dest="auto_skip_refine_on_refined_input", action="store_true", help="若输入已是 prediction_refined，自动跳过样品阶段主峰再修正")
    p3.add_argument("--no_auto_skip_refine_on_refined_input", dest="auto_skip_refine_on_refined_input", action="store_false", help="关闭自动跳过（谨慎：可能双重修框）")
    p3.set_defaults(auto_skip_refine_on_refined_input=True)
    p3.add_argument("--force_refine_main_in_sample", action="store_true", help="即使输入是 refined 也强制执行样品阶段主峰再修正")
    p3.add_argument("--sample_main_search_tol", type=float, default=0.5, help="样品主峰重定位搜索半宽(min)")
    p3.add_argument("--sample_main_max_width", type=float, default=0.35, help="样品主峰重定位后区间最大宽度(min)")
    p3.add_argument("--sample_main_boundary_noise_percentile", type=float, default=18.0, help="样品主峰切割噪声分位(越小区间越宽)")
    p3.add_argument(
        "--boundary_posterior_lookahead",
        type=int,
        default=5,
        help="[主峰再修正] 同 post-newtest：阈值后验外向点数，0 关闭",
    )
    p3.add_argument(
        "--boundary_posterior_mean_scale",
        type=float,
        default=1.25,
        help="[主峰再修正] 后验均值相对阈值倍数上限",
    )
    p3.add_argument(
        "--boundary_peer_thr_scale",
        type=float,
        default=2.0,
        help="[主峰再修正] 同伴撞峰强度倍数（样品宽表通常无同伴，可忽略）",
    )
    p3.add_argument(
        "--boundary_peer_min_overlap_rt",
        type=float,
        default=0.02,
        help="[主峰再修正] 同伴重叠最小 RT(min)",
    )
    p3.add_argument("--recover_small", action="store_true", help="样品阶段启用二次小峰恢复（ROI次峰+谷分裂）")
    p3.add_argument(
        "--sample_recover_weak_small",
        action="store_true",
        help="在已有小峰但区间过窄或与主峰重叠过多时，尝试重新识别小峰（失败则保留原结果）",
    )
    p3.add_argument("--sample_weak_small_max_width", type=float, default=0.04, help="判定为弱小的最大小峰宽度(min)")
    p3.add_argument(
        "--sample_weak_small_overlap_frac",
        type=float,
        default=0.5,
        help="小峰与主峰重叠长度占小峰宽度比例≥该值时判定为弱小事后重试",
    )
    p3.add_argument("--sample_roi_half_window", type=float, default=0.5, help="样品阶段候选搜索ROI半宽(min)，中心为标品RT")
    p3.add_argument("--sample_small_peak_rt_tol", type=float, default=0.3, help="样品阶段小峰距主峰峰顶容差(min)")
    p3.add_argument("--sample_min_secondary_ratio", type=float, default=0.05, help="样品阶段次峰最小相对高度比例")
    p3.add_argument("--sample_noise_barrier_ratio", type=float, default=1.0, help="样品阶段噪声门槛系数")
    p3.add_argument("--sample_small_noise_window_half", type=float, default=0.30, help="样品阶段小峰局部噪声窗口半宽(min)，噪声取窗口后25%")
    p3.add_argument("--sample_max_small_width", type=float, default=0.30, help="样品阶段小峰最大区间宽度(min)")
    p3.add_argument("--sample_min_small_width", type=float, default=0.03, help="样品阶段小峰最小区间宽度(min)")
    p3.add_argument("--sample_keep_small_min_ai", type=float, default=0.35, help="样品阶段小峰保留的AI下限（与skew判据为或关系）")
    p3.add_argument(
        "--final_conf_threshold",
        type=float,
        default=0.90,
        help="最终置信度阈值：低于该值建议交人工识别（图中 >90% 视为正常）",
    )
    p3.add_argument(
        "--final_rt_max_shift",
        type=float,
        default=0.50,
        help="RT 置信度线性映射的最大偏移(min)：0->1，达到该值->0（图中为0.5）",
    )
    p3.add_argument(
        "--final_rt_power",
        type=float,
        default=2.0,
        help="RT 偏移非线性衰减幂指数：越大下降越快（final_conf 中 RT 为主导项）",
    )
    p3.add_argument(
        "--final_snr_ref",
        type=float,
        default=10.0,
        help="SNR 归一化参考值：snr/snr_ref 作为SNR项（越低惩罚越大）",
    )
    p3.add_argument(
        "--final_snr_weight",
        type=float,
        default=0.30,
        help="SNR 惩罚强度（0~1）：越大越依赖 SNR（最终置信度以 AI*RT 为基底）",
    )
    p3.add_argument(
        "--final_skew_weight",
        type=float,
        default=0.10,
        help="偏度项强度（0~1）：建议较低；小峰还会额外乘偏度方向性连续加权",
    )
    p3.add_argument(
        "--final_skew_high",
        type=float,
        default=1.0,
        help="标品偏度绝对值高于该阈值时，按偏度方向对小峰置信度做方向性加权",
    )
    p3.add_argument(
        "--final_skew_scale",
        type=float,
        default=1.0,
        help="偏度方向性加权的增强尺度：越小越快达到最大加权（使用 tanh((|sk|-skew_high)/scale)）",
    )
    p3.add_argument(
        "--final_skew_boost_max",
        type=float,
        default=1.15,
        help="偏度方向匹配时的小峰最大提升倍数（随 |ref_skew| 增大逐渐接近该值）",
    )
    p3.add_argument(
        "--final_skew_penalty_min",
        type=float,
        default=0.90,
        help="偏度方向不匹配时的小峰最低惩罚倍数（随 |ref_skew| 增大逐渐接近该值）",
    )
    p3.add_argument("--plot", action="store_true", help="输出样品最终结果图到输出目录")
    p3.add_argument("--plot_sigma", type=float, default=1.0, help="样品图平滑 sigma")
    p3.add_argument(
        "--plot_dir",
        default=None,
        help="样品图目录。默认同级目录下 sample_final_plots；相对路径则相对于本次实际输出 CSV 所在目录。",
    )

    p4 = sp.add_parser(
        "predict_from_ref_rt",
        help="从样品 xic_matrix 按各通道最高峰 RT 裁 ROI 并运行 MRMPFormer，输出 prediction.csv（与历史子命令名兼容）。",
    )
    p4.add_argument("--sample_xic_dir", required=True, help="样品目录（需含 feature.csv + xic_matrix.npy）")
    p4.add_argument(
        "--standard_refs_csv",
        default=None,
        help="已弃用；保留仅为兼容旧命令行，不再读取。",
    )
    p4.add_argument("--model", required=True, help="MRMPFormer checkpoint 路径")
    p4.add_argument("--output_dir", required=True, help="输出基目录（默认会创建子目录 <sample_xic_dir 名称>）")
    p4.add_argument("--flat_output", action="store_true", help="不创建样品子目录，直接写到 --output_dir")
    p4.add_argument("--threshold", type=float, default=0.99, help="模型置信度阈值")
    p4.add_argument("--roi_smooth_sigma", type=float, default=0.0, help="ROI 生成时 XIC 高斯平滑 sigma")
    p4.add_argument("--predict_smooth_sigma", type=float, default=0.0, help="模型输入图像高斯平滑 sigma")
    p4.add_argument("--plot", action="store_true", help="生成 predicted_plots 可视化")
    p4.add_argument("--verbose", action="store_true", help="打印模型逐图调试信息")
    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "post_newtest":
        run_post_newtest(args)
    elif args.cmd == "standard_mode":
        run_standard_mode(args)
    elif args.cmd == "sample_mode":
        run_sample_mode(args)
    elif args.cmd == "predict_from_ref_rt":
        run_predict_from_standard_rt(args)


if __name__ == "__main__":
    main()

