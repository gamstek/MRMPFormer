# -*- coding: utf-8 -*-
"""
XIC 峰分析工具：SNR 计算、次峰检测。
用于两轮识别流程中的 XIC 条件筛选。
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def get_noise_regions_outside_box(rt_array, intensity_row, rt_min, rt_max, min_points=3, frac=0.2):
    """
    在预测框 [rt_min, rt_max] 外取左右噪声区。
    返回 (left_region, right_region)，均为 1d array。
    """
    rt = np.asarray(rt_array, dtype=np.float64)
    intensity = np.asarray(intensity_row, dtype=np.float64)
    inside = (rt >= rt_min) & (rt <= rt_max)
    n_inside = int(np.sum(inside))
    n_want = max(min_points, int(frac * n_inside))

    left_mask = rt < rt_min
    right_mask = rt > rt_max
    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]

    left_region = intensity[left_idx][-n_want:] if len(left_idx) >= n_want else intensity[left_idx]
    right_region = intensity[right_idx][:n_want] if len(right_idx) >= n_want else intensity[right_idx]
    return left_region, right_region


def compute_snr_outside_box(rt_array, intensity_row, rt_min, rt_max):
    """
    以预测框外区域为噪声参考的峰-峰信噪比。
    SNR = 2 * (peak_max - baseline) / max(noise_pp_left, noise_pp_right)
    """
    mask = (rt_array >= rt_min) & (rt_array <= rt_max)
    int_seg = intensity_row[mask].astype(np.float64)
    int_seg = np.maximum(int_seg, 0.0)
    if int_seg.size < 5 or np.max(int_seg) <= 0:
        return np.nan
    peak_max = float(np.max(int_seg))

    left_region, right_region = get_noise_regions_outside_box(
        rt_array, intensity_row, rt_min, rt_max, min_points=3, frac=0.2
    )
    all_noise = []
    noise_pp_left = np.nan
    noise_pp_right = np.nan
    if left_region.size >= 2:
        left_region = np.maximum(left_region.astype(np.float64), 0.0)
        all_noise.extend(left_region.tolist())
        noise_pp_left = float(np.max(left_region) - np.min(left_region))
    if right_region.size >= 2:
        right_region = np.maximum(right_region.astype(np.float64), 0.0)
        all_noise.extend(right_region.tolist())
        noise_pp_right = float(np.max(right_region) - np.min(right_region))

    if np.isnan(noise_pp_left) and np.isnan(noise_pp_right):
        return _compute_snr_peak_to_peak(int_seg)
    noise_pp = np.nanmax([x for x in (noise_pp_left, noise_pp_right) if not np.isnan(x)])
    if noise_pp <= 0:
        noise_pp = 1e-10
    baseline = float(np.median(all_noise)) if all_noise else np.nan
    if not all_noise or np.isnan(baseline):
        return _compute_snr_peak_to_peak(int_seg)
    signal = peak_max - baseline
    if signal <= 0:
        return np.nan
    return 2.0 * signal / noise_pp


def _compute_snr_peak_to_peak(intensity):
    """段内估计 SNR（备用）。"""
    arr = np.asarray(intensity, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    n = arr.size
    if n < 5 or np.max(arr) <= 0:
        return np.nan
    peak_max = float(np.max(arr))
    noise_region_size = max(int(0.15 * n), 3)
    if 2 * noise_region_size >= n:
        return np.nan
    left_region = arr[0:noise_region_size]
    right_region = arr[n - noise_region_size : n]
    noise_pp_left = float(np.max(left_region) - np.min(left_region)) if left_region.size >= 2 else np.nan
    noise_pp_right = float(np.max(right_region) - np.min(right_region)) if right_region.size >= 2 else np.nan
    if np.isnan(noise_pp_left) and np.isnan(noise_pp_right):
        return np.nan
    noise_pp = np.nanmax([x for x in (noise_pp_left, noise_pp_right) if not np.isnan(x)])
    if noise_pp <= 0:
        noise_pp = 1e-10
    baseline = float(np.median(np.concatenate([left_region, right_region])))
    signal = peak_max - baseline
    if signal <= 0:
        return np.nan
    return 2.0 * signal / noise_pp


def get_last_25pct_avg_noise(rt_array, intensity_row, rt_lo, rt_hi, frac=0.25):
    """图像后 frac（默认25%）RT 区间的平均强度，作为噪声水平。"""
    rt = np.asarray(rt_array, dtype=np.float64)
    intensity = np.asarray(intensity_row, dtype=np.float64)
    rt_cut = rt_hi - frac * (rt_hi - rt_lo)
    mask = (rt >= rt_cut) & (rt <= rt_hi)
    vals = intensity[mask]
    return float(np.mean(vals)) if vals.size >= 1 else 0.0


def roi_full_low_decile_mean_intensity(intensity_row, bottom_frac: float = 0.10) -> float:
    """
    全 ROI（整条 XIC 采样）强度排序后，取最低 bottom_frac 比例的点对其取平均，
    作为全局噪声截停参考（对应「全图低强度尾区噪声均值」）。
    """
    y = np.maximum(np.asarray(intensity_row, dtype=np.float64), 0.0)
    if y.size < 3:
        return float(np.mean(y)) if y.size else 0.0
    ys = np.sort(y)
    k = max(1, int(np.ceil(float(bottom_frac) * float(ys.size))))
    return float(np.mean(ys[:k]))


def one_sided_edge_stop_threshold_stable_tail_mean(
    rt_array,
    intensity_row,
    peak_rt: float,
    toward_left: bool,
    max_span_min: float = 0.6,
    tail_frac: float = 0.10,
    stable_fluctuation_quantile: float = 50.0,
    min_points: int = 4,
    fallback_noise_percentile: float = 55.0,
) -> float:
    """
    边框外推截停阈值：在峰顶沿该侧 max_span 内，取靠外侧（沿 RT）tail_frac 比例的采样点，
    在其中筛选「波动不大」的点（一阶差分绝对值不超过该尾部分布的分位数），
    对其强度取平均作为阈值。点数不足时退回单侧低分位估计。
    """
    rt = np.asarray(rt_array, dtype=np.float64)
    y = np.maximum(np.asarray(intensity_row, dtype=np.float64), 0.0)
    if rt.size != y.size or rt.size < 5:
        return float(np.percentile(y, fallback_noise_percentile))
    pk = float(peak_rt)
    span = float(max_span_min)
    if toward_left:
        m = (rt >= pk - span) & (rt <= pk)
    else:
        m = (rt >= pk) & (rt <= pk + span)
    if int(np.sum(m)) < min_points:
        m = rt <= pk if toward_left else rt >= pk
    rr = rt[m]
    yy = y[m]
    if rr.size < min_points:
        return float(np.percentile(y, fallback_noise_percentile))
    order = np.argsort(rr)
    rr_s = rr[order]
    yy_s = yy[order]
    n = int(rr_s.size)
    k_tail = max(min_points, int(np.ceil(float(tail_frac) * float(n))))
    k_tail = min(k_tail, n)
    if toward_left:
        tail_yy = yy_s[:k_tail]
    else:
        tail_yy = yy_s[-k_tail:]
    if tail_yy.size < 2:
        return float(np.mean(tail_yy)) if tail_yy.size else float(np.percentile(y, fallback_noise_percentile))
    ty = tail_yy.astype(np.float64)
    fluc = np.zeros(ty.size, dtype=np.float64)
    for j in range(ty.size):
        ds = []
        if j > 0:
            ds.append(abs(ty[j] - ty[j - 1]))
        if j + 1 < ty.size:
            ds.append(abs(ty[j + 1] - ty[j]))
        fluc[j] = max(ds) if ds else 0.0
    thr_fl = float(np.percentile(fluc, stable_fluctuation_quantile))
    stable_mask = fluc <= thr_fl
    if int(np.count_nonzero(stable_mask)) < max(2, min_points // 2):
        thr_fl = float(np.percentile(fluc, 75.0))
        stable_mask = fluc <= thr_fl
    sel = tail_yy[stable_mask]
    if sel.size < 1:
        sel = tail_yy
    return float(np.mean(sel))


def one_sided_low_noise_baseline(
    rt_array,
    intensity_row,
    peak_rt: float,
    toward_left: bool,
    max_span_min: float = 0.6,
    noise_percentile: float = 25.0,
    min_points: int = 3,
) -> float:
    """
    从峰顶沿移动方向一侧（左：rt<=peak；右：rt>=peak）在有限 span 内，
    用强度分位数估计局部低噪声基线，用于边界截停，避免 ROI 尾部全局噪声导致停不下来。
    """
    rt = np.asarray(rt_array, dtype=np.float64)
    y = np.maximum(np.asarray(intensity_row, dtype=np.float64), 0.0)
    if rt.size != y.size or rt.size < 3:
        return float(np.percentile(y, noise_percentile))
    pk = float(peak_rt)
    span = float(max_span_min)
    if toward_left:
        m = (rt >= pk - span) & (rt <= pk)
    else:
        m = (rt >= pk) & (rt <= pk + span)
    n = int(np.sum(m))
    if n < min_points:
        if toward_left:
            m = rt <= pk
        else:
            m = rt >= pk
        n = int(np.sum(m))
    if n < min_points:
        return float(np.percentile(y, noise_percentile))
    return float(np.percentile(y[m], noise_percentile))


def has_secondary_peak_in_roi(rt_array, intensity_row, rt_min, rt_max, rt_lo, rt_hi,
                             min_secondary_ratio=0.05, smooth_sigma=1.0, noise_barrier_ratio=0.0):
    """
    在 ROI 窗口 [rt_lo, rt_hi] 内检查是否存在次峰。
    次峰高度需 >= baseline + min_secondary_ratio * dynamic + noise_barrier_ratio * noise_avg。
    noise_barrier_ratio * noise_avg 为噪声阻碍：后25%平均噪声越大，所需峰高越高。
    返回 (has_secondary: bool, main_height: float, secondary_height: float or None)
    """
    rt = np.asarray(rt_array, dtype=np.float64)
    intensity = np.asarray(intensity_row, dtype=np.float64)
    mask_roi = (rt >= rt_lo) & (rt <= rt_hi)
    if np.sum(mask_roi) < 10:
        return False, 0.0, None

    rt_roi = rt[mask_roi]
    int_roi = intensity[mask_roi].copy()
    if smooth_sigma > 0 and int_roi.size >= 10:
        int_roi = gaussian_filter1d(np.maximum(int_roi, 0.0), sigma=min(smooth_sigma, int_roi.size / 25.0), mode="nearest")

    mask_main = (rt_roi >= rt_min) & (rt_roi <= rt_max)
    main_height = float(np.max(int_roi[mask_main])) if np.any(mask_main) else float(np.max(int_roi))
    if main_height <= 0:
        return False, 0.0, None

    baseline = float(np.percentile(int_roi, 25))
    dynamic = main_height - baseline
    if dynamic <= 0:
        return False, main_height, None

    noise_avg = get_last_25pct_avg_noise(rt_array, intensity_row, rt_lo, rt_hi, frac=0.25)
    noise_barrier = noise_barrier_ratio * max(noise_avg, 1e-9)
    min_height = baseline + min_secondary_ratio * dynamic + noise_barrier
    peaks = []
    for i in range(1, int_roi.size - 1):
        if int_roi[i] >= int_roi[i - 1] and int_roi[i] >= int_roi[i + 1] and int_roi[i] >= min_height:
            peaks.append(i)

    if len(peaks) < 2:
        return False, main_height, None

    peaks_sorted = sorted(peaks, key=lambda i: int_roi[i], reverse=True)
    main_idx = peaks_sorted[0]
    for p in peaks_sorted[1:]:
        if abs(p - main_idx) >= 2:
            secondary_height = float(int_roi[p])
            if secondary_height >= min_height:
                return True, main_height, secondary_height
    return False, main_height, None
