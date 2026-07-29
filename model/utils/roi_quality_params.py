"""
ROI 积分段质量参数：SNR、基线斜率、峰宽等，用于预测积分误差分析与自适应策略。

降低预测积分相对误差的推荐参数及用途：
- snr: 信噪比，低 SNR 时宜用噪声门限或 raw 积分
- noise_std: 噪声标准差，用于门限设定
- baseline_slope: 端点基线斜率（相对峰高），大斜率时线性基线易抬高
- peak_width_ratio: 半高宽/总宽，尖锐峰（小值）对基线敏感
- dynamic_range: (peak-baseline)/peak，低对比度时易受噪声影响
- point_counts: 连续非零点数（已有），反映峰宽
- 可选扩展: valley_ratio（双峰谷深）、peak_asymmetry（峰对称性）
"""
import numpy as np


def compute_roi_quality_params(x, y):
    """
    从积分段 (x=RT, y=intensity) 计算质量参数。
    返回 dict: snr, noise_std, baseline_level, signal, baseline_slope, peak_width_ratio, dynamic_range
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 0.0)
    n = y.size
    if n < 3:
        return {"snr": np.nan, "noise_std": np.nan, "baseline_level": np.nan, "signal": np.nan,
                "baseline_slope": np.nan, "peak_width_ratio": np.nan, "dynamic_range": np.nan}

    # 基线水平（25% 分位）
    baseline_level = float(np.percentile(y, 25))
    peak = float(np.max(y))
    signal = peak - baseline_level
    # 噪声：低强度区标准差
    low_mask = y <= np.percentile(y, 50)
    noise_region = y[low_mask]
    noise_std = float(np.std(noise_region)) if np.sum(low_mask) > 2 else float(np.std(y))
    if noise_std <= 0:
        noise_std = 1e-10
    snr = signal / noise_std

    # 基线斜率（端点连线）：y_right - y_left，归一化到 [0,1] 区间
    y_left, y_right = float(y[0]), float(y[-1])
    baseline_slope = (y_right - y_left) / (peak + 1e-10)  # 相对峰高的斜率

    # 峰宽比：半高宽点数 / 总点数（粗略估计）
    half_max = (peak + baseline_level) / 2
    above_half = y >= half_max
    if np.any(above_half):
        peak_width = np.sum(above_half)
        peak_width_ratio = peak_width / n
    else:
        peak_width_ratio = np.nan

    # 动态范围：(peak - baseline) / peak
    dynamic_range = (peak - baseline_level) / (peak + 1e-10) if peak > 0 else np.nan

    return {
        "snr": float(snr),
        "noise_std": float(noise_std),
        "baseline_level": float(baseline_level),
        "signal": float(signal),
        "baseline_slope": float(baseline_slope),
        "peak_width_ratio": float(peak_width_ratio),
        "dynamic_range": float(dynamic_range),
    }


def compute_roi_quality_params_minimal(y):
    """仅计算 SNR，用于轻量调用。返回 (snr, noise_std, baseline_level)。"""
    params = compute_roi_quality_params(np.arange(len(y)), y)
    return params["snr"], params["noise_std"], params["baseline_level"]
