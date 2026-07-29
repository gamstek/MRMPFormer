"""
根据 ROI 质量参数自适应选择积分方法，降低相对误差。

阈值基于 error_analysis_report 统计：
- 高误差组 SNR 中位数 121，低误差组 929
- 高误差组 noise_std 中位数 2807，低误差组 700
- 高误差组中 |baseline_slope|>0.05 占 5/12
"""
import numpy as np

# 阈值（可调）
SNR_RAW_THRESHOLD = 150          # SNR < 此值用 raw（低信噪比，噪声主导）
BASELINE_SLOPE_THRESHOLD = 0.05  # |baseline_slope| > 此值避免线性基线（易抬高）
PEAK_WIDTH_RAW_THRESHOLD = 0.12  # peak_width_ratio < 此值用 raw（尖锐峰对基线敏感）
SNR_SHARP_PEAK_THRESHOLD = 400   # 尖锐峰且 SNR < 此值用 raw


def select_integration_method(qparams):
    """
    根据质量参数选择积分方法。
    返回 "linear" | "raw" | "peak_adaptive"
    """
    snr = qparams.get("snr", np.nan)
    baseline_slope = qparams.get("baseline_slope", 0.0)
    peak_width_ratio = qparams.get("peak_width_ratio", np.nan)
    if np.isnan(snr):
        snr = 1e6  # 缺失时视为高 SNR
    if np.isnan(peak_width_ratio):
        peak_width_ratio = 0.2  # 缺失时视为中等峰宽
    abs_slope = abs(baseline_slope)

    # 1. 低 SNR：用 raw，避免线性基线在噪声下不稳定
    if snr < SNR_RAW_THRESHOLD:
        return "raw"

    # 2. 基线斜率大：线性基线易抬高，面积偏小
    if abs_slope > BASELINE_SLOPE_THRESHOLD:
        if peak_width_ratio < PEAK_WIDTH_RAW_THRESHOLD:
            return "raw"  # 尖锐峰 + 大斜率 -> raw
        return "peak_adaptive"  # 宽峰/双峰 + 大斜率 -> 谷-谷

    # 3. 尖锐峰且 SNR 不高：对基线敏感
    if peak_width_ratio < PEAK_WIDTH_RAW_THRESHOLD and snr < SNR_SHARP_PEAK_THRESHOLD:
        return "raw"

    return "linear"
