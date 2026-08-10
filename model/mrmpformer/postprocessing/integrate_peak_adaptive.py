"""
峰自适应积分：针对双峰、宽峰等场景，解决线性端点基线过高导致积分偏小的问题。

问题：548_mz413.1000 等物质预测框正确，但线性基线 y_left--y_right 在双峰/宽峰时
     会穿过峰肩，baseline 过高 → corrected 被掏空 → 面积远小于人工。

方案：谷-谷基线（valley-to-valley）+ 保守回退
  - 检测局部谷点，用谷点参与基线构造，使基线在双峰间下凹，避免削峰
  - 若无谷点或谷点不可靠，用 min(y_left, y_right) 作平坦基线（比线性更保守）
"""
import numpy as np
from scipy.signal import find_peaks

AREA_TIME_UNIT_SCALE = 60.0


def _find_valleys(y, prominence_ratio=0.05):
    """
    在 y 中找局部谷点（局部最小值）。
    prominence_ratio: 相对峰高的最小 prominence，用于过滤噪声谷。
    返回谷点索引数组（可能为空）。
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < 5:
        return np.array([], dtype=int)
    peak_to_valley = np.max(y) - np.min(y)
    if peak_to_valley <= 0:
        return np.array([], dtype=int)
    prominence = max(peak_to_valley * prominence_ratio, 1e-10)
    # 谷 = -y 的峰
    valley_idx, _ = find_peaks(-y, prominence=prominence, distance=2)
    return valley_idx


def integrate_peak_adaptive(x, y, scale=AREA_TIME_UNIT_SCALE):
    """
    峰自适应积分：谷-谷基线 + 保守回退。

    1. 找局部谷点；若有谷点，用端点 + 谷点 做分段线性基线（谷点使基线在双峰间下凹）
    2. 若无谷点或线性基线面积过小，用 min(y[0], y[-1]) 作平坦基线
    3. corrected = max(0, y - baseline)，trapz(corrected, x) * scale

    适用于：双峰、肩峰、宽峰等线性基线易削峰的场景。
    """
    if x.size < 2 or y.size < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 0.0)

    # 线性端点基线（原逻辑）
    y_left, y_right = float(y[0]), float(y[-1])
    baseline_linear = y_left + (y_right - y_left) * (x - x[0]) / (x[-1] - x[0])
    corrected_linear = np.maximum(0.0, y - baseline_linear)
    area_linear = float(np.trapz(corrected_linear, x) * scale)

    # 谷点
    valleys = _find_valleys(y)
    baseline = baseline_linear.copy()

    if len(valleys) > 0:
        # 构造分段线性基线：端点 + 谷点，按 x 排序，去重
        pts_x = np.concatenate([[x[0]], x[valleys], [x[-1]]])
        pts_y = np.concatenate([[y[0]], y[valleys], [y[-1]]])
        order = np.argsort(pts_x)
        pts_x = pts_x[order]
        pts_y = pts_y[order]
        keep = np.ones(len(pts_x), dtype=bool)
        for i in range(1, len(pts_x)):
            if pts_x[i] <= pts_x[i - 1] + 1e-12:
                keep[i] = False
        pts_x = pts_x[keep]
        pts_y = pts_y[keep]
        if len(pts_x) >= 2:
            baseline = np.interp(x, pts_x, pts_y)

    corrected = np.maximum(0.0, y - baseline)
    area_valley = float(np.trapz(corrected, x) * scale)

    # 平坦基线：min(y_left, y_right)，更保守
    baseline_flat = np.minimum(y_left, y_right)
    corrected_flat = np.maximum(0.0, y - baseline_flat)
    area_flat = float(np.trapz(corrected_flat, x) * scale)

    # 选择策略：优先谷基线；若谷基线仍过小（< 平坦基线的 20%）且平坦基线 > 0，用平坦
    # 原因：谷点可能不准（噪声），平坦基线是安全的下界
    if area_valley <= 0 and area_flat > 0:
        return area_flat
    if area_valley > 0 and area_flat > 0 and area_valley < 0.2 * area_flat:
        return area_flat
    return area_valley if area_valley > 0 else area_linear
