"""
预测框像素坐标 → XIC 上 RT 窗口（分钟）的映射。

与 testXIC.py 生成 ROI jpeg 时的窗口一致：
  - 理论窗口 [true_rt - 1, true_rt + 1]（±1 min）
  - 与数据边界裁剪：不超出 xic 的 RT 范围
  - 图像宽 400px 线性对应 [rt_lo, rt_hi]，与 testXIC 中 ax.set_xlim(rt_lo, rt_hi) 一致

旧逻辑 true_rt + (x - 200) * 0.005 假定中心 200 = apex，在 matplotlib 未固定 xlim 时
与图中像素列不对应，会导致积分区间偏移。
"""
import numpy as np

ROI_IMAGE_WIDTH_PX = 400.0
ROI_IMAGE_HEIGHT_PX = 300.0  # testXIC figsize (4,3) dpi 100
WINDOW_HALF_MIN = 1.0  # 与 testXIC window_half_min 一致


def rt_to_pixel_x(rt, rt_lo, rt_hi):
    """RT（分钟）→ 图像像素 x，与 testXIC 一致。"""
    if rt_hi <= rt_lo:
        return ROI_IMAGE_WIDTH_PX / 2.0
    t = (float(rt) - rt_lo) / (rt_hi - rt_lo)
    return max(0.0, min(ROI_IMAGE_WIDTH_PX, t * ROI_IMAGE_WIDTH_PX))


def intensity_to_pixel_y(intensity, y_min, y_max):
    """强度 → 图像像素 y（0=顶），与 testXIC 绘图一致。"""
    if y_max <= y_min:
        return ROI_IMAGE_HEIGHT_PX / 2.0
    t = (float(intensity) - y_min) / (y_max - y_min)
    t = max(0.0, min(1.0, t))
    return ROI_IMAGE_HEIGHT_PX - t * ROI_IMAGE_HEIGHT_PX  # 图像 y 向下


def rt_window_bounds_minutes(true_rt, rt_axis_minutes):
    """
    与 testXIC 裁剪窗口一致：apex±1 min，再夹到 rt_axis 的 [min, max]。
    rt_axis_minutes: xic 第一条数组 rt（与 xic_matrix 第 0 行一致，单位分钟）。
    返回 (rt_lo, rt_hi) 供像素线性映射。
    """
    true_rt = float(true_rt)
    rt_arr = np.asarray(rt_axis_minutes, dtype=np.float64)
    rt_min_data = float(np.nanmin(rt_arr))
    rt_max_data = float(np.nanmax(rt_arr))
    rt_lo = max(true_rt - WINDOW_HALF_MIN, rt_min_data)
    rt_hi = min(true_rt + WINDOW_HALF_MIN, rt_max_data)
    if rt_hi <= rt_lo:
        rt_lo, rt_hi = rt_min_data, rt_max_data
    return rt_lo, rt_hi


def box_x_to_rt_minutes(x_pixel, rt_lo, rt_hi):
    """单边界：像素 x ∈ [0, 400] 线性对应 [rt_lo, rt_hi]（分钟）。"""
    x = float(x_pixel)
    x = max(0.0, min(x, ROI_IMAGE_WIDTH_PX))
    return rt_lo + (x / ROI_IMAGE_WIDTH_PX) * (rt_hi - rt_lo)


def box_to_rt_range(x1, y1, x2, y2, true_rt, rt_axis_minutes, rt_window=None):
    """
    由预测框 (x1,y1,x2,y2) 得到积分用 RT 区间 [left, right]（分钟）。
    仅 x1、x2 参与 RT 映射；y 为强度轴，不参与。

    rt_window: 可选 (rt_lo, rt_hi)。若提供（如从 roi_windows.csv 读取），则用此窗口做像素→RT 映射，
               与 testXIC 绘图时 set_xlim(rt_lo, rt_hi) 一致，避免积分窗口相对人工标注偏移。
    """
    if rt_window is not None:
        rt_lo, rt_hi = float(rt_window[0]), float(rt_window[1])
    else:
        rt_lo, rt_hi = rt_window_bounds_minutes(true_rt, rt_axis_minutes)
    left = box_x_to_rt_minutes(x1, rt_lo, rt_hi)
    right = box_x_to_rt_minutes(x2, rt_lo, rt_hi)
    if right < left:
        left, right = right, left
    return left, right, rt_lo, rt_hi
