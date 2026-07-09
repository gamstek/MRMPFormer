import os
import numpy as np
from scipy.integrate import trapz
from utils.io_utils import time_master
from utils.roi_rt_mapping import box_to_rt_range

# 面积缩放：使积分结果与常见 LC-MS 软件/人工标注单位一致。
# 若 RT 轴为分钟，trapz 得到 intensity·min；人工常为 intensity·sec 等，约 60 倍关系。
# 预测值约为人工 1.7% 时，将此设为 60.0 可对齐量纲。
AREA_TIME_UNIT_SCALE = 60.0

# 线性基线端点：取窗口外/附近多点，用中位数或分位数，减少噪声导致基线偏低
BASELINE_AVG_MARGIN_MIN = 0.05  # 端点附近时间半宽（分钟），略放宽以获取更多点
BASELINE_PERCENTILE = 60  # 使用 60% 分位数（高于中位数），避免负向噪声拉低基线

# minval_noise_right 模式：右侧噪声区
NOISE_RIGHT_MARGIN_MIN = 0.05   # 固定窗口半宽（分钟）
NOISE_RIGHT_MARGIN_MAX = 0.2   # 扩展模式最大向右搜索范围（分钟）
RIGHT_BASELINE_HIGH_RATIO = 1.5   # 某点 > 当前平均*该系数时停止累加
RIGHT_BASELINE_MIN_POINTS = 3    # 至少累加点数再判断“远高于平均”


def get_baseline_endpoint_heights(rt_full, intensity_full, left, right, margin_min=BASELINE_AVG_MARGIN_MIN,
                                  percentile=BASELINE_PERCENTILE):
    """
    用积分窗口左/右端附近多点的分位数作为基线端点高度。
    - 优先取窗口外：左端 [left-margin, left]（峰前），右端 [right, right+margin]（峰后），避免峰脚干扰。
    - 若无窗口外点则用窗口内边缘。
    - 使用分位数（默认 60%）替代均值，抗噪声且略偏高，避免基线过低。
    返回 (y_left, y_right)。
    """
    rt_full = np.asarray(rt_full, dtype=np.float64)
    intensity_full = np.asarray(intensity_full, dtype=np.float64)
    # 左端：优先峰前 [left-margin, left]
    mask_left = (rt_full >= left - margin_min) & (rt_full <= left)
    if not np.any(mask_left):
        mask_left = (rt_full >= left) & (rt_full <= left + margin_min)
    # 右端：优先峰后 [right, right+margin]
    mask_right = (rt_full >= right) & (rt_full <= right + margin_min)
    if not np.any(mask_right):
        mask_right = (rt_full >= right - margin_min) & (rt_full <= right)

    def _robust_height(arr):
        if arr.size == 0:
            return None
        return float(np.percentile(arr, percentile))

    y_left = _robust_height(intensity_full[mask_left]) if np.any(mask_left) else None
    y_right = _robust_height(intensity_full[mask_right]) if np.any(mask_right) else None
    return y_left, y_right


def _right_baseline_expand_until_high(rt_full, intensity_full, right,
                                      max_margin_min=NOISE_RIGHT_MARGIN_MAX,
                                      high_ratio=RIGHT_BASELINE_HIGH_RATIO,
                                      min_points=RIGHT_BASELINE_MIN_POINTS):
    """
    从右侧窗口起点向右逐点累加求平均，直到某一点远高于当前平均值则停止，
    用当前累加平均作为右侧基线高度。避免峰尾或下一峰被算进噪声，效果更稳定。
    返回 float 或 None。
    """
    rt_full = np.asarray(rt_full, dtype=np.float64)
    intensity_full = np.asarray(intensity_full, dtype=np.float64)
    mask = (rt_full >= right) & (rt_full <= right + max_margin_min)
    if not np.any(mask):
        return None
    rt_seg = rt_full[mask]
    y_seg = intensity_full[mask]
    order = np.argsort(rt_seg)
    rt_seg = rt_seg[order]
    y_seg = y_seg[order]
    run_sum = 0.0
    run_count = 0
    for k in range(rt_seg.size):
        run_sum += y_seg[k]
        run_count += 1
        run_avg = run_sum / run_count
        if run_count >= min_points and y_seg[k] > run_avg * high_ratio:
            return float((run_sum - y_seg[k]) / (run_count - 1))
    if run_count > 0:
        return float(run_sum / run_count)
    return None


def _get_minval_noise_right_baseline_params(rt_full, intensity_full, left, right, noise_right_margin=NOISE_RIGHT_MARGIN_MIN,
                                            use_right_expand_until_high=True):
    """
    谷点-噪声基线参数。返回 (rt_left_lowest, y_left_lowest, y_right_avg)。
    use_right_expand_until_high=True 时右侧用“向右逐点累加平均直至某点远高于平均”，更稳定。
    """
    rt_full = np.asarray(rt_full, dtype=np.float64)
    intensity_full = np.asarray(intensity_full, dtype=np.float64)
    mask_win = (rt_full >= left) & (rt_full <= right)
    x = rt_full[mask_win]
    y = intensity_full[mask_win]
    if x.size < 2 or y.size < 2:
        return None
    rt_peak = float(x[np.argmax(y)])
    mask_left = (rt_full >= left) & (rt_full <= rt_peak)
    if not np.any(mask_left):
        rt_left_lowest, y_left_lowest = left, float(y[0])
    else:
        idx_min = np.argmin(intensity_full[mask_left])
        rt_left_lowest = float(rt_full[mask_left][idx_min])
        y_left_lowest = float(intensity_full[mask_left][idx_min])
    if use_right_expand_until_high:
        y_right_avg = _right_baseline_expand_until_high(rt_full, intensity_full, right)
    else:
        mask_right = (rt_full >= right) & (rt_full <= right + noise_right_margin)
        if not np.any(mask_right):
            y_right_avg = float(np.percentile(y, 60))
        else:
            y_right_avg = float(np.mean(intensity_full[mask_right]))
    if y_right_avg is None:
        y_right_avg = float(np.percentile(y, 60))
    return rt_left_lowest, y_left_lowest, y_right_avg


def get_baseline_minval_noise_right(rt_full, intensity_full, left, right, x_out, noise_right_margin=NOISE_RIGHT_MARGIN_MIN):
    """返回 minval_noise_right 基线的 y 值数组（用于绘图）。"""
    params = _get_minval_noise_right_baseline_params(rt_full, intensity_full, left, right, noise_right_margin)
    if params is None:
        return None
    rt_left_lowest, y_left_lowest, y_right_avg = params
    denom = right - rt_left_lowest if right != rt_left_lowest else 1e-10
    return y_left_lowest + (y_right_avg - y_left_lowest) * (np.asarray(x_out) - rt_left_lowest) / denom


def integrate_with_baseline_minval_noise_right(rt_full, intensity_full, left, right, scale=AREA_TIME_UNIT_SCALE,
                                               noise_right_margin=NOISE_RIGHT_MARGIN_MIN):
    """
    谷点-噪声基线积分：适用于前肩峰/双峰场景。
    - 左锚点：在 [left, rt_peak] 内取强度最低点（峰与左框间的谷点）
    - 右锚点：右侧预测框外 [right, right+noise_right_margin] 的噪声平均值
    - 基线为两点连线，积分窗口内高于基线的面积
    """
    rt_full = np.asarray(rt_full, dtype=np.float64)
    intensity_full = np.asarray(intensity_full, dtype=np.float64)
    mask_win = (rt_full >= left) & (rt_full <= right)
    x = rt_full[mask_win]
    y = intensity_full[mask_win]
    if x.size < 2 or y.size < 2:
        return 0.0
    params = _get_minval_noise_right_baseline_params(rt_full, intensity_full, left, right, noise_right_margin)
    if params is None:
        return 0.0
    rt_left_lowest, y_left_lowest, y_right_avg = params
    denom = right - rt_left_lowest if right != rt_left_lowest else 1e-10
    baseline = y_left_lowest + (y_right_avg - y_left_lowest) * (x - rt_left_lowest) / denom
    corrected = np.maximum(0.0, y - baseline)
    return float(np.trapz(corrected, x) * scale)


def integrate_with_baseline_correction_avg(rt_full, intensity_full, left, right, scale=AREA_TIME_UNIT_SCALE,
                                           margin_min=BASELINE_AVG_MARGIN_MIN, percentile=BASELINE_PERCENTILE):
    """
    线性基线校正积分：用窗口两端附近多点的分位数作为端点高度（优先窗口外），再线性插值作基线。
    可减少高噪声下因单点抖动导致的基线过低。
    """
    rt_full = np.asarray(rt_full, dtype=np.float64)
    intensity_full = np.asarray(intensity_full, dtype=np.float64)
    mask_win = (rt_full >= left) & (rt_full <= right)
    x = rt_full[mask_win]
    y = intensity_full[mask_win]
    if x.size < 2 or y.size < 2:
        return 0.0
    y_left_avg, y_right_avg = get_baseline_endpoint_heights(
        rt_full, intensity_full, left, right, margin_min, percentile
    )
    if y_left_avg is None:
        y_left_avg = float(np.percentile(y, percentile)) if y.size > 0 else float(y[0])
    if y_right_avg is None:
        y_right_avg = float(np.percentile(y, percentile)) if y.size > 0 else float(y[-1])
    baseline = y_left_avg + (y_right_avg - y_left_avg) * (x - x[0]) / (x[-1] - x[0])
    corrected = np.maximum(0.0, y - baseline)
    return float(np.trapz(corrected, x) * scale)


def integrate_with_external_baseline(rt_full, intensity_full, left, right, baseline_x, baseline_y,
                                     scale=AREA_TIME_UNIT_SCALE):
    """
    外部基线积分：用用户提供的 (x[], y[]) 定义基线，在模型输出的积分上下限 [left, right] 内，
    对 (XIC 强度 - 基线) 的正值部分积分。
    baseline_x, baseline_y: 基线曲线上的点，将插值到 [left, right] 内的 RT 网格。
    """
    from scipy.interpolate import interp1d
    rt_full = np.asarray(rt_full, dtype=np.float64)
    intensity_full = np.asarray(intensity_full, dtype=np.float64)
    baseline_x = np.asarray(baseline_x, dtype=np.float64)
    baseline_y = np.asarray(baseline_y, dtype=np.float64)
    mask_win = (rt_full >= left) & (rt_full <= right)
    x = rt_full[mask_win]
    y = intensity_full[mask_win]
    if x.size < 2 or y.size < 2:
        return 0.0
    if baseline_x.size < 2 or baseline_y.size < 2:
        return float(np.trapz(y, x) * scale)
    order = np.argsort(baseline_x)
    bx_sorted = baseline_x[order]
    by_sorted = baseline_y[order]
    f_baseline = interp1d(bx_sorted, by_sorted, kind="linear", bounds_error=False,
                         fill_value=(float(by_sorted[0]), float(by_sorted[-1])))
    baseline_at_x = f_baseline(x)
    corrected = np.maximum(0.0, y - baseline_at_x)
    return float(np.trapz(corrected, x) * scale)


def integrate_with_baseline_correction(x, y, scale=AREA_TIME_UNIT_SCALE):
    """
    线性基线校正积分：用窗口两端点做线性基线，只积分高于基线的部分。
    baseline(t) = y_left + (y_right - y_left) * (t - t_left) / (t_right - t_left)
    corrected_y = max(0, y - baseline)
    area = trapz(corrected_y, x) * scale
    可降低因积分区间过宽导致的预测偏高。
    """
    if x.size < 2 or y.size < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y_left = float(y[0])
    y_right = float(y[-1])
    baseline = y_left + (y_right - y_left) * (x - x[0]) / (x[-1] - x[0])
    corrected = np.maximum(0.0, y - baseline)
    raw = trapz(corrected, x)
    return float(raw * scale)


@time_master
def quantify(mzml, prediction, info, baseline_correction=False):
    """
    定量积分函数
    
    Parameters:
    - mzml: xic_list, list of [(2, S)] arrays, 每个化合物一个 XIC
    - prediction: list of (image_path, scores_array, boxes_array), 每个图像一个预测
    - info: feature DataFrame, 包含 Compound Name, mz, RT
    
    Returns:
    - area: list of tuples with quantification results
    """
    area = []
    
    # 新的一一对应逻辑：每个 prediction 对应一个 compound
    for i, (path, scores, box) in enumerate(prediction):
        dir = os.path.dirname(path)
        
        # 直接从 info 中获取对应的化合物信息（一一对应）
        if i >= len(info):
            print(f"[WARN] Prediction index {i} exceeds feature count {len(info)}, skipping.")
            break
            
        name = info.loc[i, 'Compound Name']
        mz = info.loc[i, 'mz']
        true_rt = info.loc[i, 'RT']
        
        # 从 xic_list 中获取对应的 XIC 数据
        if i >= len(mzml):
            print(f"[WARN] Prediction index {i} exceeds XIC count {len(mzml)}, using empty data.")
            rt = np.array([])
            intensity = np.array([])
        else:
            rt = mzml[i][0]  # RT in minutes
            intensity = mzml[i][1]  # Intensity
        
        max_rt = max(rt) if len(rt) > 0 else 0

        if len(scores) > 0:
            for j in range(len(scores)):
                score = scores[j][0]
                x1, y1 = float(box[j][0]), float(box[j][1])
                x2, y2 = float(box[j][2]), float(box[j][3])
                # 与 testXIC 固定 xlim 后的 ROI 一致：400px 线性映射到 [rt_lo, rt_hi]
                left, right, _, _ = box_to_rt_range(x1, y1, x2, y2, true_rt, rt)

                mask = (rt >= left) & (rt <= right)
                filter_x = rt[mask]
                filter_y = intensity[mask]
                point_count = max_consecutive(filter_y)
                max_intensity = max(filter_y)
                max_index = filter_y.argmax()
                max_x = filter_x[max_index]
                if baseline_correction and filter_x.size >= 2:
                    scaled_area = integrate_with_baseline_correction(filter_x, filter_y)
                else:
                    raw_area = trapz(filter_y, filter_x)
                    scaled_area = raw_area * AREA_TIME_UNIT_SCALE
                area.append((dir, name, mz, true_rt, left, right, max_x, max_intensity, scaled_area, score, point_count))
        else:
            area.append((dir, name, mz, true_rt, 0, 0, 0, 0, 0, 0, 0))
    return area


def max_consecutive(arr):
    greater_than_zero = arr > 0
    diff = np.diff(greater_than_zero.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if greater_than_zero[0]:
        starts = np.insert(starts, 0, 0)
    if greater_than_zero[-1]:
        ends = np.append(ends, len(arr))

    if len(starts) == len(ends) == 0:
        max_c = 0
    else:
        max_c = np.max(ends - starts)
    return max_c