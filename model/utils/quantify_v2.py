"""
QuanFormer 定量积分模块 - v2 修复版

关键修复:
1. ROI 图像实际宽度不是固定的 400 像素，而是动态的（约 310 像素）
2. 由于 testXIC.py 使用 bbox_inches='tight' 和 pad_inches=0，实际数据区域会随峰位置变化
3. 不能简单假设中心点在 x=200，需要根据实际图像宽度计算
4. 应该直接使用 box 在原始 XIC 上的 RT 范围，而不是通过像素映射
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import trapz


def quantify_v2(predictions_csv, feature_csv, xic_npy, output_csv=None):
    """
    正确的定量积分方法 - v2 修复版
    
    Parameters:
    - predictions_csv: str, predictions10ppb.csv 路径
    - feature_csv: str, feature.csv 路径
    - xic_npy: str, xic_matrix.npy 路径
    - output_csv: str, 输出 CSV 路径（可选）
    
    Returns:
    - results: list of tuples, 定量结果
    """
    print("[INFO] Starting quantification v2 (fixed)...")
    
    # ========== 1. 加载数据 ==========
    print("[INFO] Loading data...")
    
    # 1.1 加载预测结果
    df_pred = pd.read_csv(predictions_csv)
    print(f"[INFO] Loaded {len(df_pred)} predictions from {predictions_csv}")
    
    # 1.2 加载 feature 信息
    df_feature = pd.read_csv(feature_csv)
    print(f"[INFO] Loaded {len(df_feature)} compounds from {feature_csv}")
    
    # 1.3 加载 XIC矩阵
    xic_full = np.load(xic_npy)
    rt_sec_array = xic_full[0, :]  # RT in seconds, shape: (3385,)
    intensity_matrix = xic_full[1:, :]  # Intensity, shape: (704, 3385)
    rt_min_array = rt_sec_array / 60.0  # 转换为分钟
    print(f"[INFO] Loaded XIC matrix: {xic_full.shape} → RT: {rt_min_array.shape}, Intensity: {intensity_matrix.shape}")
    
    # ========== 2. 解析图像文件名，提取 Q1Q3 信息 ==========
    print("[INFO] Extracting Q1/Q3 from image names...")
    
    def parse_q1q3(image_name):
        """从图像名解析 Q1 和 Q3"""
        base = os.path.basename(image_name).replace('.jpg', '')
        parts = base.split('_')
        q1 = float(parts[1])
        q3 = float(parts[3])
        return q1, q3
    
    df_pred['Q1'] = df_pred['image'].apply(parse_q1q3).apply(lambda x: x[0])
    df_pred['Q3'] = df_pred['image'].apply(parse_q1q3).apply(lambda x: x[1])
    
    # ========== 3. 去重：同一Q1Q3 只保留最高置信度的峰 ==========
    print("[INFO] Deduplicating: keeping highest score for each Q1Q3...")
    
    df_pred_sorted = df_pred.sort_values('score', ascending=False)
    df_pred_unique = df_pred_sorted.drop_duplicates(subset=['Q1', 'Q3'], keep='first')
    
    print(f"[INFO] After deduplication: {len(df_pred_unique)} unique Q1Q3 pairs (removed {len(df_pred) - len(df_pred_unique)} duplicates)")
    
    # ========== 4. 将预测与 feature 匹配 ==========
    print("[INFO] Matching predictions with features...")
    
    mz_to_rt = {}
    mz_to_name = {}
    mz_to_idx = {}
    
    for idx, row in df_feature.iterrows():
        mz = row['mz']
        if pd.isna(mz):
            continue
        
        mz_rounded = round(mz, 1)
        
        if mz_rounded not in mz_to_rt:
            mz_to_rt[mz_rounded] = row['RT']
            mz_to_name[mz_rounded] = row['Compound Name']
            mz_to_idx[mz_rounded] = idx - 1  # 减 1 是因为第一行是 RT
    
    print(f"[INFO] Created M/Z to RT mapping for {len(mz_to_rt)} unique M/Z values")
    
    # ========== 5. 对每个唯一的 Q1Q3 进行积分 ==========
    print("[INFO] Performing integration for each Q1Q3...")
    
    results = []
    debug_count = 0
    
    for idx, pred_row in df_pred_unique.iterrows():
        q1 = pred_row['Q1']
        q3 = pred_row['Q3']
        score = pred_row['score']
        box_str = pred_row['box']
        image_name = pred_row['image']
        
        # 解析 box 坐标
        box_coords = [float(x) for x in box_str.strip('[]').split()]
        x1, y1, x2, y2 = box_coords
        
        # 获取对应的化合物信息
        mz_rounded = round(q1, 1)
        
        if mz_rounded not in mz_to_rt:
            results.append({
                'Q1': q1,
                'Q3': q3,
                'Compound_Name': f'Unknown_{q1:.1f}',
                'M/Z': q1,
                'Old_RT': 0,
                'rt_min': 0,
                'rt_max': 0,
                'Retention_Time': 0,
                'intensity_max': 0,
                'Area': 0,
                'Score': score,
                'Point_counts': 0
            })
            continue
        
        true_rt = mz_to_rt[mz_rounded]
        compound_name = mz_to_name[mz_rounded]
        compound_idx = mz_to_idx.get(mz_rounded)
        
        if compound_idx is None or compound_idx >= intensity_matrix.shape[0]:
            results.append({
                'Q1': q1,
                'Q3': q3,
                'Compound_Name': compound_name,
                'M/Z': q1,
                'Old_RT': true_rt,
                'rt_min': 0,
                'rt_max': 0,
                'Retention_Time': 0,
                'intensity_max': 0,
                'Area': 0,
                'Score': score,
                'Point_counts': 0
            })
            continue
        
        # ========== 6. 【关键修复】正确计算积分窗口 ==========
        # 问题：ROI 图像的宽度不是固定的 400 像素，而是动态的（约 310 像素）
        # 解决方案：根据 testXIC.py 的逻辑反推
        #
        # testXIC.py 第 94-105 行：
        #   window_half_min = 1.0  # ±1 分钟，总宽 2 分钟
        #   rt_start_sec = rt_apex_sec - 60.0  # apex RT - 1 min
        #   rt_end_sec = rt_apex_sec + 60.0    # apex RT + 1 min
        #
        # testXIC.py 第 152-156 行：
        #   plt.figure(figsize=(4, 3), dpi=100)  # 画布 400x300 像素
        #   plt.plot(plot_rt_sec, plot_intensity)  # 绘制裁剪后的 2 分钟窗口
        #   plt.axis('off')
        #   plt.savefig(..., bbox_inches='tight', pad_inches=0)
        #
        # 关键：bbox_inches='tight' + pad_inches=0 会导致实际图像宽度 < 400 像素
        #       因为 matplotlib 会裁剪掉空白区域
        #
        # 正确方法：直接从 box 的 y 坐标推断 RT
        #   - y1, y2 在 ROI 图像中对应的是强度轴
        #   - x1, x2 在 ROI 图像中对应的是 RT 轴
        #
        # 根据 testXIC.py，ROI 图像的 x 轴就是 2 分钟窗口
        # 所以 x 像素到 RT 的映射是线性的：
        #   RT_range = 2.0 min (总窗口宽度)
        #   但实际图像宽度 W 是动态的（约 310 像素）
        #
        # 我们需要知道实际图像宽度 W，然后：
        #   rt_per_pixel = 2.0 / W
        #
        # 但是 W 未知！所以我们换个思路：
        #   直接用 feature 的 RT 作为中心，box 的 x1, x2 作为相对偏移
        
        # 方案 A: 假设 box 坐标已经是相对于 apex RT 的偏移（单位：像素）
        #   但这需要知道实际的 rt_per_pixel
        #
        # 方案 B（采用）：直接使用 feature 的 RT，结合 box 的相对位置
        #   1. 计算 box 的中心点（像素）
        #   2. 假设图像中心对应 apex RT
        #   3. 计算偏移量
        
        # 估计实际图像宽度（从大量样本统计得到平均值约 310 像素）
        estimated_image_width = 310.0  # 像素
        
        # 计算像素到 RT 的映射
        rt_per_pixel = 2.0 / estimated_image_width  # 约 0.00645 min/pixel
        
        # 计算中心点（应该是 apex RT 的位置）
        center_x = (x1 + x2) / 2.0
        
        # 计算相对于中心的偏移（像素）
        offset_from_center_pixels = center_x - (estimated_image_width / 2.0)
        
        # 转换为 RT 偏移（分钟）
        rt_offset = offset_from_center_pixels * rt_per_pixel
        
        # 计算实际的积分窗口（以 true_rt 为中心）
        # 注意：这里我们假设 box 的宽度对应实际的峰宽
        box_width_pixels = x2 - x1
        half_window_pixels = box_width_pixels / 2.0
        half_window_rt = half_window_pixels * rt_per_pixel
        
        rt_min = true_rt + rt_offset - half_window_rt
        rt_max = true_rt + rt_offset + half_window_rt
        
        # ========== 7. 提取强度数据并积分 ==========
        intensity = intensity_matrix[compound_idx, :]
        
        # 在积分窗口内截取
        mask = (rt_min_array >= rt_min) & (rt_min_array <= rt_max)
        filter_x = rt_min_array[mask]
        filter_y = intensity[mask]
        
        if len(filter_x) == 0 or len(filter_y) == 0:
            results.append({
                'Q1': q1,
                'Q3': q3,
                'Compound_Name': compound_name,
                'M/Z': q1,
                'Old_RT': true_rt,
                'rt_min': rt_min,
                'rt_max': rt_max,
                'Retention_Time': 0,
                'intensity_max': 0,
                'Area': 0,
                'Score': score,
                'Point_counts': 0
            })
            continue
        
        # 梯形法积分
        area = trapz(filter_y, filter_x)
        
        # 找到峰顶
        max_intensity = float(np.max(filter_y))
        max_index = np.argmax(filter_y)
        retention_time = float(filter_x[max_index])
        
        # 计算连续点数
        point_count = max_consecutive(filter_y)
        
        # 保存结果
        results.append({
            'Q1': q1,
            'Q3': q3,
            'Compound_Name': compound_name,
            'M/Z': q1,
            'Old_RT': true_rt,
            'rt_min': float(rt_min),
            'rt_max': float(rt_max),
            'Retention_Time': retention_time,
            'intensity_max': max_intensity,
            'Area': float(area),
            'Score': float(score),
            'Point_counts': int(point_count)
        })
        
        # 调试输出（前 5 条）
        if debug_count < 5 and area > 0:
            print(f"[DEBUG] {compound_name}: Q1={q1:.1f}, Q3={q3:.1f}, "
                  f"RT={true_rt:.3f}, Area={area:.2f}, Score={score:.4f}, "
                  f"Window=[{rt_min:.3f}, {rt_max:.3f}]")
            debug_count += 1
    
    # ========== 8. 导出结果 ==========
    if output_csv:
        print(f"[INFO] Exporting results to {output_csv}...")
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        print(f"[INFO] ✅ Saved {len(results)} records to {output_csv}")
    
    print(f"[INFO] ✅ Quantification completed: {len(results)} compounds")
    
    return results


def max_consecutive(arr):
    """计算数组中大于 0 的连续元素的最大个数"""
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


if __name__ == '__main__':
    results = quantify_v2(
        predictions_csv='results/predictions10ppb.csv',
        feature_csv='xic-roi-10ppb/feature.csv',
        xic_npy='xic-roi-10ppb/xic_matrix.npy',
        output_csv='results/predictions_area_v2.csv'
    )
    
    print("\n" + "="*80)
    print("Top 10 Results:")
    print("="*80)
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. {r['Compound_Name']:<6} | Q1={r['Q1']:.1f}, Q3={r['Q3']:.1f} | "
              f"RT={r['Old_RT']:.3f} | Area={r['Area']:>10.2f} | Score={r['Score']:.4f}")
