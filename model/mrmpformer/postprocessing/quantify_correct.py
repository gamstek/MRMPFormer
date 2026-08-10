"""
MRMPFormer 定量积分模块 - 正确版本

基于预测结果 (predictions10ppb.csv) 和 feature.csv 的 RT 信息，
对 xic_matrix.npy 进行正确的峰面积积分。

核心逻辑:
1. 从 predictions10ppb.csv 读取模型预测的峰位置（box）和置信度（score）
2. 从 feature.csv 读取每个化合物的标准 RT时间
3. 从 xic_matrix.npy 读取 XIC矩阵数据
4. 对于同一Q1Q3 化合物，只保留置信度最高的峰
5. 根据预测的 box 坐标和 feature 的 RT，计算正确的积分窗口
6. 使用梯形法积分计算峰面积
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import trapz


def quantify_correct(predictions_csv, feature_csv, xic_npy, output_csv=None):
    """
    正确的定量积分方法
    
    Parameters:
    - predictions_csv: str, predictions10ppb.csv 路径
    - feature_csv: str, feature.csv 路径
    - xic_npy: str, xic_matrix.npy 路径
    - output_csv: str, 输出 CSV 路径（可选）
    
    Returns:
    - results: list of tuples, 定量结果
    """
    print("[INFO] Starting correct quantification...")
    
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
        # 格式：Q1_142.0000_Q3_125.0.jpg
        base = os.path.basename(image_name).replace('.jpg', '')
        parts = base.split('_')
        q1 = float(parts[1])
        q3 = float(parts[3])
        return q1, q3
    
    df_pred['Q1'] = df_pred['image'].apply(parse_q1q3).apply(lambda x: x[0])
    df_pred['Q3'] = df_pred['image'].apply(parse_q1q3).apply(lambda x: x[1])
    
    # ========== 3. 去重：同一Q1Q3 只保留最高置信度的峰 ==========
    print("[INFO] Deduplicating: keeping highest score for each Q1Q3...")
    
    # 按 Q1, Q3 分组，选择 Score 最高的记录
    df_pred_sorted = df_pred.sort_values('score', ascending=False)
    df_pred_unique = df_pred_sorted.drop_duplicates(subset=['Q1', 'Q3'], keep='first')
    
    print(f"[INFO] After deduplication: {len(df_pred_unique)} unique Q1Q3 pairs (removed {len(df_pred) - len(df_pred_unique)} duplicates)")
    
    # ========== 4. 将预测与 feature 匹配 ==========
    print("[INFO] Matching predictions with features...")
    
    # 创建 feature 的查找表：mz → RT
    # 注意：feature.csv 中 mz 可能有重复（双峰），我们取第一个非空值
    mz_to_rt = {}
    mz_to_name = {}
    
    for idx, row in df_feature.iterrows():
        mz = row['mz']
        if pd.isna(mz):
            continue
        
        mz_rounded = round(mz, 1)  # 四舍五入到 0.1
        
        # 如果这个 M/Z 还没有记录，或者当前记录的 RT 更小（可能是主峰）
        if mz_rounded not in mz_to_rt:
            mz_to_rt[mz_rounded] = row['RT']
            mz_to_name[mz_rounded] = row['Compound Name']
    
    print(f"[INFO] Created M/Z to RT mapping for {len(mz_to_rt)} unique M/Z values")
    
    # ========== 5. 对每个唯一的 Q1Q3 进行积分 ==========
    print("[INFO] Performing integration for each Q1Q3...")
    
    results = []
    
    for idx, pred_row in df_pred_unique.iterrows():
        q1 = pred_row['Q1']
        q3 = pred_row['Q3']
        score = pred_row['score']
        box = pred_row['box']
        image_name = pred_row['image']
        
        # 解析 box 坐标
        box_coords = box.strip('[]').split()
        x1, y1, x2, y2 = [float(x) for x in box_coords]
        
        # 在 feature 中查找对应的化合物（使用 Q1 作为 M/Z）
        mz_rounded = round(q1, 1)
        
        if mz_rounded not in mz_to_rt:
            # 没有找到对应的 M/Z
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
        
        # 获取标准 RT（来自 feature.csv）
        true_rt = mz_to_rt[mz_rounded]
        compound_name = mz_to_name[mz_rounded]
        
        # ========== 6. 计算积分窗口 ==========
        # 关键：使用预测的 box 坐标，但基于 feature 的 RT 进行校正
        
        # 6.1 计算像素到 RT 的映射关系
        # ROI 图像参数：figsize=(4, 3), dpi=100 → 400x300 像素
        # window_half_min = 1.0 → ±1 分钟，总宽 2 分钟
        rt_per_pixel = 2.0 / 400.0  # 0.005 分钟/像素
        
        # 6.2 计算中心点偏移（以 apex RT 为中心）
        center_pixel = 200  # 图像中心 x=200
        left_offset = (x1 - center_pixel) * rt_per_pixel
        right_offset = (x2 - center_pixel) * rt_per_pixel
        
        # 6.3 转换为实际 RT时间（以 true_rt 为中心）
        rt_min = true_rt + left_offset
        rt_max = true_rt + right_offset
        
        # ========== 7. 从 XIC矩阵中提取该化合物的色谱图 ==========
        # 需要找到这个化合物在 intensity_matrix 中的索引
        
        # 方法：在 df_feature 中查找匹配的 mz 和 RT
        compound_idx = None
        
        for feat_idx, feat_row in df_feature.iterrows():
            feat_mz = feat_row['mz']
            feat_rt = feat_row['RT']
            
            if pd.isna(feat_mz):
                continue
            
            # 匹配 M/Z 和 RT（允许小误差）
            if abs(round(feat_mz, 1) - mz_rounded) < 0.1 and abs(feat_rt - true_rt) < 0.05:
                compound_idx = feat_idx - 1  # 减 1 是因为第一行是 RT
                break
        
        if compound_idx is None or compound_idx >= intensity_matrix.shape[0]:
            # 没有找到对应的 XIC
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
        
        # 提取强度数据
        intensity = intensity_matrix[compound_idx, :]
        
        # ========== 8. 在积分窗口内积分 ==========
        mask = (rt_min_array >= rt_min) & (rt_min_array <= rt_max)
        filter_x = rt_min_array[mask]
        filter_y = intensity[mask]
        
        if len(filter_x) == 0 or len(filter_y) == 0:
            # 窗口内没有数据
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
        
        # 计算峰面积（梯形法）
        area = trapz(filter_y, filter_x)
        
        # 找到峰顶（最大强度点和对应的RT）
        max_intensity = float(np.max(filter_y))
        max_index = np.argmax(filter_y)
        retention_time = float(filter_x[max_index])
        
        # 计算连续点数（峰形质量指标）
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
    
    # ========== 9. 导出结果 ==========
    if output_csv:
        print(f"[INFO] Exporting results to {output_csv}...")
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        print(f"[INFO] ✅ Saved {len(results)} records to {output_csv}")
    
    print(f"[INFO] ✅ Quantification completed: {len(results)} compounds")
    
    return results


def max_consecutive(arr):
    """
    计算数组中大于 0 的连续元素的最大个数（峰形连续性指标）
    
    Parameters:
    - arr: numpy array, 强度值
    
    Returns:
    - max_c: int, 最大连续点数
    """
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


# ========== 使用示例 ==========
if __name__ == '__main__':
    results = quantify_correct(
        predictions_csv='results/predictions10ppb.csv',
        feature_csv='xic-roi-10ppb/feature.csv',
        xic_npy='xic-roi-10ppb/xic_matrix.npy',
        output_csv='results/predictions_area_corrected.csv'
    )
    
    # 打印前 10 条结果
    print("\n" + "="*80)
    print("Top 10 Results:")
    print("="*80)
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. Compound: {r['Compound_Name']}, "
              f"M/Z: {r['M/Z']:.1f}, "
              f"RT: {r['Old_RT']:.3f}, "
              f"Area: {r['Area']:.2f}, "
              f"Score: {r['Score']:.4f}")
