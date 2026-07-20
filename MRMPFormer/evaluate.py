"""
MRMPFormer 特征评估脚本。

对提取的 2048-d 特征计算:
  - Retrieval P@K (需要化合物标签)
  - Uniformity (无需标签)

支持单文件评估和双文件对比 (基线 vs 训练后)。

用法:
    # 单文件评估 (仅 Uniformity)
    python evaluate.py --features features.npy

    # 双文件对比 (基线 vs 训练后, 含 Retrieval + Uniformity)
    python evaluate.py --baseline features_baseline.npy --trained features_mrmp.npy

标签自动从图片文件名推断（中文名-序号格式），也可手动指定:
    python evaluate.py --features features.npy --labels labels.csv
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ===================================================================
# 标签推断
# ===================================================================

def infer_labels_from_paths(paths):
    """
    从文件路径中推断化合物标签。

    文件名格式: {id}_mz{mz}_q{q}_中文名-{序号}.jpeg
    取中文名部分（去掉末尾 -数字）作为化合物 ID。

    Returns:
        labels: list of str, 与 paths 一一对应
    """
    labels = []
    for p in paths:
        stem = Path(p).stem                       # 去掉扩展名
        parts = stem.rsplit('-', maxsplit=1)       # 从右分割一次
        if len(parts) == 2 and parts[1].isdigit():
            labels.append(parts[0])                # "氯磺隆"
        else:
            labels.append(stem)                    # 兜底
    return labels


def load_labels(paths, labels_file=None):
    """
    加载标签。

    优先级: labels_file > 自动推断

    labels_file 格式 (CSV):
        image_path,compound_id
        data/images/001_xxx.jpeg,氯磺隆
    """
    if labels_file is not None:
        path_to_label = {}
        with open(labels_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',', maxsplit=1)
                if len(parts) == 2:
                    path_to_label[parts[0].strip()] = parts[1].strip()
        labels = [path_to_label.get(p, 'unknown') for p in paths]
    else:
        labels = infer_labels_from_paths(paths)

    return labels


# ===================================================================
# 评估指标
# ===================================================================

def compute_retrieval_metrics(features, labels, k_values=(1, 5)):
    """
    计算 Retrieval Precision@K。

    对每张图，在其余图中找 K 个最近邻，计算其中同化合物占比。

    Args:
        features: (N, D) numpy array, 特征向量
        labels: list of str, 化合物 ID
        k_values: tuple of int, 要计算的 K 值

    Returns:
        dict: {f'P@{k}': float, ...}
    """
    N = features.shape[0]
    if N < 2:
        return {f'P@{k}': float('nan') for k in k_values}

    # L2 归一化 → 余弦相似度 = 内积
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim = features @ features.T                          # (N, N) 余弦相似度
    np.fill_diagonal(sim, -np.inf)                       # 排除自身

    results = {}
    for k in k_values:
        top_k_indices = np.argpartition(-sim, k, axis=1)[:, :k]  # O(N² log k)
        correct = 0
        for i in range(N):
            neighbor_labels = [labels[j] for j in top_k_indices[i]]
            correct += (labels[i] in neighbor_labels)
        results[f'P@{k}'] = correct / N

    return results


def compute_uniformity(features, t=2.0):
    """
    计算 Uniformity 指标。

    Args:
        features: (N, D) numpy array, L2 归一化
        t: 高斯核带宽

    Returns:
        float, 越负越好 (128-d 理论最优 ≈ -4.0)
    """
    N = features.shape[0]
    if N <= 1:
        return 0.0

    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    dist2 = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    potentials = np.exp(-t * dist2)
    return float(np.log(potentials[dist2 != np.inf].mean()))


# ===================================================================
# 主逻辑
# ===================================================================

def print_table(metrics_baseline, metrics_trained):
    """打印对比表格。"""
    keys = list(metrics_trained.keys())
    header = f"{'指标':<16} {'基线':>10} {'训练后':>10} {'变化':>10}"
    print("\n" + "=" * 50)
    print(header)
    print("-" * 50)

    for key in keys:
        b = metrics_baseline.get(key, float('nan'))
        t = metrics_trained.get(key, float('nan'))
        if 'Uniformity' in key or 'Alignment' in key:
            # 越小越好
            delta = t - b
            arrow = '↓' if delta < 0 else '↑'
        else:
            # 越大越好
            delta = t - b
            arrow = '↑' if delta > 0 else '↓'
        delta_str = f"{arrow} {delta:+.4f}"
        print(f"{key:<16} {b:>10.4f} {t:>10.4f} {delta_str:>10}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='MRMPFormer 特征评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--features', type=str, default=None,
                        help='单文件模式: 特征 .npy 路径')
    parser.add_argument('--baseline', type=str, default=None,
                        help='对比模式: 基线特征 .npy')
    parser.add_argument('--trained', type=str, default=None,
                        help='对比模式: 训练后特征 .npy')
    parser.add_argument('--labels', type=str, default=None,
                        help='标签 CSV (image_path,compound_id), 不指定则从文件名推断')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5],
                        help='Retrieval 的 K 值')
    args = parser.parse_args()

    # 模式判断
    if args.baseline and args.trained:
        # ---- 对比模式 ----
        print(f"[对比模式] 基线: {args.baseline}\n         训练后: {args.trained}")

        feats_b = np.load(args.baseline)
        feats_t = np.load(args.trained)

        # 加载路径
        paths_b_file = Path(args.baseline).with_suffix('.paths.txt')
        paths_t_file = Path(args.trained).with_suffix('.paths.txt')

        if paths_b_file.exists() and paths_t_file.exists():
            paths_b = paths_b_file.read_text(encoding='utf-8').strip().split('\n')
            paths_t = paths_t_file.read_text(encoding='utf-8').strip().split('\n')
        else:
            print("[警告] 未找到 .paths.txt, 标签统一设为 'unknown'")
            paths_b = [f'img_{i}' for i in range(len(feats_b))]
            paths_t = [f'img_{i}' for i in range(len(feats_t))]

        labels_b = load_labels(paths_b, args.labels)
        labels_t = load_labels(paths_t, args.labels)

        print(f"[数据] 基线: {feats_b.shape}  |  训练后: {feats_t.shape}")
        unique_labels = len(set(labels_b))
        print(f"[标签] 化合物种类: {unique_labels}  (来源: {'文件' if args.labels else '文件名自动推断'})")

        print("\n计算中...")
        metrics_b = compute_retrieval_metrics(feats_b, labels_b, tuple(args.k_values))
        metrics_b['Uniformity'] = compute_uniformity(feats_b)

        metrics_t = compute_retrieval_metrics(feats_t, labels_t, tuple(args.k_values))
        metrics_t['Uniformity'] = compute_uniformity(feats_t)

        print_table(metrics_b, metrics_t)

    elif args.features:
        # ---- 单文件模式 ----
        print(f"[单文件模式] 特征: {args.features}")
        feats = np.load(args.features)
        paths_file = Path(args.features).with_suffix('.paths.txt')

        if paths_file.exists():
            paths = paths_file.read_text(encoding='utf-8').strip().split('\n')
        else:
            paths = [f'img_{i}' for i in range(len(feats))]

        labels = load_labels(paths, args.labels)
        print(f"[数据] {feats.shape}  |  化合物种类: {len(set(labels))}")

        print("\n计算中...")
        metrics = compute_retrieval_metrics(feats, labels, tuple(args.k_values))
        metrics['Uniformity'] = compute_uniformity(feats)

        print("\n" + "=" * 30)
        for key, val in metrics.items():
            print(f"  {key:<16} {val:.4f}")
        print("=" * 30)

    else:
        parser.print_help()
        print("\n请指定 --features (单文件) 或 --baseline + --trained (对比)")


if __name__ == '__main__':
    main()
