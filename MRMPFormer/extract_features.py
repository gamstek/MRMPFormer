"""
MRMPFormer 特征提取脚本。

加载训练好的 SimCLR 模型, 对指定目录下的所有图像提取 2048-d backbone 特征。
支持保存为 CSV 或 .npy 格式。

用法:
    python extract_features.py --weights checkpoints/best_model.pth --input_dir data/images --output features.npy
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SimCLR
from augmentations import get_test_augmentations
from dataset import SimCLRDataset


def get_best_device():
    """自动选择最佳可用设备: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser(
        description='MRMPFormer — 图像特征提取',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--weights', type=str, required=True,
                        help='训练好的模型权重路径')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='待提取特征的图像目录')
    parser.add_argument('--output', type=str, default='features.npy',
                        help='特征输出路径 (.npy 或 .csv)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='推理 batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作进程数')
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    遍历 DataLoader, 对所有图像提取 backbone 特征。

    Returns:
        features: (N, 2048) numpy array
        paths: list of str, 对应每个特征的文件路径
    """
    model.eval()
    all_features = []
    all_paths = []

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc='提取特征')):
        images = images.to(device)
        features = model.get_features(images)  # (B, 2048)
        all_features.append(features.cpu().numpy())

        # 恢复文件路径 (batch 索引 → dataset paths)
        start_idx = batch_idx * dataloader.batch_size
        end_idx = start_idx + images.shape[0]
        batch_paths = [str(p) for p in dataloader.dataset.paths[start_idx:end_idx]]
        all_paths.extend(batch_paths)

    features = np.concatenate(all_features, axis=0)
    return features, all_paths


def save_features(features, paths, output_path):
    """保存特征到文件。"""
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix == '.npy':
        np.save(output_path, features)
        # 同时保存路径映射
        path_file = output_path.with_suffix('.paths.txt')
        path_file.write_text('\n'.join(paths), encoding='utf-8')
        print(f"特征已保存: {output_path}  ({features.shape})")
        print(f"路径映射:   {path_file}")

    elif suffix == '.csv':
        header = ','.join([f'f_{i}' for i in range(features.shape[1])])
        np.savetxt(
            output_path, features, delimiter=',',
            header=f'file,{header}', comments='',
        )
        print(f"特征已保存: {output_path}  ({features.shape})")
    else:
        raise ValueError(f"不支持的输出格式: {suffix} (仅支持 .npy 或 .csv)")


def main():
    args = get_args()
    device = get_best_device()
    print(f"[设备] 使用: {device}")

    # ---- 加载模型 ----
    print(f"[模型] 加载权重: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model = SimCLR(pretrained=False).to(device)

    # 兼容不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ↳ Epoch: {checkpoint.get('epoch', '?')} | Loss: {checkpoint.get('loss', '?'):.4f}")
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # ---- 数据集 ----
    print(f"[数据] 加载目录: {args.input_dir}")
    transform = get_test_augmentations()
    dataset = SimCLRDataset(args.input_dir, transform=transform)
    print(f"[数据] 图像数量: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # ---- 提取特征 ----
    features, paths = extract_features(model, dataloader, device)
    print(f"[完成] 特征矩阵: {features.shape}")

    # ---- 保存 ----
    save_features(features, paths, args.output)


if __name__ == '__main__':
    main()
