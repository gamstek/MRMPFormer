"""
SimCLR 数据增强策略。

五种增强组合, 对同一张图像产生两个语义相同但视觉差异显著的 view:
  1. RandomResizedCrop  — 空间变换 (最关键)
  2. RandomHorizontalFlip — 翻转不变性
  3. ColorJitter          — 颜色不变性
  4. RandomGrayscale      — 抗颜色依赖
  5. GaussianBlur         — 抗细节噪声

色谱图专用增强 (Chromatogram):
  1. RandomRTShift        — 保留时间漂移模拟
  2. Resize + Pad         — 零损失宽高比处理
  3. RandomHorizontalFlip — 翻转不变性
  4. GaussianBlur(轻度)   — 仪器分辨率差异
"""
import random

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


def get_simclr_augmentations(size=224, imagenet_norm=True):
    """
    构建 SimCLR 训练增强 pipeline。

    Args:
        size: 图像输入尺寸 (default: 224)
        imagenet_norm: 是否使用 ImageNet 均值/标准差归一化 (default: True)

    Returns:
        torchvision.transforms.Compose
    """
    augment_list = [
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ]

    if imagenet_norm:
        augment_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(augment_list)


def get_test_augmentations(size=224, imagenet_norm=True):
    """
    构建测试/推理增强 pipeline (仅 resize + center crop + normalize)。
    """
    augment_list = [
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ]
    if imagenet_norm:
        augment_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(augment_list)


class RandomRTShift(nn.Module):
    """
    模拟保留时间漂移：沿水平方向随机平移图像。

    XIC 图 x 轴 = 保留时间 (RT)。LC 色谱柱老化/温度波动会导致
    RT 整体偏移，这是色谱领域最高频、最真实的变异源。

    Args:
        max_shift: 最大平移比例 (default: 0.08, 即 ±8% 图像宽度)
        padding_mode: 填充模式
            - 'edge': 边缘像素延续（基线自然延伸，推荐）
            - 'constant': 纯白边填充 (fill=255)
    """

    def __init__(self, max_shift=0.08, padding_mode='edge'):
        super().__init__()
        self.max_shift = max_shift
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img: PIL Image

        Returns:
            PIL Image: 水平平移后的图像，尺寸不变
        """
        w, h = img.size
        max_px = int(w * self.max_shift)
        if max_px < 1:
            return img

        shift_px = random.randint(-max_px, max_px)
        if shift_px == 0:
            return img

        img_np = np.array(img)  # (H, W, C)
        shifted = np.zeros_like(img_np)

        if self.padding_mode == 'edge':
            if shift_px > 0:
                # 右移：右侧留空，左侧用第一列填充
                shifted[:, shift_px:, :] = img_np[:, :-shift_px, :]
                shifted[:, :shift_px, :] = img_np[:, :1, :]
            else:
                # 左移：左侧留空，右侧用最后一列填充
                shift_px = -shift_px
                shifted[:, :-shift_px, :] = img_np[:, shift_px:, :]
                shifted[:, -shift_px:, :] = img_np[:, -1:, :]
        elif self.padding_mode == 'constant':
            # 白边填充
            shifted[:, :, :] = 255
            if shift_px > 0:
                shifted[:, shift_px:, :] = img_np[:, :-shift_px, :]
            else:
                shift_px = -shift_px
                shifted[:, :-shift_px, :] = img_np[:, shift_px:, :]

        return Image.fromarray(shifted)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"max_shift={self.max_shift}, "
                f"padding_mode='{self.padding_mode}')")


def get_chromatogram_augmentations(config, imagenet_norm=True):
    """
    构建色谱图专用增强 pipeline。

    设计原则：
      - 零信息损失：Resize + Pad 而非 Crop，保留全图
      - 物理合理的变异：RT 漂移、仪器分辨率差异
      - 不引入无物理意义的增强：无 ColorJitter、Grayscale

    Pipeline:
      RandomRTShift → Resize → Pad → HorizontalFlip →
      GaussianBlur → ToTensor → Normalize

    Args:
        config: ExperimentConfig 对象，从中读取增强参数 (rt_shift, blur_kernel, pad_mode)
        imagenet_norm: 是否使用 ImageNet 均值/标准差归一化

    Returns:
        torchvision.transforms.Compose
    """
    size = 224  # 模型需要的方图尺寸

    # 计算 Resize 目标：保持 4:3 比例，宽边 = size
    # 原始 400×300 (4:3) → 224×168
    h_resize = int(size * 3 / 4)   # 168
    pad_total = size - h_resize     # 56
    pad_top = pad_total // 2        # 28
    pad_bottom = pad_total - pad_top  # 28

    augment_list = [
        # ① 保留时间漂移模拟
        RandomRTShift(
            max_shift=config.rt_shift,
            padding_mode=config.pad_mode,
        ),
        # ② 等比缩放：保持 4:3 宽高比 → 224×168
        transforms.Resize((h_resize, size)),
        # ③ 补边成方图：上下补 28px → 224×224，零信息损失
        transforms.Pad(
            (0, pad_top, 0, pad_bottom),
            fill=255 if config.pad_mode == 'constant' else 0,
            padding_mode='edge' if config.pad_mode == 'edge' else 'constant',
        ),
        # ④ 水平翻转：色谱峰天然对称
        transforms.RandomHorizontalFlip(p=0.5),
        # ⑤ 轻度高斯模糊：模拟仪器分辨率差异
        transforms.GaussianBlur(
            kernel_size=config.blur_kernel,
            sigma=(0.1, 1.0),
        ),
        # ⑥ 转为张量
        transforms.ToTensor(),
    ]

    if imagenet_norm:
        augment_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(augment_list)
