"""
SimCLR 数据增强策略。

五种增强组合, 对同一张图像产生两个语义相同但视觉差异显著的 view:
  1. RandomResizedCrop  — 空间变换 (最关键)
  2. RandomHorizontalFlip — 翻转不变性
  3. ColorJitter          — 颜色不变性
  4. RandomGrayscale      — 抗颜色依赖
  5. GaussianBlur         — 抗细节噪声
"""
import torchvision.transforms as transforms


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
