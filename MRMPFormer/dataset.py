"""
SimCLR 图像数据集。

从指定目录加载所有图像, 每张图像通过数据增强产生两个 view 作为正样本对。
对比学习无需标签, 仅需图像文件。
"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SimCLRDataset(Dataset):
    """
    SimCLR 无标签图像数据集。

    每次 __getitem__ 返回 (view_a, view_b) 两个增强视图,
    它们来自同一张原图, 构成正样本对。

    Args:
        root_dir: 图像存放目录 (支持 .png / .jpg / .jpeg / .bmp / .tiff)
        transform: torchvision 增强 pipeline
    """

    SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        self.transform = transform
        self.paths = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not self.paths:
            raise ValueError(
                f"No supported images found in '{root_dir}'. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_EXTS))}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform is None:
            return img, img
        view_a = self.transform(img)
        view_b = self.transform(img)
        return view_a, view_b
