# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
import os
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from . import transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        # 绕过 torchvision.datasets.CocoDetection.__init__：其在 Windows 下用系统默认编码
        # (GBK) 读标注 json，而本数据集标注为 UTF-8（含中文化合物名）会抛 UnicodeDecodeError。
        # 此处显式 UTF-8 读入并手动重建 COCO 索引；保留继承关系以满足
        # get_coco_api_from_dataset 的 isinstance 检查（评估链路的 coco api 依赖它）。
        self.root = str(img_folder)
        with open(ann_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        self.coco = COCO()
        self.coco.dataset = dataset
        self.coco.createIndex()
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def _load_image_and_annotations(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        return img, target

    def __getitem__(self, idx):
        img, target = self._load_image_and_annotations(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target




def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_transforms(image_set):
    """
    本项目色谱图专用的数据变换（非 DETR 原版自然照片增强）。

    推理端预处理（utils/predict_utils.py）仅做 ToTensor + Normalize，且输入为固定
    400x300 的 ROI 图。训练端必须与推理完全对齐，否则分布错位会导致检测全崩：
      - 关 RandomHorizontalFlip：色谱峰形镜像无物理意义，且推理不翻转
      - 关 RandomResize 放大（480~800px）：推理不缩放，训练放大造成尺度错位
      - 关 RandomSizeCrop(384,600)：裁高 600 > 图高 300，越界语义崩溃
    因此 train/val 使用同一组变换（仅 ToTensor + Normalize）。
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return normalize


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    # 兼容两种标注布局（优先 split 目录内，回退根目录）：
    #   <root>/<split>/<split>_coco.json  （coco_annotation.py 生成，如 train/train_coco.json）
    #   <root>/<split>_coco.json          （COCO 常规布局）
    img_folder = root / image_set
    ann_candidates = [
        img_folder / f"{image_set}_coco.json",
        root / f"{image_set}_coco.json",
    ]
    ann_file = next((p for p in ann_candidates if p.is_file()), None)
    if ann_file is None:
        raise FileNotFoundError(
            f'COCO annotation not found for split {image_set!r}: '
            f'checked {ann_candidates}'
        )

    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
