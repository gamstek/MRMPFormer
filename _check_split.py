# -*- coding: utf-8 -*-
"""核查 train/val 正负样本分布 + 标注行与图像的对应。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"d:\work\MRMPFormer\model")

root = Path(r"d:\work\MRMPFormer\data\coco\merged")
for split in ("train", "val"):
    coco = json.loads((root / split / f"{split}_coco.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i["file_name"] for i in coco["images"]}
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    n_pos = len(ann_by_img)
    n_neg = len(imgs) - n_pos
    print(f"== {split}: {len(imgs)} 图 | 正样本 {n_pos} | 负样本 {n_neg} ==")
    neg_files = [imgs[i] for i in imgs if i not in ann_by_img]
    for f in neg_files:
        print("  负样本:", f)

# 每个负样本图像尾缀（化合物标识）
print("\n（train+val 合计正负比见上）")
