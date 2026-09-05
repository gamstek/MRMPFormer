# -*- coding: utf-8 -*-
"""
从现成 COCO 训练集随机抽取 N 张图像，复制生成独立数据集目录（可直接被 train.py 消费）。

背景：微调/消融实验需要 traindatav1 的随机 1000 图子集。train.py 的 COCO 读取约定为
<coco_path>/<split>/<split>_coco.json（见 framework/datasets/coco.py），子集数据集必须
自含 train 图像 + 标注文件。val 不抽样，整份沿用源数据集 val（QC 样），保证与全量训练的
验证指标可直接对比；train/val 源样品文件本就互斥，抽样无泄漏。

用法（model/ 目录下执行；纯标准库，无需 torch/pycocotools）：
  python -m tools.maintenance.subsample_coco \
      --src ../data/coco/traindatav1 --out ../data/coco/traindatav1_sub1000 \
      --n 1000 --seed 42
"""
import argparse
import json
import random
import shutil
from pathlib import Path


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def _copy_images(imgs, src_dir, dst_dir):
    n_missing = 0
    for im in imgs:
        rel = Path(im["file_name"])
        src = src_dir / rel
        dst = dst_dir / rel
        if not src.is_file():
            n_missing += 1
            print(f"[WARN] 源图缺失，跳过: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return n_missing


def main():
    ap = argparse.ArgumentParser(description="随机抽取 N 张 COCO 训练图像并复制为独立数据集目录")
    ap.add_argument("--src", default="../data/coco/traindatav1", help="源 COCO 数据集根目录")
    ap.add_argument("--out", default="../data/coco/traindatav1_sub1000", help="输出子集数据集根目录")
    ap.add_argument("--n", type=int, default=1000, help="随机抽取的训练图像数")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子（固定可复现）")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    train_src, val_src = src / "train", src / "val"
    train_ann, val_ann = train_src / "train_coco.json", val_src / "val_coco.json"
    for p in (train_ann, val_ann):
        if not p.is_file():
            raise SystemExit(f"[ERROR] 缺少 {p}，请先运行 preprocessing.coco_annotation 构建数据集")

    with open(train_ann, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_ann, encoding="utf-8") as f:
        val_data = json.load(f)

    images = train_data["images"]
    if args.n >= len(images):
        raise SystemExit(f"--n {args.n} >= 源训练图像数 {len(images)}，无需抽样")
    rng = random.Random(args.seed)
    picked = rng.sample(images, args.n)
    picked_ids = {im["id"] for im in picked}
    anns = [a for a in train_data["annotations"] if a["image_id"] in picked_ids]
    neg_imgs = len(picked) - len({a["image_id"] for a in anns})

    out_train, out_val = out / "train", out / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    n_miss = _copy_images(picked, train_src, out_train)
    n_miss += _copy_images(val_data["images"], val_src, out_val)

    _write_json(out_train / "train_coco.json",
                {"images": picked, "annotations": anns, "categories": train_data["categories"]})
    _write_json(out_val / "val_coco.json", val_data)

    print(f"[subsample] 源 train: {len(images)} 图 / {len(train_data['annotations'])} bbox")
    print(f"[subsample] 抽样 {len(picked)} 图 (seed={args.seed}): "
          f"{len(anns)} 条 bbox 标注，{neg_imgs} 张无标注纯负样本图")
    print(f"[subsample] val 沿用源: {len(val_data['images'])} 图 / {len(val_data['annotations'])} bbox")
    if n_miss:
        print(f"[WARN] 共 {n_miss} 张源图缺失已跳过")
    print(f"[subsample] 输出: {out.resolve()}")


if __name__ == "__main__":
    main()
