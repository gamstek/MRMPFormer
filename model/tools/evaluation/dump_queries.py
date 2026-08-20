# -*- coding: utf-8 -*-
"""
逐 query 诊断：打印每张 ROI 图全部 query 的 P(峰)/P(背景) + 框位置 + 与 GT 的偏差，
验证"好框挂在背景概率高的 query 上"（v1 取类 bug 的铁证）。

每图输出形如：
  阿维菌素-1  GT[16.428, 16.969]
    q0  P(峰) 0.003  P(背) 0.997  框RT[16.427, 17.115]  起偏 +0.00 / 止偏 +0.15   ← 旧bug选中(背>0.9)
    q1  P(峰) 0.995  P(背) 0.005  框RT[17.295, 17.577]  起偏 +0.87 / 止偏 +0.61   ← 新代码选中(峰>0.9)
    q2  ...

末尾统计：两类 query 的框谁更贴 GT（起止总偏差），给出占比。

用法（model/ 目录下，系统终端执行）：
  python -m tools.evaluation.dump_queries --model checkpoint/quanformer.pth \
      --xic_root ../data/coco/20260715_shiyaoyuan_test/_xic --labels ../data/label/20260715_shiyaoyuan_test.xlsx --limit 8
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocessing.coco_annotation import (
    group_labels_by_sample,
    label_key,
    map_samples_to_mzmls,
    parse_labels_xlsx,
    parse_rt_field,
)


def load_model(model_path):
    """与 build_predictor 同源的加载方式。"""
    from utils.torch_device import resolve_torch_device, load_torch_checkpoint
    from models.quanformer.detr import build

    device = resolve_torch_device(verbose=False)
    checkpoint = load_torch_checkpoint(model_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    train_args = checkpoint.get("args", None)
    if train_args is None:
        raise ValueError("checkpoint 内无 args，无法构建模型")
    train_args.device = str(device)
    model = build(train_args)[0]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, device


def main():
    ap = argparse.ArgumentParser(description="逐 query 打印 P(峰)/P(背) 与框位置")
    ap.add_argument("--model", default="checkpoint/quanformer.pth")
    ap.add_argument("--xic_root", required=True, help="ROI 根目录（其下样品子目录含 jpeg + roi_windows.csv + feature.csv）")
    ap.add_argument("--labels", required=True, help="人工标注 xlsx")
    ap.add_argument("--limit", type=int, default=8, help="每样品最多打印前 N 张（0=全部）")
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    import torchvision.transforms as T
    from PIL import Image
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model, device = load_model(args.model)
    print(f"[INFO] 模型: {args.model} | 设备: {device}")

    xic_root = Path(args.xic_root)
    stems = sorted(p.name for p in xic_root.iterdir() if p.is_dir())

    labels = parse_labels_xlsx(args.labels)
    sample_order, groups = group_labels_by_sample(labels)
    stem2sample = map_samples_to_mzmls(stems, sample_order, None)

    # 统计：背概率最高 query vs 峰概率最高 query，谁的框更贴 GT
    stat = {"bg_query_better": 0, "peak_query_better": 0, "ties_or_missing": 0}

    for stem in stems:
        sample_dir = xic_root / stem
        rw = pd.read_csv(sample_dir / "roi_windows.csv")
        feat = pd.read_csv(sample_dir / "feature.csv")

        by_key = {}
        for rec in groups[stem2sample[stem]]:
            k = label_key(rec.get("compound"), rec.get("channel"))
            if k:
                by_key.setdefault(k, rec)

        img_rows = list(rw.iterrows())[: args.limit] if args.limit else list(rw.iterrows())
        for _, rw_row in img_rows:
            img_name = str(rw_row["image"]).strip()
            rt_lo, rt_hi = float(rw_row["rt_lo"]), float(rw_row["rt_hi"])
            img_path = sample_dir / img_name
            if not img_path.is_file():
                continue
            prefix = img_name.split("_mz")[0]
            native_id = (str(feat.iloc[int(prefix) - 1]["native_id"]).strip()
                         if prefix.isdigit() and int(prefix) <= len(feat) else None)
            if native_id and native_id.startswith("TIC"):
                continue  # TIC 负样本无 GT

            rec = by_key.get(native_id)
            gt = None
            if rec is not None:
                s, e = parse_rt_field(rec.get("peak_start")), parse_rt_field(rec.get("peak_end"))
                if s is not None and e is not None:
                    gt = (min(s, e), max(s, e))
            if gt is None:
                continue

            im = Image.open(img_path).convert("RGB")
            x = transform(im).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
            logits = out["pred_logits"][0].softmax(-1).cpu().numpy()   # [Q, 2]: [背景, 峰]
            boxes = out["pred_boxes"][0].cpu().numpy()                 # [Q, 4] 归一化 (cx,cy,w,h)
            w_img, _ = im.size

            print()
            print(f"{native_id}  GT[{gt[0]:.3f}, {gt[1]:.3f}]  ({img_name})")

            devs = {}
            for q in range(logits.shape[0]):
                p_bg, p_peak = float(logits[q, 0]), float(logits[q, 1])
                cx, _, bw, _ = boxes[q]
                x1 = max(0.0, (cx - bw / 2)) * w_img
                x2 = min(1.0, (cx + bw / 2)) * w_img
                # 像素 → RT（与 roi_windows 映射一致）
                rt_s = rt_lo + (x1 / w_img) * (rt_hi - rt_lo)
                rt_e = rt_lo + (x2 / w_img) * (rt_hi - rt_lo)
                d_s, d_e = rt_s - gt[0], rt_e - gt[1]
                devs[q] = abs(d_s) + abs(d_e)

                tags = []
                if p_peak > args.threshold:
                    tags.append(f"← 新代码选中(峰>{args.threshold})")
                if p_bg > args.threshold:
                    tags.append(f"← 旧bug选中(背>{args.threshold})")
                print(f"  q{q}  P(峰) {p_peak:.3f}  P(背) {p_bg:.3f}  "
                      f"框RT[{rt_s:.3f}, {rt_e:.3f}]  起偏 {d_s:+.2f} / 止偏 {d_e:+.2f}  "
                      + " ".join(tags))

            q_peak = int(np.argmax(logits[:, 1]))
            q_bg = int(np.argmax(logits[:, 0]))
            if q_peak != q_bg and q_peak in devs and q_bg in devs:
                if devs[q_bg] < devs[q_peak] - 1e-9:
                    stat["bg_query_better"] += 1
                elif devs[q_peak] < devs[q_bg] - 1e-9:
                    stat["peak_query_better"] += 1
                else:
                    stat["ties_or_missing"] += 1
            else:
                stat["ties_or_missing"] += 1

    total = sum(stat.values())
    print()
    print("=" * 64)
    print(f"统计（{total} 张有 GT 的图，容差无关，按起止总偏差比较）")
    print(f"  背概率最高 query 的框更贴 GT : {stat['bg_query_better']}")
    print(f"  峰概率最高 query 的框更贴 GT : {stat['peak_query_better']}")
    print(f"  平局/同 query/无对比         : {stat['ties_or_missing']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
