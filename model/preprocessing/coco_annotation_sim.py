# -*- coding: utf-8 -*-
"""
从 MRM-XIC 模拟数据集（V6.0：xic_data JSON + V5.0 label.csv）生成 COCO 格式训练数据集。

与 coco_annotation.py（mzML + xlsx 路线）并列的另一条构建路径，输出格式与其完全一致，
framework/datasets/coco.py 无改动即可加载：
  {output_dir}/train/*.jpeg + train_coco.json
  {output_dir}/val/*.jpeg   + val_coco.json

流程：
1. 逐样本读取 xic_data/<sample_id>.json（每条 XIC：compound_id/ion_type/q1/q3/retention_time/intensity）
2. 按 (sample_id, compound_id, ion_type) 关联 label.csv（V5.0）：
   peak_label=1 → 正样本（peak_startN/peak_endN → bbox，最多 3 个）；
   peak_label=0 → 负样本（纯噪声，有图无框）
3. ROI 渲染复用 xic_extraction.render_roi_jpeg（400x300、无坐标轴，与推理管线像素级同款）。
   窗口策略（模拟峰宽中位约 0.77 min，固定 ±1 min 会截掉大量 double/triple 峰 2/3）：
   - 单峰：中心 = peak_position1（与 v1 标注 rt 居中一致），半宽 = max(1.0, 峰宽/2 + margin)
     （普通峰保持 ±1 min 与 v1 同款，宽峰整体可见）
   - 多峰：中心 = [首峰起点, 末峰终点] 中点，半宽 = max(1.0, 跨度/2 + margin)，保证全部标注峰可见
   - 噪声：中心 = 强度最高点（模拟"最可疑位置"），半宽固定 1.0
   窗口最后夹到数据 RT 范围内；每图实际窗口写入 roi_windows.csv 供审计/积分映射。
4. bbox 与 v1 同款：peak_startN/peak_endN 经窗口线性映射为像素 x（utils.roi_rt_mapping.rt_to_pixel_x），
   高度固定全高 [x1, 0, w, 300]；完全在窗口外的峰跳过，映射后宽 <1px 跳过。
5. 按"样本"整组划分 train/val（随机抽 N 个样本，同一化合物双通道必同侧，零泄漏）。

category_id=0 峰类、num_classes=1（DETR 约定 category_id 直接作类别索引，
no-object 占 num_classes 索引——若写成 1 会导致训练被教成"全输出无目标"）。

用法（model/ 目录下执行）：
  python -m preprocessing.coco_annotation_sim --config configs/coco_annotation_traindatav2.json

可加 --limit_samples N 只构建前 N 个样本（冒烟测试）、--force 忽略已有输出重新生成。
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.xic_extraction import render_roi_jpeg
from utils.roi_rt_mapping import (
    ROI_IMAGE_WIDTH_PX,
    ROI_IMAGE_HEIGHT_PX,
    rt_to_pixel_x,
)


def load_labels(label_csv):
    """读 V5.0 label.csv → {(sample_id, compound_id, ion_type): row}；空峰字段为 NaN。"""
    df = pd.read_csv(label_csv, dtype={"sample_id": str, "compound_id": str, "ion_type": str})
    dup = df.duplicated(subset=["sample_id", "compound_id", "ion_type"]).sum()
    if dup:
        raise ValueError(f"[ERROR] label.csv 主键重复 {dup} 行")
    table = {}
    for r in df.to_dict("records"):
        table[(r["sample_id"], r["compound_id"], r["ion_type"])] = r
    return table


def collect_peak_intervals(row):
    """V5.0 行 → [(start, end), ...]（按 RT 升序，最多 3 组）；空/非法区间剔除。"""
    intervals = []
    for k in (1, 2, 3):
        s, e = row.get(f"peak_start{k}"), row.get(f"peak_end{k}")
        if s is None or e is None or (isinstance(s, float) and np.isnan(s)) \
                or (isinstance(e, float) and np.isnan(e)):
            continue
        s, e = float(s), float(e)
        if e > s:
            intervals.append((s, e))
    return intervals


def decide_window(row, rt, inten, window_half_min, multi_peak_margin_min):
    """由标注行 + 序列决定 ROI 窗口 (rt_lo, rt_hi, center, center_source)。

    有峰：窗口覆盖全部标注峰；单峰中心=peak_position1（v1 语义），多峰中心=跨度中点。
    噪声：中心=强度最高点，半宽固定 window_half_min。
    """
    rt_min, rt_max = float(rt[0]), float(rt[-1])
    intervals = collect_peak_intervals(row)
    if int(row["peak_label"]) == 1 and intervals:
        starts = [s for s, _ in intervals]
        ends = [e for _, e in intervals]
        if len(intervals) == 1:
            center = float(row["peak_position1"])
            source = "peak_position1"
        else:
            center = (min(starts) + max(ends)) / 2.0
            source = "peak_span_mid"
        half = max(window_half_min, (max(ends) - min(starts)) / 2.0 + multi_peak_margin_min)
    else:
        center = float(rt[int(np.argmax(inten))])
        source = "intensity_argmax"
        half = window_half_min
    rt_lo = max(center - half, rt_min)
    rt_hi = min(center + half, rt_max)
    if rt_hi <= rt_lo:
        rt_lo, rt_hi = rt_min, rt_max
    return rt_lo, rt_hi, center, source


def build(args):
    sim_root = Path(args.sim_root).resolve()
    xic_dir = sim_root / "xic_data"
    label_csv = sim_root / "label" / "label.csv"
    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    for sub in ("train", "val"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    labels = load_labels(label_csv)
    print(f"[INFO] label.csv: {len(labels)} 行（主键 sample_id+compound_id+ion_type 唯一）")

    json_files = sorted(xic_dir.glob("*.json"))
    if args.limit_samples:
        json_files = json_files[: args.limit_samples]
    if not json_files:
        raise FileNotFoundError(f"[ERROR] {xic_dir} 下无 JSON")
    sample_ids = [p.stem for p in json_files]

    # 按"样本"整组抽 val（固定种子，双通道化合物两图必同侧）
    rng = np.random.default_rng(args.seed)
    n_val = min(args.val_samples, max(0, len(sample_ids) - 1))
    val_set = set(rng.choice(len(sample_ids), size=n_val, replace=False).tolist()) if n_val else set()
    val_samples = sorted(sample_ids[i] for i in val_set)
    print(f"[INFO] 划分: {len(sample_ids) - n_val} 样本 train / {n_val} 样本 val；"
          f"val 样本: {val_samples[:5]}{'...' if len(val_samples) > 5 else ''}")

    log_lines = [
        f"===== build @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====",
        f"sim_root={sim_root}  samples={len(sample_ids)}  seed={args.seed}  val_samples={val_samples}",
    ]

    coco = {"train": {"images": [], "annotations": []}, "val": {"images": [], "annotations": []}}
    windows_rows = []
    stats = {
        "images": 0, "positive": 0, "negative": 0, "annotations": 0,
        "skipped_peaks": 0, "thin_bbox_skipped": 0,
        "per_type": {},
    }
    t0 = time.time()

    for sample_idx, jp in enumerate(json_files):
        sample_id = jp.stem
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("sample_id") != sample_id:
            raise ValueError(f"[ERROR] {jp.name}: sample_id 不一致")
        split = "val" if sample_idx in val_set else "train"

        n_pos = n_neg = 0
        for seq, xic in enumerate(data["xics"], start=1):
            key = (sample_id, xic["compound_id"], xic["ion_type"])
            row = labels.get(key)
            if row is None:
                raise KeyError(f"[ERROR] {sample_id}/{xic['compound_id']}/{xic['ion_type']} 在 label.csv 无对应行")
            rt = np.asarray(xic["retention_time"], dtype=np.float64)
            inten = np.asarray(xic["intensity"], dtype=np.float64)

            file_name = (f"{sample_id}__{seq:03d}_mz{float(xic['q1']):.4f}"
                         f"_q3{float(xic['q3']):.4f}_{xic['compound_id']}_{xic['ion_type']}.jpeg")
            img_id = stats["images"] + 1
            coco[split]["images"].append({
                "id": img_id,
                "file_name": file_name,
                "width": int(ROI_IMAGE_WIDTH_PX),
                "height": int(ROI_IMAGE_HEIGHT_PX),
            })

            rt_lo, rt_hi, center, center_source = decide_window(
                row, rt, inten, args.window_half_min, args.multi_peak_margin_min)

            mask = (rt >= rt_lo) & (rt <= rt_hi)
            if mask.sum() >= 2:
                plot_rt, plot_inten = rt[mask], inten[mask]
            else:
                plot_rt, plot_inten = rt, inten
            render_roi_jpeg(plot_rt, plot_inten, rt_lo, rt_hi,
                            str(output_dir / split / file_name))
            stats["images"] += 1

            peak_type = str(row["peak_type"])
            stats["per_type"].setdefault(peak_type, {"images": 0, "annotations": 0})
            stats["per_type"][peak_type]["images"] += 1

            anns_n = 0
            if int(row["peak_label"]) == 1:
                for s, e in collect_peak_intervals(row):
                    if s > rt_hi + 1e-6 or e < rt_lo - 1e-6:
                        stats["skipped_peaks"] += 1
                        continue
                    x1 = rt_to_pixel_x(s, rt_lo, rt_hi)
                    x2 = rt_to_pixel_x(e, rt_lo, rt_hi)
                    if x2 < x1:
                        x1, x2 = x2, x1
                    w = x2 - x1
                    if w < 1.0:
                        stats["thin_bbox_skipped"] += 1
                        continue
                    coco[split]["annotations"].append({
                        "id": stats["annotations"] + 1,
                        "image_id": img_id,
                        "category_id": 0,
                        "bbox": [round(x1, 2), 0.0, round(w, 2), float(ROI_IMAGE_HEIGHT_PX)],
                        "area": round(w * ROI_IMAGE_HEIGHT_PX, 2),
                        "iscrowd": 0,
                    })
                    stats["annotations"] += 1
                    anns_n += 1
                stats["per_type"][peak_type]["annotations"] += anns_n
            if anns_n:
                stats["positive"] += 1
                n_pos += 1
            else:
                stats["negative"] += 1
                n_neg += 1

            windows_rows.append({
                "split": split, "sample_id": sample_id, "image": file_name,
                "compound_id": xic["compound_id"], "ion_type": xic["ion_type"],
                "peak_type": peak_type, "rt_lo": round(rt_lo, 4), "rt_hi": round(rt_hi, 4),
                "center": round(center, 4), "center_source": center_source,
                "n_annotations": anns_n,
            })

        msg = (f"[INFO] {sample_id}: {len(data['xics'])} 图 → {split}（有峰 {n_pos}，负样本 {n_neg}）")
        print(msg)
        log_lines.append(msg)

    # ===== 写 COCO json / roi_windows.csv / build_log / 构建报告 =====
    categories = [{"id": 0, "name": "peak", "supercategory": "chromatographic_peak"}]
    for split in ("train", "val"):
        obj = {
            "images": coco[split]["images"],
            "annotations": coco[split]["annotations"],
            "categories": categories,
        }
        name = "train_coco.json" if split == "train" else "val_coco.json"
        with open(output_dir / split / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"[DONE] {split}: {len(obj['images'])} 张图、{len(obj['annotations'])} 条标注 → "
              f"{output_dir / split / name}")

    pd.DataFrame(windows_rows).to_csv(output_dir / "roi_windows.csv",
                                      index=False, encoding="utf-8-sig")
    report = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sim_root": str(sim_root),
        "dataset_definition": "MRM-XIC V6.0 / Label V5.0",
        "samples": len(sample_ids),
        "val_samples": val_samples,
        "seed": args.seed,
        "window_policy": {
            "half_min_default": args.window_half_min,
            "multi_peak_margin_min": args.multi_peak_margin_min,
            "rule": "单峰中心=peak_position1；多峰中心=跨度中点且窗口覆盖全部标注峰；噪声中心=强度最高点",
        },
        "total_images": stats["images"],
        "positive_images": stats["positive"],
        "negative_images": stats["negative"],
        "annotations": stats["annotations"],
        "skipped_peaks_outside_window": stats["skipped_peaks"],
        "thin_bbox_skipped": stats["thin_bbox_skipped"],
        "per_peak_type": stats["per_type"],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(output_dir / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output_dir / "build_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"[DONE] 合计 {stats['images']} 图（有峰 {stats['positive']} / 负样本 {stats['negative']}）、"
          f"{stats['annotations']} 条标注；窗口外跳过峰 {stats['skipped_peaks']} 个；"
          f"耗时 {report['elapsed_seconds']}s → {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate COCO dataset from MRM-XIC simulated data (V6.0 JSON + V5.0 label.csv).")
    parser.add_argument("--sim_root", default=None, help="模拟数据集根目录（含 xic_data/、label/label.csv）")
    parser.add_argument("--output_dir", default=None, help="COCO 输出根目录")
    parser.add_argument("--val_samples", type=int, default=10,
                        help="随机划入 val 的样本数（按样本整组划分，默认 10）")
    parser.add_argument("--window_half_min", type=float, default=1.0,
                        help="基础 ROI 半宽（分钟），与推理管线 ±1 min 一致")
    parser.add_argument("--multi_peak_margin_min", type=float, default=0.15,
                        help="多峰/宽峰窗口在峰跨度外的余量（分钟）")
    parser.add_argument("--seed", type=int, default=61002, help="val 样本抽样种子")
    parser.add_argument("--limit_samples", type=int, default=0,
                        help="只构建前 N 个样本（0=全部；冒烟测试用）")
    parser.add_argument("--force", action="store_true", help="删除已有输出目录后重建")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径（作为默认参数，CLI 可覆盖）")

    _cfg_path = None
    for _i, _tok in enumerate(sys.argv[1:]):
        if _tok == "--config" and _i + 1 < len(sys.argv[1:]):
            _cfg_path = sys.argv[2 + _i]
            break
        if _tok.startswith("--config="):
            _cfg_path = _tok.split("=", 1)[1]
            break
    if _cfg_path:
        with open(_cfg_path, encoding="utf-8") as _f:
            _cfg = json.load(_f)
        _cfg.pop("config", None)
        _cfg = {_k: _v for _k, _v in _cfg.items() if not _k.startswith("_")}
        parser.set_defaults(**_cfg)
        print(f"[INFO] 已加载数据集构建配置: {_cfg_path}")
    args = parser.parse_args()

    if not args.sim_root:
        parser.error("--sim_root 必填（命令行或 --config 提供）")
    if not args.output_dir:
        parser.error("--output_dir 必填（命令行或 --config 提供）")
    build(args)


if __name__ == "__main__":
    main()
