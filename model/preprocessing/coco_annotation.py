# -*- coding: utf-8 -*-
"""
从人工标注 xlsx + mzML 生成 COCO 格式训练数据集（QuanFormer/MRMPFormer 通用）。

流程：
1. 对每个 mzML 调用 preprocessing.xic_extraction.extract_xic_with_pyopenms 生成 ROI 图像
   （与推理管线完全一致：400x300、无坐标轴、apex±1min 窗口，roi_windows.csv 记录 RT 窗口）
2. 解析标注 xlsx（testcase_data.xlsx 布局，纯标准库解析无需 openpyxl），
   按 native_id「{化合物名}-1/-2」匹配 ROI：-1=定量离子，-2=定性离子；失败回退组内行序
3. peak_start/peak_end（分钟）经 roi_windows 窗口线性映射为像素 x（utils.roi_rt_mapping）
   bbox 高度固定全高 [0, 300]（y 不参与 RT 映射，见 roi_rt_mapping.box_to_rt_range）
4. 按 mzML 文件分组划分 train/val，输出：
     {output_dir}/train/*.jpeg + train_coco.json
     {output_dir}/val/*.jpeg   + val_coco.json
   与 framework/datasets/coco.py 的读取约定一致（category_id=1 峰类，num_classes=1）

用法（model/ 目录下执行）：
  python -m preprocessing.coco_annotation \
      --mzmls ../data/test/mzml/20260715_shiyaoyuan_test_1.mzML \
              ../data/test/mzml/20260715_shiyaoyuan_test_2.mzML \
      --labels ../data/test/label/testcase_data.xlsx \
      --output_dir ../data/test/coco \
      --val_stems 20260715_shiyaoyuan_test_2

复用已有 XIC 输出：work_dir/<stem>/ 下已存在 feature.csv + roi_windows.csv 时默认跳过提取，
加 --force 重新提取。
"""
import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

from preprocessing.xic_extraction import extract_xic_with_pyopenms
from utils.roi_rt_mapping import (
    ROI_IMAGE_WIDTH_PX,
    ROI_IMAGE_HEIGHT_PX,
    rt_to_pixel_x,
)

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# 标注 xlsx 期望列名 -> 语义（comonent 为原始表头拼写错误，保留兼容）
_LABEL_COLS = {
    "roi_id": "roi_id",
    "comonent": "compound",
    "component": "compound",
    "channel": "channel",
    "rt": "rt",
    "ert": "ert",
    "peak_start": "peak_start",
    "peak_end": "peak_end",
    "area": "area",
    "snr": "snr",
    "sample_id": "sample_id",
}


def _col_letter(ref: str) -> str:
    return "".join(ch for ch in ref if ch.isalpha())


def parse_labels_xlsx(xlsx_path):
    """纯标准库解析标注 xlsx sheet1（按列字母定位，天然免疫稀疏空单元格错位）。

    返回 list[dict]，键：roi_id/compound/channel/rt/ert/peak_start/peak_end/sample_id。
    """
    z = zipfile.ZipFile(xlsx_path)
    shared = [
        "".join(t.text or "" for t in si.iter(_NS + "t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml"))
    ]
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    matrix = []  # 每行 {col_letter: value_str}
    for row in sheet.iter(_NS + "row"):
        vals = {}
        for c in row.iter(_NS + "c"):
            v = c.find(_NS + "v")
            if v is None:
                continue
            vals[_col_letter(c.get("r", ""))] = (
                shared[int(v.text)] if c.get("t") == "s" else v.text
            )
        matrix.append(vals)
    if not matrix:
        return []

    header = {col: (name or "").strip() for col, name in matrix[0].items()}
    col_map = {}  # col_letter -> 语义键
    for col, raw in header.items():
        if raw in _LABEL_COLS:
            col_map[col] = _LABEL_COLS[raw]
    missing = [k for k in ("compound", "channel", "peak_start", "peak_end") if k not in col_map.values()]
    if missing:
        raise ValueError(f"[ERROR] 标注 xlsx 缺少必需列: {missing}（表头: {header}）")

    rows = []
    for vals in matrix[1:]:
        rec = {sem: (vals.get(col) or "").strip() for col, sem in col_map.items()}
        if not any(rec.values()):
            continue
        rows.append(rec)
    return rows


def parse_rt_field(s):
    """解析 '16.428(0.000)' / '16.428' → 16.428（分钟）；空/非法 → None。"""
    s = (s or "").strip()
    if not s:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else None


def label_key(compound, channel):
    """xlsx 行 → mzML native_id 键：定量离子→-1，定性离子→-2。"""
    ch = (channel or "").strip()
    suffix = "1" if "定量" in ch else ("2" if "定性" in ch else None)
    if suffix is None:
        return None
    return f"{(compound or '').strip()}-{suffix}"


def group_labels_by_sample(labels):
    """按 sample_id 分组并保留 xlsx 行序；返回 (ordered_sample_ids, {sample_id: [rows]})。"""
    order, groups = [], {}
    for rec in labels:
        sid = rec.get("sample_id") or ""
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(rec)
    return order, groups


def map_samples_to_mzmls(mzml_stems, sample_order, sample_map_str):
    """sample_id ↔ mzML stem 映射。优先 --sample_map（stem=sample_id,...），否则按两者出现顺序。"""
    mapping = {}
    if sample_map_str:
        for pair in sample_map_str.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            stem, sid = (x.strip() for x in pair.split("=", 1))
            mapping[stem] = sid
        unknown = [s for s in mzml_stems if s not in mapping]
        if unknown:
            raise ValueError(f"[ERROR] --sample_map 未覆盖 mzML: {unknown}")
        return mapping
    if len(sample_order) != len(mzml_stems):
        raise ValueError(
            f"[ERROR] xlsx 中 sample_id 数 ({len(sample_order)}) 与 mzML 数 ({len(mzml_stems)}) 不等"
            f"且未提供 --sample_map。sample_id={sample_order}, stems={mzml_stems}"
        )
    return dict(zip(mzml_stems, sample_order))


def build_coco_for_mzml(mzml_path, work_dir, labels_group, smooth_sigma, force=False, id_offset=0):
    """提取（或复用）单 mzML 的 XIC 输出，生成 COCO images/annotations 条目。

    labels_group: 该 mzML 对应 sample_id 的 xlsx 行列表（保持行序）。
    id_offset: image_id 全局偏移（跨 mzML 保证唯一，pycocotools 要求 int id）。
    返回 (images, annotations, stats)：file_name 已带 {stem}__ 前缀。
    """
    stem = Path(mzml_path).stem
    out_dir = work_dir / stem
    feature_csv = out_dir / "feature.csv"
    windows_csv = out_dir / "roi_windows.csv"

    by_key = {}
    for i, rec in enumerate(labels_group):
        k = label_key(rec.get("compound"), rec.get("channel"))
        if k:
            by_key.setdefault(k, (i, rec))

    # ROI 窗口中心覆盖表（improve.md 第 7 项）：以标注 rt 列（人工审核 RT）为窗口中心，
    # 替代默认的最高强度点 —— 训练图窗口与 bbox 标注严格同源
    rt_overrides = {}
    for k, (_i, rec) in by_key.items():
        rt_val = parse_rt_field(rec.get("rt"))
        if rt_val is not None:
            rt_overrides[k] = rt_val

    if force or not (feature_csv.exists() and windows_csv.exists()):
        out_dir.mkdir(parents=True, exist_ok=True)
        extract_xic_with_pyopenms(
            str(mzml_path), str(out_dir), smooth_sigma=smooth_sigma,
            rt_center_overrides=rt_overrides,
            exclude_tic=exclude_tic,
        )

    feats = pd.read_csv(feature_csv)
    wins = pd.read_csv(windows_csv)
    if len(feats) != len(wins):
        raise ValueError(
            f"[ERROR] {stem}: feature.csv 行数 {len(feats)} != roi_windows.csv 行数 {len(wins)}"
        )

    images, annotations = [], []
    stats = {"stem": stem, "n_roi": len(feats), "labeled": 0, "negative": 0,
             "outside_window": 0, "fallback_order": 0}
    ann_id = 0
    for i in range(len(feats)):
        native_id = str(feats.loc[i, "native_id"]) if "native_id" in feats.columns else ""
        image_name = str(wins.loc[i, "image"])
        rec = None
        if native_id and native_id in by_key:
            rec = by_key[native_id][1]
        elif i < len(labels_group):  # 回退：组内行序（xlsx 行序 = 色谱序，已验证 60/60）
            cand = labels_group[i]
            if not cand.get("_qc_excluded"):
                rec = cand
                stats["fallback_order"] += 1

        img_id = id_offset + i + 1
        images.append({
            "id": img_id,
            "file_name": f"{stem}__{image_name}",
            "width": int(ROI_IMAGE_WIDTH_PX),
            "height": int(ROI_IMAGE_HEIGHT_PX),
        })

        rt_lo, rt_hi = float(wins.loc[i, "rt_lo"]), float(wins.loc[i, "rt_hi"])
        rt_start = parse_rt_field(rec.get("peak_start")) if rec else None
        rt_end = parse_rt_field(rec.get("peak_end")) if rec else None

        if rt_start is None or rt_end is None:
            stats["negative"] += 1
            continue
        if rt_start > rt_hi + 1e-6 or rt_end < rt_lo - 1e-6:
            print(f"[WARN] {img_id} ({native_id}): 标注 RT [{rt_start}, {rt_end}] "
                  f"完全在 ROI 窗口 [{rt_lo:.3f}, {rt_hi:.3f}] 外，按负样本处理")
            stats["outside_window"] += 1
            stats["negative"] += 1
            continue

        x1 = rt_to_pixel_x(rt_start, rt_lo, rt_hi)
        x2 = rt_to_pixel_x(rt_end, rt_lo, rt_hi)
        if x2 < x1:
            x1, x2 = x2, x1
        w = x2 - x1
        if w < 1.0:
            print(f"[WARN] {img_id} ({native_id}): 映射后 bbox 宽 {w:.2f}px < 1px，按负样本处理")
            stats["negative"] += 1
            continue
        ann_id += 1
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [round(x1, 2), 0.0, round(w, 2), float(ROI_IMAGE_HEIGHT_PX)],
            "area": round(w * ROI_IMAGE_HEIGHT_PX, 2),
            "iscrowd": 0,
        })
        stats["labeled"] += 1
    return images, annotations, stats


def write_split(split, entries, output_dir, json_name, include_unlabeled=True):
    """entries: list[(image_dict, [annotation, ...])]。写图像副本 + COCO json。"""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 0
    for img, anns in entries:
        if not anns and not include_unlabeled:
            continue
        src = img["_src_path"]
        shutil.copy2(src, split_dir / img["file_name"])
        images.append({k: v for k, v in img.items() if not k.startswith("_")})
        for a in anns:
            ann_id += 1
            a = dict(a, id=ann_id)
            annotations.append(a)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "peak", "supercategory": "chromatographic_peak"}],
    }
    json_path = split_dir / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    return len(images), len(annotations), json_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate COCO-format dataset (train/val) from labeled xlsx + mzML files."
    )
    parser.add_argument("--mzmls", nargs="+", required=True, help="mzML 文件路径列表")
    parser.add_argument("--labels", required=True, help="标注 xlsx（testcase_data.xlsx 布局）")
    parser.add_argument("--output_dir", required=True, help="COCO 数据集输出根目录")
    parser.add_argument("--val_stems", nargs="*", default=[],
                        help="划入 val 的 mzML stem 列表（按文件分组划分），其余进 train")
    parser.add_argument("--smooth_sigma", type=float, default=0.0,
                        help="XIC 高斯平滑 sigma（与推理保持一致，默认 0）")
    parser.add_argument("--work_dir", default=None,
                        help="XIC 中间输出目录（默认 <output_dir>/_xic，已存在则复用）")
    parser.add_argument("--force", action="store_true", help="强制重新提取 XIC（忽略已有输出）")
    parser.add_argument("--sample_map", default=None,
                        help="显式指定 stem=sample_id 映射（逗号分隔）；缺省按出现顺序对应")
    parser.add_argument("--no_include_unlabeled", dest="include_unlabeled",
                        action="store_false",
                        help="不把无标注 ROI（TIC/匹配失败等）作为负样本纳入数据集")
    parser.add_argument("--qc_label_rt_tol", type=float, default=1.0,
                        help="标注 RT 一致性 QC 阈值（min）：跨样品/双离子 rt 极差超此值判疑似实验有误，"
                             "剔除涉事行（不进 bbox）并警示人工复核；0=关闭。默认 1.0")
    parser.add_argument("--exclude_tic", action="store_true", default=True,
                        help="不生成 TIC 等无 (Q1,Q3) 数值通道的 ROI（默认开启；--no_exclude_tic 关闭）")
    parser.add_argument("--no_exclude_tic", dest="exclude_tic", action="store_false",
                        help="保留 TIC 通道 ROI（作为负样本）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output_dir / "_xic"
    work_dir.mkdir(parents=True, exist_ok=True)

    labels = parse_labels_xlsx(args.labels)
    print(f"[INFO] 标注 xlsx: {len(labels)} 行")

    # ===== 标注 RT 一致性 QC（防线 1，详见 docs/plan_qc.md）=====
    if args.qc_label_rt_tol > 0:
        from preprocessing.label_qc import (
            check_label_rt_consistency, mark_excluded_labels, write_qc_table,
        )
        qc_rows, exclude_keys = check_label_rt_consistency(labels, tol=args.qc_label_rt_tol)
        n_excl = mark_excluded_labels(labels, exclude_keys)
        n_review = sum(1 for r in qc_rows if r.get("suggest_review"))
        print(f"[INFO] 标注 QC: 检查 {len(qc_rows)} 项，剔除 {n_excl} 行标注"
              f"（对应 ROI 降级负样本），需人工复核 {n_review} 项")
        if qc_rows:
            from datetime import datetime
            repo_root = Path(__file__).resolve().parent.parent.parent
            exp_name = output_dir.parent.name or "exp"
            qc_dir = repo_root / "output" / "QC" / f"coco_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            qc_path = qc_dir / "qc_label_rt.csv"
            n_written = write_qc_table(qc_rows, qc_path)
            print(f"[INFO] QC 结果表: {qc_path}（{n_written} 行）")

    sample_order, groups = group_labels_by_sample(labels)
    stems = [Path(p).stem for p in args.mzmls]
    stem2sample = map_samples_to_mzmls(stems, sample_order, args.sample_map)
    for stem in stems:
        print(f"[INFO] {stem} ↔ sample_id「{stem2sample[stem]}」({len(groups[stem2sample[stem]])} 行标注)")

    all_entries = []  # (split, image_dict, [anns])
    id_offset = 0
    for mzml_path, stem in zip(args.mzmls, stems):
        images, annotations, stats = build_coco_for_mzml(
            mzml_path, work_dir, groups[stem2sample[stem]],
            smooth_sigma=args.smooth_sigma, force=args.force, id_offset=id_offset,
            exclude_tic=args.exclude_tic,
        )
        id_offset += stats["n_roi"]
        anns_by_img = {}
        for a in annotations:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        xic_dir = work_dir / stem
        for img in images:
            img["_src_path"] = str(xic_dir / img["file_name"].split("__", 1)[1])
        split = "val" if stem in args.val_stems else "train"
        for img in images:
            all_entries.append((split, img, anns_by_img.get(img["id"], [])))
        print(f"[INFO] {stem}: ROI {stats['n_roi']}, 有标注 {stats['labeled']}, "
              f"负样本 {stats['negative']}(窗口外 {stats['outside_window']}, "
              f"行序回退匹配 {stats['fallback_order']})")

    if not args.val_stems:
        print("[WARN] 未指定 --val_stems：全部图像进入 train，val 集为空")
    for split, json_name in (("train", "train_coco.json"), ("val", "val_coco.json")):
        entries = [(img, anns) for s, img, anns in all_entries if s == split]
        n_img, n_ann, json_path = write_split(
            split, entries, output_dir, json_name, args.include_unlabeled)
        print(f"[DONE] {split}: {n_img} 张图（含负样本）、{n_ann} 条标注 → {json_path}")


if __name__ == "__main__":
    main()
