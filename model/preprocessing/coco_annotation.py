# -*- coding: utf-8 -*-
"""
从人工标注 xlsx + mzML 生成 COCO 格式训练数据集（QuanFormer/MRMPFormer 通用）。

流程：
1. 对每个 mzML 调用 preprocessing.xic_extraction.extract_xic_with_pyopenms 生成 ROI 图像
   （与推理管线完全一致：400x300、无坐标轴、apex±1min 窗口，roi_windows.csv 记录 RT 窗口）
2. 解析标注 xlsx（data/label/<trial>.xlsx 布局，纯标准库解析无需 openpyxl），
   按 native_id「{化合物名}-1/-2」匹配 ROI：-1=定量离子，-2=定性离子；失败回退组内行序
3. peak_start/peak_end（分钟）经 roi_windows 窗口线性映射为像素 x（utils.roi_rt_mapping）
   bbox 高度固定全高 [0, 300]（y 不参与 RT 映射，见 roi_rt_mapping.box_to_rt_range）
4. 按 mzML 文件分组划分 train/val，输出：
     {output_dir}/train/*.jpeg + train_coco.json
     {output_dir}/val/*.jpeg   + val_coco.json
   与 framework/datasets/coco.py 的读取约定一致（category_id=1 峰类，num_classes=1）

用法（model/ 目录下执行）：
  python -m preprocessing.coco_annotation \
      --mzmls ../data/mzml/20260715_shiyaoyuan_test/20260715_shiyaoyuan_test_1.mzML \
              ../data/mzml/traindata1/traindata1_1.mzML \
      --labels auto \
      --output_dir ../data/coco/merged \
      --val_stems 20260715_shiyaoyuan_test_2

说明：
  --labels 支持多个实验标注自动合并（多文件时 sample_id 加 "<实验名>__" 前缀隔离，
  避免跨实验同名样品混淆；RT 一致性 QC 按实验分别执行，不跨实验混检）；
  传 'auto' 或缺省时按 --mzmls 的 stem 前缀自动匹配 data/label/<实验>.xlsx。
  --mzmls 顺序须与合并后各实验样品的出现顺序一致（或用 --sample_map 显式映射）。

复用已有 XIC 输出：work_dir/<stem>/ 下已存在 feature.csv + roi_windows.csv 时默认跳过提取，
加 --force 重新提取。
"""
import argparse
import contextlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
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
# 真实文件（data/label/<trial>.xlsx）为多峰格式：peak_start1/2/3 + peak_end1/2/3 + peak_label/peak_count；
# 单数 peak_start/peak_end 保留以兼容旧文件。
_LABEL_COLS = {
    "roi_id": "roi_id",
    "path": "path",
    "comonent": "compound",
    "component": "compound",
    "channel": "channel",
    "rt": "rt",
    "ert": "ert",
    "peak_label": "peak_label",
    "peak_count": "peak_count",
    "peak_start": "peak_start",
    "peak_end": "peak_end",
    "peak_start1": "peak_start1",
    "peak_end1": "peak_end1",
    "peak_start2": "peak_start2",
    "peak_end2": "peak_end2",
    "peak_start3": "peak_start3",
    "peak_end3": "peak_end3",
    "area": "area",
    "area1": "area1",
    "area2": "area2",
    "area3": "area3",
    "snr": "snr",
    "instrument": "instrument",
    "raw_file": "raw_file",
    "sample_id": "sample_id",
    "product_id": "product_id",
}


def _peak_label_val(rec):
    """peak_label 字段 → int；缺失/空 → None。0=负样本，1=正样本，其余值（如 2）不入数据集。"""
    v = (rec.get("peak_label") if rec else None)
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def resolve_label_paths(labels_arg, mzml_stems):
    """解析标注文件列表：
    --labels 传多个路径 → 直接使用；
    传 'auto' 或缺省 → 按 --mzmls 的 stem 前缀自动匹配 data/label/<实验>.xlsx。"""
    if labels_arg and labels_arg != ["auto"]:
        return [Path(p) for p in labels_arg]
    label_dir = Path(__file__).resolve().parent.parent.parent / "data" / "label"
    found, seen = [], set()
    for stem in mzml_stems:
        for x in sorted(label_dir.glob("*.xlsx")):
            if stem.startswith(x.stem) and str(x) not in seen:
                seen.add(str(x))
                found.append(x)
    if not found:
        print("[ERROR] --labels 未指定且 data/label/ 未匹配到任何实验标注（按 mzML stem 前缀匹配 <实验>.xlsx）")
        sys.exit(1)
    return found


def merge_label_files(per_file_rows):
    """合并多实验标注行为一份（入参为 [(trial, rows)]，rows 为已解析且 QC 已标记的原始行）。

    多文件时给 sample_id 加 "<实验名>__" 前缀做命名空间隔离——仅当 sample_id 是旧式样品
    逻辑名（不同实验可能同名）；新式 sample_id 直接为 mzML 文件名（全局唯一），不加前缀，
    以便 map_samples_to_mzmls 按文件名精确匹配。
    RT 一致性 QC 由调用方按文件分别执行（跨实验色谱方法不同，不应混检），且必须在合并前
    于原始行上打 _qc_excluded 标记（此时 sample_id 尚未变换，与 QC 键一致）。
    单文件时直接沿用原行（行为与旧版一致）。
    返回合并后的行列表。
    """
    n_files = len(per_file_rows)
    if n_files == 1:
        return list(per_file_rows[0][1])
    merged = []
    for trial, rows in per_file_rows:
        for r in rows:
            r2 = dict(r)  # dict() 全量复制，含 _qc_excluded 等内部标记键（标记先于复制完成）
            sid = (r.get("sample_id") or "").strip()
            if sid and not sid.lower().endswith(".mzml"):
                r2["sample_id"] = f"{trial}__{sid}"
            merged.append(r2)
    return merged


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
    missing = [k for k in ("compound", "channel") if k not in col_map.values()]
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


def map_samples_to_mzmls(mzml_stems, sample_order, sample_map_str, mzml_names=None):
    """sample_id ↔ mzML stem 映射。三层优先：
    1) --sample_map（stem=sample_id,...）显式指定；
    2) sample_id 即 mzML 文件名（新标注格式，如 '20260715_shiyaoyuan_test_1.mzML'）→ 精确匹配；
    3) 旧格式（样品逻辑名）→ 按出现顺序对应。
    """
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

    # 2) 文件名式 sample_id：带或不带扩展名均可（含 .mzML 时按 stem 匹配）
    if sample_order:
        by_stem = {}
        for sid in sample_order:
            s = sid.strip()
            if s.lower().endswith(".mzml"):
                by_stem[s[: -len(".mzML")]] = sid
        hit = [s for s in mzml_stems if s in by_stem]
        if hit:
            mapping = {s: by_stem[s] for s in hit}
            miss = [s for s in mzml_stems if s not in mapping]
            if miss:
                raise ValueError(
                    f"[ERROR] sample_id 为 mzML 文件名但未覆盖全部 mzML: 缺 {miss}"
                    f"（已匹配 {hit}）"
                )
            return mapping

    # 3) 旧格式：顺序对应
    if len(sample_order) != len(mzml_stems):
        raise ValueError(
            f"[ERROR] xlsx 中 sample_id 数 ({len(sample_order)}) 与 mzML 数 ({len(mzml_stems)}) 不等"
            f"且未提供 --sample_map。sample_id={sample_order}, stems={mzml_stems}"
        )
    return dict(zip(mzml_stems, sample_order))


def build_coco_for_mzml(mzml_path, work_dir, labels_group, smooth_sigma, force=False, id_offset=0, exclude_tic=False):
    """提取（或复用）单 mzML 的 XIC 输出，生成 COCO images/annotations 条目。

    labels_group: 该 mzML 对应 sample_id 的 xlsx 行列表（保持行序）。
    id_offset: image_id 全局偏移（跨 mzML 保证唯一，pycocotools 要求 int id）。
    返回 (images, annotations, stats)：file_name 已带 {stem}__ 前缀。
    """
    stem = Path(mzml_path).stem
    out_dir = work_dir / stem
    feature_csv = out_dir / "feature.csv"
    windows_csv = out_dir / "roi_windows.csv"

    # B 范式（ROI 由标注驱动）+ peak_label 正负样本：
    # 剔除 _qc_excluded 行与 peak_label 不在 {0,1} 的行（如 2，不入数据集）；
    # peak_label=0 保留（生成 ROI 作负样本）、peak_label 缺失/空按正样本兼容旧文件
    labels_active = [
        rec for rec in labels_group
        if not rec.get("_qc_excluded")
        and _peak_label_val(rec) in (None, 0, 1)
    ]

    by_key = {}
    for i, rec in enumerate(labels_active):
        k = label_key(rec.get("compound"), rec.get("channel"))
        if k:
            by_key.setdefault(k, (i, rec))

    if force or not (feature_csv.exists() and windows_csv.exists()):
        out_dir.mkdir(parents=True, exist_ok=True)
        extract_xic_with_pyopenms(
            str(mzml_path), str(out_dir), smooth_sigma=smooth_sigma,
            labels=labels_active,
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
        elif i < len(labels_active):  # 回退：组内行序（xlsx 行序 = 色谱序，已验证 60/60）
            rec = labels_active[i]
            stats["fallback_order"] += 1

        img_id = id_offset + i + 1
        images.append({
            "id": img_id,
            "file_name": f"{stem}__{image_name}",
            "width": int(ROI_IMAGE_WIDTH_PX),
            "height": int(ROI_IMAGE_HEIGHT_PX),
        })

        rt_lo, rt_hi = float(wins.loc[i, "rt_lo"]), float(wins.loc[i, "rt_hi"])
        if rec is None:
            # B 范式下 ROI 均由标注行生成，理论不发生；防御性按负样本处理
            stats["negative"] += 1
            continue

        # 负样本：peak_label=0 → 有 ROI 图、无 bbox（训练识别"图上无峰"）
        if _peak_label_val(rec) == 0:
            stats["negative"] += 1
            continue

        # 正样本：多峰区间 peak_start1-3 / peak_end1-3 → 最多 3 个 bbox（兼容旧单数 peak_start/peak_end）
        intervals = []
        for k in (1, 2, 3):
            s = parse_rt_field(rec.get("peak_start%d" % k))
            e = parse_rt_field(rec.get("peak_end%d" % k))
            if s is not None and e is not None and e > s:
                intervals.append((s, e))
        if not intervals:
            s = parse_rt_field(rec.get("peak_start"))
            e = parse_rt_field(rec.get("peak_end"))
            if s is not None and e is not None and e > s:
                intervals.append((s, e))
        if not intervals:
            stats["negative"] += 1
            continue

        added = 0
        for rt_start, rt_end in intervals:
            if rt_start > rt_hi + 1e-6 or rt_end < rt_lo - 1e-6:
                continue  # 该峰区间完全在窗口外 → 跳过此峰（不整体降负）
            x1 = rt_to_pixel_x(rt_start, rt_lo, rt_hi)
            x2 = rt_to_pixel_x(rt_end, rt_lo, rt_hi)
            if x2 < x1:
                x1, x2 = x2, x1
            w = x2 - x1
            if w < 1.0:
                print(f"[WARN] {img_id} ({native_id}): 映射后 bbox 宽 {w:.2f}px < 1px，跳过该峰")
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
            added += 1
        if added:
            stats["labeled"] += 1
        else:
            stats["outside_window"] += 1
            stats["negative"] += 1
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
    parser.add_argument("--labels", nargs="+", default=None,
                        help="标注 xlsx 路径（可传多个实验文件自动合并；传 'auto' 或缺省则按 --mzmls stem 前缀"
                             "自动匹配 data/label/<实验>.xlsx）")
    parser.add_argument("--output_dir", required=True, help="COCO 数据集输出根目录")
    parser.add_argument("--val_stems", nargs="*", default=[],
                        help="划入 val 的 mzML stem 列表（按文件分组划分），其余进 train")
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="按 ROI 图像级随机划分 val 比例（0.3=7:3）；与 --val_stems 互斥，"
                             "同时给出时 --val_stems 优先。数据量小（如 4 个 mzML）时用此参数替代 --val_stems")
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

    stems = [Path(p).stem for p in args.mzmls]

    # ===== 多实验标注：解析 → QC 标记（原始行，sample_id 无前缀）→ 合并（加前缀）=====
    label_paths = resolve_label_paths(args.labels, stems)
    per_file = [(Path(p).stem, parse_labels_xlsx(str(p))) for p in label_paths]
    print(f"[INFO] 标注文件: {len(label_paths)} 个")
    for trial, rows in per_file:
        print(f"  - {trial}.xlsx: {len(rows)} 行")

    # ===== 标注 RT 一致性 QC（防线 1）：按实验分别执行，避免跨实验不同色谱方法混检 =====
    # _qc_excluded 打在合并前的原始行上（sample_id 尚无实验名前缀，与 exclude_keys 三元组一致），
    # 标记随行复制传导到合并结果，最终由 build_coco_for_mzml 的 labels_active 过滤 → 不生成 ROI
    n_excl = 0
    qc_rows = []
    qc_dir = None
    if args.qc_label_rt_tol > 0:
        from preprocessing.label_qc import (
            check_label_rt_consistency, mark_excluded_labels, write_qc_table,
        )
        for _trial, rows in per_file:
            _r, _k = check_label_rt_consistency(rows, tol=args.qc_label_rt_tol)
            qc_rows += _r
            n_excl += mark_excluded_labels(rows, _k)
        n_review = sum(1 for r in qc_rows if r.get("suggest_review"))
        print(f"[INFO] 标注 QC: 检查 {len(qc_rows)} 项，剔除 {n_excl} 行标注"
              f"（不生成 ROI），需人工复核 {n_review} 项")
        if qc_rows:
            # 目录名固定为 coco_<数据集名>（同数据集多次构建汇总/覆盖到同一处，便于对照）
            repo_root = Path(__file__).resolve().parent.parent.parent
            qc_dir = repo_root / "output" / "QC" / f"coco_{output_dir.name}"
            qc_path = qc_dir / "qc_label_rt.csv"
            n_written = write_qc_table(qc_rows, qc_path)
            print(f"[INFO] QC 结果表: {qc_path}（{n_written} 行）")

    labels = merge_label_files(per_file)
    print(f"[INFO] 合并后标注: {len(labels)} 行")

    sample_order, groups = group_labels_by_sample(labels)
    stem2sample = map_samples_to_mzmls(stems, sample_order, args.sample_map)

    # 提取阶段逐 ROI 的 [INFO] 噪声重定向到日志文件，终端只保留每样品摘要行
    log_path = output_dir / "build_log.txt"
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== build @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                 f"{len(stems)} mzML | QC tol {args.qc_label_rt_tol} =====\n")

    @contextlib.contextmanager
    def _stdout_to_log():
        f = open(log_path, "a", encoding="utf-8")
        old = sys.stdout
        sys.stdout = f
        try:
            yield f
        finally:
            sys.stdout = old
            f.close()

    all_entries = []  # (split, image_dict, [anns])
    id_offset = 0
    for mzml_path, stem in zip(args.mzmls, stems):
        print(f"[INFO] {stem} ↔ sample_id「{stem2sample[stem]}」({len(groups[stem2sample[stem]])} 行标注)")
        with _stdout_to_log():
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

    if not args.val_stems and args.val_ratio > 0:
        # 图像级随机划分（7:3 等）：
        # - 正样本：按化合物分层抽样——同一化合物多通道（定量/定性）尽量同侧，防近亲泄漏
        # - 负样本：图像级直接随机（负样本=无峰图，同化合物负样本间无近亲关系，可拆）；
        #   且正负各自独立按比例抽取，保证两侧正负构成一致（否则负样本可能全落一侧）
        import random

        rng = random.Random(20260820)

        def _compound_key(entry):
            cid = entry[1]["file_name"].split("__", 1)[1]
            return cid.rsplit("_", 1)[-1] if "_" in cid else cid

        n_val = 0
        # 正样本：化合物层
        pos_group = {}
        for entry in all_entries:
            if entry[2]:
                pos_group.setdefault(_compound_key(entry), []).append(entry)
        if pos_group:
            n_pos_total = sum(len(v) for v in pos_group.values())
            n_target = max(1, round(n_pos_total * args.val_ratio))
            picked, n_acc = set(), 0
            for key in sorted(pos_group, key=lambda k: rng.random()):
                if n_acc >= n_target:
                    break
                picked.add(key)
                n_acc += len(pos_group[key])
            for key in picked:
                for idx, entry in enumerate(all_entries):
                    if entry in pos_group[key]:
                        all_entries[idx] = ("val", entry[1], entry[2])
                        n_val += 1
        # 负样本：图像级直接随机
        neg_entries = [e for e in all_entries if not e[2]]
        if neg_entries:
            n_neg_total = len(neg_entries)
            n_target = max(1, round(n_neg_total * args.val_ratio))
            picked = set(rng.sample(range(n_neg_total), min(n_target, n_neg_total)))
            for i in picked:
                e = neg_entries[i]
                idx = all_entries.index(e)
                all_entries[idx] = ("val", e[1], e[2])
                n_val += 1
        n_total = len(all_entries)
        print(f"[INFO] val_ratio={args.val_ratio}: 正(化合物层)+负(图像级)独立抽 → val {n_val}/{n_total} 图"
              f"（正 {sum(1 for e in all_entries if e[0]=='val' and e[2])} / "
              f"负 {sum(1 for e in all_entries if e[0]=='val' and not e[2])}）")
    elif not args.val_stems:
        print("[WARN] 未指定 --val_stems / --val_ratio：全部图像进入 train，val 集为空")

    # ===== 训练侧 QC 阶段表（与推理侧 output/QC 同构）：qc_label_rt + qc_roi_channels + qc_summary =====
    if qc_dir is not None:
        qc_dir.mkdir(parents=True, exist_ok=True)
        n_review = sum(1 for r in qc_rows if r.get("suggest_review"))
        # ROI 通道级剔除汇总（各样品 pipeline_qc_excluded.csv 合并）
        n_roi_excl, reason_counts = 0, {}
        qc_frames = []
        for p in sorted(work_dir.glob("*/pipeline_qc_excluded.csv")):
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"[WARN] 读取 QC 剔除表失败: {p}: {e}")
                continue
            df.insert(0, "stem", p.parent.name)
            qc_frames.append(df)
        if qc_frames:
            merged = pd.concat(qc_frames, ignore_index=True)
            merged.to_csv(qc_dir / "qc_roi_channels.csv", index=False, encoding="utf-8-sig")
            n_roi_excl = len(merged)
            reason_counts = merged["reason"].value_counts().to_dict()
        # qc_summary.md：各环节统计 + 人工复核清单（实验报告要求：需人工复核的必须成表）
        lines = [
            "# QC 汇总（数据集构建）— %s" % output_dir.name,
            "",
            "- 生成时间: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "- 标注文件: %s" % ", ".join(t for t, _ in per_file),
            "",
            "## 1. 标注 RT 一致性（qc_label_rt.csv）",
            "- 检查项: %d" % len(qc_rows),
            "- 剔除标注行: %d（不生成 ROI）" % n_excl,
            "- 需人工复核: %d" % n_review,
            "",
            "## 2. ROI 通道级剔除（qc_roi_channels.csv）",
            "- 剔除条目: %d 行" % n_roi_excl,
        ]
        if reason_counts:
            lines.append("- reason 分布: " + ", ".join(f"{k}={v}" for k, v in reason_counts.items()))
        lines += ["", "## 3. 人工复核清单（标注 RT 一致性，suggest_review=true）",
                  "| 检查类型 | 样品 | 化合物 | 通道 | RT(min) | 组中位 | 极差(min) | 动作 |",
                  "|---|---|---|---|---|---|---|---|"]
        review_rows = [r for r in qc_rows if r.get("suggest_review")]
        for r in review_rows:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r.get("check_type", ""), r.get("sample_id", ""), r.get("compound", ""),
                r.get("channel", ""), r.get("rt", ""), r.get("group_median", ""),
                r.get("rt_range", ""), r.get("action", "")))
        if not review_rows:
            lines.append("| （无） | | | | | | | |")
        (qc_dir / "qc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[INFO] QC 阶段表: {qc_dir}（qc_label_rt.csv / qc_roi_channels.csv / qc_summary.md）")

    for split, json_name in (("train", "train_coco.json"), ("val", "val_coco.json")):
        entries = [(img, anns) for s, img, anns in all_entries if s == split]
        n_img, n_ann, json_path = write_split(
            split, entries, output_dir, json_name, args.include_unlabeled)
        print(f"[DONE] {split}: {n_img} 张图（含负样本）、{n_ann} 条标注 → {json_path}")


if __name__ == "__main__":
    main()
