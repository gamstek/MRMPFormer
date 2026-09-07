# -*- coding: utf-8 -*-
"""
FDR 逐层轨迹可视化（MRMPFormer v1）：展示 FDR 三层精化对峰边界的逐层改善。

证据链设计（防选择性展示）：
1. 对每个 ROI 前向，取同源 query 链：
     initial_edges_ltrb（初始粗框，无 FDR）
     refined_lr[0..2]  （FDR 层 1/2/3 精化后的左右边界，累积）
2. 用「L3 最终框 + 最终置信度」匹配人工标注 GT 峰（起止偏差 ≤ tol，贪婪降序）——
   命中后 L1/L2/L3/初始框取同一 query 通道的框 → 同源配对，杜绝"各自找最近峰"造成的伪改善。
3. 输出：
     layer_error_line.png   全量聚合折线（X=初始→L1→L2→L3；Y=左右边界 MAE + IoU）
     cases/*.png            代表案例叠加图（ROI 底图 + 初始/L1/L2/L3/GT 框）
     fdr_trace_report.md    汇总报告

用法（model/ 目录下）：
  python -m tools.evaluation.fdr_trace --mzml_dir ../data/mzml/test1 \
      --labels ../data/label/test1.xlsx --model checkpoint/mrmpformer.pth
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # model/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing.coco_annotation import (
    group_labels_by_sample,
    label_key,
    map_samples_to_mzmls,
    parse_labels_xlsx,
    parse_rt_field,
)
from utils.roi_rt_mapping import ROI_IMAGE_WIDTH_PX, rt_to_pixel_x
from utils.torch_device import resolve_torch_device, load_torch_checkpoint

# 与推理侧一致的图像预处理（ToTensor + Normalize，不 Resize）
_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 案例图框颜色/样式（Base=初始粗框 灰虚 / L1 橙 / L2 紫 / L3 红 / GT 绿）
_LAYER_STYLE = [
    ("initial", "初始框", "grey", "--", 1.0),
    ("L1", "FDR L1", "orange", "-", 1.4),
    ("L2", "FDR L2", "purple", "-", 1.6),
    ("L3", "FDR L3", "red", "-", 1.8),
]


def parse_gt_peaks(rec):
    """标注行 → [(start_min, end_min), ...]；peak_label=0 无 GT；兼容旧单数列。"""
    if str(rec.get("peak_label") or "").strip() == "0":
        return []
    peaks = []
    for k in (1, 2, 3):
        s = parse_rt_field(rec.get("peak_start%d" % k))
        e = parse_rt_field(rec.get("peak_end%d" % k))
        if s is not None and e is not None and e > s:
            peaks.append((s, e))
    if not peaks:
        s = parse_rt_field(rec.get("peak_start"))
        e = parse_rt_field(rec.get("peak_end"))
        if s is not None and e is not None and e > s:
            peaks.append((s, e))
    return peaks


def gen_rois(mzml_dir, labels_path, out_root, smooth_sigma):
    """调 inference.cli --mode roi 生成 ROI（与正式推理管线同一路径，B 范式标注驱动）。"""
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "inference.cli",
        "--mode", "roi",
        "--batch_dir", str(mzml_dir),
        "--labels", str(labels_path),
        "--output_dir", str(out_root),
        "--smooth_sigma", str(smooth_sigma),
    ]
    print("[INFO] ROI 生成:", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(ROOT))
    if ret.returncode != 0:
        raise RuntimeError("inference.cli roi 生成失败 (exit=%d)" % ret.returncode)


def load_model(model_path):
    """从 checkpoint（含训练 args）重建模型并加载权重。"""
    device = resolve_torch_device(verbose=False)
    ckpt = load_torch_checkpoint(model_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    train_args = ckpt.get("args")
    if train_args is None:
        raise ValueError("checkpoint 缺少 args，无法重建模型结构: %s" % model_path)
    train_args.device = str(device)
    from models import build_model
    if not getattr(train_args, "model", None):
        train_args.model = "mrmpformer_v1"
    result = build_model(train_args)
    model = result[0] if isinstance(result, tuple) else result
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    print("[INFO] 模型已加载: %s (%s) | device=%s" % (model_path, getattr(train_args, "model", "?"), device))
    return model, device


def box_px_to_min(x_px, rt_lo, rt_hi):
    """像素 x ∈[0,400] → 分钟（与 roi_rt_mapping.box_x_to_rt_minutes 一致）。"""
    x = float(np.clip(x_px, 0.0, ROI_IMAGE_WIDTH_PX))
    return rt_lo + (x / ROI_IMAGE_WIDTH_PX) * (rt_hi - rt_lo)


def interval_tiou(lo1, hi1, lo2, hi2):
    inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    union = max(hi1, hi2) - min(lo1, lo2)
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="FDR 逐层轨迹可视化（test1 数据集）")
    ap.add_argument("--mzml_dir", default="../data/mzml/test1", help="mzML 目录（含子目录递归）")
    ap.add_argument("--labels", default="../data/label/test1.xlsx", help="人工标注 xlsx")
    ap.add_argument("--model", default="checkpoint/mrmpformer.pth", help="模型权重（含训练 args）")
    ap.add_argument("--out_dir", default="../output/evaluation/fdr_trace", help="输出根目录")
    ap.add_argument("--threshold", type=float, default=0.9, help="最终层置信度阈值")
    ap.add_argument("--tol", type=float, default=0.1, help="起止命中容差（min），与多模型评估口径一致")
    ap.add_argument("--smooth_sigma", type=float, default=0.0, help="ROI 平滑 sigma（与训练一致 0）")
    ap.add_argument("--n_cases", type=int, default=6, help="案例图数量（改善最大 + 中位）")
    ap.add_argument("--reuse_roi", action="store_true",
                    help="复用 <out_dir>/_xic 已有 ROI（跳过 inference.cli roi 生成）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(exist_ok=True)
    xic_root = out_dir / "_xic"
    tol = args.tol

    # ========== 1. ROI 生成 ==========
    existing = list(xic_root.glob("*/roi_windows.csv")) if xic_root.is_dir() else []
    if args.reuse_roi and existing:
        print("[INFO] 复用已有 ROI: %d 个样品（--reuse_roi）" % len(existing))
    else:
        gen_rois(Path(args.mzml_dir).resolve(), Path(args.labels).resolve(), xic_root, args.smooth_sigma)

    # ========== 2. 模型 ==========
    model, device = load_model(args.model)

    # ========== 3. 标注 → 按样品分组 ==========
    labels = parse_labels_xlsx(args.labels)
    sample_order, groups = group_labels_by_sample(labels)

    # ========== 4. 逐样品前向 + 同源配对 ==========
    all_peaks = []  # 每个命中元素: {sample, image, nid, gt_lo, gt_hi, layer_boxes:[(lo_min,hi_min)], ...}
    stats = {"n_images": 0, "n_keep_queries": 0, "n_gt_peaks": 0, "n_hit": 0, "n_fp_queries": 0}
    case_cands = []  # (improve_abs_sum, rec)

    sample_dirs = sorted(p for p in xic_root.iterdir() if p.is_dir() and (p / "roi_windows.csv").is_file())
    stem2sample = map_samples_to_mzmls([p.name for p in sample_dirs], sample_order, None)
    print("[INFO] 样品数: %d（含 ROI 缓存）" % len(sample_dirs))

    for sdir in sample_dirs:
        stem = sdir.name
        sid = stem2sample.get(stem)
        if sid is None:
            print("[WARN] %s 无标注分组，跳过" % stem)
            continue
        # GT：native_id → [gt 峰列表]
        gt_by_key = {}
        for rec in groups[sid]:
            k = label_key(rec.get("compound"), rec.get("channel"))
            peaks = parse_gt_peaks(rec)
            if k and peaks:
                gt_by_key[k] = peaks
        if not gt_by_key:
            print("[WARN] %s 无正样本 GT 行" % stem)
            continue

        wins = pd.read_csv(sdir / "roi_windows.csv")
        feats = pd.read_csv(sdir / "feature.csv")
        win_by_img = {str(w["image"]): (float(w["rt_lo"]), float(w["rt_hi"])) for _, w in wins.iterrows()}
        nid_by_row = {}
        for i, f in feats.iterrows():
            nid_by_row[i + 1] = str(f["native_id"]).strip()

        images = sorted(p for p in sdir.glob("*.jpeg"))
        for img in images:
            stats["n_images"] += 1
            window = win_by_img.get(img.name)
            if window is None:
                continue
            rt_lo, rt_hi = window
            # image 前缀 N_mz → feature 行号（1-based）
            n = None
            try:
                n = int(img.stem.split("_", 1)[0])
            except ValueError:
                continue
            nid = nid_by_row.get(n)
            gt_peaks = gt_by_key.get(nid, []) if nid else []
            stats["n_gt_peaks"] += len(gt_peaks)

            with Image.open(img).convert("RGB") as im:
                tensor = _TRANSFORM(im).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
            pred_logits = outputs["pred_logits"]                    # [1,Q,2]
            probas = pred_logits.softmax(-1)[0, :, :1]              # [Q,1] 峰类
            # 同源链：初始框(左右) + refined_lr[0..2]（每层左右，归一化 [0,1]）
            init_lr = outputs["initial_edges_ltrb"][0]              # [Q,4] xyxy
            refined = [r[0] for r in outputs["refined_lr"]]         # 3×[Q,2] 左/右
            # L3 最终框（pred_boxes cxcywh → xyxy）用于匹配
            pb = outputs["pred_boxes"][0]                           # [Q,4] cxcywh 归一化
            boxes_xyxy = torch.stack([
                pb[:, 0] - pb[:, 2] / 2, pb[:, 1] - pb[:, 3] / 2,
                pb[:, 0] + pb[:, 2] / 2, pb[:, 1] + pb[:, 3] / 2], dim=-1)

            keep_order = probas.max(-1).values.argsort(descending=True).tolist()
            remaining = list(gt_peaks)
            for q in keep_order:
                sc = float(probas[q, 0])
                if sc <= args.threshold:
                    continue
                stats["n_keep_queries"] += 1
                # 本 query 各阶段框（分钟）
                x1, x2 = float(boxes_xyxy[q, 0]), float(boxes_xyxy[q, 2])
                lo3, hi3 = box_px_to_min(x1 * 400.0, rt_lo, rt_hi), box_px_to_min(x2 * 400.0, rt_lo, rt_hi)
                if hi3 <= lo3:
                    stats["n_fp_queries"] += 1
                    continue
                # GT 贪婪匹配（起止偏差均 ≤ tol）
                best_j, best_key = -1, None
                for j, (gs, ge) in enumerate(remaining):
                    if abs(lo3 - gs) <= tol + 1e-9 and abs(hi3 - ge) <= tol + 1e-9:
                        key = -(abs(lo3 - gs) + abs(hi3 - ge))
                        if best_key is None or key > best_key:
                            best_j, best_key = j, key
                if best_j < 0:
                    stats["n_fp_queries"] += 1
                    continue
                gs, ge = remaining.pop(best_j)
                stats["n_hit"] += 1

                layer_boxes = []
                # Base = 初始框（无 FDR）
                bx = (float(init_lr[q, 0]) * 400.0, float(init_lr[q, 2]) * 400.0)
                layer_boxes.append(("initial", box_px_to_min(bx[0], rt_lo, rt_hi),
                                    box_px_to_min(bx[1], rt_lo, rt_hi)))
                for k in range(3):
                    rx = (float(refined[k][q, 0]) * 400.0, float(refined[k][q, 1]) * 400.0)
                    layer_boxes.append(("L%d" % (k + 1),
                                        box_px_to_min(rx[0], rt_lo, rt_hi),
                                        box_px_to_min(rx[1], rt_lo, rt_hi)))
                rec = {
                    "sample": stem, "image": img, "nid": nid, "gt_lo": gs, "gt_hi": ge,
                    "score": sc, "layer_boxes": layer_boxes,
                }
                all_peaks.append(rec)

                # 案例候选：总边界误差改善 初始→L3
                def _abs_err(lo, hi):
                    return abs(lo - gs) + abs(hi - ge)
                err0 = _abs_err(layer_boxes[0][1], layer_boxes[0][2])
                err3 = _abs_err(layer_boxes[3][1], layer_boxes[3][2])
                case_cands.append((err0 - err3, rec))

    # ========== 5. 聚合统计 ==========
    def _layer_stats():
        rows = []
        for stage in ("initial", "L1", "L2", "L3"):
            lo_errs, hi_errs, ious = [], [], []
            for p in all_peaks:
                lb = next(x for x in p["layer_boxes"] if x[0] == stage)
                lo_errs.append(abs(lb[1] - p["gt_lo"]))
                hi_errs.append(abs(lb[2] - p["gt_hi"]))
                ious.append(interval_tiou(lb[1], lb[2], p["gt_lo"], p["gt_hi"]))
            rows.append({
                "stage": stage,
                "left_mae_min": float(np.mean(lo_errs)) if lo_errs else None,
                "right_mae_min": float(np.mean(hi_errs)) if hi_errs else None,
                "iou": float(np.mean(ious)) if ious else None,
                "n": len(lo_errs),
            })
        return pd.DataFrame(rows)

    df = _layer_stats()
    n_hit = len(all_peaks)
    print("[INFO] 命中峰(同源): %d | GT 总峰: %d | 保留 query: %d | FP query: %d"
          % (n_hit, stats["n_gt_peaks"], stats["n_keep_queries"], stats["n_fp_queries"]))
    print(df.to_string(index=False))

    # 逐峰逐层明细落盘（可复现/复核）
    trace_rows = []
    for p in all_peaks:
        row = {"sample": p["sample"], "image": p["image"].name, "native_id": p["nid"],
               "gt_lo": p["gt_lo"], "gt_hi": p["gt_hi"], "score": p["score"]}
        for stage, lo, hi in p["layer_boxes"]:
            row[stage + "_lo"] = lo
            row[stage + "_hi"] = hi
            row[stage + "_iou"] = interval_tiou(lo, hi, p["gt_lo"], p["gt_hi"])
        trace_rows.append(row)
    pd.DataFrame(trace_rows).to_csv(out_dir / "traces.csv", index=False, encoding="utf-8-sig")
    print("[INFO] 逐层明细: %s" % (out_dir / "traces.csv"))

    # 折线图（双轴：左=MAE min，右=IoU）
    fig, ax1 = plt.subplots(figsize=(7.5, 5.2))
    xs = np.arange(len(df))
    ax1.plot(xs, df["left_mae_min"], "o-", color="tab:blue", label="left-boundary MAE (min)")
    ax1.plot(xs, df["right_mae_min"], "s-", color="tab:cyan", label="right-boundary MAE (min)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{s}" for s in df["stage"]])
    ax1.set_ylabel("Boundary MAE (min)")
    ax1.set_xlabel("Refinement stage (initial -> FDR L1/L2/L3)")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, df["iou"], "d--", color="tab:green", label="tIoU")
    ax2.set_ylabel("tIoU vs GT")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_title("FDR layer-wise boundary refinement on test1 (n=%d peaks)" % n_hit)
    fig.tight_layout()
    line_path = out_dir / "layer_error_line.png"
    fig.savefig(line_path, dpi=150)
    plt.close(fig)

    # ========== 6. 案例图 ==========
    cands = sorted(case_cands, key=lambda c: -c[0])  # 改善最大在前
    picked = []
    if cands:
        picked.append(cands[0][1])
        if len(cands) > 1:
            picked.append(cands[1][1])
        mid = cands[len(cands) // 2][1]
        if mid not in picked:
            picked.append(mid)
        for c in cands[3:]:
            if len(picked) >= args.n_cases:
                break
            if c[1] not in picked:
                picked.append(c[1])
    case_files = []
    for i, rec in enumerate(picked, 1):
        img = rec["image"]
        with Image.open(img).convert("RGB") as im:
            w, h = im.size
            fig, ax = plt.subplots(figsize=(7.5, 3.4))
            ax.imshow(im)
            rt_lo, rt_hi = 0.0, 1.0
            # 画图用的像素窗口
            def _px(lo_min, hi_min, rt_l, rt_h):
                return rt_to_pixel_x(lo_min, rt_l, rt_h), rt_to_pixel_x(hi_min, rt_l, rt_h)
            # 从 roi_windows 取窗口做像素映射（GT 分钟 → 像素）
            wins = pd.read_csv(rec["image"].parent / "roi_windows.csv")
            wrow = wins[wins["image"] == img.name]
            if len(wrow):
                rt_lo, rt_hi = float(wrow.iloc[0]["rt_lo"]), float(wrow.iloc[0]["rt_hi"])
            # GT 区间
            gx1, gx2 = _px(rec["gt_lo"], rec["gt_hi"], rt_lo, rt_hi)
            ax.axvspan(gx1, gx2, color="green", alpha=0.25, label="GT %.3f-%.3f min" % (rec["gt_lo"], rec["gt_hi"]))
            for stage, label, color, ls, lw in _LAYER_STYLE:
                lb = next(x for x in rec["layer_boxes"] if x[0] == stage)
                px1, px2 = _px(lb[1], lb[2], rt_lo, rt_hi)
                ax.plot([px1, px2], [h * 0.06, h * 0.06], color=color, ls=ls, lw=lw,
                        label="%s (%.3f-%.3f)" % (label, lb[1], lb[2]))
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("%s | %s | score=%.2f" % (rec["sample"], rec["nid"], rec["score"]))
            ax.legend(loc="upper right", fontsize=7)
            fig.tight_layout()
            cp = cases_dir / ("case_%02d.png" % i)
            fig.savefig(cp, dpi=130)
            plt.close(fig)
            case_files.append(cp)

    # ========== 7. 报告 ==========
    def _fmt(v, f="{:.4f}"):
        return f.format(v) if v is not None else "N/A"

    L = []
    L.append("# FDR 逐层轨迹报告（test1）")
    L.append("")
    L.append("- 模型: %s" % args.model)
    L.append("- 数据: %s（%d 样品 %d 张 ROI 图）" % (args.mzml_dir, len(sample_dirs), stats["n_images"]))
    L.append("- 口径: 同源 query 配对（L3 框匹配 GT，起止偏差≤±%.2f min，score≥%.2f）" % (tol, args.threshold))
    L.append("")
    L.append("## 1. 检出概况")
    L.append("")
    L.append("- GT 峰总数: %d | L3 命中: %d（%.1f%%） | 未命中保留 query(FP): %d"
            % (stats["n_gt_peaks"], n_hit, 100.0 * n_hit / stats["n_gt_peaks"] if stats["n_gt_peaks"] else 0,
               stats["n_fp_queries"]))
    L.append("")
    L.append("## 2. 逐层聚合（同源 n=%d 峰）" % n_hit)
    L.append("")
    L.append("| 阶段 | 左边界MAE(min) | 右边界MAE(min) | tIoU |")
    L.append("|---|---|---|---|")
    for _, r in df.iterrows():
        L.append("| %s | %s | %s | %s |" % (r["stage"], _fmt(r["left_mae_min"]),
                                            _fmt(r["right_mae_min"]), _fmt(r["iou"])))
    L.append("")
    if n_hit and df.iloc[3]["left_mae_min"] is not None:
        im0 = df.iloc[0]["left_mae_min"] + df.iloc[0]["right_mae_min"]
        im3 = df.iloc[3]["left_mae_min"] + df.iloc[3]["right_mae_min"]
        L.append("- 总边界误差(左+右MAE): 初始 %.4f → L3 %.4f min（改善 %.1f%%）" % (im0, im3, 100 * (1 - im3 / im0)))
        L.append("- IoU: 初始 %.4f → L3 %.4f" % (df.iloc[0]["iou"], df.iloc[3]["iou"]))
    L.append("")
    L.append("## 3. 图表产物")
    L.append("")
    L.append("- 折线图: `%s`" % line_path)
    for cp in case_files:
        L.append("- 案例图: `%s`" % cp)
    L.append("")
    L.append("## 4. 说明")
    L.append("")
    L.append("- 同源配对保证逐层变化来自同一检测目标，非选择性展示；")
    L.append("- 案例图按『总边界误差改善 初始→L3 最大』排序自动选取，中位改善峰亦含。")
    L.append("- 关于『初始框』：它是 Decoder L1 的 bbox_embed 直接输出，仅作为 FDR 残差精化的锚点/尺度"
            "（s0=w0），在训练损失中不直接受边界监督——模型把边界精度完全交给 FDR 分布头。"
            "因此 initial→L1 的落差体现的是 FDR 的承载能力（无 FDR 时该框不可作为独立检测输出），"
            "L1→L2→L3 才是逐层 FDR 对同一初始锚点的残余精化曲线。")
    L.append("- 空白样品（负样本）ROI 无 GT 峰，不进本统计（与 evaluate_baseline 口径一致）；"
            "其对硫磷化合物因标注 RT 双离子不一致被 QC 剔除，与既往 test1 评测一致。")
    (out_dir / "fdr_trace_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("[DONE] 报告: %s" % (out_dir / "fdr_trace_report.md"))
    print("[DONE] 折线图: %s | 案例图: %d 张" % (line_path, len(case_files)))


if __name__ == "__main__":
    main()
