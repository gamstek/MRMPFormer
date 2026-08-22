# -*- coding: utf-8 -*-
"""
人工标注框 vs 模型预测框 对照可视化。

一张图两层对照，让偏差可直接目视 + 读数：
  上排：ROI 原图（模型实际看到的 400x300 图）叠框——绿色=人工标注框（GT），
        红色=预测框（半透明填充便于看重叠程度），框顶标注各自 RT 区间；
  下排：该通道真实 XIC 曲线（xic_matrix）叠同样区间——GT 绿色竖线带、Pred 红色虚线，
        带分钟刻度，偏差一目了然；
  标题：化合物/native_id + 匹配结果（TP/FP/FN）+ 起止偏差(min)。

匹配口径与 tools.evaluation.evaluate_baseline 完全一致（复用其 _parse_gt_peaks/
match_image），评估报告里是什么结果，图上就是什么结果。

输入布局（evaluate_baseline --run_inference 1 的 _pipeline 产物）：
  <pipeline_dir>/xic-roi-batch/<stem>/{*.jpeg, feature.csv, roi_windows.csv, xic_matrix.npy}
  <pipeline_dir>/batch_predictions/<stem>/prediction.csv

用法（model/ 目录下，系统 PowerShell 执行）：
  python -m tools.visualization.plot_gt_vs_pred \
      --pipeline_dir ../output/evaluation/quanformerv3/_pipeline/20260715_shiyaoyuan_test_1 \
      --labels ../data/label/20260715_shiyaoyuan_test.xlsx \
      --output_dir ../output/test/gt_vs_pred/test_1

输出：
  <output_dir>/TP|FP|FN/<图名>.png   按匹配结果分类的对照图
  <output_dir>/index.csv             逐图汇总（区间 + 偏差 + score）
  <output_dir>/summary.txt           计数汇总
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from preprocessing.coco_annotation import (  # noqa: E402
    group_labels_by_sample,
    label_key,
    map_samples_to_mzmls,
    parse_labels_xlsx,
)
from tools.evaluation.evaluate_baseline import (  # noqa: E402
    _load_pred_rows,
    _parse_gt_peaks,
    match_image,
)
from utils.roi_rt_mapping import rt_to_pixel_x  # noqa: E402

IMG_W, IMG_H = 400, 300


def _cjk_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_xic(xic_dir: Path, row_idx: int, smooth_sigma: float):
    """读 xic_matrix.npy 的第 row_idx 行（0-based）→ (rt, y)，失败返回 None。"""
    p = xic_dir / "xic_matrix.npy"
    if not p.is_file():
        return None
    x = np.load(str(p))
    rt = x[0, :].astype(np.float64)
    if np.nanmax(rt) > 200:
        rt = rt / 60.0
    if row_idx >= x.shape[0] - 1:
        return None
    y = x[1 + row_idx, :].astype(np.float64)
    if smooth_sigma > 0:
        y = gaussian_filter1d(y, sigma=smooth_sigma)
    return rt, y


def _fmt(v, nd=3):
    return ("%.*f" % (nd, v)) if v is not None and np.isfinite(v) else "-"


def plot_one(img_path: Path, out_path: Path, title: str, xic_xy, rt_window,
             gts_px, preds_px, devs):
    """画单张对照图。

    gts_px: [(x1, x2, label)] 人工框（像素 x）
    preds_px: [(x1, x2, label)] 预测框（像素 x）
    devs: [(dev_start, dev_end)] 与 gts_px 对齐的起止偏差（min）；未配对为 None
    xic_xy: (rt, y) 或 None
    rt_window: (rt_lo, rt_hi) ROI 窗口（min）
    """
    _cjk_font()
    from PIL import Image
    img = Image.open(str(img_path)).convert("RGB")

    has_xic = xic_xy is not None
    fig, axes = plt.subplots(
        2 if has_xic else 1, 1, figsize=(10, 6.5) if has_xic else (10, 3.8),
        gridspec_kw={"height_ratios": [3, 2]} if has_xic else None)
    ax_img = axes[0] if has_xic else axes

    # ---- 上排：ROI 图 + 框 ----
    ax_img.imshow(img)
    for (x1, x2, lbl), dv in zip(gts_px, devs):
        ax_img.add_patch(plt.Rectangle(
            (x1, 2), max(x2 - x1, 1.5), IMG_H - 4, fill=False,
            edgecolor="lime", linewidth=2))
        txt = lbl if dv is None else lbl + "  Δ%s/%s" % (_fmt(dv[0], 2), _fmt(dv[1], 2))
        ax_img.text(x1, 12, txt, color="lime", fontsize=7.5, va="top", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55))
    for (x1, x2, lbl) in preds_px:
        ax_img.add_patch(plt.Rectangle(
            (x1, 20), max(x2 - x1, 1.5), IMG_H - 40, fill=True,
            facecolor="red", alpha=0.18, edgecolor="red", linewidth=2))
        ax_img.text(x1, IMG_H - 28, lbl, color="red", fontsize=7.5, va="top", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55))
    ax_img.set_xlim(0, IMG_W)
    ax_img.set_ylim(IMG_H, 0)
    ax_img.set_title(title, fontsize=9)
    ax_img.set_ylabel("ROI 图（绿=人工 红=预测）")

    # ---- 下排：真实 XIC + 同区间竖线（分钟轴） ----
    if has_xic:
        rt, y = xic_xy
        rt_lo, rt_hi = rt_window
        pad = max(0.3, (rt_hi - rt_lo) * 0.12)
        m = (rt >= rt_lo - pad) & (rt <= rt_hi + pad)
        ax_x = axes[1]
        ax_x.plot(rt[m], y[m], color="steelblue", linewidth=0.9)
        for (x1, x2, lbl), _dv in zip(gts_px, devs):
            gs = rt_lo + (rt_hi - rt_lo) * x1 / IMG_W
            ge = rt_lo + (rt_hi - rt_lo) * x2 / IMG_W
            ax_x.axvspan(gs, ge, color="lime", alpha=0.12)
            ax_x.axvline(gs, color="lime", lw=1.2)
            ax_x.axvline(ge, color="lime", lw=1.2)
        for (x1, x2, _lbl) in preds_px:
            ps = rt_lo + (rt_hi - rt_lo) * x1 / IMG_W
            pe = rt_lo + (rt_hi - rt_lo) * x2 / IMG_W
            ax_x.axvline(ps, color="red", ls="--", lw=1.2)
            ax_x.axvline(pe, color="red", ls="--", lw=1.2)
        ax_x.set_xlim(rt_lo - pad, rt_hi + pad)
        ax_x.set_xlabel("RT (min)   绿实线=人工区间，红虚线=预测区间")
        ax_x.set_ylabel("Intensity")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="人工标注 vs 预测框 对照可视化（TP/FP/FN 分类输出）")
    ap.add_argument("--pipeline_dir", required=True,
                    help="evaluate_baseline _pipeline/<stem> 目录（含 xic-roi-batch/ 与 batch_predictions/）")
    ap.add_argument("--labels", required=True, help="人工标注 xlsx")
    ap.add_argument("--output_dir", required=True, help="输出目录")
    ap.add_argument("--threshold", type=float, default=0.9, help="预测框纳入的最低置信度")
    ap.add_argument("--tolerance", type=float, default=0.1, help="命中容差（min，与评估一致）")
    ap.add_argument("--quant_tolerance", type=float, default=0.2, help="宽松配对容差（min，图上标 loose）")
    ap.add_argument("--smooth_sigma", type=float, default=0.8, help="XIC 平滑（与管线一致）")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="处理 pipeline_dir 下指定 stem（缺省=自动发现全部）")
    args = ap.parse_args()

    pipeline_dir = Path(args.pipeline_dir)
    roi_root = pipeline_dir / "xic-roi-batch"
    pred_root = pipeline_dir / "batch_predictions"
    if not roi_root.is_dir():
        print("[ERROR] 未找到 %s（需为 _pipeline/<stem> 或含 xic-roi-batch 的目录）" % roi_root)
        sys.exit(1)

    stems = args.stems or sorted(p.name for p in roi_root.iterdir() if p.is_dir())
    # 多 stem 时标注映射（与 evaluate_baseline 相同规则）
    labels = parse_labels_xlsx(args.labels)
    sample_order, groups = group_labels_by_sample(labels)
    stem2sample = map_samples_to_mzmls(stems, sample_order, None)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    idx_rows, counts = [], {"TP": 0, "FP": 0, "FN": 0, "loose": 0}

    for stem in stems:
        xic_dir = roi_root / stem
        pred_csv = pred_root / stem / "prediction.csv"
        if not pred_csv.is_file():
            print("[WARN] 跳过 %s：无 prediction.csv" % stem)
            continue
        feat = pd.read_csv(xic_dir / "feature.csv")
        wins = pd.read_csv(xic_dir / "roi_windows.csv").set_index("image")

        by_key = {}
        for rec in groups[stem2sample[stem]]:
            k = label_key(rec.get("compound"), rec.get("channel"))
            if k:
                by_key.setdefault(k, rec)

        pred_rows = _load_pred_rows(str(pred_csv), args.threshold)
        pred_by_img = {}
        for r in pred_rows:
            pred_by_img.setdefault(str(r["image"]).strip(), []).append(dict(r))
        for v in pred_by_img.values():
            v.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)

        for i, frow in feat.iterrows():
            n = i + 1
            native_id = str(frow["native_id"]).strip()
            img_name = next((im for im in pred_by_img if im.startswith("%d_mz" % n)), None)
            img_name = img_name or next(
                (w for w in wins.index if w.startswith("%d_mz" % n)), None)
            img_path = xic_dir / img_name if img_name else None
            if img_path is None or not img_path.is_file():
                continue

            rw = wins.loc[img_name]
            rt_lo, rt_hi = float(rw["rt_lo"]), float(rw["rt_hi"])
            rec = by_key.get(native_id)
            gt_peaks = _parse_gt_peaks(rec) if rec is not None else []
            rows = pred_by_img.get(img_name, [])

            pairs, fps, fns, loose = match_image(
                rows, [(g[0], g[1]) for g in gt_peaks], args.tolerance,
                loose_tol=args.quant_tolerance)
            loose_pred_ids = {id(pr) for pr, _g in loose}

            compound = str(frow.get("Compound Name", "")).strip()
            n_tp, n_fp, n_fn = len(pairs), len(fps), len(fns)
            result = ("TP" if (n_tp and not n_fp and not n_fn)
                      else "FN" if n_fn else ("FP" if n_fp and not gt_peaks else "MIX"))

            gts_px, devs = [], []
            matched_gt = {id(g): pr for pr, g in pairs}
            for g in gt_peaks:
                x1 = rt_to_pixel_x(g[0], rt_lo, rt_hi)
                x2 = rt_to_pixel_x(g[1], rt_lo, rt_hi)
                pr = matched_gt.get(id(g))
                dv = (pr["rt_min"] - g[0], pr["rt_max"] - g[1]) if pr else None
                gts_px.append((min(x1, x2), max(x1, x2), "GT %s-%s" % (_fmt(g[0], 2), _fmt(g[1], 2))))
                devs.append(dv)
            for g in fns + [(gl[0], gl[1], float("nan")) for _pr, gl in loose]:
                if any(abs(g[0] - gg[0]) < 1e-9 and abs(g[1] - gg[1]) < 1e-9 for gg in gt_peaks):
                    continue  # 已画过的 loose/严格区间不重复画框

            preds_px = []
            for pr in rows:
                lo, hi = float(pr["rt_min"]), float(pr["rt_max"])
                x1 = rt_to_pixel_x(lo, rt_lo, rt_hi)
                x2 = rt_to_pixel_x(hi, rt_lo, rt_hi)
                tag = "loose" if id(pr) in loose_pred_ids else "P"
                preds_px.append((min(x1, x2), max(x1, x2),
                                 "%s %s-%s s%.2f" % (tag, _fmt(lo, 2), _fmt(hi, 2), float(pr.get("score") or 0))))

            counts["TP"] += n_tp
            counts["FP"] += n_fp
            counts["FN"] += n_fn
            counts["loose"] += len(loose)
            for pr, g in pairs:
                idx_rows.append({"stem": stem, "image": img_name, "native_id": native_id,
                                 "compound": compound, "result": "TP",
                                 "gt_start": g[0], "gt_end": g[1],
                                 "pred_start": pr["rt_min"], "pred_end": pr["rt_max"],
                                 "dev_start_min": round(pr["rt_min"] - g[0], 3),
                                 "dev_end_min": round(pr["rt_max"] - g[1], 3),
                                 "score": round(float(pr.get("score") or 0), 4)})
            for pr, g in loose:
                idx_rows.append({"stem": stem, "image": img_name, "native_id": native_id,
                                 "compound": compound, "result": "loose",
                                 "gt_start": g[0], "gt_end": g[1],
                                 "pred_start": pr["rt_min"], "pred_end": pr["rt_max"],
                                 "dev_start_min": round(pr["rt_min"] - g[0], 3),
                                 "dev_end_min": round(pr["rt_max"] - g[1], 3),
                                 "score": round(float(pr.get("score") or 0), 4)})
            for pr in fps:
                if id(pr) in loose_pred_ids:
                    continue
                idx_rows.append({"stem": stem, "image": img_name, "native_id": native_id,
                                 "compound": compound, "result": "FP",
                                 "gt_start": None, "gt_end": None,
                                 "pred_start": pr["rt_min"], "pred_end": pr["rt_max"],
                                 "dev_start_min": None, "dev_end_min": None,
                                 "score": round(float(pr.get("score") or 0), 4)})
            for g in fns:
                idx_rows.append({"stem": stem, "image": img_name, "native_id": native_id,
                                 "compound": compound, "result": "FN",
                                 "gt_start": g[0], "gt_end": g[1],
                                 "pred_start": None, "pred_end": None,
                                 "dev_start_min": None, "dev_end_min": None, "score": None})

            xic_xy = _load_xic(xic_dir, i, args.smooth_sigma)
            title = "[%s] %s | TP %d / FP %d / FN %d / loose %d" % (
                result, native_id, n_tp, n_fp, n_fn, len(loose))
            plot_one(img_path, out_root / result / ("%s.png" % Path(img_name).stem),
                     title, xic_xy, (rt_lo, rt_hi), gts_px, preds_px, devs)

    pd.DataFrame(idx_rows).to_csv(out_root / "index.csv", index=False, encoding="utf-8-sig")
    with open(out_root / "summary.txt", "w", encoding="utf-8") as f:
        f.write("GT vs Pred 可视化汇总\npipeline: %s\nlabels: %s\n"
                "threshold=%g tol=%g quant_tol=%g\nTP %d | FP %d | FN %d | loose %d\n" % (
                    pipeline_dir, args.labels, args.threshold, args.tolerance,
                    args.quant_tolerance, counts["TP"], counts["FP"], counts["FN"], counts["loose"]))
    print("[DONE] 图与 index.csv → %s（TP %d | FP %d | FN %d | loose %d）" % (
        out_root, counts["TP"], counts["FP"], counts["FN"], counts["loose"]))


if __name__ == "__main__":
    main()
