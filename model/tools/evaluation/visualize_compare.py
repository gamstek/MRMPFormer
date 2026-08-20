# -*- coding: utf-8 -*-
"""
双模型预测 vs 人工标注 可视化复核工具。

在同一张 ROI 图上叠加：
  - 人工标注区间（绿色实线 + 淡绿带）
  - 模型1 预测区间（蓝色虚线 + 淡蓝带，取最高分框）
  - 模型2 预测区间（红色虚线 + 淡红带，取最高分框）
并标注起止偏差（min）与命中判定（起止偏差均 <= tolerance 判 TP）。

产物（output_dir）：
  <sample>/<image_stem>.png   每通道一张叠加图
  <sample>.html / index.html  画廊（浏览器打开逐张复核）
  compare_summary.csv         全通道汇总（GT/v1/v2 区间、偏差、判定、面积）

用法（model/ 目录下，系统终端执行，涉及 matplotlib 渲染）：
  python -m tools.evaluation.visualize_compare \
      --labels ../data/label/20260715_shiyaoyuan_test.xlsx \
      --xic_root ../data/test/coco/_xic \
      --pred1 ../output/test/pred_v1_fixed --name1 v1 \
      --pred2 ../output/test/pred_v2_fixed --name2 v2 \
      --output_dir ../output/evaluation/vis_v1_v2
"""
import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.coco_annotation import (
    group_labels_by_sample,
    label_key,
    map_samples_to_mzmls,
    parse_labels_xlsx,
    parse_rt_field,
)

DEFAULT_SCORE = 0.90
DEFAULT_TOL = 0.1


def load_pred(csv_path, min_score):
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError(f"{csv_path}: 缺 image 列")
    df = df[pd.to_numeric(df.get("score"), errors="coerce").fillna(0) >= min_score]
    groups = {}
    for _, r in df.iterrows():
        groups.setdefault(str(r["image"]).strip(), []).append(dict(r))
    for v in groups.values():
        v.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return groups


def verdict_of(rows, gt, tol):
    """返回 (best_row, dev_start, dev_end, verdict)。verdict: TP / FP / FN / 无标注"""
    if gt is None:
        return (rows[0] if rows else None), None, None, "无标注"
    if not rows:
        return None, None, None, "FN(漏检)"
    best = rows[0]
    try:
        ds = abs(float(best["rt_min"]) - gt[0])
        de = abs(float(best["rt_max"]) - gt[1])
    except (TypeError, ValueError, KeyError):
        return best, None, None, "FP(区间无效)"
    ok = ds <= tol + 1e-9 and de <= tol + 1e-9
    return best, ds, de, ("TP" if ok else f"FP(偏{ds:.2f}/{de:.2f})")


def main():
    ap = argparse.ArgumentParser(description="双模型预测 vs 人工标注可视化复核")
    ap.add_argument("--labels", required=True, help="人工标注 xlsx")
    ap.add_argument("--xic_root", required=True, help="ROI 根目录（其下各样品子目录含 jpeg + roi_windows.csv + feature.csv）")
    ap.add_argument("--pred1", required=True, help="模型1 prediction 根目录（其下样品子目录含 prediction.csv）")
    ap.add_argument("--name1", default="v1", help="模型1 显示名")
    ap.add_argument("--pred2", required=True, help="模型2 prediction 根目录")
    ap.add_argument("--name2", default="v2", help="模型2 显示名")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_SCORE, help="置信度阈值")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOL, help="起止偏差容差（min），TP 判定")
    args = ap.parse_args()

    # 渲染依赖放 main 内（沙箱 import 安全）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    xic_root = Path(args.xic_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 样品 = xic_root 下同时有 pred1/pred2 prediction.csv 的子目录
    stems = []
    for sub in sorted(p for p in xic_root.iterdir() if p.is_dir()):
        if (Path(args.pred1) / sub.name / "prediction.csv").is_file() and \
           (Path(args.pred2) / sub.name / "prediction.csv").is_file():
            stems.append(sub.name)
    if not stems:
        sys.exit(f"[ERROR] 未找到配对的样品目录（检查 --xic_root/--pred1/--pred2）: {xic_root}")

    labels = parse_labels_xlsx(args.labels)
    sample_order, groups = group_labels_by_sample(labels)
    stem2sample = map_samples_to_mzmls(stems, sample_order, None)

    summary_rows = []
    for stem in stems:
        sample_dir = xic_root / stem
        rw = pd.read_csv(sample_dir / "roi_windows.csv")
        feat = pd.read_csv(sample_dir / "feature.csv")
        g1 = load_pred(Path(args.pred1) / stem / "prediction.csv", args.threshold)
        g2 = load_pred(Path(args.pred2) / stem / "prediction.csv", args.threshold)

        by_key = {}
        for rec in groups[stem2sample[stem]]:
            k = label_key(rec.get("compound"), rec.get("channel"))
            if k:
                by_key.setdefault(k, rec)

        img_out = out_root / stem
        img_out.mkdir(parents=True, exist_ok=True)
        entries = []  # (png_rel, caption)

        all_imgs = list(dict.fromkeys(
            [str(r["image"]).strip() for _, r in rw.iterrows()]
            + list(g1.keys()) + list(g2.keys())))
        n_rows = 0
        for img_name in all_imgs:
            rw_row = rw[rw["image"].astype(str).str.strip() == img_name]
            rt_lo, rt_hi = (float(rw_row.iloc[0]["rt_lo"]), float(rw_row.iloc[0]["rt_hi"])) if len(rw_row) else (None, None)
            # image 前缀 N ↔ feature 第 N 行（1-based），与评测同规则
            native_id = None
            prefix = img_name.split("_mz")[0]
            if prefix.isdigit() and int(prefix) <= len(feat):
                native_id = str(feat.iloc[int(prefix) - 1]["native_id"]).strip()

            gt = None
            gt_area = None
            if native_id:
                rec = by_key.get(native_id)
                if rec is not None:
                    s, e = parse_rt_field(rec.get("peak_start")), parse_rt_field(rec.get("peak_end"))
                    if s is not None and e is not None:
                        gt = (min(s, e), max(s, e))
                    try:
                        gt_area = float(rec.get("area")) if rec.get("area") else None
                    except (TypeError, ValueError):
                        gt_area = None

            r1, ds1, de1, v1 = verdict_of(g1.get(img_name, []), gt, args.tolerance)
            r2, ds2, de2, v2 = verdict_of(g2.get(img_name, []), gt, args.tolerance)

            img_path = sample_dir / img_name
            if not img_path.is_file():
                continue  # 缺图跳过
            if native_id and native_id.startswith("TIC"):
                continue  # TIC 负样本：无标注，无复核意义

            fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=110)
            ax.imshow(np.asarray(Image.open(img_path)))
            ax.set_xlim(0, 400)   # 像素坐标系（与 ROI 图一致）
            ax.set_ylim(300, 0)

            # RT(分钟) → 像素 x，与生成 ROI 图时的 xlim 映射一致
            from utils.roi_rt_mapping import rt_to_pixel_x, ROI_IMAGE_WIDTH_PX

            def _rt2px(rt):
                if rt_lo is None:
                    return None
                return rt_to_pixel_x(rt, rt_lo, rt_hi)

            def _span_px(s_rt, e_rt, color, ls, lw, alpha_band):
                ps, pe = _rt2px(s_rt), _rt2px(e_rt)
                if ps is None or pe is None:
                    return
                if pe < ps:
                    ps, pe = pe, ps
                ax.axvspan(ps, pe, color=color, alpha=alpha_band)
                for x in (ps, pe):
                    ax.axvline(x, color=color, ls=ls, lw=lw)

            if gt is not None:
                _span_px(gt[0], gt[1], "green", "-", 2.0, 0.12)

            def _span(row, color):
                if row is None:
                    return
                try:
                    s, e = float(row["rt_min"]), float(row["rt_max"])
                except (TypeError, ValueError, KeyError):
                    return
                _span_px(s, e, color, "--", 1.6, 0.10)

            _span(r1, "tab:blue")
            _span(r2, "tab:red")

            # x 轴刻度：像素位置，标签显示对应 RT（分钟）
            if rt_lo is not None and rt_hi is not None:
                span = rt_hi - rt_lo
                ticks_px = [0, 100, 200, 300, 400]
                ax.set_xticks(ticks_px)
                ax.set_xticklabels(
                    [f"{rt_lo + (p / ROI_IMAGE_WIDTH_PX) * span:.2f}" for p in ticks_px])
                ax.set_xlabel("RT (min)")
            ax.set_yticks([])

            def _fmt_row(row, ds, de, verdict, color):
                if row is None:
                    return f"{verdict}", color
                try:
                    rng = f"[{float(row['rt_min']):.3f}, {float(row['rt_max']):.3f}]"
                    sc = float(row.get("score") or 0)
                except (TypeError, ValueError, KeyError):
                    rng, sc = "?", float("nan")
                dev = "" if ds is None else f" 偏差 {ds:.3f}/{de:.3f}"
                return f"{verdict} {rng} score={sc:.3f}{dev}", color

            t1, c1 = _fmt_row(r1, ds1, de1, v1, "tab:blue")
            t2, c2 = _fmt_row(r2, ds2, de2, v2, "tab:red")
            gt_txt = f"人工 [{gt[0]:.3f}, {gt[1]:.3f}]" if gt else "人工 无"
            ax.set_title(f"{native_id or img_name}\n{gt_txt}", fontsize=9)
            ax.text(0.5, -0.30, t1, transform=ax.transAxes, ha="center",
                    fontsize=8, color=c1)
            ax.text(0.5, -0.38, t2, transform=ax.transAxes, ha="center",
                    fontsize=8, color=c2)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            png_name = Path(img_name).stem + ".png"
            fig.savefig(img_out / png_name, bbox_inches="tight")
            plt.close(fig)
            n_rows += 1

            cap = (f"{native_id or img_name} | 人工 {gt_txt}"
                   f" | {args.name1}: {t1} | {args.name2}: {t2}")
            entries.append((f"{stem}/{png_name}", cap))

            summary_rows.append({
                "sample": stem, "native_id": native_id,
                "gt_start": gt[0] if gt else None, "gt_end": gt[1] if gt else None,
                "gt_area": gt_area,
                f"{args.name1}_start": float(r1["rt_min"]) if r1 else None,
                f"{args.name1}_end": float(r1["rt_max"]) if r1 else None,
                f"{args.name1}_score": float(r1.get("score") or 0) if r1 else None,
                f"{args.name1}_verdict": v1,
                f"{args.name2}_start": float(r2["rt_min"]) if r2 else None,
                f"{args.name2}_end": float(r2["rt_max"]) if r2 else None,
                f"{args.name2}_score": float(r2.get("score") or 0) if r2 else None,
                f"{args.name2}_verdict": v2,
            })

        # 每样品画廊
        parts = [f"<html><head><meta charset='utf-8'><title>{stem}</title></head><body>",
                 f"<h2>{stem}（{n_rows} 张）</h2>",
                 f"<p>绿=人工 | 蓝={args.name1} | 红={args.name2} | 容差 ±{args.tolerance} min</p>"]
        for rel, cap in entries:
            parts.append(f"<div style='margin:8px 0'><div style='font-size:14px'>{html.escape(cap)}</div>"
                         f"<img src='{html.escape(rel)}' style='max-width:520px;border:1px solid #ccc'></div>")
        parts.append("</body></html>")
        (out_root / f"{stem}.html").write_text("\n".join(parts), encoding="utf-8")

    # index
    idx = ["<html><head><meta charset='utf-8'><title>v1/v2 复核</title></head><body><h2>样品索引</h2><ul>"]
    for stem in stems:
        idx.append(f"<li><a href='{stem}.html'>{stem}</a></li>")
    idx.append("</ul></body></html>")
    (out_root / "index.html").write_text("\n".join(idx), encoding="utf-8")

    pd.DataFrame(summary_rows).to_csv(out_root / "compare_summary.csv",
                                      index=False, encoding="utf-8-sig")

    # 终端汇总
    df = pd.DataFrame(summary_rows)
    print()
    print("=" * 64)
    print(f"可视化复核 | 容差 ±{args.tolerance} min | score>={args.threshold}")
    print("=" * 64)
    for name in (args.name1, args.name2):
        vc = df[f"{name}_verdict"].value_counts()
        tp = int(vc.get("TP", 0))
        fn = int(vc.get("FN(漏检)", 0))
        fp = int(len(df) - tp - fn - vc.get("无标注", 0))
        print(f"{name:4s} | TP {tp} | FN {fn} | FP {fp} | 共 {len(df)} 通道")
    print("-" * 64)
    print(f"画廊  : {out_root / 'index.html'}（浏览器打开）")
    print(f"明细  : {out_root / 'compare_summary.csv'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
