# -*- coding: utf-8 -*-
"""Plot MRMPFormer and CentreWave boundaries on the original XIC traces.

The script consumes the already generated five-model benchmark artifacts.  It
applies the same label QC and matching function, supports all manual/predicted
peaks in an ROI, and writes per-ROI PNG galleries plus aggregate diagnostics.

Run from ``model/``::

    python -m tools.evaluation.visualize_mrmp_centrewave
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.coco_annotation import label_key, parse_rt_field
from tools.evaluation.evaluate_baseline import _parse_gt_peaks, match_image
from tools.evaluation.evaluate_centrewave_grid import (
    _configure_centrewave,
    _find_mzml,
    _prepare_labels,
)


COLORS = {"GT": "#2ca02c", "MRMPFormer": "#1f77b4", "CentreWave": "#ff7f0e"}
TOLERANCE = 0.1


def _safe_name(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(text)).strip(" .")
    return text[:120] or "roi"


def _load_predictions(pred_path: Path, feature_path: Path, threshold: float):
    pred = pd.read_csv(pred_path)
    feat = pd.read_csv(feature_path)
    if "peak_start" in pred.columns:
        pred = pred.rename(columns={"peak_start": "rt_min", "peak_end": "rt_max"})
    if "score" not in pred.columns:
        raise ValueError(f"{pred_path}: 缺少 score")
    pred = pred[pd.to_numeric(pred["score"], errors="coerce").fillna(0) >= threshold]
    result = {}
    for _, row in pred.iterrows():
        image = str(row.get("image") or "").strip()
        prefix = image.split("_mz", 1)[0]
        if not prefix.isdigit() or not (1 <= int(prefix) <= len(feat)):
            continue
        native_id = str(feat.iloc[int(prefix) - 1]["native_id"]).strip()
        item = dict(row)
        try:
            item["rt_min"] = float(item["rt_min"])
            item["rt_max"] = float(item["rt_max"])
            item["score"] = float(item["score"])
        except (TypeError, ValueError):
            continue
        result.setdefault(native_id, []).append(item)
    for rows in result.values():
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return result


def _mrmp_paths(project: Path, dataset: str, stem: str):
    grid = "grid4" if dataset == "test1" else "grid4_test2"
    sample = project / "output/evaluation" / grid / "mrmpformer/_pipeline" / stem
    pred = sample / "predictions_model" / stem / f"model_prediction_{stem}.csv"
    feat = sample / "xic_roi" / stem / "feature.csv"
    return pred, feat


def _cw_paths(cw_benchmark: Path, dataset: str, stem: str, snr_tag: str):
    sample = cw_benchmark / dataset / "eval_inputs" / snr_tag / stem
    return sample / "predictions.csv", sample / "feature.csv"


def _status(preds, gt, tol=TOLERANCE):
    pairs, fps, fns, _ = match_image(preds, gt, tol=tol, loose_tol=0)
    if not gt and not preds:
        label = "TN"
    else:
        label = f"TP {len(pairs)} / FP {len(fps)} / FN {len(fns)}"
    return label, pairs, fps, fns


def _interval_text(rows):
    if not rows:
        return "无"
    return "; ".join(f"{float(r['rt_min']):.3f}–{float(r['rt_max']):.3f}" for r in rows)


def _plot_roi(
    out_path: Path,
    sample_id: str,
    native_id: str,
    t_min,
    intensity,
    center_min: float,
    gt,
    mrmp,
    cw,
    mrmp_status: str,
    cw_status: str,
    tag: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    mask = (t_min >= center_min - 1.0) & (t_min <= center_min + 1.0)
    tx = np.asarray(t_min[mask], dtype=float)
    yy = np.asarray(intensity[mask], dtype=float)
    if not len(tx):
        return False

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=130)
    ax.plot(tx, yy, color="#202020", lw=1.25, zorder=2)
    ax.fill_between(tx, 0, yy, color="#777777", alpha=0.10, zorder=1)

    def draw_intervals(intervals, color, linestyle, alpha, zorder):
        for start, end in intervals:
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=zorder)
            ax.axvline(start, color=color, ls=linestyle, lw=1.8, zorder=zorder + 1)
            ax.axvline(end, color=color, ls=linestyle, lw=1.8, zorder=zorder + 1)

    draw_intervals([(g[0], g[1]) for g in gt], COLORS["GT"], "-", 0.12, 3)
    draw_intervals([(r["rt_min"], r["rt_max"]) for r in mrmp], COLORS["MRMPFormer"], "--", 0.09, 4)
    draw_intervals([(r["rt_min"], r["rt_max"]) for r in cw], COLORS["CentreWave"], ":", 0.09, 5)

    tag_text = f" | tag: {tag}" if tag else ""
    ax.set_title(f"{sample_id} | {native_id}{tag_text}", fontsize=12)
    ax.set_xlabel("保留时间 RT (min)")
    ax.set_ylabel("强度")
    ax.grid(axis="x", alpha=0.16)
    ax.margins(x=0)
    ymax = float(np.nanmax(yy)) if len(yy) else 1.0
    ax.set_ylim(bottom=0, top=max(1.0, ymax * 1.18))
    ax.text(
        0.01,
        0.98,
        f"MRMPFormer: {mrmp_status}\nCentreWave: {cw_status}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.86, edgecolor="#bbbbbb"),
    )
    handles = [
        Line2D([0], [0], color="#202020", lw=1.4, label="原始 XIC"),
        Patch(facecolor=COLORS["GT"], alpha=0.20, label="人工标签"),
        Patch(facecolor=COLORS["MRMPFormer"], alpha=0.20, label="MRMPFormer"),
        Patch(facecolor=COLORS["CentreWave"], alpha=0.20, label="CentreWave"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.88, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _paired_errors(preds, gt, loose_tol=0.5):
    pairs, _fps, _fns, _ = match_image(preds, gt, tol=loose_tol, loose_tol=0)
    return [
        {
            "start_error": float(pr["rt_min"]) - g[0],
            "end_error": float(pr["rt_max"]) - g[1],
            "width_error": (float(pr["rt_max"]) - float(pr["rt_min"])) - (g[1] - g[0]),
        }
        for pr, g in pairs
    ]


def _case_type(gt, mrmp_pairs, mrmp_fps, mrmp_fns, cw_pairs, cw_fps, cw_fns):
    if not gt:
        if not mrmp_fps and not cw_fps:
            return "两者均正确无峰"
        if not mrmp_fps:
            return "MRMPFormer 更准"
        if not cw_fps:
            return "CentreWave 更准"
        return "两者均误报"
    mrmp_ok = not mrmp_fps and not mrmp_fns
    cw_ok = not cw_fps and not cw_fns
    if mrmp_ok and cw_ok:
        return "两者均命中"
    if mrmp_ok:
        return "MRMPFormer 更准"
    if cw_ok:
        return "CentreWave 更准"
    if len(mrmp_pairs) > len(cw_pairs):
        return "MRMPFormer 更准"
    if len(cw_pairs) > len(mrmp_pairs):
        return "CentreWave 更准"
    if not mrmp_pairs and not cw_pairs:
        return "两者均未命中"
    return "两者均有误差"


def _plot_aggregate(out_root: Path, comparison, error_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    tolerances = [0.01, 0.05, 0.1, 0.2, 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140, sharey=True)
    for ax, dataset in zip(axes, ("test1", "test2")):
        for model, color, marker in (
            ("mrmpformer", COLORS["MRMPFormer"], "o"),
            ("centrewave", COLORS["CentreWave"], "s"),
        ):
            vals = [comparison[dataset]["grid"][model]["0.9"][str(t)]["f1"] for t in tolerances]
            ax.plot(tolerances, vals, marker=marker, lw=2, color=color, label=model)
        ax.set_title(dataset)
        ax.set_xlabel("起止容差 (min)")
        ax.set_xticks(tolerances)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("F1")
    axes[1].legend()
    fig.suptitle("MRMPFormer vs CentreWave：F1 随边界容差变化")
    fig.tight_layout()
    fig.savefig(out_root / "f1_vs_tolerance.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140, sharey=True)
    metrics = ("precision", "recall", "f1")
    x = np.arange(len(metrics))
    for ax, dataset in zip(axes, ("test1", "test2")):
        for offset, model, color in (
            (-0.18, "mrmpformer", COLORS["MRMPFormer"]),
            (0.18, "centrewave", COLORS["CentreWave"]),
        ):
            m = comparison[dataset]["grid"][model]["0.9"]["0.1"]
            vals = [m[k] for k in metrics]
            bars = ax.bar(x + offset, vals, width=0.34, color=color, alpha=0.85, label=model)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_title(dataset)
        ax.set_xticks(x, ["Precision", "Recall", "F1"])
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("指标值（阈值 0.9 / 容差 0.1 min）")
    axes[1].legend()
    fig.suptitle("MRMPFormer vs CentreWave：核心指标")
    fig.tight_layout()
    fig.savefig(out_root / "core_metrics.png", bbox_inches="tight")
    plt.close(fig)

    err = pd.DataFrame(error_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140, sharey=True)
    for row_i, dataset in enumerate(("test1", "test2")):
        for col_i, (field, title) in enumerate(
            (("start_error", "起点有符号误差"), ("end_error", "终点有符号误差"))
        ):
            ax = axes[row_i, col_i]
            values = [
                err[(err.dataset == dataset) & (err.model == model)][field].dropna().to_numpy()
                for model in ("MRMPFormer", "CentreWave")
            ]
            bp = ax.boxplot(values, tick_labels=["MRMPFormer", "CentreWave"], patch_artist=True, showfliers=True)
            for patch, color in zip(bp["boxes"], (COLORS["MRMPFormer"], COLORS["CentreWave"])):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            ax.axhline(0, color="black", lw=0.8)
            ax.axhline(0.1, color="#999999", ls="--", lw=0.8)
            ax.axhline(-0.1, color="#999999", ls="--", lw=0.8)
            ax.set_title(f"{dataset}：{title}")
            ax.set_ylabel("预测 - 标签 (min)")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("0.5 min 内可配对峰的边界误差分布")
    fig.tight_layout()
    fig.savefig(out_root / "boundary_error_boxplots.png", bbox_inches="tight")
    plt.close(fig)


def _make_montage(out_root: Path, rows, max_per_dataset=8):
    from PIL import Image, ImageDraw, ImageFont

    chosen = []
    for dataset in ("test1", "test2"):
        categories = (
            "MRMPFormer 更准", "CentreWave 更准", "两者均未命中",
            "两者均有误差", "两者均误报",
        )
        per_category = max(1, max_per_dataset // 4)
        dataset_chosen = []
        for category in categories:
            part = [
                r for r in rows
                if r["dataset"] == dataset and r["case_type"] == category
            ]
            part.sort(key=lambda r: -float(r["difference_score"]))
            dataset_chosen.extend(part[:per_category])
        chosen.extend(dataset_chosen[:max_per_dataset])
    if not chosen:
        return None
    thumb_w, thumb_h = 720, 430
    cols = 2
    rows_n = int(np.ceil(len(chosen) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows_n * (thumb_h + 34)), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    for i, rec in enumerate(chosen):
        image = Image.open(rec["plot_path"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + 34)
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + thumb_h + 5), f"{rec['dataset']} | {rec['case_type']}", fill="black", font=font)
    path = out_root / "representative_failures_montage.jpg"
    canvas.save(path, quality=92)
    return path


def _write_gallery(dataset_dir: Path, dataset: str, entries):
    grouped = {}
    for rec in entries:
        grouped.setdefault(rec["sample"], []).append(rec)
    index = [
        "<html><head><meta charset='utf-8'><title>MRMPFormer vs CentreWave</title></head><body>",
        f"<h1>{html.escape(dataset)}：MRMPFormer vs CentreWave</h1>",
        "<p>绿色=人工标签，蓝色=MRMPFormer，橙色=CentreWave；判定容差 0.1 min。</p>",
    ]
    for sample, rows in grouped.items():
        anchor = _safe_name(sample)
        index.append(f"<h2 id='{anchor}'>{html.escape(sample)}（{len(rows)} 个 ROI）</h2>")
        for rec in rows:
            rel = Path(rec["plot_path"]).relative_to(dataset_dir).as_posix()
            caption = (
                f"{rec['native_id']} | {rec['case_type']} | "
                f"MRMP {rec['mrmp_status']} | CW {rec['cw_status']}"
            )
            index.append(
                f"<div style='margin:14px 0'><div>{html.escape(caption)}</div>"
                f"<img src='{html.escape(rel)}' style='max-width:1000px;width:96%;border:1px solid #ccc'></div>"
            )
    index.append("</body></html>")
    (dataset_dir / "index.html").write_text("\n".join(index), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="..")
    parser.add_argument("--cw-root", default="../CW/CW")
    parser.add_argument("--cw-benchmark", default="project_review/cw_benchmark")
    parser.add_argument("--comparison-json", default="project_review/cw_benchmark/comparison/five_model_current_labels.json")
    parser.add_argument("--output-dir", default="project_review/cw_benchmark/visualizations")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--cw-snr-tag", default="snr_10")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    cw_benchmark = Path(args.cw_benchmark).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    comparison = json.loads(Path(args.comparison_json).resolve().read_text(encoding="utf-8"))
    api = _configure_centrewave(Path(args.cw_root).resolve())

    summary_rows = []
    error_rows = []
    all_entries = []
    for dataset in ("test1", "test2"):
        print(f"[plot] {dataset}", flush=True)
        label_path = project / "data/label" / f"{dataset}.xlsx"
        label_info = _prepare_labels(label_path, 1.0)
        dataset_dir = out_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_entries = []
        for sample_id in label_info["order"]:
            stem = Path(sample_id).stem
            mzml = _find_mzml(project / "data/mzml" / dataset, sample_id)
            channels, _meta = api["extract_channels"](str(mzml))
            mrmp_pred, mrmp_feat = _mrmp_paths(project, dataset, stem)
            cw_pred, cw_feat = _cw_paths(cw_benchmark, dataset, stem, args.cw_snr_tag)
            mrmp_by_native = _load_predictions(mrmp_pred, mrmp_feat, args.threshold)
            cw_by_native = _load_predictions(cw_pred, cw_feat, args.threshold)

            for roi_i, rec in enumerate(label_info["groups"][sample_id], 1):
                native_id = label_key(rec.get("compound"), rec.get("channel"))
                if native_id not in channels:
                    continue
                center = parse_rt_field(rec.get("rt"))
                if center is None:
                    continue
                gt = _parse_gt_peaks(rec)
                mrmp = mrmp_by_native.get(native_id, [])
                cw = cw_by_native.get(native_id, [])
                ms, mpairs, mfps, mfns = _status(mrmp, gt, args.tolerance)
                cs, cpairs, cfps, cfns = _status(cw, gt, args.tolerance)
                case = _case_type(gt, mpairs, mfps, mfns, cpairs, cfps, cfns)
                sample_dir = dataset_dir / _safe_name(stem)
                png = sample_dir / f"{roi_i:03d}_{_safe_name(native_id)}.png"
                t, x = channels[native_id]
                ok = _plot_roi(
                    png,
                    sample_id,
                    native_id,
                    np.asarray(t) / 60.0,
                    np.asarray(x),
                    center,
                    gt,
                    mrmp,
                    cw,
                    ms,
                    cs,
                    str(rec.get("tag") or "").strip(),
                )
                if not ok:
                    continue
                for model, preds in (("MRMPFormer", mrmp), ("CentreWave", cw)):
                    for e in _paired_errors(preds, gt, 0.5):
                        error_rows.append({"dataset": dataset, "sample": stem, "native_id": native_id, "model": model, **e})
                def interval_error(rows):
                    errors = _paired_errors(rows, gt, 0.5)
                    return sum(abs(e["start_error"]) + abs(e["end_error"]) for e in errors)
                entry = {
                    "dataset": dataset,
                    "sample": stem,
                    "native_id": native_id,
                    "tag": str(rec.get("tag") or "").strip(),
                    "gt_count": len(gt),
                    "gt_intervals": json.dumps([(g[0], g[1]) for g in gt], ensure_ascii=False),
                    "mrmp_intervals": _interval_text(mrmp),
                    "cw_intervals": _interval_text(cw),
                    "mrmp_status": ms,
                    "cw_status": cs,
                    "mrmp_tp": len(mpairs),
                    "cw_tp": len(cpairs),
                    "case_type": case,
                    "difference_score": abs(interval_error(mrmp) - interval_error(cw)),
                    "plot_path": str(png),
                }
                dataset_entries.append(entry)
                all_entries.append(entry)
                summary_rows.append(entry)
        pd.DataFrame(dataset_entries).to_csv(dataset_dir / "roi_visual_summary.csv", index=False, encoding="utf-8-sig")
        _write_gallery(dataset_dir, dataset, dataset_entries)

    pd.DataFrame(summary_rows).to_csv(out_root / "roi_visual_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(out_root / "paired_boundary_errors_tol0p5.csv", index=False, encoding="utf-8-sig")
    _plot_aggregate(out_root, comparison, error_rows)
    montage = _make_montage(out_root, all_entries)

    counts = pd.DataFrame(all_entries).groupby(["dataset", "case_type"]).size().to_dict()
    report = [
        "# MRMPFormer vs CentreWave 可视化复核",
        "",
        "绿色为人工标签，蓝色为 MRMPFormer，橙色为 CentreWave。单 ROI 判定容差为 0.1 min。",
        "",
        "## 总览图",
        "",
        "![核心指标](core_metrics.png)",
        "",
        "![F1-容差曲线](f1_vs_tolerance.png)",
        "",
        "![边界误差箱线图](boundary_error_boxplots.png)",
        "",
    ]
    if montage:
        report.extend(["![典型差异案例](representative_failures_montage.jpg)", ""])
    report.extend(["## ROI 画廊", "", "- [test1 全部 ROI](test1/index.html)", "- [test2 全部 ROI](test2/index.html)", "", "## 0.1 min 命中数量对比", ""])
    for dataset in ("test1", "test2"):
        report.append(f"### {dataset}")
        report.append("")
        for case in (
            "MRMPFormer 更准", "CentreWave 更准", "两者均命中",
            "两者均正确无峰", "两者均未命中", "两者均有误差", "两者均误报",
        ):
            report.append(f"- {case}: {counts.get((dataset, case), 0)} 个 ROI")
        report.append("")
    (out_root / "VISUAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(out_root / "VISUAL_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
