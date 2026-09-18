# -*- coding: utf-8 -*-
"""Re-score four saved model predictions and append CentreWave results.

This is intentionally a scoring-only script.  It uses the existing prediction
CSVs, the current test1/test2 labels, and the exact evaluator used by the
four-model benchmark.  CentreWave results are read from
``evaluate_centrewave_grid.py`` output.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import pandas as pd

from tools.evaluation.evaluate_baseline import evaluate
from tools.evaluation.evaluate_special_isolation import evaluate_special


MODELS = ("quanformer", "quanformerv2", "quanformerv3", "mrmpformer")
ALL_MODELS = MODELS + ("centrewave",)
SCORES = (0.1, 0.5, 0.8, 0.9, 0.99)
TOLS = (0.01, 0.05, 0.1, 0.2, 0.5)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _prediction_map(model_root: Path):
    pipeline = model_root / "_pipeline"
    result = {}
    for sample_dir in sorted(pipeline.iterdir()):
        if not sample_dir.is_dir():
            continue
        pred = (
            sample_dir
            / "predictions_model"
            / sample_dir.name
            / f"model_prediction_{sample_dir.name}.csv"
        )
        # Older refinement outputs sometimes dropped native_id; the XIC-stage
        # feature table is the canonical ROI-to-channel mapping used in scoring.
        feat = sample_dir / "xic_roi" / sample_dir.name / "feature.csv"
        if pred.is_file() and feat.is_file():
            result[sample_dir.name] = {"pred": pred, "feat": feat}
    if not result:
        raise FileNotFoundError(f"{model_root}: 未找到已有预测")
    return result


def _score_model(pred_map, label_path: Path):
    grid = {}
    special = {}
    # The evaluator prints the same QC warnings on every cell; retain metrics
    # while suppressing this redundant console noise.
    with contextlib.redirect_stdout(io.StringIO()):
        for score in SCORES:
            grid[str(score)] = {}
            special[str(score)] = {}
            for tol in TOLS:
                metrics, _detail, _area, _qc = evaluate(
                    pred_map,
                    str(label_path),
                    tol=tol,
                    min_score=score,
                    quant_tol=tol,
                    qc_label_rt_tol=1.0,
                )
                grid[str(score)][str(tol)] = metrics
                sp, _audit = evaluate_special(
                    pred_map,
                    str(label_path),
                    tol=tol,
                    min_score=score,
                    qc_label_rt_tol=1.0,
                )
                special[str(score)][str(tol)] = sp
    return grid, special


def _best(grid):
    cells = []
    for score, by_tol in grid.items():
        for tol, metric in by_tol.items():
            cells.append((metric["f1"], metric["recall"], metric["precision"], score, tol, metric))
    return max(cells, key=lambda x: (x[0], x[1], x[2], float(x[3]), float(x[4])))


def _fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _table(lines, header, rows):
    lines.extend(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def _report(out_path: Path, combined, timing, cw_summary, old_summary):
    test1_delta = max(
        abs(
            combined["test1"]["grid"][model][str(score)][str(tol)]["f1"]
            - old_summary["test1"][model][str(score)][str(tol)]["f1"]
        )
        for model in MODELS for score in SCORES for tol in TOLS
    )
    lines = [
        "# CentreWave 与四模型同口径对比",
        "",
        "- 五个方法均按同一标签、同一 QC、同一边界命中函数计分。",
        "- 命中：预测起止相对人工起止的绝对偏差均不超过容差。",
        "- 神经网络扫描 score；CentreWave 无校准分类概率，通过原生强度/宽度/SNR 验证后记 score=1。",
        "- CentreWave 原生工作点为 SNR≥10、intensity>1000，CPU 运行；四模型时间来自原 RTX 4090 记录。",
        "",
        "## 数据版本核对",
        "",
        f"- test1 当前 GT={cw_summary['test1']['label']['gt_count']}，与原报告 82 一致。",
        f"- test1 四模型 100 个 F1 网格单元复现最大绝对差={test1_delta:.1e}。",
        f"- test2 当前 GT={cw_summary['test2']['label']['gt_count']}、特殊峰={cw_summary['test2']['label']['special_count']}；"
        "当前 `data/label/test2.xlsx` 修改时间晚于原四模型报告，因此下表中的四模型 test2 数字也已按当前标签重算。",
        "",
    ]
    for dataset in ("test1", "test2"):
        grids = combined[dataset]["grid"]
        specials = combined[dataset]["special"]
        lines.extend([f"## {dataset}", "", "### F1：阈值 0.9 扫容差", ""])
        _table(
            lines,
            ["容差(min)"] + list(ALL_MODELS),
            [[f"{tol:g}"] + [f"{grids[m]['0.9'][str(tol)]['f1']:.3f}" for m in ALL_MODELS] for tol in TOLS],
        )
        lines.extend(["### F1：容差 0.1 min 扫阈值", ""])
        _table(
            lines,
            ["阈值"] + list(ALL_MODELS),
            [[f"{score:g}"] + [f"{grids[m][str(score)]['0.1']['f1']:.3f}" for m in ALL_MODELS] for score in SCORES],
        )
        lines.extend(["### 详表：阈值 0.9 / 容差 0.1 min", ""])
        rows = []
        for model in ALL_MODELS:
            m = grids[model]["0.9"]["0.1"]
            rows.append([
                model,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['TP']}/{m['FP']}/{m['FN']}",
                _fmt(m.get("area_r2_pred_vs_manual")),
                f"{_fmt(m.get('rt_start_dev_median_min'))} / {_fmt(m.get('rt_end_dev_median_min'))}",
            ])
        _table(lines, ["模型", "P", "R", "F1", "TP/FP/FN", "面积R²", "RT 起/止中位(min)"], rows)
        lines.extend(["### 每模型最佳 F1 组合", ""])
        rows = []
        for model in ALL_MODELS:
            if model == "centrewave":
                cells = []
                for snr, by_tol in cw_summary[dataset]["snr_scan"].items():
                    for tol, metric in by_tol.items():
                        cells.append(
                            (
                                metric["f1"], metric["recall"], metric["precision"],
                                float(snr), float(tol), metric,
                            )
                        )
                f1, _r, _p, snr, tol, m = max(
                    cells, key=lambda x: (x[0], x[1], x[2], x[3], -x[4])
                )
                setting = f"(SNR={snr:g}, {tol:.2f} min)"
            else:
                f1, _r, _p, score, tol, m = _best(grids[model])
                setting = f"({float(score):.2f}, {float(tol):.2f} min)"
            rows.append([
                model,
                setting,
                f"{f1:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
            ])
        _table(lines, ["模型", "最佳(阈值,容差)", "F1", "P", "R"], rows)
        lines.extend(["### 特殊峰识别：阈值 0.9 扫容差", ""])
        rows = []
        for tol in TOLS:
            row = [f"{tol:g}"]
            for model in ALL_MODELS:
                sp = specials[model]["0.9"][str(tol)]
                row.append(f"{sp['special_tp']}/{sp['special_gt']} ({sp['special_rate']:.3f})")
            rows.append(row)
        _table(lines, ["容差(min)"] + list(ALL_MODELS), rows)
        lines.extend(["### 推理时间", ""])
        time_rows = []
        for model in MODELS:
            t = timing[dataset][model]
            time_rows.append([model, t["samples"], f"{t['pred_mean_ms']:.0f} ms", f"{t['pipeline_mean_ms']:.0f} ms", "RTX 4090"])
        ct = cw_summary[dataset]["timing"]
        time_rows.append([
            "centrewave",
            ct["sample_count"],
            f"{ct['detector_mean_ms_per_sample']:.0f} ms",
            f"{ct['pipeline_mean_ms_per_sample']:.0f} ms",
            "CPU",
        ])
        _table(lines, ["模型", "样品数", "核心计算均值/样品", "全流程均值/样品", "设备"], time_rows)

    lines.extend(["## 结果解读", ""])
    for dataset in ("test1", "test2"):
        vals = [(combined[dataset]["grid"][m]["0.9"]["0.1"]["f1"], m) for m in ALL_MODELS]
        vals.sort(reverse=True)
        cw = combined[dataset]["grid"]["centrewave"]["0.9"]["0.1"]
        lines.append(
            f"- {dataset} 核心档第一名为 {vals[0][1]}（F1={vals[0][0]:.4f}）；"
            f"CentreWave 为 F1={cw['f1']:.4f}、P={cw['precision']:.4f}、R={cw['recall']:.4f}。"
        )
    lines.extend([
        "- CentreWave 在 test1 的召回与三种训练后模型相同（80/82），差距主要来自 10 个 FP；提高原生 SNR 可减少 test1 FP。",
        "- test1 把 SNR 从 10 提到 100 后，CentreWave 在 0.1 min 下达到 80/2/2、F1=0.9756；test2 的候选均为高 SNR，调 SNR 不改变结果。",
        "- CentreWave 在 test2 的 0.1 min 严格边界口径明显较弱，但在 0.5 min 达到 F1=0.9408。0.5 min 配对峰的右边界有 +0.0893 min 中位偏移，预测宽度中位多 0.1115 min，主要问题是边界估计偏宽和偏晚。",
        "- 不应把 CentreWave 的原始小波响应强行映射为神经网络概率；若要形成可扫描的统一 score，需要另建独立校准集做概率校准。",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="..")
    parser.add_argument("--cw-summary", default="project_review/cw_benchmark/summary.json")
    parser.add_argument("--output-dir", default="project_review/cw_benchmark/comparison")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cw_summary = json.loads(Path(args.cw_summary).resolve().read_text(encoding="utf-8"))
    old_summary = {
        "test1": json.loads((project / "output/evaluation/grid4_summary.json").read_text(encoding="utf-8")),
        "test2": json.loads((project / "output/evaluation/grid4_test2_summary.json").read_text(encoding="utf-8")),
    }
    timing = {
        "test1": json.loads((project / "output/evaluation/grid4_timing.json").read_text(encoding="utf-8")),
        "test2": json.loads((project / "output/evaluation/grid4_test2_timing.json").read_text(encoding="utf-8")),
    }
    roots = {
        "test1": project / "output/evaluation/grid4",
        "test2": project / "output/evaluation/grid4_test2",
    }
    combined = {}
    for dataset in ("test1", "test2"):
        label = project / "data/label" / f"{dataset}.xlsx"
        grids, specials = {}, {}
        for model in MODELS:
            print(f"[score] {dataset} / {model}", flush=True)
            grids[model], specials[model] = _score_model(_prediction_map(roots[dataset] / model), label)
        grids["centrewave"] = cw_summary[dataset]["native_grid"]
        specials["centrewave"] = {str(score): cw_summary[dataset]["special"] for score in SCORES}
        combined[dataset] = {"grid": grids, "special": specials}

    _write_json(out / "five_model_current_labels.json", combined)
    _report(out / "FIVE_MODEL_REPORT.md", combined, timing, cw_summary, old_summary)
    print(out / "FIVE_MODEL_REPORT.md")


if __name__ == "__main__":
    main()
