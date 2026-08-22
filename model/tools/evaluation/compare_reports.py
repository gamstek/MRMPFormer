# -*- coding: utf-8 -*-
"""
汇总多份 evaluation_report.json → 模型对比报告（markdown）。

用法（model/ 目录下）：
  python -m tools.evaluation.compare_reports \
      --reports baseline=../output/evaluation/v3_cmp/baseline/evaluation_report.json \
                v2=../output/evaluation/v3_cmp/v2/evaluation_report.json \
                v3=../output/evaluation/v3_cmp/v3/evaluation_report.json \
      --out ../output/evaluation/v3_cmp/comparison.md
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

_ROW_KEYS = [
    ("precision", "Precision", "{:.4f}"),
    ("recall", "Recall", "{:.4f}"),
    ("f1", "F1", "{:.4f}"),
    ("TP", "TP", "{}"),
    ("FP", "FP", "{}"),
    ("FN", "FN", "{}"),
    ("area_r2_pred_vs_manual", "面积 R²（预测 vs 人工）", "{:.5f}"),
    ("n_area_pairs", "面积配对对数", "{}"),
    ("rt_start_dev_median_min", "RT 起偏差中位(min)", "{:.4f}"),
    ("rt_end_dev_median_min", "RT 止偏差中位(min)", "{:.4f}"),
    ("rsd_median", "RSD 中位", "{:.4f}"),
    ("n_rsd_compounds", "RSD 化合物数", "{}"),
]


def main():
    ap = argparse.ArgumentParser(description="多模型评估报告对比汇总")
    ap.add_argument("--reports", nargs="+", required=True,
                    help="name=path/to/evaluation_report.json（可多个）")
    ap.add_argument("--out", default="../output/evaluation/comparison.md",
                    help="对比报告输出路径")
    args = ap.parse_args()

    reports = []
    for item in args.reports:
        name, path = item.split("=", 1)
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        reports.append((name, data))

    lines = [
        "# 模型评估对比报告",
        "",
        "- 生成时间: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "- 评估数据: %s" % reports[0][1]["labels"] if reports else "",
        "- 协议: 命中判据 起止偏差 ≤ ±%s min | 定量配对 ±%s min | score ≥ %s" % (
            reports[0][1].get("tolerance"), reports[0][1].get("quant_tolerance"),
            "0.90"),
        "",
        "| 指标 | " + " | ".join(n for n, _ in reports) + " |",
        "|---|---" * len(reports) + "|",
    ]
    for key, label, fmt in _ROW_KEYS:
        row = []
        for _n, data in reports:
            v = data.get("metrics", {}).get(key)
            row.append(fmt.format(v) if isinstance(v, (int, float)) else "N/A")
        lines.append("| %s | %s |" % (label, " | ".join(row)))

    lines += ["", "## 结论（人工解读）", "",
              "按 F1 / 面积 R² / RSD 综合评估各模型在 shiyaoyuan 数据上的检测与定量能力。"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[DONE] 对比报告: %s" % out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
