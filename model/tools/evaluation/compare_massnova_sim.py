# -*- coding: utf-8 -*-
"""汇总 massnova 在模拟集三档（test_easy / test_medium / test_hard）上的评测结果，
生成跨数据集对比报告与可视化。

前置：各数据集已用 run_massnova_sim.py --eval_after 跑完，即存在
      output/inference/massnova_<dataset>/report/metrics_headline.csv 等文件。

用法（仓库根目录执行）：
    python model/tools/evaluation/compare_massnova_sim.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 三集对比关注的核心列（存在于 metrics_headline.csv）
HEAD_COLS = ["Precision", "Recall", "F1", "TP", "FP", "FN",
             "RT 绝对偏差中位数(min)", "边界 IoU 中位数", "面积 log-log 相关系数",
             "噪声通道误报率", "噪声通道平均误报峰数", "含峰通道检出率(≥1峰)"]
COL_CN = {
    "Precision": "精确率", "Recall": "召回率", "F1": "F1",
    "RT 绝对偏差中位数(min)": "RT偏差中位(min)", "边界 IoU 中位数": "IoU中位",
    "面积 log-log 相关系数": "面积log-log相关",
    "噪声通道误报率": "噪声通道误报率", "噪声通道平均误报峰数": "噪声通道FP/通道",
    "含峰通道检出率(≥1峰)": "含峰通道检出率",
}


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df is None or not len(df):
        return "_（无数据）_\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "-"
            return ("%.0f" % v) if abs(v) >= 1000 else float_fmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def load_one(root: Path, dataset: str):
    rep = root / ("massnova_%s" % dataset) / "report"
    if not (rep / "metrics_headline.csv").exists():
        return None
    head = pd.read_csv(rep / "metrics_headline.csv", encoding="utf-8-sig").iloc[0].to_dict()
    by_type = pd.read_csv(rep / "metrics_by_peak_type.csv", encoding="utf-8-sig") \
        if (rep / "metrics_by_peak_type.csv").exists() else pd.DataFrame()
    by_cond = pd.read_csv(rep / "metrics_by_condition.csv", encoding="utf-8-sig") \
        if (rep / "metrics_by_condition.csv").exists() else pd.DataFrame()
    return {"dataset": dataset, "report_dir": rep, "head": head,
            "by_type": by_type, "by_cond": by_cond}


def grouped_bar(ax, categories, series: dict, title, ylabel, ylim=None, rot=30):
    """series: {series_name: [values per category]}"""
    n = len(series)
    x = np.arange(len(categories))
    width = 0.8 / max(1, n)
    for k, (name, vals) in enumerate(series.items()):
        ax.bar(x + k * width - 0.4 + width / 2, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=rot, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def make_figures(items, figs_dir: Path):
    figs_dir.mkdir(parents=True, exist_ok=True)
    names = [it["dataset"] for it in items]
    figs = []

    # 1) P/R/F1 三集对比
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    grouped_bar(axes[0], ["Precision", "Recall", "F1"],
                {nm: [it["head"].get(k, np.nan) for k in ("Precision", "Recall", "F1")]
                 for nm, it in zip(names, items)}, "核心指标对比", "值", ylim=(0, 1.05), rot=0)
    grouped_bar(axes[1], ["TP", "FP", "FN"],
                {nm: [it["head"].get(k, np.nan) for k in ("TP", "FP", "FN")]
                 for nm, it in zip(names, items)}, "匹配计数对比", "峰数", rot=0)
    p = figs_dir / "cmp1_core_metrics.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 2) 按峰型召回率跨集对比
    types = []
    for it in items:
        if len(it["by_type"]):
            types.extend(it["by_type"].loc[it["by_type"]["GT 峰数"] > 0, "peak_type"].tolist())
    types = sorted(set(types))
    if types:
        series = {}
        for nm, it in zip(names, items):
            bt = it["by_type"]
            m = dict(zip(bt.get("peak_type", []), bt.get("召回率", []))) if len(bt) else {}
            series[nm] = [m.get(t, np.nan) for t in types]
        fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(types) + 4), 4.2))
        grouped_bar(ax, types, series, "按峰型召回率（三集对比）", "召回率", ylim=(0, 1.05))
        p = figs_dir / "cmp2_recall_by_peak_type.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 3) 按困难条件召回率跨集对比（取三集并集前 15）
    conds = []
    for it in items:
        if len(it["by_cond"]):
            conds.extend(it["by_cond"].loc[it["by_cond"]["GT 峰数"] > 0, "困难条件"].tolist())
    conds = sorted(set(conds))
    if conds:
        series = {}
        for nm, it in zip(names, items):
            bc = it["by_cond"]
            m = dict(zip(bc.get("困难条件", []), bc.get("召回率", []))) if len(bc) else {}
            series[nm] = [m.get(c, np.nan) for c in conds]
        fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(conds) + 4), 4.6))
        grouped_bar(ax, conds, series, "按困难条件召回率（三集对比）", "召回率", ylim=(0, 1.05), rot=35)
        p = figs_dir / "cmp3_recall_by_condition.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 4) 过检出与噪声误报对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    grouped_bar(axes[0], ["FP/TP", "噪声通道误报率"],
                {nm: [it["head"].get("FP", np.nan) / max(1e-9, it["head"].get("TP", np.nan)),
                      it["head"].get("噪声通道误报率", np.nan)] for nm, it in zip(names, items)},
                "过检出指标", "值/比例", rot=0)
    grouped_bar(axes[1], names,
                {"噪声通道FP/通道": [it["head"].get("噪声通道平均误报峰数", np.nan) for it in items]},
                "噪声通道平均误报峰数", "FP 峰数", rot=0)
    p = figs_dir / "cmp4_overdetection.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)
    return figs


def write_report(out_dir: Path, items, figs):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# massnova 模拟集三档对比测试报告\n"]
    lines.append("## 1. 数据与口径\n")
    lines.append("- 数据集：test_easy / test_medium / test_hard（MRM-XIC V7.2 模拟集，各 50 样本 × 176 通道）")
    lines.append("- 匹配口径：同通道内 apex RT 最近贪心一对一匹配，容差 ±0.2 min")
    lines.append("- 指标：TP=匹配成功的预测峰；FP=未匹配预测峰；FN=未匹配真值峰\n")

    lines.append("## 2. 核心指标对比\n")
    rows = []
    for it in items:
        r = {"数据集": it["dataset"]}
        for c in HEAD_COLS:
            if c in it["head"]:
                r[COL_CN.get(c, c)] = float(it["head"][c])
        rows.append(r)
    lines.append(_md_table(pd.DataFrame(rows)))
    lines.append("\n![核心指标](figures/cmp1_core_metrics.png)\n")

    lines.append("## 3. 分峰型召回率\n")
    for it in items:
        lines.append(f"\n### {it['dataset']}\n")
        lines.append(_md_table(it["by_type"]))
    lines.append("\n![峰型对比](figures/cmp2_recall_by_peak_type.png)\n")

    lines.append("## 4. 分困难条件召回率\n")
    for it in items:
        lines.append(f"\n### {it['dataset']}\n")
        lines.append(_md_table(it["by_cond"]))
    lines.append("\n![困难条件对比](figures/cmp3_recall_by_condition.png)\n")

    lines.append("## 5. 过检出与噪声误报\n")
    lines.append("![过检出对比](figures/cmp4_overdetection.png)\n")

    # 自动差异要点
    lines.append("## 6. 差异要点（自动汇总）\n")
    try:
        by = {it["dataset"]: it["head"] for it in items}
        def f(ds, k):
            return float(by.get(ds, {}).get(k, np.nan))
        lines.append("- 召回率：%s" % "；".join(
            "%s=%.4f" % (ds, f(ds, "Recall")) for ds in by))
        lines.append("- 精确率：%s" % "；".join(
            "%s=%.4f" % (ds, f(ds, "Precision")) for ds in by))
        lines.append("- FP/TP 比：%s" % "；".join(
            "%s=%.2f" % (ds, f(ds, "FP") / max(1e-9, f(ds, "TP"))) for ds in by))
        lines.append("- 噪声通道平均误报峰数：%s" % "；".join(
            "%s=%.2f" % (ds, f(ds, "噪声通道平均误报峰数")) for ds in by))
        # 最弱峰型（以 hard 为主）
        for it in items:
            bt = it["by_type"]
            if len(bt):
                bt = bt[bt["GT 峰数"] >= 50]
                if len(bt):
                    w = bt.sort_values("召回率").iloc[0]
                    lines.append("- 最弱峰型（%s，GT≥50）：%s 召回 %.4f"
                                 % (it["dataset"], w["peak_type"], w["召回率"]))
    except Exception as exc:  # 汇总失败不影响主报告
        lines.append(f"_（自动要点生成失败：{exc}）_")
    lines.append("\n## 7. 复现命令\n")
    lines.append("```powershell")
    lines.append("python model/tools/evaluation/run_massnova_sim.py --sim_root data/<dataset> \\")
    lines.append("  --model checkpoint/mrmpformerv2.pth --eval_after")
    lines.append("python model/tools/evaluation/compare_massnova_sim.py")
    lines.append("```\n")

    report = out_dir / "report_compare.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def build_args(argv=None):
    ap = argparse.ArgumentParser(description="massnova 模拟集三档对比报告")
    ap.add_argument("--root", type=str, default="output/inference", help="各数据集输出根目录")
    ap.add_argument("--datasets", type=str, default="test_easy,test_medium,test_hard")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="对比报告输出目录；缺省 <root>/sim_compare")
    return ap.parse_args(argv)


def main(argv=None):
    args = build_args(argv)
    root = Path(args.root)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    items = []
    for ds in datasets:
        it = load_one(root, ds)
        if it is None:
            print(f"[WARN] 缺少 {ds} 的评测结果（{root}/massnova_{ds}/report/metrics_headline.csv），已跳过")
            continue
        items.append(it)
    if not items:
        raise SystemExit("没有任何可用的评测结果，请先运行 run_massnova_sim.py --eval_after")

    out_dir = Path(args.out_dir) if args.out_dir else root / "sim_compare"
    figs = make_figures(items, out_dir / "figures")
    report = write_report(out_dir, items, figs)

    print(f"[DONE] 对比报告: {report}")
    print(f"[DONE] 图: {out_dir / 'figures'}（{len(figs)} 张）")
    for it in items:
        h = it["head"]
        print("  %-12s P=%.4f R=%.4f F1=%.4f (TP=%d FP=%d FN=%d)"
              % (it["dataset"], h.get("Precision", float("nan")), h.get("Recall", float("nan")),
                 h.get("F1", float("nan")), h.get("TP", 0), h.get("FP", 0), h.get("FN", 0)))


if __name__ == "__main__":
    main()
