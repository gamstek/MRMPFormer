"""Generate review-oriented plots and statistics for comparison confidence output."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "GREEN": "#2ca25f",
    "YELLOW": "#f0b429",
    "RED": "#d7301f",
    "UNAVAILABLE": "#7a7a7a",
    "INFO": "#4292c6",
}

REFERENCE_STATUSES = {
    "MATCHED_UNIQUE",
    "MATCHED_AMBIGUOUS",
    "TRAD_ONLY_AI_MISS",
    "REFERENCE_MISSING",
}

REASON_LABELS = {
    "SCORE_AI_MISSING": "AI模型分缺失",
    "TRAD_ONLY_AI_MISS": "传统峰未匹配到AI峰",
    "TRAD_REFERENCE_LOW_QUALITY": "传统参照质量低",
    "RT_HARD_LIMIT": "RT差超过0.30 min",
    "AREA_HARD_LIMIT": "面积差超过4倍",
    "MATCH_AMBIGUOUS": "匹配存在歧义",
    "AI_ONLY_EXTRA": "AI竞争性额外峰",
    "COMPOSITE_SCORE_THRESHOLD": "复合置信度低",
}


def _set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DengXian",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else math.nan


def _natural_sample_key(value: str) -> tuple[str, int]:
    match = re.match(r"^(.*?)(\d+)$", str(value))
    return (match.group(1), int(match.group(2))) if match else (str(value), -1)


def build_statistics(peaks: pd.DataFrame, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = peaks.match_status.isin(REFERENCE_STATUSES)
    matched = peaks.match_status.isin({"MATCHED_UNIQUE", "MATCHED_AMBIGUOUS"})
    extras = peaks.match_status.eq("AI_ONLY_EXTRA")
    scored_warning = peaks.alert_level.isin({"RED", "YELLOW"})
    review = peaks.review_required.fillna(False).astype(bool)

    n_all = len(peaks)
    n_reference = int(reference.sum())
    n_matched = int(matched.sum())
    n_extras = int(extras.sum())
    n_samples = len(samples)
    with_reference = samples.sample_alert_level.ne("UNAVAILABLE")
    n_samples_reference = int(with_reference.sum())

    rows = []

    def add(metric: str, count: int, denominator_name: str, denominator: int, note: str) -> None:
        rows.append(
            {
                "metric": metric,
                "count": int(count),
                "denominator_name": denominator_name,
                "denominator": int(denominator),
                "percentage": _pct(count, denominator),
                "note": note,
            }
        )

    add("全部输出行", n_all, "全部输出行", n_all, "每个传统峰记录或AI额外峰记录占一行")
    add("红黄正式预警", int(scored_warning.sum()), "全部输出行", n_all, "RED + YELLOW")
    add("红色预警", int(peaks.alert_level.eq("RED").sum()), "全部输出行", n_all, "高风险")
    add("黄色预警", int(peaks.alert_level.eq("YELLOW").sum()), "全部输出行", n_all, "中风险")
    add("无法评分", int(peaks.alert_level.eq("UNAVAILABLE").sum()), "全部输出行", n_all, "缺少传统参照或传统行字段不全")
    add("需要人工复核", int(review.sum()), "全部输出行", n_all, "红色、黄色和无法评分")
    add("传统参考峰", n_reference, "传统参考峰", n_reference, "传统Excel的全部行")
    add("传统参考峰红黄预警", int((reference & scored_warning).sum()), "传统参考峰", n_reference, "传统峰口径的RED + YELLOW")
    add("传统参考峰需复核", int((reference & review).sum()), "传统参考峰", n_reference, "包含传统字段缺失")
    add("成功匹配峰", n_matched, "传统参考峰", n_reference, "唯一匹配 + 歧义匹配")
    add("成功匹配峰红黄预警", int((matched & scored_warning).sum()), "成功匹配峰", n_matched, "匹配后比较结果为RED或YELLOW")
    add("成功匹配但AI分缺失", int((matched & peaks.score_ai.isna()).sum()), "成功匹配峰", n_matched, "触发硬红规则")
    add("传统峰未匹配到AI峰", int(peaks.match_status.eq("TRAD_ONLY_AI_MISS").sum()), "传统参考峰", n_reference, "AI漏检候选")
    add("AI额外峰", n_extras, "AI额外峰", n_extras, "没有占用传统峰的一对一匹配结果")
    add("竞争性AI额外峰预警", int((extras & scored_warning).sum()), "AI额外峰", n_extras, "可能与已分配AI峰竞争，黄色")
    add("样本总数", n_samples, "样本总数", n_samples, "test3样本")
    add("有传统参照的样本", n_samples_reference, "样本总数", n_samples, "可计算样本级结果")
    add("样本级红色", int((samples.sample_alert_level.eq("RED") & with_reference).sum()), "有传统参照的样本", n_samples_reference, "当前规则取样本内最严重峰")
    add("样本级无法评分", int(samples.sample_alert_level.eq("UNAVAILABLE").sum()), "样本总数", n_samples, "样本无传统参照")

    statistics = pd.DataFrame(rows)

    warning_rows = peaks.loc[scored_warning, ["alert_level", "alert_reasons"]].copy()
    warning_rows["alert_reasons"] = warning_rows.alert_reasons.fillna("").replace("", "COMPOSITE_SCORE_THRESHOLD")
    exploded = warning_rows.assign(reason=warning_rows.alert_reasons.str.split("|")).explode("reason")
    reason_counts = exploded.groupby("reason").size().sort_values(ascending=False)
    reason_stats = reason_counts.rename("count").reset_index()
    reason_stats["reason_label"] = reason_stats.reason.map(REASON_LABELS).fillna(reason_stats.reason)
    reason_stats["percentage_of_red_yellow"] = 100.0 * reason_stats["count"] / max(len(warning_rows), 1)
    reason_stats["note"] = "原因可以重叠，各行比例不能相加"
    return statistics, reason_stats


def _annotate_bars(ax: plt.Axes, bars, total: int, horizontal: bool = False) -> None:
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        label = f"{int(value):,}\n({_pct(value, total):.2f}%)"
        if horizontal:
            ax.text(value, bar.get_y() + bar.get_height() / 2, "  " + label.replace("\n", " "), va="center", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=9)


def plot_alert_overview(peaks: pd.DataFrame, output: Path) -> None:
    order = ["GREEN", "YELLOW", "RED", "UNAVAILABLE", "INFO"]
    names = ["绿色\n无需复核", "黄色\n预警", "红色\n预警", "无法评分\n需复核", "INFO\n额外峰记录"]
    counts = peaks.alert_level.value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(names, counts.values, color=[COLORS[x] for x in order], width=0.68)
    _annotate_bars(ax, bars, len(peaks))
    ax.set_ylabel("记录数")
    fig.suptitle(f"test3 全部输出状态（总计 {len(peaks):,} 条）", fontsize=15, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "红+黄为有风险分的正式预警；UNAVAILABLE虽无风险分，仍进入人工复核；INFO不复核",
        ha="center",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reference_alerts(peaks: pd.DataFrame, output: Path) -> None:
    reference = peaks[peaks.match_status.isin(REFERENCE_STATUSES)]
    order = ["GREEN", "YELLOW", "RED", "UNAVAILABLE"]
    names = ["绿色", "黄色", "红色", "无法评分"]
    counts = reference.alert_level.value_counts().reindex(order, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = [COLORS[x] for x in order]
    axes[0].pie(
        counts.values,
        labels=[f"{name}\n{count:,} ({_pct(count, len(reference)):.2f}%)" for name, count in zip(names, counts)],
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    axes[0].text(0, 0, f"传统参考峰\n{len(reference):,}", ha="center", va="center", fontsize=14, weight="bold")
    bars = axes[1].bar(names, counts.values, color=colors)
    _annotate_bars(axes[1], bars, len(reference))
    axes[1].set_ylabel("传统参考峰数")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("test3 传统参考峰口径的预警分布", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_warning_reasons(reason_stats: pd.DataFrame, warning_total: int, output: Path) -> None:
    data = reason_stats.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(data.reason_label, data["count"], color="#e6550d")
    _annotate_bars(ax, bars, warning_total, horizontal=True)
    ax.set_xlabel("涉及该原因的红/黄预警数")
    ax.set_title(f"test3 红/黄预警原因（共 {warning_total:,} 条预警）", fontsize=15, weight="bold")
    ax.text(0.99, 0.02, "同一条预警可有多个原因，比例不可相加", transform=ax.transAxes, ha="right", fontsize=10)
    ax.grid(axis="x", alpha=0.2)
    max_count = max(data["count"].max(), 1)
    ax.set_xlim(0, max_count * 1.28)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sample_profile(samples: pd.DataFrame, output: Path) -> None:
    data = samples.copy()
    data["_key"] = data.sample_id.map(_natural_sample_key)
    data = data.sort_values("_key").reset_index(drop=True)
    data["warning_rate"] = np.where(
        data.n_reference_rows.gt(0),
        100.0 * (data.n_red + data.n_yellow) / data.n_reference_rows,
        np.nan,
    )
    x = np.arange(len(data))
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    confidence_colors = data.sample_alert_level.map(COLORS).fillna("#7a7a7a")
    axes[0].bar(x, data.sample_confidence.fillna(0), color=confidence_colors)
    axes[0].plot(x, data.match_coverage, color="#2166ac", marker=".", linewidth=1, label="匹配覆盖率")
    axes[0].axhline(0.8, color=COLORS["GREEN"], linestyle="--", linewidth=1, label="绿色阈值 0.8")
    axes[0].axhline(0.6, color=COLORS["YELLOW"], linestyle="--", linewidth=1, label="黄色阈值 0.6")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("样本置信度 / 匹配覆盖率")
    axes[0].legend(ncol=3, loc="lower right")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, data.warning_rate.fillna(0), color="#d95f0e")
    axes[1].set_ylabel("传统参考峰红黄预警率 (%)")
    axes[1].set_xlabel("样本")
    axes[1].set_ylim(0, max(100, float(data.warning_rate.max(skipna=True)) * 1.08))
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(data.sample_id, rotation=70, ha="right", fontsize=8)
    fig.suptitle("test3 各样本置信度、匹配覆盖率与逐峰预警率", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_matched_diagnostics(peaks: pd.DataFrame, output: Path) -> None:
    data = peaks[peaks.match_status.isin({"MATCHED_UNIQUE", "MATCHED_AMBIGUOUS"})].copy()
    data["area_log2_fold"] = np.log2(data.area_fold.clip(lower=1.0))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for level in ["GREEN", "YELLOW", "RED"]:
        subset = data[data.alert_level.eq(level)]
        axes[0].scatter(
            subset.delta_rt_abs.clip(upper=0.55),
            subset.area_log2_fold.clip(upper=7.5),
            s=9,
            alpha=0.30,
            c=COLORS[level],
            label=f"{level} ({len(subset):,})",
            edgecolors="none",
        )
    axes[0].axvline(0.30, color="#222222", linestyle="--", linewidth=1, label="RT硬阈值")
    axes[0].axhline(2.0, color="#555555", linestyle=":", linewidth=1, label="面积4倍阈值")
    axes[0].set_xlabel("|AI RT - 传统 RT| (min，显示上限0.55)")
    axes[0].set_ylabel("log2(面积倍数，显示上限7.5)")
    axes[0].set_title("RT差与面积差")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.15)

    available_score = data[data.score_ai.notna()]
    missing_score = int(data.score_ai.isna().sum())
    axes[1].hist(
        [available_score.loc[available_score.alert_level.eq(x), "score_ai"] for x in ["GREEN", "YELLOW", "RED"]],
        bins=np.linspace(0.8, 1.0, 31),
        stacked=True,
        color=[COLORS[x] for x in ["GREEN", "YELLOW", "RED"]],
        label=["GREEN", "YELLOW", "RED"],
    )
    axes[1].set_xlabel("score_ai（仅非缺失记录）")
    axes[1].set_ylabel("成功匹配峰数")
    axes[1].set_title(f"AI模型分分布；另有 {missing_score:,} 个匹配峰缺少score_ai")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.15)
    fig.suptitle("test3 成功匹配峰的预警诊断", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(statistics: pd.DataFrame, reason_stats: pd.DataFrame, output: Path) -> None:
    lines = [
        "# test3 置信度预警统计与图表",
        "",
        "## 核心统计",
        "",
        "| 指标 | 数量 | 分母 | 占比 | 说明 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in statistics.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.count:,} | {row.denominator_name} {row.denominator:,} | "
            f"{row.percentage:.2f}% | {row.note} |"
        )
    lines.extend(
        [
            "",
            "## 红黄预警原因",
            "",
            "同一条记录可以同时触发多个原因，因此下面的占比不能相加。没有硬规则原因但复合置信度低于阈值的记录归入“复合置信度低”。",
            "",
            "| 原因 | 数量 | 占全部红黄预警 |",
            "|---|---:|---:|",
        ]
    )
    for row in reason_stats.itertuples(index=False):
        lines.append(f"| {row.reason_label} (`{row.reason}`) | {row.count:,} | {row.percentage_of_red_yellow:.2f}% |")
    lines.extend(
        [
            "",
            "## 图表",
            "",
            "- `01_alert_overview.png`：全部输出状态。",
            "- `02_reference_peak_alerts.png`：传统参考峰口径的预警率。",
            "- `03_warning_reasons.png`：红黄预警原因。",
            "- `04_sample_profile.png`：每个样本的置信度、匹配覆盖率和逐峰预警率。",
            "- `05_matched_diagnostics.png`：成功匹配峰的RT、面积和AI模型分诊断。",
            "",
            "## 解释限制",
            "",
            "当前样本级状态取样本内传统参考峰的最严重等级。因此，一个样本只要出现一个红峰，样本状态就是红色。样本级红色比例不能解释为该样本内全部峰均错误。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak_csv", required=True, type=Path)
    parser.add_argument("--sample_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    _set_chinese_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peaks = pd.read_csv(args.peak_csv, low_memory=False)
    samples = pd.read_csv(args.sample_csv, low_memory=False)
    statistics, reason_stats = build_statistics(peaks, samples)

    statistics.to_csv(args.output_dir / "warning_statistics.csv", index=False, encoding="utf-8-sig")
    reason_stats.to_csv(args.output_dir / "warning_reason_statistics.csv", index=False, encoding="utf-8-sig")
    plot_alert_overview(peaks, args.output_dir / "01_alert_overview.png")
    plot_reference_alerts(peaks, args.output_dir / "02_reference_peak_alerts.png")
    warning_total = int(peaks.alert_level.isin({"RED", "YELLOW"}).sum())
    plot_warning_reasons(reason_stats, warning_total, args.output_dir / "03_warning_reasons.png")
    plot_sample_profile(samples, args.output_dir / "04_sample_profile.png")
    plot_matched_diagnostics(peaks, args.output_dir / "05_matched_diagnostics.png")
    write_markdown(statistics, reason_stats, args.output_dir / "WARNING_STATISTICS.md")

    print(f"Output: {args.output_dir.resolve()}")
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
