"""Randomly sample channels and draw all AI/traditional peaks in one XIC plot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from inference.massnova import extract_full_xics


ALERT_COLORS = {
    "GREEN": "#238b45",
    "YELLOW": "#e6a700",
    "RED": "#d7301f",
    "UNAVAILABLE": "#756bb1",
    "INFO": "#8c8c8c",
}
SEVERITY = {"INFO": 0, "GREEN": 1, "YELLOW": 2, "UNAVAILABLE": 3, "RED": 4}


def _set_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _sample_number(value: object, prefix: str) -> float:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", str(value).strip(), flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def _safe_name(value: object, limit: int = 90) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "channel")).strip(" ._")
    return (text or "channel")[:limit]


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number:.{digits}f}" if np.isfinite(number) else "--"


def _valid_interval(lo: object, hi: object) -> bool:
    try:
        lo_value, hi_value = float(lo), float(hi)
    except (TypeError, ValueError):
        return False
    return np.isfinite(lo_value) and np.isfinite(hi_value) and hi_value > lo_value


def _boundary_text(lo: object, hi: object) -> str:
    """Format a peak interval as the a/b boundary pair shown in the plot table."""
    if not _valid_interval(lo, hi):
        return "--"
    return f"[{float(lo):.4f}, {float(hi):.4f}]"


def _traditional_boundary_for_ai(row: pd.Series) -> tuple[object, object]:
    """Use the assigned traditional peak, or the nearest diagnostic reference for extras."""
    if _valid_interval(row.get("trad_rt_min"), row.get("trad_rt_max")):
        return row.get("trad_rt_min"), row.get("trad_rt_max")
    return row.get("nearest_trad_rt_min"), row.get("nearest_trad_rt_max")


def _worst_alert(values) -> str:
    levels = [str(value) for value in values if str(value) in SEVERITY]
    return max(levels, key=lambda value: SEVERITY[value]) if levels else "UNAVAILABLE"


def _channel_catalog(rows: pd.DataFrame) -> pd.DataFrame:
    ai = rows[rows["ai_row_id"].notna()].copy()
    grouped = ai.groupby(["sample_id", "ai_chrom_index", "uid"], sort=False, dropna=False)
    catalog = grouped.agg(
        n_ai_peaks=("ai_row_id", "nunique"),
        n_ignored_weak=("weak_extra_ignored", "sum"),
        n_review_required=("review_required", "sum"),
    ).reset_index()
    alert = grouped["alert_level"].agg(_worst_alert).reset_index(name="channel_alert_level")
    return catalog.merge(alert, on=["sample_id", "ai_chrom_index", "uid"], how="left")


def _draw_channel(channel_rows: pd.DataFrame, feature: dict, output: Path) -> dict:
    rt = np.asarray(feature["rt"], dtype=float)
    intensity = np.asarray(feature["intensity"], dtype=float)
    ai = (
        channel_rows[channel_rows["ai_row_id"].notna()]
        .drop_duplicates("ai_row_id").sort_values("ai_rt_peak").copy()
    )
    trad = (
        channel_rows[channel_rows["trad_row_id"].notna()]
        .drop_duplicates("trad_row_id").sort_values("trad_rt_peak").copy()
    )
    n_table = max(1, len(ai))
    fig_height = 6.5 + 0.38 * n_table
    fig = plt.figure(figsize=(17.2, fig_height), dpi=115)
    grid = fig.add_gridspec(2, 1, height_ratios=[5.0, max(1.45, 0.43 * n_table)], hspace=0.16)
    ax = fig.add_subplot(grid[0])
    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    ax.plot(rt, intensity, color="#202020", linewidth=1.55, label="XIC", zorder=3)
    ymax = max(float(np.nanmax(intensity)) if len(intensity) else 1.0, 1.0)

    trad_labels: dict[float, str] = {}
    for number, (_, row) in enumerate(trad.iterrows(), start=1):
        trad_labels[float(row["trad_row_id"])] = f"T{number}"
        if _valid_interval(row.trad_rt_min, row.trad_rt_max):
            ax.axvspan(float(row.trad_rt_min), float(row.trad_rt_max), color="#f16913", alpha=0.16, zorder=1)
            ax.axvline(float(row.trad_rt_min), color="#f16913", linestyle="--", linewidth=1.0, zorder=2)
            ax.axvline(float(row.trad_rt_max), color="#f16913", linestyle="--", linewidth=1.0, zorder=2)
        if pd.notna(row.trad_rt_peak):
            peak_rt = float(row.trad_rt_peak)
            peak_y = float(np.interp(peak_rt, rt, intensity))
            ax.scatter([peak_rt], [peak_y], marker="v", s=48, color="#f16913", zorder=8)
            ax.annotate(f"T{number}", (peak_rt, peak_y), xytext=(0, 10), textcoords="offset points",
                        ha="center", color="#b33c00", fontsize=8.5, weight="bold")

    table_rows = []
    for display_no, (_, row) in enumerate(ai.iterrows(), start=1):
        level = str(row.alert_level)
        color = ALERT_COLORS.get(level, "#3182bd")
        ignored = bool(row.get("weak_extra_ignored", False))
        linestyle = ":" if ignored else "-"
        alpha = 0.10 if ignored else 0.18
        if _valid_interval(row.ai_rt_min, row.ai_rt_max):
            ax.axvspan(float(row.ai_rt_min), float(row.ai_rt_max), color=color, alpha=alpha, zorder=2)
            ax.axvline(float(row.ai_rt_min), color=color, linestyle=linestyle, linewidth=1.0, zorder=4)
            ax.axvline(float(row.ai_rt_max), color=color, linestyle=linestyle, linewidth=1.0, zorder=4)
        peak_rt = float(row.ai_rt_peak)
        peak_y = float(np.interp(peak_rt, rt, intensity))
        marker = "x" if ignored else "o"
        ax.scatter([peak_rt], [peak_y], marker=marker, s=52, color=color, zorder=9)
        short_level = "忽略" if ignored else level
        ax.annotate(
            f"A{display_no} {short_level}", (peak_rt, peak_y), xytext=(0, 18 + 12 * ((display_no - 1) % 2)),
            textcoords="offset points", ha="center", fontsize=8.2, color=color, weight="bold",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7}, zorder=10,
        )
        match_text = "EXTRA"
        if str(row.match_status).startswith("MATCHED") and pd.notna(row.trad_row_id):
            match_text = f"匹配{trad_labels.get(float(row.trad_row_id), 'T?')}"
        elif row.match_status == "NO_TRAD_SAMPLE":
            match_text = "无传统样本"
        pair_text = "是" if bool(row.get("paired_transition_supported", False)) else "否"
        trad_lo, trad_hi = _traditional_boundary_for_ai(row)
        table_rows.append([
            f"A{display_no}/peak#{int(float(row.ai_peak_no)) if pd.notna(row.ai_peak_no) else '?'}",
            "模型" if row.score_source == "model" else "信号",
            _fmt(row.ai_rt_peak, 4),
            _boundary_text(row.ai_rt_min, row.ai_rt_max),
            _boundary_text(trad_lo, trad_hi),
            f"{100 * float(row.relative_height_baseline_corrected):.1f}%" if pd.notna(row.relative_height_baseline_corrected) else "--",
            f"{100 * float(row.relative_area_in_channel):.1f}%" if pd.notna(row.relative_area_in_channel) else "--",
            _fmt(row.score_peak, 3),
            match_text,
            _fmt(row.get("delta_rt_abs"), 4),
            _fmt(row.get("final_confidence"), 3),
            pair_text,
            short_level,
        ])

    channel_level = _worst_alert(ai.alert_level)
    title_color = ALERT_COLORS.get(channel_level, "#222222")
    sample_id = str(ai.iloc[0].sample_id)
    uid = str(ai.iloc[0].uid)
    ax.set_title(
        f"{sample_id} | {uid} | 同通道全部峰：AI {len(ai)} / 传统 {len(trad)} | 通道状态 {channel_level}",
        fontsize=14, color=title_color, weight="bold", pad=12,
    )
    ax.set_xlabel("保留时间 RT (min)")
    ax.set_ylabel("强度")
    ax.set_xlim(float(np.min(rt)), float(np.max(rt)))
    ax.set_ylim(bottom=min(0.0, float(np.nanmin(intensity))), top=ymax * 1.17)
    ax.grid(alpha=0.18)
    ax.legend(handles=[
        Line2D([0], [0], color="#202020", lw=1.6, label="XIC"),
        Patch(facecolor="#f16913", alpha=0.20, label="MassNova传统峰"),
        Patch(facecolor="#d7301f", alpha=0.20, label="AI红色报警峰"),
        Patch(facecolor="#e6a700", alpha=0.20, label="AI黄色报警峰"),
        Patch(facecolor="#238b45", alpha=0.20, label="AI绿色峰"),
        Line2D([0], [0], marker="x", color="#8c8c8c", lw=0, label="极弱额外峰：保留但不报警"),
    ], loc="upper right", fontsize=8.5, ncol=2)

    columns = ["AI峰", "来源", "峰顶RT", "AI边界[a,b]", "传统边界[a,b]", "相对峰高", "相对面积",
               "峰自身分", "匹配", "|ΔRT|", "最终置信度", "配对通道", "报警"]
    column_widths = [0.085, 0.055, 0.065, 0.118, 0.118, 0.075, 0.075,
                     0.075, 0.075, 0.065, 0.083, 0.070, 0.055]
    table = table_ax.table(cellText=table_rows, colLabels=columns, colWidths=column_widths,
                           cellLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    table.scale(1.0, 1.28)
    for (row_no, col_no), cell in table.get_celld().items():
        if row_no == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif col_no == len(columns) - 1:
            level_text = table_rows[row_no - 1][-1]
            lookup = "INFO" if level_text == "忽略" else level_text
            cell.set_facecolor(ALERT_COLORS.get(lookup, "#ffffff"))
            cell.set_alpha(0.20)
    table_ax.text(
        0.0, -0.02,
        "弱峰规则：AI_ONLY_EXTRA + signal_rule + 基线校正相对峰高<10% + 相对面积<10% + 无配对transition支持 → 灰色INFO，不需复核。",
        transform=table_ax.transAxes, fontsize=8.5, color="#555555", va="top",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return {
        "sample_id": sample_id,
        "uid": uid,
        "ai_chrom_index": int(float(ai.iloc[0].ai_chrom_index)),
        "n_ai_peaks": len(ai),
        "n_trad_peaks": len(trad),
        "n_ignored_weak": int(ai.weak_extra_ignored.fillna(False).sum()),
        "n_review_required": int(ai.review_required.fillna(False).sum()),
        "channel_alert_level": channel_level,
    }


def _write_html(index: pd.DataFrame, output_dir: Path) -> None:
    cards = []
    for row in index.itertuples(index=False):
        path = str(row.plot_path).replace("\\", "/")
        title = html.escape(f"{row.sample_id} | {row.uid}")
        meta = html.escape(
            f"{row.channel_alert_level} | AI峰 {row.n_ai_peaks} | 传统峰 {row.n_trad_peaks} | "
            f"忽略弱峰 {row.n_ignored_weak} | 需复核 {row.n_review_required}"
        )
        cards.append(
            f'<article class="card" data-level="{html.escape(str(row.channel_alert_level))}">'
            f'<a href="{path}"><img loading="lazy" src="{path}" alt="{title}"></a>'
            f'<div class="title">{title}</div><div class="meta">{meta}</div></article>'
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>test3随机3000通道多峰置信度图</title><style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:22px;background:#f4f6f8;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}}
.card{{background:#fff;border-radius:8px;padding:9px;box-shadow:0 1px 5px #bbb}}
.card img{{width:100%;border:1px solid #ddd}}.title{{font-weight:700;margin-top:7px}}.meta{{font-size:13px;color:#666}}
</style></head><body><h1>test3_3～test3_56：随机3000通道多峰置信度与报警</h1>
<p>每张图展示一个MRM通道中的全部AI峰和传统峰。灰色×为极弱额外信号峰，保留审计但不报警。</p>
<div class="grid">{''.join(cards)}</div></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _write_readme(output_dir: Path, selected: pd.DataFrame, index: pd.DataFrame,
                  failures: pd.DataFrame, args: argparse.Namespace, elapsed: float) -> None:
    levels = index.channel_alert_level.value_counts()
    counts = "\n".join(f"| {name} | {int(value)} |" for name, value in levels.items())
    text = f"""# test3随机3000通道多峰置信度与报警图

- 样本范围：`{args.sample_prefix}_{args.sample_min}`～`{args.sample_prefix}_{args.sample_max}`
- 抽样单位：同一样本中的一个MRM通道
- 随机种子：`{args.seed}`
- 可抽样通道：{args.eligible_channel_count}
- 请求/实际抽样：{args.random_n}/{len(selected)}
- 成功出图：{len(index)}
- 失败：{len(failures)}
- 图中AI峰：{int(index.n_ai_peaks.sum())}
- 图中传统峰：{int(index.n_trad_peaks.sum())}
- 灰色不报警弱峰：{int(index.n_ignored_weak.sum())}
- 仍需复核的AI峰：{int(index.n_review_required.sum())}
- 耗时：{elapsed:.1f}秒
- 图下方每个AI峰均列出 `AI边界[a,b]` 与对应的 `传统边界[a,b]`；额外AI峰显示同通道最近传统参考的边界。

## 通道最高报警等级

| 等级 | 通道数 |
|---|---:|
{counts}

## 弱峰规则

仅将同时满足下列条件的峰设为 `INFO / IGNORED_WEAK_AI_EXTRA`：

1. `AI_ONLY_EXTRA`；
2. `score_source=signal_rule`；
3. 基线校正相对峰高小于10%；
4. 同通道相对面积小于10%；
5. 同化合物另一transition在0.10 min内没有峰支持。

峰数据、置信度和原报警状态仍保留在CSV中。正式匹配峰、模型峰和无传统样本峰不被该规则抑制。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _process_sample(
    sample_id: str,
    sample_selection: pd.DataFrame,
    sample_rows: pd.DataFrame,
    mzml_path: Path,
    output_dir: Path,
    smooth_sigma: float,
) -> tuple[list[dict], list[dict]]:
    """Draw one sample's selected channels; suitable for a worker process."""
    _set_font()
    index_rows: list[dict] = []
    failures: list[dict] = []
    try:
        features, _ = extract_full_xics(mzml_path, smooth_sigma=smooth_sigma)
    except Exception as exc:
        return [], [{"sample_id": sample_id, "reason": f"mzML extraction failed: {exc!r}"}]
    by_index = {int(feature["chrom_index"]): feature for feature in features}
    for selection in sample_selection.sort_values("random_order").itertuples(index=False):
        serial = int(selection.random_order)
        chrom_index = int(float(selection.ai_chrom_index))
        feature = by_index.get(chrom_index)
        if feature is None:
            failures.append({"sample_id": sample_id, "ai_chrom_index": chrom_index,
                             "uid": selection.uid, "reason": "channel missing from mzML extraction"})
            continue
        channel_rows = sample_rows[sample_rows["uid"] == selection.uid].copy()
        folder = output_dir / "plots" / _safe_name(sample_id)
        name = f"{serial:04d}_chrom{chrom_index:03d}_{_safe_name(selection.uid)}_{selection.channel_alert_level}.png"
        output = folder / name
        try:
            record = _draw_channel(channel_rows, feature, output)
            record["random_order"] = serial
            record["plot_path"] = str(output.relative_to(output_dir))
            index_rows.append(record)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "ai_chrom_index": chrom_index,
                             "uid": selection.uid, "reason": repr(exc)})
    return index_rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence_csv", required=True, type=Path)
    parser.add_argument("--mzml_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--random_n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--sample_prefix", default="test3")
    parser.add_argument("--sample_min", type=int, default=3)
    parser.add_argument("--sample_max", type=int, default=56)
    parser.add_argument("--smooth_sigma", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--selection_csv", type=Path, default=None,
                        help="复用既有selected_channels.csv，保持同一批通道和random_order，不重新随机抽样")
    args = parser.parse_args()

    started = time.perf_counter()
    _set_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.confidence_csv, low_memory=False)
    rows["_sample_number"] = rows["sample_id"].map(lambda value: _sample_number(value, args.sample_prefix))
    rows = rows[rows["_sample_number"].between(args.sample_min, args.sample_max, inclusive="both")].copy()
    catalog = _channel_catalog(rows)
    args.eligible_channel_count = len(catalog)
    if len(catalog) < args.random_n:
        raise SystemExit(f"Only {len(catalog)} eligible channels, fewer than {args.random_n}")
    if args.selection_csv is not None:
        selected = pd.read_csv(args.selection_csv).copy()
        required = {"sample_id", "ai_chrom_index", "uid", "random_order"}
        missing = sorted(required - set(selected.columns))
        if missing:
            raise SystemExit(f"Selection CSV is missing columns: {missing}")
        selected = selected.merge(
            catalog,
            on=["sample_id", "ai_chrom_index", "uid"],
            how="left",
            suffixes=("_old", ""),
        )
        if selected["channel_alert_level"].isna().any():
            raise SystemExit("Selection CSV contains channels absent from current confidence data")
        args.random_n = len(selected)
    else:
        selected = catalog.sample(n=args.random_n, random_state=args.seed, replace=False).copy()
        selected["random_order"] = np.arange(1, len(selected) + 1)
    selected.to_csv(args.output_dir / "selected_channels.csv", index=False, encoding="utf-8-sig")

    mzml_paths = {path.stem: path for path in args.mzml_dir.glob("*.mzML")}
    index_rows, failures = [], []
    jobs = []
    for sample_id, sample_selection in selected.groupby("sample_id", sort=True):
        mzml_path = mzml_paths.get(str(sample_id))
        if mzml_path is None:
            failures.append({"sample_id": sample_id, "reason": "mzML missing"})
            continue
        jobs.append((
            str(sample_id), sample_selection.copy(), rows[rows["sample_id"] == sample_id].copy(),
            mzml_path, args.output_dir, args.smooth_sigma,
        ))
    if args.workers <= 1:
        for job in jobs:
            made, failed = _process_sample(*job)
            index_rows.extend(made)
            failures.extend(failed)
            print(f"[{job[0]}] plots={len(index_rows)} failures={len(failures)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_sample, *job): job[0] for job in jobs}
            for future in as_completed(futures):
                sample_id = futures[future]
                try:
                    made, failed = future.result()
                    index_rows.extend(made)
                    failures.extend(failed)
                except Exception as exc:
                    failures.append({"sample_id": sample_id, "reason": f"worker failed: {exc!r}"})
                print(f"[{sample_id}] plots={len(index_rows)} failures={len(failures)}", flush=True)

    index = pd.DataFrame(index_rows).sort_values("random_order") if index_rows else pd.DataFrame()
    failure_frame = pd.DataFrame(failures)
    index.to_csv(args.output_dir / "channel_plot_index.csv", index=False, encoding="utf-8-sig")
    failure_frame.to_csv(args.output_dir / "plot_failures.csv", index=False, encoding="utf-8-sig")
    if len(index):
        _write_html(index, args.output_dir)
    elapsed = time.perf_counter() - started
    _write_readme(args.output_dir, selected, index, failure_frame, args, elapsed)
    print(f"Generated {len(index)} channel plots; failures={len(failure_frame)}; elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
