"""Draw XIC peak overlays for confidence warnings and green controls."""

from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from inference.massnova import extract_full_xics


COLORS = {
    "GREEN": "#238b45",
    "YELLOW": "#e6a700",
    "RED": "#d7301f",
    "UNAVAILABLE": "#777777",
    "INFO": "#3182bd",
}

CATEGORY_SPECS = [
    ("01_ai_score_missing", "AI模型分缺失", "SCORE_AI_MISSING", "comparison_confidence", True),
    ("02_trad_peak_missed", "传统峰未匹配到AI峰", "TRAD_ONLY_AI_MISS", "delta_rt_abs", False),
    ("03_rt_hard_limit", "RT差超过0.30 min", "RT_HARD_LIMIT", "delta_rt_abs", False),
    ("04_area_hard_limit", "面积差超过4倍", "AREA_HARD_LIMIT", "area_fold", False),
    ("05_composite_low", "复合置信度低", "COMPOSITE_SCORE_THRESHOLD", "comparison_confidence", True),
    ("06_trad_low_quality", "传统参照质量低", "TRAD_REFERENCE_LOW_QUALITY", "quality_trad_rule", True),
    ("07_ambiguous_match", "匹配存在歧义", "MATCH_AMBIGUOUS", "match_margin", True),
    ("08_competitive_extra", "AI竞争性额外峰", "AI_ONLY_EXTRA", "match_cost", True),
]


def _set_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _safe_name(value: object, max_len: int = 72) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return (text or "unknown")[:max_len]


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def _reason_mask(series: pd.Series, reason: str) -> pd.Series:
    text = series.fillna("").astype(str)
    if reason == "COMPOSITE_SCORE_THRESHOLD":
        return text.eq("")
    if reason == "AI_ONLY_EXTRA":
        return text.str.contains(r"(?:^|\|)AI_ONLY_EXTRA(?:$|\|)", regex=True)
    return text.str.contains(rf"(?:^|\|){re.escape(reason)}(?:$|\|)", regex=True)


def _diverse_take(data: pd.DataFrame, limit: int, sort_col: str, ascending: bool) -> pd.DataFrame:
    if data.empty or limit <= 0:
        return data.iloc[0:0]
    if sort_col in data.columns:
        data = data.sort_values(sort_col, ascending=ascending, na_position="last")
    first = data.drop_duplicates("sample_id", keep="first")
    picked = first.head(limit)
    if len(picked) < limit:
        remaining = data.loc[~data._source_index.isin(picked._source_index)]
        picked = pd.concat([picked, remaining.head(limit - len(picked))], ignore_index=False)
    return picked.head(limit)


def select_gallery_rows(peaks: pd.DataFrame, per_category: int, green_controls: int) -> pd.DataFrame:
    data = peaks.copy()
    data["_source_index"] = np.arange(len(data), dtype=int)
    warnings = data[data.alert_level.isin({"RED", "YELLOW"})]
    selected = []
    used: set[int] = set()

    for folder, label, reason, sort_col, ascending in CATEGORY_SPECS:
        mask = _reason_mask(warnings.alert_reasons, reason)
        candidates = warnings.loc[mask & ~warnings._source_index.isin(used)]
        if reason == "AI_ONLY_EXTRA":
            candidates = candidates[candidates.match_status.eq("AI_ONLY_EXTRA")]
        chosen = _diverse_take(candidates, per_category, sort_col, ascending).copy()
        if chosen.empty:
            continue
        chosen["gallery_category"] = folder
        chosen["gallery_category_label"] = label
        selected.append(chosen)
        used.update(chosen._source_index.astype(int))

    green = data[data.alert_level.eq("GREEN") & ~data._source_index.isin(used)]
    green = _diverse_take(green, green_controls, "comparison_confidence", False).copy()
    if not green.empty:
        green["gallery_category"] = "09_green_controls"
        green["gallery_category_label"] = "绿色对照峰"
        selected.append(green)

    unavailable = data[data.alert_level.eq("UNAVAILABLE") & data.trad_row_id.notna() & ~data._source_index.isin(used)]
    unavailable = _diverse_take(unavailable, min(6, per_category), "trad_row_id", True).copy()
    if not unavailable.empty:
        unavailable["gallery_category"] = "10_reference_missing"
        unavailable["gallery_category_label"] = "传统字段缺失"
        selected.append(unavailable)

    if not selected:
        return data.iloc[0:0]
    return pd.concat(selected, ignore_index=True)


def _feature_lookup(features: list[dict]) -> tuple[dict[str, dict], dict[int, dict]]:
    by_uid = {}
    by_index = {}
    for feature in features:
        by_uid.setdefault(str(feature.get("uid", "")), feature)
        by_index[int(feature["chrom_index"])] = feature
    return by_uid, by_index


def _resolve_feature(row: pd.Series, by_uid: dict[str, dict], by_index: dict[int, dict]) -> dict | None:
    for column in ("ai_chrom_index",):
        value = row.get(column)
        if pd.notna(value) and int(value) in by_index:
            return by_index[int(value)]
    uid = str(row.get("uid", ""))
    if uid in by_uid:
        return by_uid[uid]
    folded = uid.casefold()
    return next((feature for key, feature in by_uid.items() if key.casefold() == folded), None)


def _valid_interval(lo: object, hi: object) -> bool:
    try:
        lo_float, hi_float = float(lo), float(hi)
    except (TypeError, ValueError):
        return False
    return np.isfinite(lo_float) and np.isfinite(hi_float) and hi_float > lo_float


def _plot_interval(ax: plt.Axes, lo: float, hi: float, apex: object, color: str, alpha: float,
                   linestyle: str, linewidth: float, label: str | None = None) -> None:
    ax.axvspan(float(lo), float(hi), color=color, alpha=alpha, label=label)
    ax.axvline(float(lo), color=color, linestyle=linestyle, linewidth=linewidth)
    ax.axvline(float(hi), color=color, linestyle=linestyle, linewidth=linewidth)
    try:
        apex_float = float(apex)
    except (TypeError, ValueError):
        return
    if np.isfinite(apex_float):
        ax.axvline(apex_float, color=color, linestyle=":", linewidth=linewidth + 0.3)


def _selected_traditional_reference(row: pd.Series) -> dict[str, object]:
    """Return assigned or nearest diagnostic traditional values for one AI event."""
    nearest = str(row.get("comparison_reference_type", "")) == "NEAREST_UNASSIGNED"
    prefix = "nearest_trad_" if nearest else "trad_"
    return {
        "rt_min": row.get(f"{prefix}rt_min", math.nan),
        "rt_peak": row.get(f"{prefix}rt_peak", math.nan),
        "rt_max": row.get(f"{prefix}rt_max", math.nan),
        "area": row.get(f"{prefix}area", math.nan),
        "snr": row.get(f"{prefix}snr", math.nan),
        "label": "最近传统参考" if nearest else "传统算法正式匹配峰",
        "nearest": nearest,
    }


def draw_peak_plot(row: pd.Series, channel_rows: pd.DataFrame, feature: dict, output: Path) -> None:
    rt = np.asarray(feature["rt"], dtype=float)
    intensity = np.asarray(feature["intensity"], dtype=float)
    fig = plt.figure(figsize=(11.5, 8.1), dpi=120)
    grid = fig.add_gridspec(2, 1, height_ratios=[4.8, 1.8], hspace=0.17)
    ax = fig.add_subplot(grid[0])
    info_ax = fig.add_subplot(grid[1])
    info_ax.axis("off")

    ax.plot(rt, intensity, color="#171717", linewidth=1.7, label="XIC（推理同口径平滑）", zorder=3)
    selected_source = int(row["_source_index"])

    # Context intervals are deliberately faint; the selected pair is drawn afterwards.
    for source_index, other in channel_rows.iterrows():
        if int(other["_source_index"]) == selected_source:
            continue
        if _valid_interval(other.trad_rt_min, other.trad_rt_max):
            _plot_interval(ax, other.trad_rt_min, other.trad_rt_max, other.trad_rt_peak,
                           "#41ab5d", 0.06, "--", 0.7)
        if pd.notna(other.ai_row_id) and _valid_interval(other.ai_rt_min, other.ai_rt_max):
            _plot_interval(ax, other.ai_rt_min, other.ai_rt_max, other.ai_rt_peak,
                           "#3182bd", 0.05, "--", 0.7)

    reference = _selected_traditional_reference(row)
    ref_color = "#e6550d" if reference["nearest"] else "#31a354"
    owner = None
    if reference["nearest"] and pd.notna(row.get("nearest_trad_row_id")):
        trad_ids = pd.to_numeric(channel_rows.get("trad_row_id"), errors="coerce")
        owner_candidates = channel_rows[
            trad_ids.eq(float(row["nearest_trad_row_id"]))
            & channel_rows.match_status.astype(str).str.startswith("MATCHED")
            & channel_rows.ai_row_id.notna()
        ]
        if not owner_candidates.empty:
            owner = owner_candidates.iloc[0]
            if _valid_interval(owner.ai_rt_min, owner.ai_rt_max):
                _plot_interval(ax, owner.ai_rt_min, owner.ai_rt_max, owner.ai_rt_peak,
                               "#756bb1", 0.24, "-.", 1.3,
                               "已匹配该传统峰的AI峰")
    if _valid_interval(reference["rt_min"], reference["rt_max"]):
        _plot_interval(ax, reference["rt_min"], reference["rt_max"], reference["rt_peak"],
                       ref_color, 0.27, "--", 1.4, str(reference["label"]))
    if pd.notna(row.ai_row_id) and _valid_interval(row.ai_rt_min, row.ai_rt_max):
        _plot_interval(ax, row.ai_rt_min, row.ai_rt_max, row.ai_rt_peak,
                       "#2b8cbe", 0.27, "-.", 1.4, "本次评估AI峰")
        selected_rt = float(row.ai_rt_peak)
        selected_y = float(np.interp(selected_rt, rt, intensity))
        ax.annotate(
            f"本次评估 AI peak#{int(float(row.ai_peak_no))}",
            xy=(selected_rt, selected_y), xytext=(8, 18), textcoords="offset points",
            fontsize=9, color="#0868ac", weight="bold",
            arrowprops={"arrowstyle": "->", "color": "#0868ac", "lw": 1.0},
            zorder=8,
        )
    if owner is not None and pd.notna(owner.ai_rt_peak):
        owner_rt = float(owner.ai_rt_peak)
        owner_y = float(np.interp(owner_rt, rt, intensity))
        ax.annotate(
            f"已正式匹配 AI peak#{int(float(owner.ai_peak_no))}",
            xy=(owner_rt, owner_y), xytext=(-12, -42), textcoords="offset points",
            fontsize=8.5, color="#54278f", weight="bold",
            ha="right",
            arrowprops={"arrowstyle": "->", "color": "#54278f", "lw": 0.9},
            zorder=8,
        )

    relevant = []
    for value in (reference["rt_min"], reference["rt_peak"], reference["rt_max"],
                  row.get("ai_rt_min"), row.get("ai_rt_peak"), row.get("ai_rt_max")):
        if pd.notna(value) and np.isfinite(float(value)):
            relevant.append(float(value))
    if relevant and rt.size:
        lo = max(float(rt.min()), min(relevant) - 0.35)
        hi = min(float(rt.max()), max(relevant) + 0.35)
        if hi - lo < 0.8:
            center = 0.5 * (lo + hi)
            lo = max(float(rt.min()), center - 0.4)
            hi = min(float(rt.max()), center + 0.4)
        ax.set_xlim(lo, hi)
    elif rt.size:
        ax.set_xlim(float(rt.min()), float(rt.max()))

    alert = str(row.alert_level)
    alert_color = COLORS.get(alert, "#333333")
    category_label = str(row.get("gallery_category_label", "置信度随机抽样"))
    selected_peak_text = (f"AI peak#{int(float(row.ai_peak_no))}"
                          if pd.notna(row.get("ai_peak_no")) else "无AI峰")
    title = f"{row.sample_id} | {row.uid} | {selected_peak_text} | {alert} | {category_label}"
    ax.set_title(title, color=alert_color, fontsize=14, weight="bold", pad=10)
    ax.set_ylabel("强度")
    ax.grid(alpha=0.2)
    handles = [Line2D([0], [0], color="#171717", lw=1.7, label="XIC")]
    if _valid_interval(reference["rt_min"], reference["rt_max"]):
        handles.append(Patch(facecolor=ref_color, alpha=0.27, label=str(reference["label"])))
    if owner is not None and _valid_interval(owner.ai_rt_min, owner.ai_rt_max):
        handles.append(Patch(facecolor="#756bb1", alpha=0.24,
                             label="已匹配该传统峰的AI峰"))
    handles.append(Patch(facecolor="#2b8cbe", alpha=0.27, label="本次评估AI峰区间"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    left_text = (
        f"传统参考  RT={_fmt(reference['rt_peak'])}  区间=[{_fmt(reference['rt_min'])}, {_fmt(reference['rt_max'])}]  "
        f"面积={_fmt(reference['area'], 2)}  SNR={_fmt(reference['snr'], 2)}  "
        f"类型={row.get('comparison_reference_type', 'NONE')}\n"
        f"AI组合算法 RT={_fmt(row.ai_rt_peak)}  区间=[{_fmt(row.ai_rt_min)}, {_fmt(row.ai_rt_max)}]  "
        f"面积={_fmt(row.ai_area, 2)}  峰分={_fmt(row.get('score_peak'), 3)}  "
        f"来源={row.get('score_source', '--')}  score_ai={_fmt(row.get('score_ai'), 3)}"
    )
    right_text = (
        f"匹配状态={row.match_status}\n"
        f"|ΔRT|={_fmt(row.delta_rt_abs)} min  |Δ面积|={_fmt(row.get('area_diff_abs'), 2)}  "
        f"面积差={_fmt(row.get('area_diff_pct'), 2)}%\n"
        f"比较置信度={_fmt(row.comparison_confidence, 3)}  最终置信度={_fmt(row.get('final_confidence'), 3)}  "
        f"状态系数={_fmt(row.get('conf_status'), 2)}  风险分={_fmt(row.final_rule_risk_score, 1)}\n"
        f"原因={row.alert_reasons if pd.notna(row.alert_reasons) and str(row.alert_reasons) else '仅由复合置信度阈值决定'}"
    )
    if owner is not None:
        right_text += (
            f"\n最近传统峰已由 AI peak#{int(float(owner.ai_peak_no))} "
            f"(RT={_fmt(owner.ai_rt_peak)}) 正式匹配；本图评估 peak#{int(float(row.ai_peak_no))}"
        )
    info_ax.text(0.01, 0.96, left_text, va="top", fontsize=9.7, linespacing=1.40)
    info_ax.text(0.01, 0.48, right_text, va="top", fontsize=9.7, linespacing=1.36,
                 bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": alert_color})
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_html(index: pd.DataFrame, output_dir: Path) -> None:
    cards = []
    for row in index.itertuples(index=False):
        rel = str(row.plot_path).replace("\\", "/")
        title = html.escape(f"{row.sample_id} | {row.uid}")
        meta = html.escape(f"{row.alert_level} · {row.gallery_category_label} · {row.match_status}")
        cards.append(
            f'<article class="card"><a href="{rel}"><img loading="lazy" src="{rel}" alt="{title}"></a>'
            f'<div class="title">{title}</div><div class="meta">{meta}</div></article>'
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>test3 峰预警图集</title>
<style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:24px;background:#f4f6f8;color:#222}}
h1{{margin-bottom:4px}} p{{color:#555}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:18px}}
.card{{background:white;border-radius:9px;padding:10px;box-shadow:0 1px 5px #bbb}}
.card img{{width:100%;height:auto;border:1px solid #ddd}} .title{{font-weight:700;margin-top:8px}} .meta{{font-size:13px;color:#666;margin-top:4px}}
</style></head><body><h1>test3 传统算法与AI组合算法峰对比图集</h1>
<p>绿色阴影为MassNova传统峰，蓝色阴影为MRMPFormer＋信号算法峰。点击缩略图查看原图。</p>
<div class="grid">{''.join(cards)}</div></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence_csv", required=True, type=Path)
    parser.add_argument("--mzml_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--per_category", type=int, default=12)
    parser.add_argument("--green_controls", type=int, default=16)
    parser.add_argument("--smooth_sigma", type=float, default=0.8)
    args = parser.parse_args()

    _set_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peaks = pd.read_csv(args.confidence_csv, low_memory=False)
    peaks["_source_index"] = np.arange(len(peaks), dtype=int)
    selected = select_gallery_rows(peaks, args.per_category, args.green_controls)
    if selected.empty:
        raise SystemExit("No rows selected")

    mzml_paths = {path.stem: path for path in args.mzml_dir.glob("*.mzML")}
    mzml_paths.update({path.stem: path for path in args.mzml_dir.glob("*.mzml")})
    index_rows = []
    failures = []
    serial = 0

    for sample_id, sample_selection in selected.groupby("sample_id", sort=True):
        mzml_path = mzml_paths.get(str(sample_id))
        if mzml_path is None:
            for _, row in sample_selection.iterrows():
                failures.append({"source_index": int(row["_source_index"]), "sample_id": sample_id, "uid": row.uid,
                                 "error": "mzML not found"})
            continue
        try:
            features, _ = extract_full_xics(mzml_path, smooth_sigma=args.smooth_sigma, verbose=False)
            by_uid, by_index = _feature_lookup(features)
        except Exception as exc:
            for _, row in sample_selection.iterrows():
                failures.append({"source_index": int(row["_source_index"]), "sample_id": sample_id, "uid": row.uid,
                                 "error": str(exc)})
            continue

        sample_rows = peaks[peaks.sample_id.eq(sample_id)]
        for _, row in sample_selection.iterrows():
            feature = _resolve_feature(row, by_uid, by_index)
            if feature is None:
                failures.append({"source_index": int(row["_source_index"]), "sample_id": sample_id, "uid": row.uid,
                                 "error": "XIC channel not found"})
                continue
            serial += 1
            folder = args.output_dir / str(row.gallery_category)
            folder.mkdir(parents=True, exist_ok=True)
            filename = f"{serial:03d}_{_safe_name(sample_id)}_{_safe_name(row.uid)}_{row.alert_level}.png"
            out_path = folder / filename
            channel_rows = sample_rows[sample_rows.uid.eq(row.uid)]
            draw_peak_plot(row, channel_rows, feature, out_path)
            index_rows.append(
                {
                    "plot_no": serial,
                    "source_index": int(row["_source_index"]),
                    "sample_id": sample_id,
                    "uid": row.uid,
                    "gallery_category": row.gallery_category,
                    "gallery_category_label": row.gallery_category_label,
                    "alert_level": row.alert_level,
                    "match_status": row.match_status,
                    "alert_reasons": row.alert_reasons,
                    "score_ai": row.score_ai,
                    "comparison_confidence": row.comparison_confidence,
                    "delta_rt_abs": row.delta_rt_abs,
                    "area_fold": row.area_fold,
                    "plot_path": out_path.relative_to(args.output_dir),
                }
            )

    index = pd.DataFrame(index_rows)
    index.to_csv(args.output_dir / "peak_plot_index.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        failures,
        columns=["source_index", "sample_id", "uid", "error"],
    ).to_csv(args.output_dir / "plot_failures.csv", index=False, encoding="utf-8-sig")
    _write_html(index, args.output_dir)
    print(f"Generated {len(index)} peak plots; failures={len(failures)}; output={args.output_dir.resolve()}")
    if len(index):
        print(index.groupby(["gallery_category", "gallery_category_label"]).size().to_string())


if __name__ == "__main__":
    main()
