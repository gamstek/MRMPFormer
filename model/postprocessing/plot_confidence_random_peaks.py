"""Randomly sample final AI peaks and draw confidence/alert annotated XIC plots."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from inference.massnova import extract_full_xics
from postprocessing.plot_confidence_peak_gallery import (
    _feature_lookup,
    _resolve_feature,
    _safe_name,
    _set_font,
    _write_html,
    draw_peak_plot,
)


def _sample_number(value: object, prefix: str) -> float:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", str(value).strip(), flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def _markdown_counts(series: pd.Series) -> str:
    counts = series.fillna("<missing>").astype(str).value_counts()
    total = int(counts.sum())
    lines = ["| value | count | proportion |", "|---|---:|---:|"]
    lines.extend(
        f"| `{name}` | {int(count)} | {100.0 * int(count) / total:.2f}% |"
        for name, count in counts.items()
    )
    return "\n".join(lines)


def _write_summary(path: Path, selected: pd.DataFrame, index: pd.DataFrame,
                   failures: pd.DataFrame, args: argparse.Namespace, elapsed: float) -> None:
    alert_counts = selected.alert_level.value_counts()
    match_counts = selected.match_status.value_counts()
    review_count = int(selected.alert_level.isin(["YELLOW", "RED", "UNAVAILABLE"]).sum())
    matched = selected[selected.match_status.astype(str).str.startswith("MATCHED")]
    matched_review = int(matched.alert_level.isin(["YELLOW", "RED", "UNAVAILABLE"]).sum())
    extra = selected[selected.match_status.eq("AI_ONLY_EXTRA")]
    no_trad = selected[selected.match_status.eq("NO_TRAD_SAMPLE")]
    reasons = (
        selected.alert_reasons.dropna().astype(str).str.split("|").explode()
        .loc[lambda values: values.ne("")].value_counts().head(12)
    )
    reason_lines = ["| reason | count |", "|---|---:|"]
    reason_lines.extend(f"| `{name}` | {int(count)} |" for name, count in reasons.items())
    threshold_note = (
        f"\n- 推理阈值说明：{args.inference_threshold_note}"
        if args.inference_threshold_note else ""
    )
    text = f"""# test3 随机峰置信度与预警图集

- 输入置信度表：`{args.confidence_csv.resolve()}`
- mzML 目录：`{args.mzml_dir.resolve()}`
- 样本范围：`{args.sample_prefix}_{args.sample_min}` ～ `{args.sample_prefix}_{args.sample_max}`
- 固定随机种子：`{args.seed}`
- 请求抽样数：{args.random_n}
- 实际抽样数：{len(selected)}
- 成功出图：{len(index)}
- 失败：{len(failures)}
- 耗时：{elapsed:.1f} 秒
{threshold_note}

## 预警等级

{_markdown_counts(selected.alert_level)}

## 匹配状态

{_markdown_counts(selected.match_status)}

## 峰分来源

{_markdown_counts(selected.score_source)}

## 图片说明

- 蓝色：MRMPFormer + 信号算法最终 AI 峰。
- 绿色：正式一对一匹配的传统峰。
- 橙色：`AI_ONLY_EXTRA` 使用的最近传统诊断参考，不是正式匹配。
- 紫色：已经正式匹配该最近传统峰的另一个 AI 峰。
- 图中显示峰自身分、分数来源、RT 绝对差、面积绝对差、面积差百分比、比较置信度、最终置信度、风险分、预警等级和原因。
- `NO_TRAD_SAMPLE` 没有传统参考，比较置信度和最终置信度显示为 `--`，预警为 `UNAVAILABLE`。

## 结果解读

- 默认无需复核的 GREEN：{int(alert_counts.get('GREEN', 0))} 个，占 {100.0 * int(alert_counts.get('GREEN', 0)) / len(selected):.2f}%。
- 需要复核的 YELLOW、RED、UNAVAILABLE：{review_count} 个，占 {100.0 * review_count / len(selected):.2f}%。
- 正式匹配峰：{len(matched)} 个，占 {100.0 * len(matched) / len(selected):.2f}%；其中 {matched_review} 个需要复核，占正式匹配峰的 {100.0 * matched_review / max(len(matched), 1):.2f}%。
- 额外 AI 峰：{len(extra)} 个，占 {100.0 * len(extra) / len(selected):.2f}%。
- 无传统样本参考：{len(no_trad)} 个，占 {100.0 * len(no_trad) / len(selected):.2f}%。

## 主要预警原因

{chr(10).join(reason_lines)}

## 数据边界

本图集用于检查置信度、最近传统参考和预警绘图逻辑。若输入峰表来自旧阈值推理，本图集中的峰数量和预警比例不代表新阈值重新推理后的结果。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence_csv", required=True, type=Path)
    parser.add_argument("--mzml_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--random_n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--sample_prefix", default="test3")
    parser.add_argument("--sample_min", type=int, default=3)
    parser.add_argument("--sample_max", type=int, default=58)
    parser.add_argument("--smooth_sigma", type=float, default=0.8)
    parser.add_argument("--inference_threshold_note", default="")
    args = parser.parse_args()

    if args.random_n <= 0:
        raise SystemExit("--random_n must be positive")
    if args.sample_max < args.sample_min:
        raise SystemExit("--sample_max must be >= --sample_min")

    started = time.perf_counter()
    _set_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peaks = pd.read_csv(args.confidence_csv, low_memory=False)
    peaks["_source_index"] = np.arange(len(peaks), dtype=int)
    peaks["_sample_number"] = peaks.sample_id.map(
        lambda value: _sample_number(value, args.sample_prefix)
    )
    eligible = peaks[
        peaks.ai_row_id.notna()
        & peaks._sample_number.between(args.sample_min, args.sample_max, inclusive="both")
    ].copy()
    if len(eligible) < args.random_n:
        raise SystemExit(f"Only {len(eligible)} eligible AI peaks, fewer than {args.random_n}")

    selected = eligible.sample(n=args.random_n, random_state=args.seed, replace=False).copy()
    selected["random_order"] = np.arange(1, len(selected) + 1, dtype=int)
    selected["gallery_category"] = selected.sample_id.astype(str)
    selected["gallery_category_label"] = "随机抽样置信度与预警"
    selected.to_csv(args.output_dir / "selected_peaks.csv", index=False, encoding="utf-8-sig")

    mzml_paths = {path.stem: path for path in args.mzml_dir.glob("*.mzML")}
    mzml_paths.update({path.stem: path for path in args.mzml_dir.glob("*.mzml")})
    index_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    processed = 0
    total_samples = int(selected.sample_id.nunique())

    ordered = selected.sort_values(["_sample_number", "random_order"])
    for sample_no, (sample_id, sample_selection) in enumerate(
        ordered.groupby("sample_id", sort=False), start=1
    ):
        mzml_path = mzml_paths.get(str(sample_id))
        if mzml_path is None:
            for _, row in sample_selection.iterrows():
                failures.append({"random_order": int(row.random_order),
                                 "source_index": int(row["_source_index"]),
                                 "sample_id": sample_id, "uid": row.uid,
                                 "error": "mzML not found"})
            processed += len(sample_selection)
            print(f"[{sample_no}/{total_samples}] {sample_id}: mzML not found; "
                  f"processed={processed}/{len(selected)}", flush=True)
            continue

        try:
            features, _ = extract_full_xics(
                mzml_path, smooth_sigma=args.smooth_sigma, verbose=False
            )
            by_uid, by_index = _feature_lookup(features)
        except Exception as exc:
            for _, row in sample_selection.iterrows():
                failures.append({"random_order": int(row.random_order),
                                 "source_index": int(row["_source_index"]),
                                 "sample_id": sample_id, "uid": row.uid,
                                 "error": str(exc)})
            processed += len(sample_selection)
            print(f"[{sample_no}/{total_samples}] {sample_id}: extraction failed: {exc}; "
                  f"processed={processed}/{len(selected)}", flush=True)
            continue

        sample_rows = peaks[peaks.sample_id.eq(sample_id)]
        sample_success = 0
        sample_failure = 0
        for _, row in sample_selection.iterrows():
            feature = _resolve_feature(row, by_uid, by_index)
            if feature is None:
                failures.append({"random_order": int(row.random_order),
                                 "source_index": int(row["_source_index"]),
                                 "sample_id": sample_id, "uid": row.uid,
                                 "error": "XIC channel not found"})
                sample_failure += 1
                continue
            chrom_index = int(float(row.ai_chrom_index))
            peak_no = int(float(row.ai_peak_no))
            sample_dir = args.output_dir / "plots" / str(sample_id)
            sample_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{int(row.random_order):04d}_chrom{chrom_index:03d}_peak{peak_no:03d}_"
                f"{_safe_name(row.uid)}_{row.alert_level}.png"
            )
            out_path = sample_dir / filename
            channel_rows = sample_rows[sample_rows.uid.eq(row.uid)]
            try:
                draw_peak_plot(row, channel_rows, feature, out_path)
            except Exception as exc:
                failures.append({"random_order": int(row.random_order),
                                 "source_index": int(row["_source_index"]),
                                 "sample_id": sample_id, "uid": row.uid,
                                 "error": str(exc)})
                sample_failure += 1
                continue
            sample_success += 1
            index_rows.append({
                "random_order": int(row.random_order),
                "source_index": int(row["_source_index"]),
                "sample_id": sample_id,
                "uid": row.uid,
                "ai_chrom_index": chrom_index,
                "ai_peak_no": peak_no,
                "score_peak": row.score_peak,
                "score_source": row.score_source,
                "match_status": row.match_status,
                "comparison_reference_type": row.comparison_reference_type,
                "delta_rt_abs": row.delta_rt_abs,
                "area_diff_abs": row.area_diff_abs,
                "area_diff_pct": row.area_diff_pct,
                "comparison_confidence": row.comparison_confidence,
                "final_confidence": row.final_confidence,
                "alert_level": row.alert_level,
                "alert_reasons": row.alert_reasons,
                "gallery_category": row.gallery_category,
                "gallery_category_label": row.gallery_category_label,
                "plot_path": out_path.relative_to(args.output_dir),
            })
        processed += len(sample_selection)
        print(f"[{sample_no}/{total_samples}] {sample_id}: plots={sample_success}, "
              f"failures={sample_failure}, processed={processed}/{len(selected)}", flush=True)

    index = pd.DataFrame(index_rows).sort_values("random_order") if index_rows else pd.DataFrame()
    failure_frame = pd.DataFrame(
        failures,
        columns=["random_order", "source_index", "sample_id", "uid", "error"],
    )
    index.to_csv(args.output_dir / "peak_plot_index.csv", index=False, encoding="utf-8-sig")
    failure_frame.to_csv(args.output_dir / "plot_failures.csv", index=False, encoding="utf-8-sig")
    (index.groupby(["match_status", "alert_level"], dropna=False).size()
     .rename("count").reset_index()
     .to_csv(args.output_dir / "match_alert_crosstab.csv", index=False, encoding="utf-8-sig"))
    (index.groupby(["score_source", "alert_level"], dropna=False).size()
     .rename("count").reset_index()
     .to_csv(args.output_dir / "source_alert_crosstab.csv", index=False, encoding="utf-8-sig"))
    if len(index):
        _write_html(index, args.output_dir)
    elapsed = time.perf_counter() - started
    _write_summary(args.output_dir / "README.md", selected, index, failure_frame, args, elapsed)
    print(f"Generated {len(index)} plots; failures={len(failure_frame)}; "
          f"elapsed={elapsed:.1f}s; output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
