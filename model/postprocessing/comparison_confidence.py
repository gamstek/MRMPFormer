# -*- coding: utf-8 -*-
"""AI peak / MassNova traditional peak matching and review confidence v1.

The module deliberately keeps event matching, numeric confidence and manual
review state as separate concepts.  Defaults are transparent starting values;
they must be calibrated on an independently reviewed dataset before release.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


SCHEMA_VERSION = "comparison_confidence_v1"
BIG_COST = 1e9


@dataclass(frozen=True)
class ConfidenceConfig:
    match_rt_gate_min: float = 0.50
    match_max_cost: float = 1.00
    match_rt_weight: float = 0.70
    match_boundary_weight: float = 0.25
    match_ai_tiebreak_weight: float = 0.05
    ambiguity_margin: float = 0.10
    ambiguous_match_factor: float = 0.60
    rt_half_confidence_min: float = 0.10
    area_half_confidence_log2: float = 1.00
    missing_ai_score_confidence: float = 0.25
    confidence_weight_ai: float = 0.35
    confidence_weight_rt: float = 0.40
    confidence_weight_area: float = 0.25
    green_confidence_min: float = 0.80
    yellow_confidence_min: float = 0.60
    hard_red_rt_min: float = 0.30
    hard_red_area_fold: float = 4.00
    trad_rt_half_confidence_min: float = 0.10
    trad_snr_low: float = 3.00
    trad_snr_good: float = 10.00
    trad_quality_weight_rt: float = 0.60
    trad_quality_weight_snr: float = 0.40
    trad_quality_green_min: float = 0.80
    trad_quality_yellow_min: float = 0.60

    @classmethod
    def from_json(cls, path: str | Path | None) -> "ConfidenceConfig":
        if not path:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        known = cls.__dataclass_fields__
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(f"Unknown confidence config fields: {unknown}")
        return cls(**data)

    def validate(self) -> None:
        if self.match_rt_gate_min <= 0 or self.rt_half_confidence_min <= 0:
            raise ValueError("RT thresholds must be positive")
        if self.area_half_confidence_log2 <= 0:
            raise ValueError("area_half_confidence_log2 must be positive")
        if self.trad_rt_half_confidence_min <= 0 or self.trad_snr_good <= self.trad_snr_low:
            raise ValueError("traditional quality thresholds are invalid")
        mw = self.match_rt_weight + self.match_boundary_weight + self.match_ai_tiebreak_weight
        if not math.isclose(mw, 1.0, abs_tol=1e-8):
            raise ValueError(f"matching weights must sum to 1, got {mw}")
        cw = self.confidence_weight_ai + self.confidence_weight_rt + self.confidence_weight_area
        if not math.isclose(cw, 1.0, abs_tol=1e-8):
            raise ValueError(f"confidence weights must sum to 1, got {cw}")
        tw = self.trad_quality_weight_rt + self.trad_quality_weight_snr
        if not math.isclose(tw, 1.0, abs_tol=1e-8):
            raise ValueError(f"traditional quality weights must sum to 1, got {tw}")


TRAD_COLUMNS = {
    "sample_id": "Unnamed: 0",
    "sample_name": "样品名",
    "uid": "化合物",
    "ion_type": "离子类型",
    "rt_peak": "保留时间",
    "rt_expected": "预期保留时间",
    "rt_min": "峰起始时间",
    "rt_max": "峰结束时间",
    "area": "分析物峰面积",
    "height": "分析物峰高度",
    "snr": "信噪比",
    "ion_ratio": "离子比率",
}


def _number(value: Any) -> float:
    """Read a number or the leading RT from strings such as ``1.23(456)``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else math.nan


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_traditional_xlsx(path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name)
    missing = [v for v in TRAD_COLUMNS.values() if v not in raw.columns]
    if missing:
        raise ValueError(f"Traditional workbook is missing columns: {missing}")
    out = pd.DataFrame(index=raw.index)
    out["trad_row_id"] = np.arange(len(raw), dtype=int)
    for dst in ("sample_id", "sample_name", "uid", "ion_type"):
        out[f"trad_{dst}"] = raw[TRAD_COLUMNS[dst]].map(_text)
    for dst in ("rt_peak", "rt_expected", "rt_min", "rt_max", "area", "height", "snr", "ion_ratio"):
        out[f"trad_{dst}"] = raw[TRAD_COLUMNS[dst]].map(_number)
    out["sample_id"] = out["trad_sample_id"]
    out["uid"] = out["trad_uid"]
    return out


def load_ai_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"mzml_stem", "uid", "rt_min", "rt_peak", "rt_max", "area", "model_score"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"AI CSV is missing columns: {missing}")
    out = pd.DataFrame(index=raw.index)
    out["ai_row_id"] = np.arange(len(raw), dtype=int)
    out["sample_id"] = raw["mzml_stem"].map(_text)
    out["uid"] = raw["uid"].map(_text)
    for col in ("chrom_index", "peak_no", "q1", "rt_min", "rt_peak", "rt_max",
                "apex_intensity", "area", "snr", "n_points", "model_score"):
        out[f"ai_{col}"] = pd.to_numeric(raw[col], errors="coerce") if col in raw else math.nan
    text_blank = pd.Series("", index=raw.index)
    bool_false = pd.Series(False, index=raw.index)
    out["ai_compound_name"] = raw.get("compound_name", text_blank).map(_text)
    out["ai_boundary_source"] = raw.get("boundary_source", text_blank).map(_text)
    out["ai_validated"] = raw.get("validated", bool_false).map(_bool)
    return out


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    vals = (a0, a1, b0, b1)
    if not all(np.isfinite(vals)) or a1 <= a0 or b1 <= b0:
        return math.nan
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return intersection / union if union > 0 else 0.0


def _pair_metrics(t: pd.Series, a: pd.Series, cfg: ConfidenceConfig) -> dict[str, float | bool]:
    delta_rt = abs(float(a.ai_rt_peak) - float(t.trad_rt_peak))
    iou = interval_iou(float(a.ai_rt_min), float(a.ai_rt_max),
                       float(t.trad_rt_min), float(t.trad_rt_max))
    overlaps = bool(np.isfinite(iou) and iou > 0)
    eligible = bool(np.isfinite(delta_rt) and (delta_rt <= cfg.match_rt_gate_min or overlaps))
    rt_part = min(delta_rt / cfg.match_rt_gate_min, 2.0) if np.isfinite(delta_rt) else 2.0
    boundary_part = 1.0 - iou if np.isfinite(iou) else 0.5
    score = float(a.ai_model_score)
    ai_part = 1.0 - float(np.clip(score, 0, 1)) if np.isfinite(score) else 1.0
    cost = (cfg.match_rt_weight * rt_part +
            cfg.match_boundary_weight * boundary_part +
            cfg.match_ai_tiebreak_weight * ai_part)
    return {"eligible": eligible, "delta_rt_abs": delta_rt,
            "interval_iou": iou, "match_cost": cost}


def _base_row(status: str, sample_id: str, uid: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "match_status": status,
            "sample_id": sample_id, "uid": uid}


def _add_trad(row: dict[str, Any], t: pd.Series) -> None:
    for col, value in t.items():
        if col.startswith("trad_"):
            row[col] = value


def _add_ai(row: dict[str, Any], a: pd.Series) -> None:
    for col, value in a.items():
        if col.startswith("ai_"):
            row[col] = value
    row["score_ai"] = a.ai_model_score


def match_peak_events(trad: pd.DataFrame, ai: pd.DataFrame,
                      cfg: ConfidenceConfig) -> pd.DataFrame:
    """Perform exact-channel, one-to-one event matching and preserve all rows."""
    cfg.validate()
    trad_groups = {k: g for k, g in trad.groupby(["sample_id", "uid"], sort=False)}
    ai_groups = {k: g for k, g in ai.groupby(["sample_id", "uid"], sort=False)}
    trad_samples = set(trad.sample_id)
    rows: list[dict[str, Any]] = []

    for key in sorted(set(trad_groups) | set(ai_groups)):
        sample_id, uid = key
        tg = trad_groups.get(key)
        ag = ai_groups.get(key)
        if tg is None:
            status = "NO_TRAD_SAMPLE" if sample_id not in trad_samples else "AI_ONLY_EXTRA"
            for _, a in ag.iterrows():
                row = _base_row(status, sample_id, uid)
                _add_ai(row, a)
                rows.append(row)
            continue
        if ag is None or ag.empty:
            for _, t in tg.iterrows():
                status = "REFERENCE_MISSING" if not np.isfinite(t.trad_rt_peak) else "TRAD_ONLY_AI_MISS"
                row = _base_row(status, sample_id, uid)
                _add_trad(row, t)
                rows.append(row)
            continue

        valid_t = tg[np.isfinite(tg.trad_rt_peak)]
        invalid_t = tg[~np.isfinite(tg.trad_rt_peak)]
        for _, t in invalid_t.iterrows():
            row = _base_row("REFERENCE_MISSING", sample_id, uid)
            _add_trad(row, t)
            rows.append(row)

        if valid_t.empty:
            for _, a in ag.iterrows():
                row = _base_row("AI_ONLY_EXTRA", sample_id, uid)
                _add_ai(row, a)
                rows.append(row)
            continue

        trows = [r for _, r in valid_t.iterrows()]
        arows = [r for _, r in ag.iterrows()]
        group_rows: list[dict[str, Any]] = []
        matrix = np.full((len(trows), len(arows)), BIG_COST, dtype=float)
        metrics: dict[tuple[int, int], dict[str, Any]] = {}
        for i, t in enumerate(trows):
            for j, a in enumerate(arows):
                m = _pair_metrics(t, a, cfg)
                metrics[i, j] = m
                if m["eligible"]:
                    matrix[i, j] = float(m["match_cost"])

        assigned_t: set[int] = set()
        assigned_a: set[int] = set()
        assigned_cost_by_t: dict[int, float] = {}
        for i, j in zip(*linear_sum_assignment(matrix)):
            m = metrics[i, j]
            if matrix[i, j] >= BIG_COST or matrix[i, j] > cfg.match_max_cost:
                continue
            alternatives = sorted(matrix[i, k] for k in range(len(arows))
                                  if k != j and matrix[i, k] < BIG_COST)
            second = alternatives[0] if alternatives else math.nan
            margin = second - matrix[i, j] if np.isfinite(second) else math.inf
            status = "MATCHED_AMBIGUOUS" if margin < cfg.ambiguity_margin else "MATCHED_UNIQUE"
            row = _base_row(status, sample_id, uid)
            _add_trad(row, trows[i])
            _add_ai(row, arows[j])
            row.update({"delta_rt_abs": m["delta_rt_abs"], "interval_iou": m["interval_iou"],
                        "match_cost": matrix[i, j], "second_best_cost": second,
                        "match_margin": margin,
                        "ai_candidates_total": len(arows),
                        "ai_candidates_eligible": int(np.sum(matrix[i] < BIG_COST))})
            group_rows.append(row)
            assigned_t.add(i)
            assigned_a.add(j)
            assigned_cost_by_t[i] = float(matrix[i, j])

        for i, t in enumerate(trows):
            if i not in assigned_t:
                row = _base_row("TRAD_ONLY_AI_MISS", sample_id, uid)
                _add_trad(row, t)
                row["ai_candidates_total"] = len(arows)
                row["ai_candidates_eligible"] = int(np.sum(matrix[i] < BIG_COST))
                group_rows.append(row)
        for j, a in enumerate(arows):
            if j not in assigned_a:
                row = _base_row("AI_ONLY_EXTRA", sample_id, uid)
                _add_ai(row, a)
                eligible = any(bool(metrics[i, j]["eligible"]) for i in range(len(trows)))
                deltas = [float(metrics[i, j]["delta_rt_abs"]) for i in range(len(trows))]
                competitive = any(
                    matrix[i, j] < BIG_COST and
                    matrix[i, j] <= assigned_cost_by_t.get(i, cfg.match_max_cost) + cfg.ambiguity_margin
                    for i in range(len(trows))
                )
                row["extra_is_match_eligible"] = eligible
                row["extra_is_competitive"] = competitive
                row["extra_nearest_trad_rt_abs"] = min(deltas) if deltas else math.nan
                group_rows.append(row)
        unmatched_count = len(arows) - len(assigned_a)
        competitive_count = sum(
            1 for row in group_rows
            if row["match_status"] == "AI_ONLY_EXTRA" and bool(row.get("extra_is_competitive", False))
        )
        for row in group_rows:
            if str(row["match_status"]).startswith("MATCHED") or row["match_status"] == "TRAD_ONLY_AI_MISS":
                row["unmatched_ai_peak_count"] = unmatched_count
                row["competitive_ai_peak_count"] = competitive_count
        rows.extend(group_rows)

    return pd.DataFrame(rows)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _traditional_quality(row: dict[str, Any], cfg: ConfidenceConfig) -> dict[str, Any]:
    rt = row.get("trad_rt_peak", math.nan)
    expected = row.get("trad_rt_expected", math.nan)
    snr = row.get("trad_snr", math.nan)
    rt_diff = abs(float(rt) - float(expected)) if _finite(rt) and _finite(expected) else math.nan
    conf_rt = (2.0 ** (-rt_diff / cfg.trad_rt_half_confidence_min)
               if np.isfinite(rt_diff) else math.nan)
    if _finite(snr):
        snr_value = float(snr)
        if snr_value <= cfg.trad_snr_low:
            conf_snr = 0.0
        elif snr_value >= cfg.trad_snr_good:
            conf_snr = 1.0
        else:
            numerator = math.log1p(snr_value) - math.log1p(cfg.trad_snr_low)
            denominator = math.log1p(cfg.trad_snr_good) - math.log1p(cfg.trad_snr_low)
            conf_snr = numerator / denominator
    else:
        conf_snr = math.nan

    components = []
    if np.isfinite(conf_rt):
        components.append((conf_rt, cfg.trad_quality_weight_rt))
    if np.isfinite(conf_snr):
        components.append((conf_snr, cfg.trad_quality_weight_snr))
    if components:
        weight_sum = sum(weight for _, weight in components)
        quality = math.exp(sum((weight / weight_sum) * math.log(max(value, 1e-12))
                               for value, weight in components))
        if quality >= cfg.trad_quality_green_min:
            level = "GREEN"
        elif quality >= cfg.trad_quality_yellow_min:
            level = "YELLOW"
        else:
            level = "RED"
    else:
        quality, level = math.nan, "UNAVAILABLE"
    return {
        "trad_expected_rt_diff_abs": rt_diff,
        "conf_trad_expected_rt": conf_rt,
        "conf_trad_snr": conf_snr,
        "quality_trad_rule": quality,
        "quality_trad_level": level,
    }


def score_confidence(matches: pd.DataFrame, cfg: ConfidenceConfig) -> pd.DataFrame:
    """Add auditable sub-scores, composite confidence and rule-based alert state."""
    out = matches.copy()
    scored: list[dict[str, Any]] = []
    matched_states = {"MATCHED_UNIQUE", "MATCHED_AMBIGUOUS"}
    for _, source in out.iterrows():
        row = source.to_dict()
        status = row["match_status"]
        reasons: list[str] = []
        row.update(_traditional_quality(row, cfg))
        conf = 0.0
        conf_match = conf_ai = conf_rt = conf_area = math.nan
        area_abs = area_rel = area_log_ratio = area_fold = math.nan

        if status in matched_states:
            drt = float(row["delta_rt_abs"])
            conf_rt = 2.0 ** (-drt / cfg.rt_half_confidence_min)
            ai_score = row.get("score_ai", math.nan)
            ai_missing = not _finite(ai_score)
            conf_ai = cfg.missing_ai_score_confidence if ai_missing else float(np.clip(float(ai_score), 0, 1))

            aa, ta = row.get("ai_area", math.nan), row.get("trad_area", math.nan)
            if _finite(aa) and _finite(ta) and float(aa) >= 0 and float(ta) >= 0:
                eps = max(1e-12, 1e-9 * max(float(aa), float(ta), 1.0))
                area_abs = abs(float(aa) - float(ta))
                area_rel = area_abs / max(float(aa), float(ta), eps)
                area_log_ratio = abs(math.log((float(aa) + eps) / (float(ta) + eps)))
                area_fold = math.exp(area_log_ratio)
                log2_fold = area_log_ratio / math.log(2.0)
                conf_area = 2.0 ** (-log2_fold / cfg.area_half_confidence_log2)
            else:
                conf_area = 0.0
                reasons.append("AREA_MISSING")

            cost = float(row.get("match_cost", cfg.match_max_cost))
            conf_match = math.exp(-max(cost, 0.0))
            if status == "MATCHED_AMBIGUOUS":
                conf_match *= cfg.ambiguous_match_factor
                reasons.append("MATCH_AMBIGUOUS")
            eps = 1e-12
            base = math.exp(
                cfg.confidence_weight_ai * math.log(max(conf_ai, eps)) +
                cfg.confidence_weight_rt * math.log(max(conf_rt, eps)) +
                cfg.confidence_weight_area * math.log(max(conf_area, eps))
            )
            conf = float(np.clip(conf_match * base, 0, 1))
            if ai_missing:
                reasons.append("SCORE_AI_MISSING")
            if drt >= cfg.hard_red_rt_min:
                reasons.append("RT_HARD_LIMIT")
            if _finite(area_fold) and area_fold >= cfg.hard_red_area_fold:
                reasons.append("AREA_HARD_LIMIT")
            if row["quality_trad_level"] == "RED":
                reasons.append("TRAD_REFERENCE_LOW_QUALITY")
        else:
            reasons.append(status)

        if status == "TRAD_ONLY_AI_MISS":
            level = "RED"
        elif status in {"REFERENCE_MISSING", "NO_TRAD_SAMPLE"}:
            level = "UNAVAILABLE"
        elif status == "AI_ONLY_EXTRA":
            level = "YELLOW" if _bool(row.get("extra_is_competitive", False)) else "INFO"
        elif status == "MATCHED_AMBIGUOUS" or "SCORE_AI_MISSING" in reasons:
            level = "RED"
        elif "RT_HARD_LIMIT" in reasons or "AREA_HARD_LIMIT" in reasons:
            level = "RED"
        elif "TRAD_REFERENCE_LOW_QUALITY" in reasons:
            level = "YELLOW"
        elif conf >= cfg.green_confidence_min:
            level = "GREEN"
        elif conf >= cfg.yellow_confidence_min:
            level = "YELLOW"
        else:
            level = "RED"
        review_required = level in {"YELLOW", "RED", "UNAVAILABLE"}
        row.update({
            "area_diff_abs": area_abs, "area_diff_rel": area_rel,
            "area_log_ratio": area_log_ratio, "area_fold": area_fold,
            "conf_match": conf_match, "conf_ai": conf_ai, "conf_rt": conf_rt,
            "conf_area": conf_area, "comparison_confidence": conf,
            "final_rule_risk_score": 100.0 * (1.0 - conf),
            "alert_level": level, "alert_reasons": "|".join(reasons),
            "review_required": review_required,
            "review_status": "PENDING" if review_required else "NOT_REQUIRED",
            "review_decision": "", "reviewer": "", "reviewed_at": "",
            "review_comment": "",
        })
        scored.append(row)
    return pd.DataFrame(scored)


def summarize_samples(rows: pd.DataFrame) -> pd.DataFrame:
    severity = {"INFO": 0, "GREEN": 0, "YELLOW": 1, "RED": 2}
    summaries = []
    for sample_id, group in rows.groupby("sample_id", sort=True):
        counts = group.alert_level.value_counts()
        reference = group[group.get("trad_row_id", pd.Series(index=group.index, dtype=float)).notna()]
        available = reference[reference.match_status != "REFERENCE_MISSING"]
        matched = available[available.match_status.str.startswith("MATCHED")]
        coverage = len(matched) / len(available) if len(available) else math.nan
        p10 = float(matched.comparison_confidence.quantile(0.10)) if len(matched) else math.nan
        median = float(matched.comparison_confidence.median()) if len(matched) else math.nan
        sample_confidence = coverage * p10 if np.isfinite(coverage) and np.isfinite(p10) else math.nan
        if reference.empty or available.empty:
            worst = "UNAVAILABLE"
        else:
            ref_levels = [x for x in reference.alert_level if x in severity]
            worst = max(ref_levels, key=lambda x: severity[x]) if ref_levels else "UNAVAILABLE"
        summaries.append({
            "schema_version": SCHEMA_VERSION, "sample_id": sample_id,
            "sample_alert_level": worst, "sample_confidence": sample_confidence,
            "matched_confidence_p10": p10, "matched_confidence_median": median,
            "match_coverage": coverage, "n_rows": len(group),
            "n_reference_rows": len(reference), "n_matched": len(matched),
            "n_info": int(counts.get("INFO", 0)),
            "n_green": int(counts.get("GREEN", 0)),
            "n_yellow": int(counts.get("YELLOW", 0)),
            "n_red": int(counts.get("RED", 0)),
            "n_unavailable": int(counts.get("UNAVAILABLE", 0)),
            "n_review_required": int(group.review_required.sum()),
            "n_ai_miss": int((group.match_status == "TRAD_ONLY_AI_MISS").sum()),
            "n_ai_extra": int((group.match_status == "AI_ONLY_EXTRA").sum()),
            "n_ambiguous": int((group.match_status == "MATCHED_AMBIGUOUS").sum()),
            "min_confidence": float(group.comparison_confidence.min()),
        })
    return pd.DataFrame(summaries)


def _counts_markdown(counts: pd.Series) -> str:
    lines = ["| value | count |", "|---|---:|"]
    lines.extend(f"| `{index}` | {int(value)} |" for index, value in counts.items())
    return "\n".join(lines)


def _write_report(path: Path, peaks: pd.DataFrame, samples: pd.DataFrame,
                  cfg: ConfidenceConfig, traditional_path: Path, ai_path: Path) -> None:
    ms = peaks.match_status.value_counts()
    al = peaks.alert_level.value_counts()
    text = f"""# Comparison confidence v1 run report

- Schema: `{SCHEMA_VERSION}`
- Traditional input: `{traditional_path.resolve()}`
- AI input: `{ai_path.resolve()}`
- Peak/event rows: {len(peaks)}
- Samples: {len(samples)}

## Match states

{_counts_markdown(ms)}

## Alert levels

{_counts_markdown(al)}

## Interpretation

`score_ai` is the raw MRMPFormer softmax score. `comparison_confidence` combines
match quality, AI score, RT agreement and area agreement. `final_rule_risk_score`
is `100 * (1 - comparison_confidence)`; hard rules determine `alert_level` and
`review_status`. Thresholds in this run are provisional and require calibration
against independent manual review labels.

## Configuration

```json
{json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def run(traditional_xlsx: str | Path, ai_csv: str | Path,
        output_dir: str | Path, config: ConfidenceConfig | None = None) -> dict[str, Path]:
    cfg = config or ConfidenceConfig()
    cfg.validate()
    traditional_path, ai_path = Path(traditional_xlsx), Path(ai_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trad = load_traditional_xlsx(traditional_path)
    ai = load_ai_csv(ai_path)
    peaks = score_confidence(match_peak_events(trad, ai, cfg), cfg)
    samples = summarize_samples(peaks)
    peak_path = out / "peak_confidence.csv"
    sample_path = out / "sample_alert_summary.csv"
    metadata_path = out / "confidence_run.json"
    report_path = out / "confidence_report.md"
    peaks.to_csv(peak_path, index=False, encoding="utf-8-sig")
    samples.to_csv(sample_path, index=False, encoding="utf-8-sig")
    metadata_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "traditional_input": str(traditional_path.resolve()),
        "ai_input": str(ai_path.resolve()),
        "config": asdict(cfg),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report_path, peaks, samples, cfg, traditional_path, ai_path)
    return {"peaks": peak_path, "samples": sample_path,
            "metadata": metadata_path, "report": report_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match AI/traditional peaks and compute review confidence")
    parser.add_argument("--traditional", required=True, help="MassNova traditional result xlsx")
    parser.add_argument("--ai", required=True, help="AI inference all.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None, help="Optional confidence JSON")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = run(args.traditional, args.ai, args.output_dir,
                ConfidenceConfig.from_json(args.config))
    for name, path in paths.items():
        print(f"[{name}] {path.resolve()}")


if __name__ == "__main__":
    main()
