"""Suppress review alerts for visually tiny, unsupported signal-only AI extras.

The rule is intentionally applied after AI/traditional one-to-one matching.  It
preserves every peak and every confidence value for audit, while changing only
the alert disposition of weak ``AI_ONLY_EXTRA`` signal peaks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_VERSION = "weak_ai_extra_v1"


def _transition_parts(uid: object) -> tuple[str, str]:
    text = str(uid or "").strip()
    match = re.match(r"^(.*)-([12])$", text)
    return (match.group(1).casefold(), match.group(2)) if match else (text.casefold(), "")


def _add_pair_support(rows: pd.DataFrame, rt_tolerance_min: float) -> pd.DataFrame:
    rows = rows.copy()
    rows["paired_transition_supported"] = False
    rows["paired_transition_rt_abs"] = math.nan
    ai = rows[rows["ai_row_id"].notna()].copy()
    parts = ai["uid"].map(_transition_parts)
    ai["_compound_base"] = parts.map(lambda value: value[0])
    ai["_transition_suffix"] = parts.map(lambda value: value[1])
    ai["_row_position"] = ai.index

    for _, group in ai.groupby(["sample_id", "_compound_base"], sort=False):
        by_suffix = {
            suffix: part for suffix, part in group.groupby("_transition_suffix", sort=False)
            if suffix in {"1", "2"}
        }
        if "1" not in by_suffix or "2" not in by_suffix:
            continue
        for suffix, other_suffix in (("1", "2"), ("2", "1")):
            other_rts = pd.to_numeric(by_suffix[other_suffix]["ai_rt_peak"], errors="coerce").dropna().to_numpy()
            if not len(other_rts):
                continue
            for index, rt_value in pd.to_numeric(by_suffix[suffix]["ai_rt_peak"], errors="coerce").items():
                if not np.isfinite(rt_value):
                    continue
                delta = float(np.min(np.abs(other_rts - float(rt_value))))
                rows.at[index, "paired_transition_rt_abs"] = delta
                rows.at[index, "paired_transition_supported"] = delta <= rt_tolerance_min
    return rows


def apply_weak_extra_alert_rule(
    rows: pd.DataFrame,
    *,
    relative_height_max: float = 0.10,
    relative_area_max: float = 0.10,
    paired_transition_rt_tolerance_min: float = 0.10,
) -> pd.DataFrame:
    """Return a copy with weak unsupported signal extras changed to INFO.

    Required protection rules:
    - matched peaks are never suppressed;
    - model-source peaks are never suppressed;
    - NO_TRAD_SAMPLE rows are never suppressed;
    - a co-eluting ``-1``/``-2`` transition pair rescues the peak.
    """
    required = {
        "ai_row_id", "sample_id", "uid", "ai_chrom_index", "ai_rt_peak",
        "ai_area", "ai_apex_intensity", "match_status", "score_source",
        "alert_level", "alert_reasons", "review_required", "review_status",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"weak extra alert input is missing columns: {missing}")
    if not 0 < relative_height_max < 1 or not 0 < relative_area_max < 1:
        raise ValueError("relative thresholds must be in (0, 1)")
    if paired_transition_rt_tolerance_min <= 0:
        raise ValueError("paired transition RT tolerance must be positive")

    out = rows.copy()
    ai_mask = out["ai_row_id"].notna()
    channel_keys = ["sample_id", "ai_chrom_index"]
    channel_max_area = out.loc[ai_mask].groupby(channel_keys)["ai_area"].transform("max")
    out["relative_area_in_channel"] = math.nan
    out.loc[ai_mask, "relative_area_in_channel"] = (
        pd.to_numeric(out.loc[ai_mask, "ai_area"], errors="coerce")
        / channel_max_area.where(channel_max_area > 0)
    ).clip(lower=0.0, upper=1.0)

    if "relative_height_baseline_corrected" not in out:
        channel_max_height = out.loc[ai_mask].groupby(channel_keys)["ai_apex_intensity"].transform("max")
        out["relative_height_baseline_corrected"] = math.nan
        out.loc[ai_mask, "relative_height_baseline_corrected"] = (
            pd.to_numeric(out.loc[ai_mask, "ai_apex_intensity"], errors="coerce")
            / channel_max_height.where(channel_max_height > 0)
        ).clip(lower=0.0, upper=1.0)
        out["relative_height_basis"] = "raw_final_peak_fallback"
    else:
        out["relative_height_basis"] = "phase1_baseline_corrected"

    out = _add_pair_support(out, paired_transition_rt_tolerance_min)
    out["alert_policy_version"] = POLICY_VERSION
    out["alert_level_before_weak_filter"] = out["alert_level"]
    out["review_required_before_weak_filter"] = out["review_required"]
    out["weak_extra_ignored"] = False
    out["alert_disposition"] = "ACTIVE"

    weak = (
        out["match_status"].eq("AI_ONLY_EXTRA")
        & out["score_source"].eq("signal_rule")
        & pd.to_numeric(out["relative_height_baseline_corrected"], errors="coerce").lt(relative_height_max)
        & pd.to_numeric(out["relative_area_in_channel"], errors="coerce").lt(relative_area_max)
        & ~out["paired_transition_supported"].fillna(False)
    )
    out.loc[weak, "weak_extra_ignored"] = True
    out.loc[weak, "alert_disposition"] = "IGNORED_WEAK_AI_EXTRA"
    out.loc[weak, "alert_level"] = "INFO"
    out.loc[weak, "review_required"] = False
    out.loc[weak, "review_status"] = "NOT_REQUIRED"
    reasons = out.loc[weak, "alert_reasons"].fillna("").astype(str)
    out.loc[weak, "alert_reasons"] = reasons.map(
        lambda value: "|".join(filter(None, [value, "IGNORED_WEAK_AI_EXTRA"]))
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--summary_json", type=Path)
    parser.add_argument("--relative_height_max", type=float, default=0.10)
    parser.add_argument("--relative_area_max", type=float, default=0.10)
    parser.add_argument("--pair_rt_tol", type=float, default=0.10)
    args = parser.parse_args()

    source = pd.read_csv(args.input_csv, low_memory=False)
    result = apply_weak_extra_alert_rule(
        source,
        relative_height_max=args.relative_height_max,
        relative_area_max=args.relative_area_max,
        paired_transition_rt_tolerance_min=args.pair_rt_tol,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "policy_version": POLICY_VERSION,
        "input_rows": len(source),
        "output_rows": len(result),
        "ignored_weak_ai_extra": int(result["weak_extra_ignored"].sum()),
        "relative_height_max": args.relative_height_max,
        "relative_area_max": args.relative_area_max,
        "paired_transition_rt_tolerance_min": args.pair_rt_tol,
        "alert_levels_before": source["alert_level"].value_counts(dropna=False).to_dict(),
        "alert_levels_after": result["alert_level"].value_counts(dropna=False).to_dict(),
    }
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
