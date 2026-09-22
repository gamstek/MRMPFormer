"""Audit a per-channel relative-apex candidate filter against saved test3 results.

This is an analysis tool only.  It does not change the production inference path.
It reconstructs the Phase-1 SciPy candidates from mzML, then measures each saved AI
peak against the strongest Phase-1 candidate in the same sample/channel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.massnova import enumerate_peaks, extract_full_xics


def _sample_number(value: str) -> int | None:
    try:
        return int(str(value).rsplit("_", 1)[1])
    except (IndexError, TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confidence-csv",
        type=Path,
        default=ROOT / "project_review/confidence_v2_test3_from_existing/peak_confidence.csv",
    )
    parser.add_argument("--mzml-dir", type=Path, default=ROOT.parent / "data/mzml/test3")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/massnova.json")
    parser.add_argument("--sample-start", type=int, default=3)
    parser.add_argument("--sample-end", type=int, default=58)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "project_review/candidate_relative_intensity_audit_test3_3_58.csv",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    phase1 = {
        "baseline_percentile": float(cfg.get("scan_baseline_percentile", 25.0)),
        "baseline_mode": str(cfg.get("scan_baseline_mode", "global_percentile")),
        "min_peak_ratio": float(cfg.get("scan_min_peak_ratio", 0.04)),
        "prominence_ratio": float(cfg.get("scan_prominence_ratio", 0.055)),
        "min_prominence_abs": float(cfg.get("scan_min_prominence_abs", 0.0)),
        "min_peak_gap_points": int(cfg.get("scan_min_peak_gap_points", 3)),
        "min_peak_width_min": float(cfg.get("scan_min_peak_width_min", 0.10)),
        "void_time_min": float(cfg.get("scan_void_time_min", 0.5)),
        "max_peaks_per_channel": int(cfg.get("scan_max_peaks_per_channel", 50)),
        "valley_ratio": 0.90,
        "min_valley_central_frac": 0.12,
    }

    confidence = pd.read_csv(args.confidence_csv, low_memory=False)
    ai = confidence[confidence["ai_row_id"].notna()].copy()
    ai = ai.drop_duplicates("ai_row_id")
    sample_no = ai["sample_id"].map(_sample_number)
    ai = ai[sample_no.between(args.sample_start, args.sample_end)].copy()

    candidate_rows: list[dict] = []
    mzml_files = sorted(
        (
            path
            for path in args.mzml_dir.glob("*.mzML")
            if (number := _sample_number(path.stem)) is not None
            and args.sample_start <= number <= args.sample_end
        ),
        key=lambda path: _sample_number(path.stem) or -1,
    )
    for index, mzml_path in enumerate(mzml_files, start=1):
        features, _ = extract_full_xics(
            mzml_path,
            smooth_sigma=float(cfg.get("smooth_sigma", 0.8)),
            min_chrom_points=int(cfg.get("pipeline_min_chrom_points", 10)),
            min_max_intensity=float(cfg.get("pipeline_min_max_intensity", 1000.0)),
        )
        for feature in features:
            rt = feature["rt"]
            intensity = feature["intensity"]
            apexes = enumerate_peaks(rt, intensity, **phase1)
            apex_values = [float(intensity[apex]) for apex in apexes]
            max_apex = max(apex_values, default=np.nan)
            body = np.asarray(rt, dtype=float) >= phase1["void_time_min"]
            if int(np.count_nonzero(body)) < 5:
                body = np.ones(len(intensity), dtype=bool)
            baseline = (
                float(np.percentile(np.asarray(intensity, dtype=float)[body], phase1["baseline_percentile"]))
                if int(np.count_nonzero(body)) else np.nan
            )
            retained_count = sum(
                value >= args.threshold * max_apex for value in apex_values
            ) if apex_values else 0
            candidate_rows.append(
                {
                    "sample_id": mzml_path.stem,
                    "ai_chrom_index": int(feature["chrom_index"]),
                    "phase1_candidate_count": len(apexes),
                    "phase1_retained_at_threshold": retained_count,
                    "phase1_baseline": baseline,
                    "phase1_max_candidate_apex": max_apex,
                }
            )
        print(f"[{index:02d}/{len(mzml_files):02d}] {mzml_path.name}")

    candidates = pd.DataFrame(candidate_rows)
    audited = ai.merge(candidates, on=["sample_id", "ai_chrom_index"], how="left")
    audited["relative_to_phase1_max"] = (
        audited["ai_apex_intensity"] / audited["phase1_max_candidate_apex"]
    )
    corrected_denominator = audited["phase1_max_candidate_apex"] - audited["phase1_baseline"]
    audited["relative_height_baseline_corrected"] = (
        (audited["ai_apex_intensity"] - audited["phase1_baseline"])
        / corrected_denominator.where(corrected_denominator > 0)
    ).clip(lower=0.0, upper=1.0)
    audited["removed_at_threshold"] = audited["relative_to_phase1_max"] < args.threshold
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    audited.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    removed = audited["removed_at_threshold"].fillna(False)
    print("\nExact Phase-1 denominator audit")
    total_candidates = int(candidates["phase1_candidate_count"].sum())
    retained_candidates = int(candidates["phase1_retained_at_threshold"].sum())
    print(
        f"Phase-1 candidates removed: {total_candidates - retained_candidates:,}/"
        f"{total_candidates:,} ({(total_candidates - retained_candidates) / max(1, total_candidates):.2%})"
    )
    print(f"AI peaks: {len(audited):,}")
    print(f"Removed at {args.threshold:.2f}: {int(removed.sum()):,} ({removed.mean():.2%})")
    for status in ("MATCHED_UNIQUE", "MATCHED_AMBIGUOUS", "AI_ONLY_EXTRA", "NO_TRAD_SAMPLE"):
        mask = audited["match_status"].eq(status)
        count = int((removed & mask).sum())
        print(f"{status}: {count:,}/{int(mask.sum()):,} ({count / max(1, int(mask.sum())):.2%})")
    print("\nRemoved by match status and score source")
    print(pd.crosstab(audited.loc[removed, "match_status"], audited.loc[removed, "score_source"], margins=True))
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
