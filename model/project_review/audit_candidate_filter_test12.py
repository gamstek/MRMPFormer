"""Estimate the effect of a Phase-1 relative-apex filter on labeled test1/test2."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.massnova import enumerate_peaks, extract_full_xics
from preprocessing.coco_annotation import label_key, parse_rt_field
from tools.evaluation.evaluate_baseline import _parse_gt_peaks, match_image, prf
from tools.evaluation.evaluate_centrewave_grid import _prepare_labels


THRESHOLD = 0.25
ROI_HALF_MIN = 1.0
TOLERANCES = (0.1, 0.2, 0.5)


def phase1_params(config: dict) -> dict:
    return {
        "baseline_percentile": float(config.get("scan_baseline_percentile", 25.0)),
        "baseline_mode": str(config.get("scan_baseline_mode", "global_percentile")),
        "min_peak_ratio": float(config.get("scan_min_peak_ratio", 0.04)),
        "prominence_ratio": float(config.get("scan_prominence_ratio", 0.055)),
        "min_prominence_abs": float(config.get("scan_min_prominence_abs", 0.0)),
        "min_peak_gap_points": int(config.get("scan_min_peak_gap_points", 3)),
        "min_peak_width_min": float(config.get("scan_min_peak_width_min", 0.10)),
        "void_time_min": float(config.get("scan_void_time_min", 0.5)),
        "max_peaks_per_channel": int(config.get("scan_max_peaks_per_channel", 50)),
        "valley_ratio": 0.90,
        "min_valley_central_frac": 0.12,
    }


def candidate_denominators(dataset: str, config: dict, sample_stems: set[str]) -> pd.DataFrame:
    rows = []
    params = phase1_params(config)
    mzml_files = [
        path
        for path in sorted((ROOT.parent / f"data/mzml/{dataset}").glob("*.mzML"))
        if path.stem in sample_stems
    ]
    for mzml in mzml_files:
        features, _ = extract_full_xics(
            mzml,
            smooth_sigma=float(config.get("smooth_sigma", 0.8)),
            min_chrom_points=int(config.get("pipeline_min_chrom_points", 10)),
            min_max_intensity=float(config.get("pipeline_min_max_intensity", 1000.0)),
        )
        for feature in features:
            apexes = enumerate_peaks(feature["rt"], feature["intensity"], **params)
            values = [float(feature["intensity"][apex]) for apex in apexes]
            maximum = max(values, default=float("nan"))
            rows.append(
                {
                    "mzml_stem": mzml.stem,
                    "chrom_index": int(feature["chrom_index"]),
                    "candidate_count": len(values),
                    "candidate_retained": sum(v >= THRESHOLD * maximum for v in values),
                    "max_candidate_apex": maximum,
                }
            )
    return pd.DataFrame(rows)


def score(labels: list[dict], predictions: pd.DataFrame, keep_filter, tol: float) -> dict:
    tp = fp = fn = 0
    selected = predictions[keep_filter].copy()
    for rec in labels:
        sample = Path(str(rec.get("sample_id") or "")).stem
        uid = label_key(rec.get("compound"), rec.get("channel"))
        center = parse_rt_field(rec.get("rt"))
        rows = selected[(selected["mzml_stem"] == sample) & (selected["uid"] == uid)]
        if center is not None:
            rows = rows[(rows["rt_peak"] >= center - ROI_HALF_MIN) & (rows["rt_peak"] <= center + ROI_HALF_MIN)]
        pred_rows = rows.sort_values("apex_intensity", ascending=False).to_dict("records")
        pairs, fps, fns, _ = match_image(pred_rows, _parse_gt_peaks(rec), tol=tol, loose_tol=0)
        tp += len(pairs)
        fp += len(fps)
        fn += len(fns)
    precision, recall, f1 = prf(tp, fp, fn)
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    config = json.loads((ROOT / "configs/massnova.json").read_text(encoding="utf-8"))
    summaries = []
    for dataset in ("test1", "test2"):
        prediction_path = ROOT.parent / f"output/inference/massnova_{dataset}_special_v2_rerun/all.csv"
        predictions = pd.read_csv(prediction_path)
        denom = candidate_denominators(dataset, config, set(predictions["mzml_stem"].astype(str)))
        predictions = predictions.merge(denom, on=["mzml_stem", "chrom_index"], how="left")
        predictions["relative_to_phase1_max"] = predictions["apex_intensity"] / predictions["max_candidate_apex"]
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = _prepare_labels(ROOT.parent / f"data/label/{dataset}.xlsx", 1.0)
        keep_all = pd.Series(True, index=predictions.index)
        keep_quarter = predictions["relative_to_phase1_max"] >= THRESHOLD
        total_candidates = int(denom["candidate_count"].sum())
        retained_candidates = int(denom["candidate_retained"].sum())
        print(f"\n{dataset}: candidates removed {total_candidates - retained_candidates}/{total_candidates}")
        print(f"{dataset}: final peaks removed {(~keep_quarter).sum()}/{len(predictions)}")
        for tol in TOLERANCES:
            before = score(prepared["active"], predictions, keep_all, tol)
            after = score(prepared["active"], predictions, keep_quarter, tol)
            summaries.append({"dataset": dataset, "tolerance": tol, "setting": "before", **before})
            summaries.append({"dataset": dataset, "tolerance": tol, "setting": "quarter_filter", **after})
            print(f"tol={tol}: before={before} after={after}")
        out = ROOT / f"project_review/candidate_filter_audit_{dataset}_predictions.csv"
        predictions.to_csv(out, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary.to_csv(ROOT / "project_review/candidate_filter_audit_test12_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
