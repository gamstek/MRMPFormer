# -*- coding: utf-8 -*-
"""Compare the mzML Python entry with the array/DLL Python entry.

The two entries intentionally have different input front ends.  This audit
starts after mzML transition selection: the normal entry receives the same
raw RT/intensity arrays through ``MassNovaArrayRuntime`` and compares final
``a/b/c`` values channel by channel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.massnova import (
    _scan_params_from_args,
    enumerate_peaks,
    extract_full_xics,
    finalize_channel_peaks,
    validate_with_model,
)
from inference.massnova_runtime import MassNovaArrayRuntime


PHASE1_KEYS = (
    "baseline_percentile", "baseline_mode", "min_peak_ratio",
    "prominence_ratio", "min_prominence_abs", "min_peak_gap_points",
    "min_peak_width_min", "void_time_min", "max_peaks_per_channel",
    "valley_ratio", "min_valley_central_frac",
)


def _model_path(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    # Deployment configs are defined relative to model/, not configs/.
    return config_path.resolve().parents[1] / candidate


def _normal_python_results(features, args, model_path):
    scan_params = _scan_params_from_args(args)
    candidates_by_channel = {}
    rts = {}
    intensities = {}
    for feature in features:
        channel = int(feature["chrom_index"])
        rt = feature["rt"]
        intensity = feature["intensity"]
        rts[channel] = rt
        intensities[channel] = intensity
        apexes = enumerate_peaks(
            rt, intensity,
            **{key: scan_params[key] for key in PHASE1_KEYS},
        )
        candidates_by_channel[channel] = [{
            "apex_idx": int(apex),
            "rt_peak": float(rt[apex]),
            "apex_intensity": float(intensity[apex]),
            "model_score": np.nan,
            "validated": False,
        } for apex in apexes]

    if any(candidates_by_channel.values()):
        validate_with_model(
            candidates_by_channel, rts, intensities, model_path,
            threshold=float(args.threshold),
            window_half_min=scan_params["window_half_min"],
            keep_windows=False,
            verbose=False,
            onnx_use_gpu=int(args.use_gpu),
            onnx_batch_size=int(args.batch_size),
        )

    output = []
    for feature in features:
        channel = int(feature["chrom_index"])
        peaks = finalize_channel_peaks(
            feature["rt"], feature["intensity"],
            candidates_by_channel.get(channel, []), scan_params,
        )
        output.append({
            "uid": str(feature["uid"]),
            "status": "ok" if peaks else "alert",
            "peaks": [{
                "a": float(peak["rt_min"]),
                "b": float(peak["rt_max"]),
                "c": float(peak["peak_score"]),
            } for peak in peaks],
        })
    return output


def audit(mzml_path: Path, config_path: Path, max_channels: int, atol: float):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(**config)
    model_path = _model_path(config_path, config["model"])

    raw_features, _ = extract_full_xics(
        mzml_path, smooth_sigma=0.0,
        min_chrom_points=0, min_max_intensity=0.0, verbose=False,
    )
    normal_features, _ = extract_full_xics(
        mzml_path, smooth_sigma=float(config["smooth_sigma"]),
        min_chrom_points=int(config["pipeline_min_chrom_points"]),
        min_max_intensity=float(config["pipeline_min_max_intensity"]),
        verbose=False,
    )
    if max_channels > 0:
        normal_features = normal_features[:max_channels]
    selected = {int(feature["chrom_index"]) for feature in normal_features}
    raw_by_channel = {
        int(feature["chrom_index"]): feature for feature in raw_features
        if int(feature["chrom_index"]) in selected
    }
    raw_selected = [raw_by_channel[int(feature["chrom_index"])]
                    for feature in normal_features]

    expected = _normal_python_results(normal_features, args, model_path)
    runtime_config = dict(config)
    runtime_config.update({
        "min_chrom_points": int(config["pipeline_min_chrom_points"]),
        "min_max_intensity": float(config["pipeline_min_max_intensity"]),
    })
    runtime = MassNovaArrayRuntime(model_path, runtime_config)
    actual = runtime.process_items([{
        "uid": feature["uid"],
        "x": feature["rt"],
        "y": feature["intensity"],
        "smooth_sigma": 0.0,
    } for feature in raw_selected])["items"]

    mismatches = []
    maximum = {"a": 0.0, "b": 0.0, "c": 0.0}
    total_peaks = 0
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left["uid"] != right["uid"] or left["status"] != right["status"]:
            mismatches.append({"channel": index, "kind": "identity/status",
                               "python": left, "array": right})
            continue
        if len(left["peaks"]) != len(right["peaks"]):
            mismatches.append({"channel": index, "kind": "peak_count",
                               "python": left, "array": right})
            continue
        total_peaks += len(left["peaks"])
        for peak_index, (p_left, p_right) in enumerate(zip(left["peaks"], right["peaks"])):
            diffs = {key: abs(p_left[key] - p_right[key]) for key in ("a", "b", "c")}
            for key, value in diffs.items():
                maximum[key] = max(maximum[key], value)
            if any(value > atol for value in diffs.values()):
                mismatches.append({"channel": index, "peak": peak_index,
                                   "kind": "numeric", "diff": diffs,
                                   "python": p_left, "array": p_right})

    result = {
        "mzml": str(mzml_path.resolve()),
        "model": str(model_path.resolve()),
        "threshold": float(config["threshold"]),
        "channels": len(expected),
        "peaks": total_peaks,
        "max_abs_diff": maximum,
        "tolerance": atol,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if mismatches else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/massnova.json"))
    parser.add_argument("--max_channels", type=int, default=12)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    raise SystemExit(audit(args.mzml, args.config, args.max_channels, args.atol))


if __name__ == "__main__":
    main()
