"""Compare a candidate ONNX through the unchanged MassNova Python pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inference.massnova_runtime import MassNovaArrayRuntime
from tools.build_wide_finetune import (
    ROOT, SOURCES, contains_all, load_run, parse_number, read_labels,
    valid_intervals, window_for,
)


def interval_iou(left, right):
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def score_channel(expected, predicted):
    candidates = sorted(
        ((interval_iou(gt, pred), gi, pi)
         for gi, gt in enumerate(expected) for pi, pred in enumerate(predicted)),
        reverse=True,
    )
    used_gt, used_pred = set(), set()
    errors = []
    for iou, gi, pi in candidates:
        if iou < 0.5 or gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        errors.append((abs(expected[gi][0] - predicted[pi][0]),
                       abs(expected[gi][1] - predicted[pi][1])))
    return len(used_gt), len(predicted) - len(used_pred), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-channels", type=int, default=48)
    parser.add_argument("--max-channels", type=int, default=0, help="0 evaluates all valid channels")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Model/final peak threshold; runtime default is 0.5")
    args = parser.parse_args()

    mzml = ROOT / "data/mzml/20251111-01/20251111-01_1.mzML"
    traces = load_run(mzml)
    rows = read_labels(SOURCES["20251111"], "20251111")
    examples = []
    for row in rows:
        intervals, status = valid_intervals(row)
        native_id = row["_transition_id"]
        if status != "ok" or native_id not in traces:
            continue
        rt, intensity = traces[native_id]
        center = parse_number(row.get("rt"))
        if center is None or not contains_all(intervals, window_for(center, rt)):
            continue
        # The vendor converter pads each chromatogram with repeated RT=-1 s
        # placeholders. The MassNova array API requires strictly increasing RT.
        valid = rt >= 0
        rt, intensity = rt[valid], intensity[valid]
        if len(rt) < 2 or np.any(np.diff(rt) <= 0):
            continue
        examples.append((native_id, rt, intensity, intervals))
    if args.max_channels > 0:
        examples = examples[:args.max_channels]

    runtime = MassNovaArrayRuntime(args.model, {
        "threshold": args.threshold,
        "smooth_sigma": 0.8,
        "use_gpu": -1,
        "batch_size": 64,
        "min_chrom_points": 10,
        "min_max_intensity": 1000.0,
    })
    gt_count = hit_count = fp_count = no_peak_channels = 0
    errors = []
    for start in range(0, len(examples), args.batch_channels):
        batch = examples[start:start + args.batch_channels]
        output = runtime.process_items([
            {"uid": name, "x": rt, "y": intensity}
            for name, rt, intensity, _ in batch
        ])["items"]
        for (_, _, _, gt), item in zip(batch, output):
            predicted = [(p["a"], p["b"]) for p in item["peaks"]]
            hit, fp, channel_errors = score_channel(gt, predicted)
            gt_count += len(gt)
            hit_count += hit
            fp_count += fp
            no_peak_channels += not predicted
            errors.extend(channel_errors)
        print(f"processed {min(start + args.batch_channels, len(examples))}/{len(examples)} channels", flush=True)
    result = {
        "model": str(args.model),
        "threshold": args.threshold,
        "channels": len(examples),
        "ground_truth_peaks": gt_count,
        "matched_iou50": hit_count,
        "recall_iou50": hit_count / gt_count if gt_count else 0.0,
        "false_positive_peaks": fp_count,
        "precision_iou50": hit_count / (hit_count + fp_count) if hit_count + fp_count else 0.0,
        "no_peak_channels": no_peak_channels,
        "left_mae_min_on_matches": float(np.mean([x[0] for x in errors])) if errors else None,
        "right_mae_min_on_matches": float(np.mean([x[1] for x in errors])) if errors else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
