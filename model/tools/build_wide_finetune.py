"""Build an audited, augmented COCO dataset for the wide-peak experiment.

Inputs are read-only. Converted mzML files must be placed under
output/wide_finetune/clean_mzml; the 20251111 mzML is read from data/mzml.
The entire 20251111 acquisition is held out for validation. No deployment
model or inference/postprocessing code is changed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pyopenms as oms
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

MODEL_DIR = Path(__file__).resolve().parents[1]
ROOT = MODEL_DIR.parent
sys.path.insert(0, str(MODEL_DIR))
from preprocessing.xic_extraction import render_roi_jpeg  # noqa: E402
from utils.mzml_chromatogram_ids import resolve_native_ids_for_chromatograms  # noqa: E402

WIDTH = 400
HEIGHT = 300
HALF_WINDOW_MIN = 1.0
SMOOTH_SIGMA = 0.8
SEED = 42
SOURCES = {
    "20260919": ROOT / "data/wide_data/20250919-01.xlsx",
    "20260910": ROOT / "data/wide_data/20260910-01(1).xlsx",
    "20251111": ROOT / "data/wide_data/20251111-宽峰.xlsx",
}


def parse_number(value):
    if value is None or str(value).strip() == "":
        return None
    match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def embedded_intensity(value):
    if value is None:
        return None
    match = re.search(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*\)", str(value))
    return float(match.group(1)) if match else None


def read_labels(path, source):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book.worksheets[0]
    values = sheet.values
    header = next(values)
    rows = []
    for excel_row, values in enumerate(values, 2):
        row = dict(zip(header, values))
        native_id = str(row.get("comonent") or "").strip()
        match = re.search(r"-(\d+)$", native_id)
        row.update({
            "_source": source,
            "_excel_row": excel_row,
            "_transition_id": native_id,
            "_compound_base": native_id[: match.start()] if match else native_id,
        })
        rows.append(row)
    return rows


def decode_id(value):
    if isinstance(value, bytes):
        # pyopenms on this Windows build returns locale-encoded native IDs.
        # A few GBK byte sequences are also valid UTF-8, so try GBK first.
        for encoding in ("gbk", "utf-8"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                pass
        return value.decode("utf-8", errors="replace")
    return str(value)


def sample_name(path):
    for _, element in ET.iterparse(path, events=("start",)):
        if element.tag.rsplit("}", 1)[-1] == "sample":
            return element.get("name", "")
    raise ValueError(f"sample name not found: {path}")


def load_run(path):
    experiment = oms.MSExperiment()
    oms.MzMLFile().load(str(path), experiment)
    traces = {}
    chromatograms = experiment.getChromatograms()
    native_ids = resolve_native_ids_for_chromatograms(str(path), chromatograms, decode_id)
    for chrom, native_id in zip(chromatograms, native_ids):
        rt_sec, intensity = chrom.get_peaks()
        if len(rt_sec) < 3 or native_id.startswith("TIC") or native_id == "0":
            continue
        if native_id in traces:
            raise ValueError(f"duplicate native ID {native_id!r} in {path}")
        traces[native_id] = (
            np.asarray(rt_sec, dtype=np.float64) / 60.0,
            np.asarray(intensity, dtype=np.float64),
        )
    return traces


def anchor_error(row, trace):
    rt, intensity = trace
    errors = []
    # In these Excel files the right endpoint intensity can be the fitted
    # baseline value, not the raw XIC intensity. The first left anchor is a
    # measured point and is suitable for matching acquisitions.
    raw = row.get("peak_start1")
    time = parse_number(raw)
    expected = embedded_intensity(raw)
    if time is not None and expected is not None:
        nearest = int(np.argmin(np.abs(rt - time)))
        if abs(rt[nearest] - time) > 0.015:
            return math.inf
        actual = max(0.0, float(intensity[nearest]))
        errors.append(abs(math.log1p(actual) - math.log1p(max(0.0, expected))))
    return float(np.mean(errors)) if errors else math.inf


def map_20260919(rows, paths, runs):
    if len(rows) % 3:
        raise ValueError("20260919 rows are not complete three-transition groups")
    groups = [rows[i:i + 3] for i in range(0, len(rows), 3)]
    for group in groups:
        if {row["_transition_id"].rsplit("-", 1)[-1] for row in group} != {"1", "2", "3"}:
            raise ValueError(f"unexpected transition group near Excel row {group[0]['_excel_row']}")
    costs = np.full((len(groups), len(paths)), math.inf)
    for gi, group in enumerate(groups):
        for pi, path in enumerate(paths):
            trace_map = runs[path]
            if not all(row["_transition_id"] in trace_map for row in group):
                continue
            costs[gi, pi] = np.median([
                anchor_error(row, trace_map[row["_transition_id"]]) for row in group
            ])
    if not np.all(np.isfinite(costs).any(axis=1)):
        raise ValueError("some 20260919 label groups have no candidate run")
    assigned_groups, assigned_paths = linear_sum_assignment(costs)
    if len(assigned_groups) != len(groups):
        raise ValueError("incomplete 20260919 assignment")
    mapping = {}
    audit = []
    for gi, pi in zip(assigned_groups, assigned_paths):
        best = float(costs[gi, pi])
        second = float(np.partition(costs[gi], 1)[1])
        # The endpoint intensities in this workbook come from the source XIC.
        # An ambiguous or poor assignment must not silently become a label.
        if not np.isfinite(best) or best > 0.001 or second - best < 0.001:
            raise ValueError(
                f"uncertain 20260919 mapping at Excel row {groups[gi][0]['_excel_row']}: "
                f"{paths[pi].name}, error={best:.4f}, margin={second-best:.4f}"
            )
        for row in groups[gi]:
            mapping[row["_excel_row"]] = paths[pi]
        audit.append({
            "first_excel_row": groups[gi][0]["_excel_row"],
            "run": paths[pi].name,
            "anchor_error": round(best, 6),
            "next_best_margin": round(second - best, 6),
        })
    return mapping, audit


def valid_intervals(row):
    intervals = []
    for k in (1, 2, 3):
        left_raw, right_raw = row.get(f"peak_start{k}"), row.get(f"peak_end{k}")
        left, right = parse_number(left_raw), parse_number(right_raw)
        if left is None and right is None:
            continue
        if left is None or right is None or not np.isfinite([left, right]).all() or right <= left:
            return None, "invalid_boundary"
        intervals.append((left, right))
    return (intervals, "ok") if intervals else (None, "missing_boundary")


def window_for(center, rt):
    left = max(center - HALF_WINDOW_MIN, float(rt[0]))
    right = min(center + HALF_WINDOW_MIN, float(rt[-1]))
    return left, right


def contains_all(intervals, window, tolerance=0.001):
    left, right = window
    return left < right and all(left - tolerance <= start < end <= right + tolerance
                                for start, end in intervals)


def noise_variant(intensity, rng):
    low = intensity[intensity <= np.quantile(intensity, 0.3)]
    background_mad = 1.4826 * np.median(np.abs(low - np.median(low))) if len(low) else 0.0
    peak = float(np.max(intensity))
    sigma = min(max(background_mad * 0.2, peak * 0.001), peak * 0.005)
    return np.maximum(0.0, intensity + rng.normal(0.0, sigma, size=len(intensity)))


def add_image(dataset, image_name, intervals, window, metadata):
    image_id = len(dataset["images"]) + 1
    dataset["images"].append({
        "id": image_id, "file_name": image_name, "width": WIDTH, "height": HEIGHT,
        **metadata,
    })
    left, right = window
    for start, end in intervals:
        # Excel RTs are rounded to three decimals, while mzML keeps the
        # original sampling time. Sub-0.001-minute edge differences are only
        # rounding and may map a fraction of a pixel beyond the canvas.
        x1 = max(0.0, WIDTH * (start - left) / (right - left))
        x2 = min(float(WIDTH), WIDTH * (end - left) / (right - left))
        width = x2 - x1
        if not 0 <= x1 < x2 <= WIDTH + 1e-4 or width < 1:
            raise ValueError(f"invalid pixel box for {image_name}: {x1}, {x2}")
        dataset["annotations"].append({
            "id": len(dataset["annotations"]) + 1,
            "image_id": image_id,
            "category_id": 0,
            "bbox": [round(x1, 3), 0.0, round(width, 3), float(HEIGHT)],
            "area": round(width * HEIGHT, 3),
            "iscrowd": 0,
        })


def replay_existing(dataset, destination, source_dir, count, rng):
    coco_path = source_dir / f"{source_dir.name}_coco.json"
    old = json.loads(coco_path.read_text(encoding="utf-8"))
    candidates = [item for item in old["images"] if (source_dir / item["file_name"]).is_file()]
    selected = rng.sample(candidates, min(count, len(candidates)))
    by_image = defaultdict(list)
    for ann in old["annotations"]:
        by_image[ann["image_id"]].append(ann)
    for image in selected:
        filename = f"replay_{image['id']}_{Path(image['file_name']).name}"
        shutil.copy2(source_dir / image["file_name"], destination / filename)
        new_id = len(dataset["images"]) + 1
        dataset["images"].append({
            "id": new_id, "file_name": filename, "width": image["width"],
            "height": image["height"], "source": "normal_replay", "augmentation": "none",
        })
        for ann in by_image[image["id"]]:
            dataset["annotations"].append({
                "id": len(dataset["annotations"]) + 1,
                "image_id": new_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })
    return len(selected)


def build(args):
    output = args.output.resolve()
    clean = args.clean_mzml.resolve()
    paths19 = sorted((clean / "20260919-01").glob("*.mzML"),
                     key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    paths10 = sorted((clean / "20260910-01").glob("*.mzML"))
    paths11 = sorted((ROOT / "data/mzml/20251111-01").glob("*.mzML"))
    if len(paths19) != 62 or len(paths10) != 9 or len(paths11) != 1:
        raise ValueError(f"expected 62/9/1 mzML files, found {len(paths19)}/{len(paths10)}/{len(paths11)}")
    all_paths = paths19 + paths10 + paths11
    runs = {path: load_run(path) for path in all_paths}
    rows = {source: read_labels(path, source) for source, path in SOURCES.items()}
    map19, map_audit = map_20260919(rows["20260919"], paths19, runs)
    map10 = {sample_name(path): path for path in paths10}
    if len(map10) != 9:
        raise ValueError("duplicate 20260910 mzML sample names")
    assignments = {}
    assignments.update(map19)
    for row in rows["20260910"]:
        path = map10.get(str(row.get("raw_file") or "").strip())
        assignments[("20260910", row["_excel_row"])] = path
    for row in rows["20251111"]:
        assignments[("20251111", row["_excel_row"])] = paths11[0]

    dataset = {
        split: {"images": [], "annotations": [], "categories": [
            {"id": 0, "name": "peak", "supercategory": "chromatographic_peak"}
        ]} for split in ("train", "val")
    }
    qc = []
    rng = np.random.default_rng(SEED)
    for source in ("20260919", "20260910", "20251111"):
        split = "val" if source == "20251111" else "train"
        for row in rows[source]:
            excel_row = row["_excel_row"]
            path = (assignments.get(excel_row) if source == "20260919"
                    else assignments.get((source, excel_row)))
            native_id = row["_transition_id"]
            result = {
                "source": source, "excel_row": excel_row, "transition_id": native_id,
                "compound_base": row["_compound_base"],
                "run": path.name if path else "", "status": "", "n_images": 0,
            }
            if path is None:
                result["status"] = "sample_not_matched"
            elif native_id not in runs[path]:
                result["status"] = "transition_not_matched"
            else:
                intervals, reason = valid_intervals(row)
                center = parse_number(row.get("rt"))
                rt, raw_intensity = runs[path][native_id]
                # All but one 20260919 left-end intensity agree with their
                # assigned raw XIC to within rounding error. Its lone outlier
                # is held for manual label review rather than trained on.
                anchor = anchor_error(row, (rt, raw_intensity)) if source == "20260919" else None
                result["anchor_error"] = round(anchor, 6) if anchor is not None else ""
                if source == "20260919" and anchor is not None and anchor > 0.05:
                    result["status"] = "anchor_mismatch_review"
                elif reason != "ok":
                    result["status"] = reason
                elif center is None or not math.isfinite(center):
                    result["status"] = "invalid_center"
                elif not contains_all(intervals, window_for(center, rt)):
                    result["status"] = "boundary_outside_original_window"
                else:
                    result["status"] = "included"
                    candidates = [(0.0, False, "original")]
                    if split == "train":
                        candidates += [
                            (-0.10, False, "shift_left"),
                            (0.10, False, "shift_right"),
                            (float(rng.uniform(-0.15, 0.15)), True, "shift_noise"),
                        ]
                    for offset, add_noise, variant in candidates:
                        window = window_for(center + offset, rt)
                        if not contains_all(intervals, window):
                            continue
                        intensity = noise_variant(raw_intensity, rng) if add_noise else raw_intensity
                        smoothed = gaussian_filter1d(intensity, sigma=SMOOTH_SIGMA)
                        image_name = f"wide_{source}_{path.stem}_{excel_row}_{variant}.jpeg"
                        image_path = output / "coco" / split / image_name
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        mask = (rt >= window[0]) & (rt <= window[1])
                        if mask.sum() < 3:
                            continue
                        render_roi_jpeg(rt[mask], smoothed[mask], *window, image_path)
                        add_image(dataset[split], image_name, intervals, window, {
                            "source": source, "run": path.name, "excel_row": excel_row,
                            "transition_id": native_id, "augmentation": variant,
                            "rt_lo": window[0], "rt_hi": window[1],
                        })
                        result["n_images"] += 1
                    if not result["n_images"]:
                        result["status"] = "no_rendered_image"
            qc.append(result)
    replay = ROOT / "data/coco/traindatav1"
    replay_rng = random.Random(SEED)
    n_train_replay = replay_existing(dataset["train"], output / "coco/train",
                                     replay / "train", args.replay_train, replay_rng)
    n_val_replay = replay_existing(dataset["val"], output / "coco/val",
                                   replay / "val", args.replay_val, replay_rng)
    for split in ("train", "val"):
        target = output / "coco" / split / f"{split}_coco.json"
        target.write_text(json.dumps(dataset[split], ensure_ascii=False), encoding="utf-8")
    with (output / "qc_rows.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qc[0]))
        writer.writeheader()
        writer.writerows(qc)
    with (output / "mapping_20260919.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(map_audit[0]))
        writer.writeheader()
        writer.writerows(map_audit)
    summary = {
        "seed": SEED, "smooth_sigma": SMOOTH_SIGMA,
        "held_out_source": "20251111", "normal_replay_train": n_train_replay,
        "normal_replay_val": n_val_replay,
        "qc_status": dict(Counter(f"{item['source']}:{item['status']}" for item in qc)),
        "train_images": len(dataset["train"]["images"]),
        "train_boxes": len(dataset["train"]["annotations"]),
        "val_images": len(dataset["val"]["images"]),
        "val_boxes": len(dataset["val"]["annotations"]),
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_config = json.loads((MODEL_DIR / "configs/mrmpformer_special_v2.json").read_text(encoding="utf-8"))
    train_config.update({
        "_comment_model": "Fine-tune delivered mrmpformer_special_v2 weights without changing architecture or loss",
        "_comment_resume": "Load delivered checkpoint, reset optimizer, and start fine-tuning at epoch 0",
        "_comment_wide": "Initial wide-peak fine-tune; 20251111 is held out; deployment remains unchanged",
        "coco_path": Path(os.path.relpath(output / "coco", MODEL_DIR)).as_posix(),
        "output_dir": Path(os.path.relpath(output / "train_v1", MODEL_DIR)).as_posix(),
        "resume": "checkpoint/mrmpformerv2.pth",
        "reset_optimizer": True,
        "start_epoch": 0,
        "lr": 1e-5,
        "lr_backbone": 1e-6,
        "batch_size": 4,
        "epochs": 8,
        "lr_drop": 6,
        "num_workers": 2,
    })
    (output / "train_config.json").write_text(
        json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/wide_finetune")
    parser.add_argument("--clean-mzml", type=Path,
                        default=ROOT / "output/wide_finetune/clean_mzml")
    parser.add_argument("--replay-train", type=int, default=1000)
    parser.add_argument("--replay-val", type=int, default=200)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
