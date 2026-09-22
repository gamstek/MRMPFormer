# -*- coding: utf-8 -*-
"""Evaluate CentreWave with the same boundary-tolerance protocol as the models.

The neural-model benchmark uses label-driven, two-minute XIC ROIs and counts a
true positive only when both predicted boundaries are within the requested
tolerance of a manual interval.  CentreWave has no calibrated [0, 1] class
probability, so its native operating point is represented by score=1 after its
intensity/width/SNR validation.  The script additionally scans CentreWave's
actual decision parameter, ``snr_min``.

Run from ``model/``::

    python -m tools.evaluation.evaluate_centrewave_grid \
      --cw-root ../CW/CW --output-dir project_review/cw_benchmark
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import grey_opening, median_filter

from preprocessing.coco_annotation import (
    group_labels_by_sample,
    label_key,
    parse_labels_xlsx,
    parse_rt_field,
)
from preprocessing.label_qc import check_label_rt_consistency, mark_excluded_labels
from tools.evaluation.evaluate_baseline import _parse_gt_peaks, evaluate
from tools.evaluation.evaluate_special_isolation import _read_tags, evaluate_special


SCORE_GRID = (0.1, 0.5, 0.8, 0.9, 0.99)
TOL_GRID = (0.01, 0.05, 0.1, 0.2, 0.5)
SNR_GRID = (0.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _peak_label_value(rec) -> int | None:
    try:
        return int(float(str(rec.get("peak_label", "")).strip()))
    except (TypeError, ValueError):
        return None


def _prepare_labels(label_path: Path, qc_tol: float):
    labels = parse_labels_xlsx(label_path)
    tags = _read_tags(label_path)
    for rec in labels:
        try:
            rid = int(float(rec.get("roi_id") or -1))
        except (TypeError, ValueError):
            rid = -1
        rec["tag"] = tags.get(rid, "")
    qc_rows, excluded = check_label_rt_consistency(labels, tol=qc_tol)
    n_excluded = mark_excluded_labels(labels, excluded)
    active = [
        rec for rec in labels
        if not rec.get("_qc_excluded") and _peak_label_value(rec) in (None, 0, 1)
    ]
    order, groups = group_labels_by_sample(active)
    gt_count = sum(len(_parse_gt_peaks(rec)) for rec in active)
    special_count = sum(
        len(_parse_gt_peaks(rec)) for rec in active if str(rec.get("tag") or "").strip()
    )
    return {
        "all": labels,
        "active": active,
        "order": order,
        "groups": groups,
        "qc_rows": qc_rows,
        "n_excluded": n_excluded,
        "gt_count": gt_count,
        "special_count": special_count,
    }


def _find_mzml(mzml_dir: Path, sample_id: str) -> Path:
    direct = mzml_dir / sample_id
    if direct.is_file():
        return direct
    stem = sample_id[:-5] if sample_id.lower().endswith(".mzml") else sample_id
    hits = [p for p in mzml_dir.glob("*.mzML") if p.stem == stem]
    if len(hits) != 1:
        raise FileNotFoundError(f"样品 {sample_id!r} 在 {mzml_dir} 中匹配到 {len(hits)} 个 mzML")
    return hits[0]


def _configure_centrewave(cw_root: Path):
    vendor = cw_root / "vendor_pywt"
    package = cw_root / "Centwave" / "CentreWave"
    for path in (vendor, package):
        if not path.is_dir():
            raise FileNotFoundError(path)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from centrewave.config import DetectorConfig, ValidationConfig
    from centrewave.detector import AdaptiveWaveletDetector
    from centrewave.mzml_io import extract_channels, sanitize_series
    from centrewave.noise import robust_std, zero_phase_highpass
    from centrewave.pipeline import subtract_baseline
    from centrewave.validation import compute_peak_metrics, estimate_peak_bounds

    return {
        "DetectorConfig": DetectorConfig,
        "ValidationConfig": ValidationConfig,
        "AdaptiveWaveletDetector": AdaptiveWaveletDetector,
        "extract_channels": extract_channels,
        "sanitize_series": sanitize_series,
        "robust_std": robust_std,
        "zero_phase_highpass": zero_phase_highpass,
        "subtract_baseline": subtract_baseline,
        "compute_peak_metrics": compute_peak_metrics,
        "estimate_peak_bounds": estimate_peak_bounds,
    }


class CentreWaveRoiRunner:
    """Thin label-ROI adapter around the current CentreWave detector."""

    def __init__(self, api, half_window_min: float):
        self.api = api
        self.half_window_s = float(half_window_min) * 60.0
        # These are the settings in CW/CW/evaluate_centrewave.py.  min_sep=2
        # is deliberate there so close double peaks can be represented.
        self.dcfg = api["DetectorConfig"](
            sigma_min=1.0,
            sigma_max=20.0,
            amin=300.0,
            min_sep=2.0,
            n_scales=100,
            n_cal=48,
        )
        self.vcfg = api["ValidationConfig"](
            intensity_min=1000.0,
            snr_min=10.0,
            noise_window_s=60.0,
        )
        self.detectors = {}

    def _detector(self, fs: float):
        key = round(float(fs), 2)
        if key not in self.detectors:
            # The official test adapter uses a fixed calibration length of 320.
            self.detectors[key] = self.api["AdaptiveWaveletDetector"](
                fs, 320, self.dcfg
            )
        return self.detectors[key]

    def _preprocess(self, x, fs):
        x = self.api["sanitize_series"](x)
        x = self.api["subtract_baseline"](x, fs, 60.0)
        x = median_filter(x, size=3)
        return grey_opening(x, size=5)

    def detect(self, t_full, x_full, center_min: float):
        center_s = float(center_min) * 60.0
        lo, hi = center_s - self.half_window_s, center_s + self.half_window_s
        mask = (t_full >= lo) & (t_full <= hi)
        tw = np.asarray(t_full[mask], dtype=float)
        xw = np.asarray(x_full[mask], dtype=float)
        if len(tw) < self.dcfg.min_n:
            return []
        fs = 1.0 / float(np.median(np.diff(tw)))
        xd = self._preprocess(xw, fs)
        peaks = self._detector(fs).detect(xd)["peaks"]
        if not peaks:
            return []

        # Detector positions are relative to the cropped array.  Anchor them to
        # the first observed scan (rather than the ideal mathematical window).
        t0 = float(tw[0])
        for peak in peaks:
            peak.t_peak = t0 + float(peak.t_peak)

        residual = self.api["zero_phase_highpass"](
            xw, fs, self.vcfg.noise_hp_hz
        )
        noise_std = self.api["robust_std"](residual) if residual.size else float("nan")
        rows = []
        for peak in peaks:
            others = [p for p in peaks if p is not peak]
            row = self.api["compute_peak_metrics"](
                tw,
                xw,
                peak,
                others,
                self.vcfg,
                noise_std_plateau=noise_std,
                residual=residual,
            )
            start, end = self.api["estimate_peak_bounds"](
                tw, xw, peak.t_peak, peak.sigma
            )
            row.update(
                confidence=float(peak.profile_value),
                amplitude=float(peak.amplitude),
                area_gauss=float(peak.area),
                det_start=float(start),
                det_end=float(end),
            )
            rows.append(row)
        rows.sort(key=lambda r: r["confidence"], reverse=True)
        return rows[: self.dcfg.max_peaks_per_channel]

    def passes(self, row, snr_min: float) -> bool:
        if not (row.get("intensity_pass") and row.get("width_pass")):
            return False
        local = float(row.get("snr_local_hf", float("nan")))
        plateau = float(row.get("snr_plateau", float("nan")))
        local_ok = np.isfinite(local) and local >= snr_min
        floor_ok = self.vcfg.snr_time_floor <= 0 or (
            np.isfinite(local) and local >= self.vcfg.snr_time_floor
        )
        plateau_ok = np.isfinite(plateau) and plateau >= snr_min and floor_ok
        return bool(local_ok or plateau_ok)


def _detect_dataset(
    runner: CentreWaveRoiRunner,
    label_info,
    mzml_dir: Path,
):
    raw_by_sample = {}
    timing = []
    missing_channels = []
    for sample_id in label_info["order"]:
        mzml = _find_mzml(mzml_dir, sample_id)
        full_start = time.perf_counter()
        channels, _meta = runner.api["extract_channels"](str(mzml))
        load_s = time.perf_counter() - full_start
        sample_rows = []
        detect_s = 0.0
        for roi_index, rec in enumerate(label_info["groups"][sample_id], 1):
            native_id = label_key(rec.get("compound"), rec.get("channel"))
            if not native_id:
                continue
            channel = channels.get(native_id)
            if channel is None:
                raw = str(rec.get("compound") or "").strip()
                channel = channels.get(raw)
            if channel is None:
                missing_channels.append({"sample_id": sample_id, "native_id": native_id})
                continue
            center = parse_rt_field(rec.get("rt"))
            if center is None:
                continue
            started = time.perf_counter()
            candidates = runner.detect(channel[0], channel[1], center)
            detect_s += time.perf_counter() - started
            for rank, cand in enumerate(candidates, 1):
                row = dict(cand)
                row.update(
                    sample_id=sample_id,
                    roi_index=roi_index,
                    native_id=native_id,
                    image=f"{roi_index}_mz_centrewave.jpeg",
                    rank=rank,
                )
                sample_rows.append(row)
        full_s = time.perf_counter() - full_start
        raw_by_sample[sample_id] = sample_rows
        timing.append(
            {
                "sample_id": sample_id,
                "mzml": str(mzml),
                "load_s": load_s,
                "detector_s": detect_s,
                "pipeline_s": full_s,
                "roi_count": len(label_info["groups"][sample_id]),
                "candidate_count": len(sample_rows),
            }
        )
        print(
            f"[CW] {sample_id}: ROI={len(label_info['groups'][sample_id])}, "
            f"candidates={len(sample_rows)}, detect={detect_s:.3f}s, total={full_s:.3f}s",
            flush=True,
        )
    return raw_by_sample, timing, missing_channels


def _candidate_record(row):
    keys = (
        "sample_id", "roi_index", "native_id", "rank", "confidence",
        "t_peak", "det_start", "det_end", "sigma", "area", "area_gauss",
        "amplitude", "intensity_max", "snr", "snr_time", "snr_local_hf",
        "snr_plateau", "pass_branch", "intensity_pass", "width_pass",
        "passes_constraints",
    )
    return {k: row.get(k) for k in keys}


def _write_eval_inputs(
    dataset_dir: Path,
    label_info,
    raw_by_sample,
    runner: CentreWaveRoiRunner,
    snr_min: float,
):
    tag = f"snr_{snr_min:g}".replace(".", "p")
    root = dataset_dir / "eval_inputs" / tag
    pred_feat_map = {}
    for sample_id in label_info["order"]:
        stem = Path(sample_id).stem
        sample_dir = root / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        features = []
        for i, rec in enumerate(label_info["groups"][sample_id], 1):
            features.append(
                {
                    "Compound Name": i,
                    "native_id": label_key(rec.get("compound"), rec.get("channel")),
                }
            )
        feature_path = sample_dir / "feature.csv"
        pd.DataFrame(features).to_csv(feature_path, index=False, encoding="utf-8-sig")

        preds = []
        for row in raw_by_sample.get(sample_id, []):
            if runner.passes(row, snr_min):
                preds.append(
                    {
                        "image": row["image"],
                        "rt_min": float(row["det_start"]) / 60.0,
                        "rt_max": float(row["det_end"]) / 60.0,
                        # Binary native validation decision.  This is intentionally
                        # not presented as a calibrated neural probability.
                        "score": 1.0,
                        "area": row.get("area"),
                        "confidence_raw": row.get("confidence"),
                        "snr_local_hf": row.get("snr_local_hf"),
                        "snr_plateau": row.get("snr_plateau"),
                    }
                )
        pred_path = sample_dir / "predictions.csv"
        pd.DataFrame(
            preds,
            columns=(
                "image", "rt_min", "rt_max", "score", "area",
                "confidence_raw", "snr_local_hf", "snr_plateau",
            ),
        ).to_csv(pred_path, index=False, encoding="utf-8-sig")
        pred_feat_map[stem] = {"pred": pred_path, "feat": feature_path}
    return pred_feat_map


def _evaluate_grid(pred_feat_map, label_path: Path, tolerances, scores, qc_tol):
    result = {}
    details = {}
    areas = {}
    for score in scores:
        result[str(score)] = {}
        for tol in tolerances:
            metrics, det, area, _qc = evaluate(
                pred_feat_map,
                str(label_path),
                tol=tol,
                min_score=score,
                quant_tol=tol,
                qc_label_rt_tol=qc_tol,
            )
            result[str(score)][str(tol)] = metrics
            details[(score, tol)] = det
            areas[(score, tol)] = area
    return result, details, areas


def _best_cell(grid):
    cells = []
    for score, by_tol in grid.items():
        for tol, metrics in by_tol.items():
            cells.append((metrics["f1"], metrics["recall"], metrics["precision"], score, tol, metrics))
    return max(cells, key=lambda x: (x[0], x[1], x[2], float(x[3]), float(x[4])))


def _timing_summary(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else None
    return {
        "sample_count": len(rows),
        "detector_mean_ms_per_sample": mean("detector_s") * 1000.0 if rows else None,
        "pipeline_mean_ms_per_sample": mean("pipeline_s") * 1000.0 if rows else None,
        "load_mean_ms_per_sample": mean("load_s") * 1000.0 if rows else None,
        "per_sample": rows,
    }


def _metric_cell(m):
    return f"{m['f1']:.3f}"


def _write_report(out_dir: Path, datasets, native_snr, scores, tolerances, snr_grid):
    lines = [
        "# CentreWave 与四模型同口径评测",
        "",
        "- 命中规则：预测起止与人工起止偏差均不超过容差。",
        "- ROI：标签 RT 中心左右各 1 min；QC：双离子/跨样品 RT 阈值 1.0 min。",
        f"- CentreWave 原生工作点：intensity > 1000、SNR >= {native_snr:g}、宽度护栏通过。",
        "- CentreWave 没有 [0,1] 分类概率；通过原生验证的候选统一记 score=1，因此模型 score 扫描行相同。",
        "- 推理在 CPU 上执行，GPU 不参与小波计算。",
        "",
    ]
    for name, data in datasets.items():
        native = data["native_grid"]
        lines.extend([
            f"## {name}",
            "",
            f"GT 峰 {data['label']['gt_count']}，特殊峰 {data['label']['special_count']}，"
            f"QC 剔除标签行 {data['label']['n_excluded']}。",
            "",
            f"### F1：原生 SNR={native_snr:g}，扫容差",
            "",
            "| 容差(min) | CentreWave | P | R | TP/FP/FN |",
            "|---:|---:|---:|---:|---:|",
        ])
        for tol in tolerances:
            m = native[str(scores[0])][str(tol)]
            lines.append(
                f"| {tol:g} | {m['f1']:.3f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['TP']}/{m['FP']}/{m['FN']} |"
            )
        lines.extend([
            "",
            "### F1：容差 0.1 min，模型 score 扫描",
            "",
            "| score 阈值 | CentreWave |",
            "|---:|---:|",
        ])
        for score in scores:
            m = native[str(score)]["0.1"]
            lines.append(f"| {score:g} | {m['f1']:.3f} |")
        h = native["0.9"]["0.1"]
        lines.extend([
            "",
            f"### 详表：原生 SNR={native_snr:g} / 容差 0.1 min",
            "",
            "| P | R | F1 | TP/FP/FN | 面积R² | RT 起/止中位(min) |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {h['precision']:.4f} | {h['recall']:.4f} | {h['f1']:.4f} | "
            f"{h['TP']}/{h['FP']}/{h['FN']} | "
            f"{h['area_r2_pred_vs_manual'] if h['area_r2_pred_vs_manual'] is not None else 'N/A'} | "
            f"{h['rt_start_dev_median_min'] if h['rt_start_dev_median_min'] is not None else 'N/A'} / "
            f"{h['rt_end_dev_median_min'] if h['rt_end_dev_median_min'] is not None else 'N/A'} |",
            "",
            f"### 特殊峰：原生 SNR={native_snr:g}，扫容差",
            "",
            "| 容差(min) | CentreWave |",
            "|---:|---:|",
        ])
        for tol in tolerances:
            sp = data["special"][str(tol)]
            rate = sp["special_rate"]
            lines.append(f"| {tol:g} | {sp['special_tp']}/{sp['special_gt']} ({rate:.3f}) |")
        lines.extend([
            "",
            "### CentreWave 原生 SNR 阈值扫描（容差 0.1 min）",
            "",
            "| SNR | P | R | F1 | TP/FP/FN |",
            "|---:|---:|---:|---:|---:|",
        ])
        for snr in snr_grid:
            m = data["snr_scan"][str(snr)]["0.1"]
            lines.append(
                f"| {snr:g} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                f"{m['TP']}/{m['FP']}/{m['FN']} |"
            )
        t = data["timing"]
        lines.extend([
            "",
            "### 时间",
            "",
            "| 样品数 | 小波检测均值/样品 | 全流程均值/样品 |",
            "|---:|---:|---:|",
            f"| {t['sample_count']} | {t['detector_mean_ms_per_sample']:.0f} ms | "
            f"{t['pipeline_mean_ms_per_sample']:.0f} ms |",
            "",
        ])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cw-root", default="../CW/CW")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--output-dir", default="project_review/cw_benchmark")
    parser.add_argument("--datasets", nargs="+", default=("test1", "test2"))
    parser.add_argument("--roi-half-window-min", type=float, default=1.0)
    parser.add_argument("--native-snr", type=float, default=10.0)
    parser.add_argument("--qc-label-rt-tol", type=float, default=1.0)
    parser.add_argument("--scores", default=",".join(map(str, SCORE_GRID)))
    parser.add_argument("--tolerances", default=",".join(map(str, TOL_GRID)))
    parser.add_argument("--snr-grid", default=",".join(map(str, SNR_GRID)))
    args = parser.parse_args()

    cw_root = Path(args.cw_root).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = _parse_csv_floats(args.scores)
    tolerances = _parse_csv_floats(args.tolerances)
    snr_grid = _parse_csv_floats(args.snr_grid)
    api = _configure_centrewave(cw_root)
    runner = CentreWaveRoiRunner(api, args.roi_half_window_min)

    all_results = {}
    for dataset in args.datasets:
        print(f"\n===== {dataset} =====", flush=True)
        label_path = data_root / "label" / f"{dataset}.xlsx"
        mzml_dir = data_root / "mzml" / dataset
        label_info = _prepare_labels(label_path, args.qc_label_rt_tol)
        raw_by_sample, timing_rows, missing = _detect_dataset(runner, label_info, mzml_dir)
        dataset_dir = out_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [_candidate_record(row) for rows in raw_by_sample.values() for row in rows]
        ).to_csv(dataset_dir / "raw_candidates.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(label_info["qc_rows"]).to_csv(
            dataset_dir / "label_qc.csv", index=False, encoding="utf-8-sig"
        )
        _write_json(dataset_dir / "missing_channels.json", missing)

        native_map = _write_eval_inputs(
            dataset_dir, label_info, raw_by_sample, runner, args.native_snr
        )
        native_grid, native_details, native_areas = _evaluate_grid(
            native_map, label_path, tolerances, scores, args.qc_label_rt_tol
        )
        native_details[(0.9, 0.1)].to_csv(
            dataset_dir / "details_native_snr10_score0p9_tol0p1.csv",
            index=False,
            encoding="utf-8-sig",
        )
        native_areas[(0.9, 0.1)].to_csv(
            dataset_dir / "areas_native_snr10_score0p9_tol0p1.csv",
            index=False,
            encoding="utf-8-sig",
        )

        special = {}
        for tol in tolerances:
            metrics, audit = evaluate_special(
                native_map,
                str(label_path),
                tol=tol,
                min_score=0.9,
                qc_label_rt_tol=args.qc_label_rt_tol,
            )
            special[str(tol)] = metrics
            if abs(tol - 0.1) < 1e-12:
                pd.DataFrame(audit).to_csv(
                    dataset_dir / "special_audit_native_snr10_tol0p1.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        snr_scan = {}
        for snr in snr_grid:
            scan_map = _write_eval_inputs(dataset_dir, label_info, raw_by_sample, runner, snr)
            scan_grid, _, _ = _evaluate_grid(
                scan_map, label_path, tolerances, (0.9,), args.qc_label_rt_tol
            )
            snr_scan[str(snr)] = scan_grid["0.9"]

        result = {
            "dataset": dataset,
            "label_path": str(label_path),
            "label": {
                "rows_total": len(label_info["all"]),
                "rows_active": len(label_info["active"]),
                "n_excluded": label_info["n_excluded"],
                "gt_count": label_info["gt_count"],
                "special_count": label_info["special_count"],
            },
            "settings": {
                "cw_root": str(cw_root),
                "roi_half_window_min": args.roi_half_window_min,
                "native_snr": args.native_snr,
                "qc_label_rt_tol": args.qc_label_rt_tol,
                "scores": scores,
                "tolerances": tolerances,
                "snr_grid": snr_grid,
                "detector": vars(runner.dcfg),
                "validation": vars(runner.vcfg),
            },
            "native_grid": native_grid,
            "special": special,
            "snr_scan": snr_scan,
            "timing": _timing_summary(timing_rows),
            "missing_channels": missing,
        }
        _write_json(dataset_dir / "summary.json", result)
        all_results[dataset] = result

    _write_json(out_dir / "summary.json", all_results)
    _write_report(out_dir, all_results, args.native_snr, scores, tolerances, snr_grid)
    print(f"\n完成: {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
