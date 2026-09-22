# -*- coding: utf-8 -*-
"""Array-based MassNova runtime shared by the embedded Cython/C DLL bridge."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from inference.massnova import (
    _scan_params_from_args,
    enumerate_peaks,
    finalize_channel_peaks,
    validate_with_model,
)
from inference.onnx_window_predictor import OnnxWindowPredictor


DEFAULT_RUNTIME_CONFIG = {
    "threshold": 0.5,
    "smooth_sigma": 0.8,
    "use_gpu": 0,
    "batch_size": 128,
    "min_chrom_points": 10,
    "min_max_intensity": 1000.0,
}

_PHASE1_KEYS = (
    "baseline_percentile", "baseline_mode", "min_peak_ratio",
    "prominence_ratio", "min_prominence_abs", "min_peak_gap_points",
    "min_peak_width_min", "void_time_min", "max_peaks_per_channel",
    "valley_ratio", "min_valley_central_frac",
)


def _finite_float(value, name):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


class MassNovaArrayRuntime:
    """Run the Python MassNova pipeline directly on C-provided XIC arrays."""

    def __init__(self, model_path, config=None, *, predictor=None):
        merged = dict(DEFAULT_RUNTIME_CONFIG)
        if config:
            merged.update(dict(config))
        self.config = merged
        self.args = SimpleNamespace(**merged)
        self.scan_params = _scan_params_from_args(self.args)
        self.predictor = predictor or OnnxWindowPredictor(
            model_path,
            use_gpu=int(merged["use_gpu"]),
            batch_size=int(merged["batch_size"]),
        )
        self.model_path = str(model_path)

    @property
    def is_gpu_enabled(self):
        return bool(getattr(self.predictor, "is_gpu_enabled", False))

    def _prepare(self, item, index):
        uid = str(item.get("uid", ""))
        rt_value = item.get("rt", item.get("x"))
        intensity_value = item.get("intensity", item.get("y"))
        if rt_value is None or intensity_value is None:
            raise ValueError(f"items[{index}] requires x/y or rt/intensity arrays")
        rt = np.asarray(rt_value, dtype=np.float64)
        raw = np.asarray(intensity_value, dtype=np.float64)
        if rt.ndim != 1 or raw.ndim != 1 or rt.size != raw.size:
            raise ValueError(f"items[{index}] RT and intensity must be equal 1-D arrays")
        if rt.size < 2:
            raise ValueError(f"items[{index}] arrays must contain at least two points")
        if not np.all(np.isfinite(rt)) or not np.all(np.isfinite(raw)):
            raise ValueError(f"items[{index}] arrays must be finite")
        if np.any(np.diff(rt) <= 0):
            raise ValueError(f"items[{index}] RT must be strictly increasing")

        # A positive per-item value overrides the process setting.  Zero keeps
        # the configured Python default, matching normal MassNova inference.
        item_sigma = _finite_float(item.get("smooth_sigma", 0.0), "smooth_sigma")
        sigma = item_sigma if item_sigma > 0 else float(self.config["smooth_sigma"])
        intensity = gaussian_filter1d(raw, sigma=sigma) if sigma > 0 else raw.copy()
        maximum = float(np.max(intensity)) if intensity.size else 0.0
        reason = None
        if int(self.config["min_chrom_points"]) > 0 and rt.size < int(self.config["min_chrom_points"]):
            reason = "too_few_points"
        elif float(self.config["min_max_intensity"]) > 0 and maximum < float(self.config["min_max_intensity"]):
            reason = "low_max_intensity"
        return {
            "index": index,
            "uid": uid,
            "rt": rt,
            "intensity": intensity,
            "max_intensity": maximum,
            "qc_reason": reason,
        }

    @staticmethod
    def _qc_result(feature):
        return {
            "uid": feature["uid"],
            "status": "alert",
            "peaks": [],
            "alerts": [{
                "level": "fail",
                "code": "CHANNEL_LOW_INTENSITY",
                "detail": {
                    "n_points": int(feature["rt"].size),
                    "max_intensity": float(feature["max_intensity"]),
                },
            }],
        }

    @staticmethod
    def _peak_result(feature, peaks):
        serialized = [{
            "a": float(peak["rt_min"]),
            "b": float(peak["rt_max"]),
            "c": float(peak["peak_score"]),
        } for peak in peaks]
        if serialized:
            # Model peaks and accepted signal-rule fallback peaks are both final
            # usable events.  Source affects c, not the channel status.
            return {"uid": feature["uid"], "status": "ok", "peaks": serialized, "alerts": []}
        return {
            "uid": feature["uid"],
            "status": "alert",
            "peaks": [],
            "alerts": [{"level": "fail", "code": "NO_PEAK_FOUND"}],
        }

    def process_items(self, items):
        prepared = [self._prepare(item, index) for index, item in enumerate(items)]
        output = [None] * len(prepared)
        accepted = [feature for feature in prepared if feature["qc_reason"] is None]
        for feature in prepared:
            if feature["qc_reason"] is not None:
                output[feature["index"]] = self._qc_result(feature)

        candidates_by_channel = {}
        rts = {}
        intensities = {}
        for feature in accepted:
            channel = feature["index"]
            rt, intensity = feature["rt"], feature["intensity"]
            rts[channel], intensities[channel] = rt, intensity
            apexes = enumerate_peaks(
                rt, intensity,
                **{key: self.scan_params[key] for key in _PHASE1_KEYS},
            )
            candidates_by_channel[channel] = [{
                "apex_idx": int(apex),
                "rt_peak": float(rt[apex]),
                "apex_intensity": float(intensity[apex]),
                "model_score": np.nan,
                "validated": False,
            } for apex in apexes]

        if accepted and any(candidates_by_channel.values()):
            validate_with_model(
                candidates_by_channel, rts, intensities, self.model_path,
                threshold=float(self.config["threshold"]),
                window_half_min=self.scan_params["window_half_min"],
                keep_windows=False,
                verbose=False,
                predictor=self.predictor,
            )

        for feature in accepted:
            channel = feature["index"]
            peaks = finalize_channel_peaks(
                feature["rt"], feature["intensity"],
                candidates_by_channel.get(channel, []), self.scan_params,
                threshold=float(self.config["threshold"]),
            )
            output[channel] = self._peak_result(feature, peaks)
        return {"items": output}

    def process_json(self, input_json):
        payload = json.loads(input_json)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("items is required and must be an array")
        return json.dumps(self.process_items(payload["items"]), ensure_ascii=False, separators=(",", ":"))
