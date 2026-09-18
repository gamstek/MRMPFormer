# -*- coding: utf-8 -*-
"""Shared, auditable intrinsic scores for final peak events."""
from __future__ import annotations

import math

import numpy as np


def signal_peak_score(
    snr: float,
    n_points: float,
    *,
    snr_pivot: float = 10.0,
    points_good: float = 10.0,
    snr_weight: float = 0.8,
    points_weight: float = 0.2,
) -> tuple[float, float, float]:
    """Return ``(score, snr_component, points_component)`` for a signal peak.

    This is an engineering evidence score, not a calibrated probability.
    ``snr_component`` is 0.5 at ``snr_pivot`` and saturates smoothly.
    ``points_component`` reaches 1 at ``points_good`` supported peak points.
    """
    raw = (snr, n_points, snr_pivot, points_good, snr_weight, points_weight)
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return math.nan, math.nan, math.nan
    snr, n_points, snr_pivot, points_good, snr_weight, points_weight = values
    if not all(np.isfinite(values)) or snr < 0 or n_points < 0:
        return math.nan, math.nan, math.nan
    if snr_pivot <= 0 or points_good <= 0 or snr_weight < 0 or points_weight < 0:
        raise ValueError("signal score parameters must be positive")
    weight_sum = snr_weight + points_weight
    if weight_sum <= 0:
        raise ValueError("signal score weights must have a positive sum")

    conf_snr = float(np.clip(snr / (snr + snr_pivot), 0.0, 1.0))
    conf_points = float(np.clip(n_points / points_good, 0.0, 1.0))
    eps = 1e-12
    score = math.exp(
        (snr_weight / weight_sum) * math.log(max(conf_snr, eps))
        + (points_weight / weight_sum) * math.log(max(conf_points, eps))
    )
    return float(np.clip(score, 0.0, 1.0)), conf_snr, conf_points


def unified_peak_score(
    boundary_source: str,
    model_score: float,
    signal_score: float,
) -> tuple[float, str]:
    """Select a final peak score while retaining its source semantics."""
    source = str(boundary_source or "").strip().lower()
    try:
        model_value = float(model_score)
    except (TypeError, ValueError):
        model_value = math.nan
    try:
        signal_value = float(signal_score)
    except (TypeError, ValueError):
        signal_value = math.nan

    if source == "model" and np.isfinite(model_value):
        return float(np.clip(model_value, 0.0, 1.0)), "model"
    if source == "signal" and np.isfinite(signal_value):
        return float(np.clip(signal_value, 0.0, 1.0)), "signal_rule"
    if np.isfinite(model_value):
        return float(np.clip(model_value, 0.0, 1.0)), "model"
    if np.isfinite(signal_value):
        return float(np.clip(signal_value, 0.0, 1.0)), "signal_rule"
    return math.nan, "missing"
