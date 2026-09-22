"""centWave-inspired chromatographic peak detection."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from wavelet import ricker, cwt


def _continuous_ridge(cwt_matrix, location, min_ridge_length,
                      max_ridge_drift):
    """Return the strongest continuous ridge supporting ``location``.

    A ridge must contain a positive local maximum at every adjacent scale.
    Its position may move by at most ``max_ridge_drift`` samples between
    neighboring scales. Missing scales terminate the current ridge.
    """
    best_run = []
    current_run = []

    for scale_idx, response in enumerate(cwt_matrix):
        maxima, _ = find_peaks(response)
        maxima = maxima[response[maxima] > 0]

        if current_run:
            previous = current_run[-1][1]
            nearby = maxima[np.abs(maxima - previous) <= max_ridge_drift]
        else:
            nearby = maxima[np.abs(maxima - location) <= max_ridge_drift]

        if len(nearby):
            ridge_idx = int(nearby[np.argmax(response[nearby])])
            current_run.append(
                (scale_idx, ridge_idx, float(response[ridge_idx]))
            )
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = []

    if len(current_run) > len(best_run):
        best_run = current_run
    if len(best_run) < min_ridge_length:
        return None

    positions = [point[1] for point in best_run]
    strongest = max(best_run, key=lambda point: point[2])
    return {
        "scale_idx": strongest[0],
        "ridge_length": len(best_run),
        "ridge_drift": max(positions) - min(positions),
    }


def _nearest_valley(signal, apex, limit, direction):
    """Return the nearest local minimum between ``apex`` and ``limit``."""
    if direction < 0:
        lo, hi = limit, apex
    else:
        lo, hi = apex, limit

    if hi <= lo:
        return int(limit)

    segment = signal[lo:hi + 1]
    minima, _ = find_peaks(-segment)
    minima = minima[(minima > 0) & (minima < len(segment) - 1)]
    if len(minima):
        candidate = minima[-1] if direction < 0 else minima[0]
        return int(lo + candidate)

    return int(lo + np.argmin(segment))


def _refine_boundary(intensity, boundary, lo, hi, radius):
    """Move a filtered boundary to a nearby minimum in the raw signal."""
    search_lo = max(lo, boundary - radius)
    search_hi = min(hi, boundary + radius)
    if search_hi <= search_lo:
        return int(boundary)
    segment = intensity[search_lo:search_hi + 1]
    return int(search_lo + np.argmin(segment))


def _initial_boundaries(intensity, apex, best_scale, max_peak_width):
    """Estimate boundaries from the best-scale smoothed response."""
    n = len(intensity)
    half_width = max(1, max_peak_width // 2)
    left_limit = max(0, apex - half_width)
    right_limit = min(n - 1, apex + half_width)

    # The best CWT scale controls smoothing and therefore the peak width used
    # for the initial valley search. Capping the search prevents a broad
    # baseline region from producing unbounded peak intervals.
    sigma = max(0.5, float(best_scale) / 2.0)
    smooth = gaussian_filter1d(intensity, sigma=sigma, mode="nearest")
    left = _nearest_valley(smooth, apex, left_limit, -1)
    right = _nearest_valley(smooth, apex, right_limit, 1)

    radius = max(1, int(np.ceil(best_scale / 2.0)))
    left = _refine_boundary(intensity, left, left_limit, apex, radius)
    right = _refine_boundary(intensity, right, apex, right_limit, radius)
    return left, right, smooth


def _split_overlapping_boundaries(candidates, intensity):
    """Split adjacent overlapping peaks at their intervening raw valley."""
    candidates.sort(key=lambda candidate: candidate["apex_idx"])
    for left_peak, right_peak in zip(candidates, candidates[1:]):
        if left_peak["right_idx"] < right_peak["left_idx"]:
            continue

        left_apex = left_peak["apex_idx"]
        right_apex = right_peak["apex_idx"]
        if right_apex <= left_apex:
            continue

        smoothing_scale = min(left_peak["best_scale"], right_peak["best_scale"])
        smooth = gaussian_filter1d(
            intensity,
            sigma=max(0.5, float(smoothing_scale) / 2.0),
            mode="nearest",
        )
        between = smooth[left_apex:right_apex + 1]
        coarse_valley = left_apex + int(np.argmin(between))
        radius = max(1, int(np.ceil(smoothing_scale / 2.0)))
        valley = _refine_boundary(
            intensity, coarse_valley, left_apex, right_apex, radius
        )

        left_peak["right_idx"] = valley
        right_peak["left_idx"] = valley


def _peak_trend_scores(intensity, left, apex, right, best_scale):
    """Measure rising-left and falling-right variation around a peak."""
    sigma = max(0.5, float(best_scale) / 4.0)
    smooth = gaussian_filter1d(intensity, sigma=sigma, mode="nearest")
    left_diff = np.diff(smooth[left:apex + 1])
    right_diff = np.diff(smooth[apex:right + 1])

    left_variation = np.sum(np.abs(left_diff))
    right_variation = np.sum(np.abs(right_diff))
    if left_variation <= 0 or right_variation <= 0:
        return 0.0, 0.0

    left_score = np.sum(np.maximum(left_diff, 0)) / left_variation
    right_score = np.sum(np.maximum(-right_diff, 0)) / right_variation
    return float(left_score), float(right_score)


def _longest_true_run(mask):
    """Return the longest consecutive True run in a boolean array."""
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _peak_quality_metrics(intensity, left, apex, right, best_scale):
    """Measure residual roughness, direction efficiency and peak support."""
    segment = np.asarray(intensity[left:right + 1], dtype=float)
    if len(segment) < 3 or not left < apex < right:
        return {
            "roughness": np.inf,
            "smoothed_snr": 0.0,
            "left_direction_efficiency": 0.0,
            "right_direction_efficiency": 0.0,
            "effective_points": 0,
            "apex_position": 0.0,
        }

    sigma = max(1.0, float(best_scale) / 3.0)
    smooth = gaussian_filter1d(segment, sigma=sigma, mode="nearest")
    smooth_apex_rel = int(np.argmax(smooth))
    smooth_apex = float(smooth[smooth_apex_rel])
    baseline = float(max(smooth[0], smooth[-1]))
    prominence = max(0.0, smooth_apex - baseline)

    residual = segment - smooth
    residual_median = np.median(residual)
    residual_noise = 1.4826 * np.median(
        np.abs(residual - residual_median)
    )
    noise_denom = max(float(residual_noise), 1e-6)
    smooth_range = max(
        float(np.percentile(smooth, 95) - np.percentile(smooth, 5)),
        1e-6,
    )

    left_raw = segment[:smooth_apex_rel + 1]
    right_raw = segment[smooth_apex_rel:]
    left_variation = np.sum(np.abs(np.diff(left_raw)))
    right_variation = np.sum(np.abs(np.diff(right_raw)))
    left_gain = max(0.0, smooth_apex - float(smooth[0]))
    right_drop = max(0.0, smooth_apex - float(smooth[-1]))

    threshold = baseline + residual_noise
    effective_points = _longest_true_run(smooth > threshold)

    return {
        "roughness": float(residual_noise / smooth_range),
        "smoothed_snr": float(prominence / noise_denom),
        "left_direction_efficiency": float(
            left_gain / max(float(left_variation), 1e-6)
        ),
        "right_direction_efficiency": float(
            right_drop / max(float(right_variation), 1e-6)
        ),
        "effective_points": int(effective_points),
        "apex_position": float(smooth_apex_rel / (len(segment) - 1)),
    }


def detect_peaks_centwave(rt, intensity, scales=np.arange(1, 15),
                           snr_thresh=3, min_peak_width=2,
                           max_peak_width=None, min_ridge_length=None,
                           max_ridge_drift=2, min_trend_score=0.7,
                           max_roughness=0.7,
                           min_direction_efficiency=0.08,
                           min_effective_points=3,
                           min_smoothed_snr=1.1,
                           apex_margin=0.15,
                           quality_filter=True):
    """
    ``min_peak_width`` is the minimum distance between candidate apexes in
    samples. ``max_peak_width`` is the maximum returned boundary width in
    samples; by default it is six times the largest CWT scale.
    Candidates must also form a positive ridge over consecutive scales.
    ``min_ridge_length`` defaults to half of the scales (at least four),
    and ``max_ridge_drift`` limits movement between adjacent scales.
    ``min_trend_score`` requires the final peak shape to rise on the left
    and fall on the right; each score is the correctly directed variation
    divided by the total variation on that side.
    The optional joint quality filter also requires low residual roughness,
    efficient directional change, enough consecutive supported points, a
    significant smoothed peak and an apex away from both boundaries.
    """
    rt = np.asarray(rt)
    intensity = np.asarray(intensity, dtype=float)
    scales = np.asarray(scales, dtype=float)
    n = len(intensity)

    if n == 0 or len(rt) != n:
        if n == 0 and len(rt) == 0:
            return [], 0.0, 1e-6
        raise ValueError("rt and intensity must be non-empty arrays of equal length")
    if not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(rt)):
        raise ValueError("rt and intensity must contain only finite values")
    if scales.ndim != 1 or len(scales) == 0 or np.any(scales <= 0):
        raise ValueError("scales must be a non-empty sequence of positive values")
    if min_ridge_length is None:
        min_ridge_length = min(
            len(scales), max(4, int(np.ceil(len(scales) / 2)))
        )
    min_ridge_length = int(min_ridge_length)
    if min_ridge_length < 1 or min_ridge_length > len(scales):
        raise ValueError("min_ridge_length must be between 1 and len(scales)")
    max_ridge_drift = int(max_ridge_drift)
    if max_ridge_drift < 0:
        raise ValueError("max_ridge_drift must be non-negative")
    min_trend_score = float(min_trend_score)
    if not 0 <= min_trend_score <= 1:
        raise ValueError("min_trend_score must be between 0 and 1")
    max_roughness = float(max_roughness)
    if max_roughness < 0:
        raise ValueError("max_roughness must be non-negative")
    min_direction_efficiency = float(min_direction_efficiency)
    if not 0 <= min_direction_efficiency <= 1:
        raise ValueError(
            "min_direction_efficiency must be between 0 and 1"
        )
    min_effective_points = int(min_effective_points)
    if min_effective_points < 1:
        raise ValueError("min_effective_points must be at least 1")
    min_smoothed_snr = float(min_smoothed_snr)
    if min_smoothed_snr < 0:
        raise ValueError("min_smoothed_snr must be non-negative")
    apex_margin = float(apex_margin)
    if not 0 <= apex_margin < 0.5:
        raise ValueError("apex_margin must be between 0 and 0.5")

    if max_peak_width is None:
        max_peak_width = int(np.ceil(6 * np.max(scales)))
    max_peak_width = int(max_peak_width)
    if max_peak_width < 2:
        raise ValueError("max_peak_width must be at least 2 samples")

    median_int = np.median(intensity)
    low_part = intensity[intensity <= median_int]
    baseline = np.mean(low_part) if len(low_part) > 0 else np.min(intensity)
    low_std = np.std(low_part)
    noise = low_std if low_std > 0 else 1e-6

    cwt_matrix = cwt(intensity, ricker, scales)
    multiscale_response = np.sum(np.maximum(cwt_matrix, 0), axis=0)
    response_range = np.ptp(multiscale_response)
    response_mad = np.median(
        np.abs(multiscale_response - np.median(multiscale_response))
    )
    noise_floor = 3.0 * 1.4826 * response_mad
    prominence = max(0.15 * response_range, noise_floor)
    peak_idx, _ = find_peaks(
        multiscale_response,
        distance=min_peak_width,
        prominence=prominence if prominence > 0 else None,
    )

    candidates = []
    for cwt_idx in peak_idx:
        ridge = _continuous_ridge(
            cwt_matrix, cwt_idx, min_ridge_length, max_ridge_drift
        )
        if ridge is None:
            continue

        scale_idx = ridge["scale_idx"]
        best_scale = float(scales[scale_idx])

        # CWT localizes a candidate; the raw maximum in one best-scale radius
        # is the reported chromatographic apex.
        apex_radius = max(1, int(np.ceil(best_scale)))
        apex_lo = max(0, cwt_idx - apex_radius)
        apex_hi = min(n - 1, cwt_idx + apex_radius)
        apex = apex_lo + int(np.argmax(intensity[apex_lo:apex_hi + 1]))
        apex_intensity = intensity[apex]
        snr = (apex_intensity - baseline) / noise
        if snr < snr_thresh:
            continue

        left, right, _ = _initial_boundaries(
            intensity, apex, best_scale, max_peak_width
        )
        candidates.append({
            "left_idx": left,
            "right_idx": right,
            "apex_idx": apex,
            "best_scale": best_scale,
            "snr": snr,
            "ridge_length": ridge["ridge_length"],
            "ridge_drift": ridge["ridge_drift"],
        })

    unique = {}
    for candidate in candidates:
        apex = candidate["apex_idx"]
        coefficient = cwt_matrix[
            int(np.argmin(np.abs(scales - candidate["best_scale"]))), apex
        ]
        if apex not in unique or coefficient > unique[apex][0]:
            unique[apex] = (coefficient, candidate)
    candidates = [item[1] for item in unique.values()]
    _split_overlapping_boundaries(candidates, intensity)

    peaks = []
    for candidate in sorted(candidates, key=lambda item: item["apex_idx"]):
        left = candidate["left_idx"]
        right = candidate["right_idx"]
        apex = candidate["apex_idx"]
        left_score, right_score = _peak_trend_scores(
            intensity, left, apex, right, candidate["best_scale"]
        )
        if left_score < min_trend_score or right_score < min_trend_score:
            continue
        quality = _peak_quality_metrics(
            intensity, left, apex, right, candidate["best_scale"]
        )
        quality_pass = (
            quality["roughness"] <= max_roughness
            and quality["smoothed_snr"] >= min_smoothed_snr
            and quality["left_direction_efficiency"]
            >= min_direction_efficiency
            and quality["right_direction_efficiency"]
            >= min_direction_efficiency
            and quality["effective_points"] >= min_effective_points
            and apex_margin <= quality["apex_position"] <= 1.0 - apex_margin
        )
        if quality_filter and not quality_pass:
            continue
        peaks.append({
            "rt_start": rt[left],
            "rt_end": rt[right],
            "apex_rt": rt[apex],
            "apex_intensity": intensity[apex],
            "snr": candidate["snr"],
            "best_scale": candidate["best_scale"],
            "ridge_length": candidate["ridge_length"],
            "ridge_drift": candidate["ridge_drift"],
            "left_trend_score": left_score,
            "right_trend_score": right_score,
            **quality,
        })

    return peaks, baseline, noise
