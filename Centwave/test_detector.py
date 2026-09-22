"""Tests for CWT-based chromatographic peak boundary estimation."""

import unittest

import numpy as np

from detector import (
    _continuous_ridge,
    _peak_quality_metrics,
    _peak_trend_scores,
    detect_peaks_centwave,
)


def _gaussian(x, center, sigma, height):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


class PeakBoundaryTests(unittest.TestCase):
    def test_joint_quality_metrics_separate_clean_and_jagged_peaks(self):
        x = np.arange(121, dtype=float)
        clean = 20 + _gaussian(x, 60, 12, 500)
        jagged = clean + 120 * np.sin(np.pi * x / 2)

        clean_quality = _peak_quality_metrics(
            clean, 10, 60, 110, best_scale=12
        )
        jagged_quality = _peak_quality_metrics(
            jagged, 10, 60, 110, best_scale=12
        )

        self.assertLess(clean_quality["roughness"], 0.7)
        self.assertGreaterEqual(
            clean_quality["left_direction_efficiency"], 0.08
        )
        self.assertGreaterEqual(
            clean_quality["right_direction_efficiency"], 0.08
        )
        self.assertGreater(
            jagged_quality["roughness"], clean_quality["roughness"]
        )
        self.assertLess(
            jagged_quality["left_direction_efficiency"],
            clean_quality["left_direction_efficiency"],
        )
        self.assertTrue(
            jagged_quality["roughness"] > 0.7
            or jagged_quality["smoothed_snr"] < 1.1
            or jagged_quality["left_direction_efficiency"] < 0.08
            or jagged_quality["right_direction_efficiency"] < 0.08
            or jagged_quality["effective_points"] < 3
        )

    def test_rejects_invalid_joint_quality_parameters(self):
        with self.assertRaisesRegex(ValueError, "max_roughness"):
            detect_peaks_centwave(
                np.arange(10), np.ones(10), max_roughness=-1,
            )

    def test_joint_filter_rejects_sharply_oscillating_candidate(self):
        rt = np.arange(241, dtype=float)
        envelope = 20 + _gaussian(rt, 120, 14, 500)
        jagged = envelope + 200 * np.sin(np.pi * rt / 2)

        unfiltered, _, _ = detect_peaks_centwave(
            rt, jagged, scales=np.arange(2, 18), snr_thresh=1,
            max_peak_width=80, quality_filter=False,
        )
        filtered, _, _ = detect_peaks_centwave(
            rt, jagged, scales=np.arange(2, 18), snr_thresh=1,
            max_peak_width=80, quality_filter=True,
        )

        self.assertEqual(len(unfiltered), 1)
        self.assertEqual(filtered, [])

    def test_peak_shape_requires_left_rise_and_right_fall(self):
        valid = np.array([0, 1, 3, 6, 10, 7, 4, 2, 0], dtype=float)
        invalid = np.array([0, 1, 3, 6, 7, 8, 9, 10, 11], dtype=float)

        valid_scores = _peak_trend_scores(valid, 0, 4, 8, best_scale=1)
        invalid_scores = _peak_trend_scores(invalid, 0, 4, 8, best_scale=1)

        self.assertGreaterEqual(valid_scores[0], 0.7)
        self.assertGreaterEqual(valid_scores[1], 0.7)
        self.assertGreaterEqual(invalid_scores[0], 0.7)
        self.assertLess(invalid_scores[1], 0.7)

    def test_rejects_invalid_trend_threshold(self):
        with self.assertRaisesRegex(ValueError, "min_trend_score"):
            detect_peaks_centwave(
                np.arange(10), np.ones(10), min_trend_score=1.1,
            )

    def test_requires_consecutive_ridge_scales(self):
        cwt_matrix = np.zeros((7, 40), dtype=float)
        for scale_idx in (0, 1, 3, 4, 6):
            cwt_matrix[scale_idx, 20] = 10

        ridge = _continuous_ridge(
            cwt_matrix, location=20, min_ridge_length=3,
            max_ridge_drift=1,
        )

        self.assertIsNone(ridge)

    def test_accepts_continuous_ridge_with_limited_drift(self):
        cwt_matrix = np.zeros((6, 40), dtype=float)
        for scale_idx, location in enumerate((19, 20, 20, 21, 21, 22)):
            cwt_matrix[scale_idx, location] = 10 + scale_idx

        ridge = _continuous_ridge(
            cwt_matrix, location=20, min_ridge_length=5,
            max_ridge_drift=1,
        )

        self.assertIsNotNone(ridge)
        self.assertEqual(ridge["ridge_length"], 6)
        self.assertEqual(ridge["ridge_drift"], 3)

    def test_rejects_ridge_with_excessive_adjacent_drift(self):
        cwt_matrix = np.zeros((5, 40), dtype=float)
        for scale_idx, location in enumerate((20, 20, 24, 24, 24)):
            cwt_matrix[scale_idx, location] = 10

        ridge = _continuous_ridge(
            cwt_matrix, location=20, min_ridge_length=4,
            max_ridge_drift=1,
        )

        self.assertIsNone(ridge)

    def test_single_peak_boundaries_enclose_apex_and_are_capped(self):
        rt = np.arange(240, dtype=float)
        intensity = 10 + _gaussian(rt, 120, 8, 1000)
        peaks, _, _ = detect_peaks_centwave(
            rt, intensity, scales=np.arange(2, 18), snr_thresh=1,
            max_peak_width=60,
        )

        self.assertEqual(len(peaks), 1)
        peak = peaks[0]
        self.assertLess(peak["rt_start"], peak["apex_rt"])
        self.assertGreater(peak["rt_end"], peak["apex_rt"])
        self.assertLessEqual(peak["rt_end"] - peak["rt_start"], 60)
        self.assertAlmostEqual(peak["apex_rt"], 120, delta=1)
        self.assertIn("best_scale", peak)
        self.assertGreaterEqual(peak["ridge_length"], 3)
        self.assertIn("ridge_drift", peak)
        self.assertGreaterEqual(peak["left_trend_score"], 0.7)
        self.assertGreaterEqual(peak["right_trend_score"], 0.7)

    def test_overlapping_peaks_are_split_at_intervening_valley(self):
        rt = np.arange(220, dtype=float)
        intensity = (
            5
            + _gaussian(rt, 90, 7, 1000)
            + _gaussian(rt, 112, 7, 850)
        )
        peaks, _, _ = detect_peaks_centwave(
            rt, intensity, scales=np.arange(2, 14), snr_thresh=1,
            min_peak_width=5, max_peak_width=70,
        )

        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(peaks[0]["apex_rt"], 90, delta=2)
        self.assertAlmostEqual(peaks[1]["apex_rt"], 112, delta=2)
        self.assertEqual(peaks[0]["rt_end"], peaks[1]["rt_start"])
        self.assertGreater(peaks[0]["rt_end"], peaks[0]["apex_rt"])
        self.assertLess(peaks[1]["rt_start"], peaks[1]["apex_rt"])

    def test_raw_signal_refines_boundary_to_local_minimum(self):
        rt = np.arange(180, dtype=float)
        intensity = 20 + _gaussian(rt, 90, 9, 700)
        intensity[58] = 0
        intensity[122] = 0
        peaks, _, _ = detect_peaks_centwave(
            rt, intensity, scales=np.arange(3, 16), snr_thresh=1,
            max_peak_width=70,
        )

        self.assertEqual(len(peaks), 1)
        self.assertIn(peaks[0]["rt_start"], range(57, 60))
        self.assertIn(peaks[0]["rt_end"], range(121, 124))


if __name__ == "__main__":
    unittest.main()
