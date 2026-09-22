import math
import unittest

import pandas as pd

from postprocessing.comparison_confidence_v2 import (
    ConfidenceConfig,
    _traditional_quality,
    match_peak_events,
    score_confidence,
)
from utils.peak_scores import signal_peak_score, unified_peak_score


class PeakScoreV2Tests(unittest.TestCase):
    def test_traditional_expected_rt_difference_within_point_one_is_normal(self):
        quality = _traditional_quality({
            "trad_rt_peak": 7.614,
            "trad_rt_expected": 7.521,
            "trad_snr": math.nan,
        }, ConfidenceConfig())
        self.assertAlmostEqual(quality["trad_expected_rt_diff_abs"], 0.093)
        self.assertAlmostEqual(quality["trad_expected_rt_excess_abs"], 0.0)
        self.assertTrue(quality["trad_expected_rt_within_tolerance"])
        self.assertAlmostEqual(quality["conf_trad_expected_rt"], 1.0)
        self.assertEqual(quality["quality_trad_level"], "GREEN")

    def test_traditional_expected_rt_confidence_uses_only_excess_over_tolerance(self):
        quality = _traditional_quality({
            "trad_rt_peak": 1.20,
            "trad_rt_expected": 1.00,
            "trad_snr": math.nan,
        }, ConfidenceConfig())
        self.assertAlmostEqual(quality["trad_expected_rt_excess_abs"], 0.10)
        self.assertFalse(quality["trad_expected_rt_within_tolerance"])
        self.assertAlmostEqual(quality["conf_trad_expected_rt"], 0.5)
        self.assertEqual(quality["quality_trad_level"], "RED")

    def test_signal_score_has_auditable_components(self):
        score, snr_part, points_part = signal_peak_score(10.0, 5.0)
        self.assertAlmostEqual(snr_part, 0.5)
        self.assertAlmostEqual(points_part, 0.5)
        self.assertAlmostEqual(score, 0.5)
        selected, source = unified_peak_score("signal", math.nan, score)
        self.assertAlmostEqual(selected, 0.5)
        self.assertEqual(source, "signal_rule")

    def test_model_score_remains_distinct_from_signal_score(self):
        selected, source = unified_peak_score("model", 0.72, 0.95)
        self.assertAlmostEqual(selected, 0.72)
        self.assertEqual(source, "model")

    def test_symmetric_area_difference_and_confidence(self):
        matches = pd.DataFrame([{
            "match_status": "MATCHED_UNIQUE",
            "sample_id": "s1",
            "uid": "c-1",
            "delta_rt_abs": 0.0,
            "interval_iou": 1.0,
            "match_cost": 0.0,
            "score_ai": 0.9,
            "score_signal": math.nan,
            "score_peak": 0.9,
            "score_source": "model",
            "ai_area": 200.0,
            "trad_area": 100.0,
        }])
        row = score_confidence(matches, ConfidenceConfig()).iloc[0]
        self.assertAlmostEqual(row.area_diff_pct, 100.0 / 150.0 * 100.0)
        self.assertAlmostEqual(row.conf_area_diff_pct, row.area_diff_pct)
        self.assertAlmostEqual(row.conf_area, 2.0 / 3.0)
        self.assertEqual(row.confidence_scope, "PAIRED_COMPARISON")
        self.assertAlmostEqual(row.final_confidence, row.comparison_confidence)

    def test_ai_only_peak_uses_nearest_traditional_reference(self):
        matches = pd.DataFrame([{
            "match_status": "AI_ONLY_EXTRA",
            "sample_id": "s1",
            "uid": "c-1",
            "ai_rt_peak": 1.20,
            "ai_area": 120.0,
            "nearest_trad_rt_peak": 1.00,
            "nearest_trad_area": 100.0,
            "nearest_trad_interval_iou": 0.0,
            "nearest_trad_match_cost": 0.50,
            "score_ai": math.nan,
            "score_signal": 0.74,
            "score_peak": 0.74,
            "score_source": "signal_rule",
            "extra_is_competitive": False,
        }])
        row = score_confidence(matches, ConfidenceConfig()).iloc[0]
        self.assertAlmostEqual(row.delta_rt_abs, 0.20)
        self.assertAlmostEqual(row.area_diff_abs, 20.0)
        self.assertAlmostEqual(row.area_diff_pct, 20.0 / 110.0 * 100.0)
        self.assertTrue(math.isfinite(row.comparison_confidence))
        self.assertLess(row.final_confidence, row.score_peak)
        self.assertEqual(row.confidence_scope, "NEAREST_REFERENCE_COMPARISON")
        self.assertEqual(row.comparison_reference_type, "NEAREST_UNASSIGNED")
        self.assertIn(row.alert_level, {"YELLOW", "RED"})

    def test_no_traditional_sample_is_not_given_a_fake_comparison(self):
        matches = pd.DataFrame([{
            "match_status": "NO_TRAD_SAMPLE",
            "sample_id": "s2",
            "uid": "c-1",
            "score_ai": 0.91,
            "score_signal": math.nan,
            "score_peak": 0.91,
            "score_source": "model",
        }])
        row = score_confidence(matches, ConfidenceConfig()).iloc[0]
        self.assertTrue(math.isnan(row.comparison_confidence))
        self.assertTrue(math.isnan(row.final_confidence))
        self.assertEqual(row.alert_level, "UNAVAILABLE")

    def test_unassigned_ai_peak_gets_nearest_traditional_diagnostics(self):
        trad = pd.DataFrame([
            {"trad_row_id": 0, "sample_id": "s1", "uid": "c-1",
             "trad_rt_peak": 1.0, "trad_rt_min": 0.9, "trad_rt_max": 1.1,
             "trad_area": 100.0}
        ])
        ai = pd.DataFrame([
            {"ai_row_id": 0, "sample_id": "s1", "uid": "c-1",
             "ai_rt_peak": 1.0, "ai_rt_min": 0.9, "ai_rt_max": 1.1,
             "ai_area": 100.0, "ai_peak_score": 0.9,
             "ai_model_score": 0.9, "ai_signal_score": math.nan,
             "ai_score_source": "model"},
            {"ai_row_id": 1, "sample_id": "s1", "uid": "c-1",
             "ai_rt_peak": 1.2, "ai_rt_min": 1.15, "ai_rt_max": 1.25,
             "ai_area": 70.0, "ai_peak_score": 0.7,
             "ai_model_score": math.nan, "ai_signal_score": 0.7,
             "ai_score_source": "signal_rule"},
        ])
        matched = match_peak_events(trad, ai, ConfidenceConfig())
        extra = matched[matched.match_status.eq("AI_ONLY_EXTRA")].iloc[0]
        self.assertEqual(extra.nearest_trad_row_id, 0)
        self.assertAlmostEqual(extra.extra_nearest_trad_rt_abs, 0.2)
        self.assertAlmostEqual(extra.nearest_trad_area, 100.0)


if __name__ == "__main__":
    unittest.main()
