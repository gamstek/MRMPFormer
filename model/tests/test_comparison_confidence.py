import math
import unittest

import pandas as pd

from postprocessing.comparison_confidence import (
    ConfidenceConfig,
    interval_iou,
    match_peak_events,
    score_confidence,
)


def _trad(rows):
    df = pd.DataFrame(rows)
    defaults = {
        "trad_sample_id": "s1", "trad_sample_name": "sample",
        "trad_uid": "c-1", "trad_ion_type": "quant",
        "trad_rt_expected": 5.0, "trad_height": 10.0,
        "trad_snr": 20.0, "trad_ion_ratio": 1.0,
    }
    for key, value in defaults.items():
        if key not in df:
            df[key] = value
    df["sample_id"] = df.trad_sample_id
    df["uid"] = df.trad_uid
    return df


def _ai(rows):
    df = pd.DataFrame(rows)
    defaults = {
        "ai_compound_name": "c-1", "ai_boundary_source": "model",
        "ai_validated": True, "ai_chrom_index": 1, "ai_peak_no": 1,
        "ai_q1": 100.0, "ai_apex_intensity": 100.0, "ai_snr": 20.0,
        "ai_n_points": 20,
    }
    for key, value in defaults.items():
        if key not in df:
            df[key] = value
    return df


class ComparisonConfidenceTests(unittest.TestCase):
    def test_interval_iou(self):
        self.assertAlmostEqual(interval_iou(1, 3, 2, 4), 1 / 3)
        self.assertEqual(interval_iou(1, 2, 3, 4), 0)

    def test_matching_is_exact_channel_and_one_to_one(self):
        trad = _trad([{
            "trad_row_id": 0, "trad_rt_peak": 5.0, "trad_rt_min": 4.8,
            "trad_rt_max": 5.2, "trad_area": 100.0,
        }])
        ai = _ai([
            {"ai_row_id": 0, "sample_id": "s1", "uid": "c-1", "ai_rt_peak": 5.02,
             "ai_rt_min": 4.81, "ai_rt_max": 5.19, "ai_area": 105.0, "ai_model_score": 0.95},
            {"ai_row_id": 1, "sample_id": "s1", "uid": "c-1", "ai_rt_peak": 5.30,
             "ai_rt_min": 5.25, "ai_rt_max": 5.40, "ai_area": 20.0, "ai_model_score": 0.99},
        ])
        result = match_peak_events(trad, ai, ConfidenceConfig())
        matched = result[result.match_status.str.startswith("MATCHED")]
        extra = result[result.match_status == "AI_ONLY_EXTRA"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(int(matched.iloc[0].ai_row_id), 0)
        self.assertEqual(len(extra), 1)
        self.assertEqual(int(extra.iloc[0].ai_row_id), 1)
        scored = score_confidence(result, ConfidenceConfig())
        self.assertEqual(scored[scored.match_status == "AI_ONLY_EXTRA"].iloc[0].alert_level,
                         "INFO")

    def test_confidence_penalizes_rt_area_and_missing_ai_score(self):
        cfg = ConfidenceConfig()
        base = pd.DataFrame([{
            "schema_version": "comparison_confidence_v1", "match_status": "MATCHED_UNIQUE",
            "sample_id": "s1", "uid": "c-1", "delta_rt_abs": 0.1,
            "interval_iou": 0.8, "match_cost": 0.1, "score_ai": 0.9,
            "ai_area": 200.0, "trad_area": 100.0,
        }])
        scored = score_confidence(base, cfg).iloc[0]
        self.assertAlmostEqual(scored.conf_rt, 0.5)
        self.assertTrue(math.isclose(scored.conf_area, 0.5, rel_tol=1e-6))
        missing = base.copy()
        missing.loc[0, "score_ai"] = math.nan
        missing_scored = score_confidence(missing, cfg).iloc[0]
        self.assertLess(missing_scored.comparison_confidence, scored.comparison_confidence)
        self.assertEqual(missing_scored.alert_level, "RED")
        self.assertEqual(missing_scored.review_status, "PENDING")


if __name__ == "__main__":
    unittest.main()
