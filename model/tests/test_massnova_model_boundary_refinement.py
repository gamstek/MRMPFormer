import unittest
from types import SimpleNamespace

import numpy as np

from inference.massnova import _scan_params_from_args, finalize_channel_peaks


class ModelBoundaryRefinementTests(unittest.TestCase):
    def setUp(self):
        self.rt = np.linspace(0.0, 2.0, 201)
        distance = np.abs(self.rt - 1.0)
        self.y = 10.0 + 1000.0 * np.maximum(0.0, 1.0 - distance / 0.20)
        self.apex_idx = int(np.argmax(self.y))
        self.params = _scan_params_from_args(SimpleNamespace())

    def _finalize(self, left, right):
        candidate = {
            "apex_idx": self.apex_idx,
            "rt_peak": float(self.rt[self.apex_idx]),
            "apex_intensity": float(self.y[self.apex_idx]),
            "rt_min": float(left),
            "rt_max": float(right),
            "model_score": 0.91,
            "validated": True,
        }
        peaks = finalize_channel_peaks(
            self.rt, self.y, [candidate], self.params, threshold=0.5)
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0]["boundary_source"], "model")
        self.assertAlmostEqual(peaks[0]["peak_score"], 0.91)
        return peaks[0]

    def test_model_box_above_baseline_expands_to_stable_baseline(self):
        peak = self._finalize(0.90, 1.10)

        self.assertAlmostEqual(peak["rt_min"], 0.80, delta=0.011)
        self.assertAlmostEqual(peak["rt_max"], 1.20, delta=0.011)

    def test_model_box_covering_surplus_baseline_is_shrunk(self):
        peak = self._finalize(0.50, 1.50)

        self.assertAlmostEqual(peak["rt_min"], 0.80, delta=0.011)
        self.assertAlmostEqual(peak["rt_max"], 1.20, delta=0.011)

    def test_model_box_can_expand_one_side_and_shrink_the_other(self):
        peak = self._finalize(0.90, 1.50)

        self.assertAlmostEqual(peak["rt_min"], 0.80, delta=0.011)
        self.assertAlmostEqual(peak["rt_max"], 1.20, delta=0.011)

    def test_model_box_already_at_baseline_remains_stable(self):
        peak = self._finalize(0.80, 1.20)

        self.assertAlmostEqual(peak["rt_min"], 0.80, delta=0.011)
        self.assertAlmostEqual(peak["rt_max"], 1.20, delta=0.011)


if __name__ == "__main__":
    unittest.main()
