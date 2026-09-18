"""Regression coverage for final peaks being persisted with intrinsic scores."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from inference.massnova import run_massnova_on_mzml


class MassNovaOutputTests(unittest.TestCase):
    def test_model_and_residual_signal_peaks_reach_final_csv(self):
        rt = np.linspace(1.0, 3.0, 401)
        intensity = (10 + 10000 * np.exp(-((rt - 1.5) / 0.06) ** 2)
                     + 8000 * np.exp(-((rt - 2.5) / 0.06) ** 2))
        feature = dict(chrom_index=0, uid="compound-1", compound_name="compound",
                       q1=100.0, rt=rt, intensity=intensity)

        def validate(candidates, *args, **kwargs):
            self.assertEqual(len(candidates[0]), 2)
            peak = candidates[0][0]
            peak.update(model_score=0.95, validated=True, boundary_source="model",
                        rt_min=1.35, rt_max=1.65)

        args = SimpleNamespace(model="fixture.pth", threshold=0.6,
                               no_plots=True, plot=False, smooth_sigma=0)
        with tempfile.TemporaryDirectory() as out, \
                patch("inference.massnova.extract_full_xics", return_value=([feature], [])), \
                patch("inference.massnova.validate_with_model", side_effect=validate):
            result = run_massnova_on_mzml("sample.mzML", "sample", args, out)
            csv = pd.read_csv(Path(out) / "prediction_refined/sample/massnova_peaks.csv")
        self.assertEqual(result["n_peaks"], 2)
        self.assertEqual(len(result["peak_rows"]), 2)
        self.assertEqual(len(csv), 2)
        self.assertEqual(list(csv.score_source), ["model", "signal_rule"])
        self.assertTrue((csv.area > 0).all())
        self.assertTrue((csv.n_points > 0).all())
        self.assertTrue(np.isfinite(csv.peak_score).all())
        self.assertAlmostEqual(csv.iloc[0].model_score, 0.95)
        self.assertTrue(np.isnan(csv.iloc[1].model_score))
        self.assertGreater(csv.iloc[1].signal_score, 0)
        np.testing.assert_allclose(csv.rt_peak, [1.5, 2.5])


if __name__ == "__main__":
    unittest.main()
