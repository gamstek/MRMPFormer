import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inference import massnova


class DuplicateWindowTests(unittest.TestCase):
    def test_same_channel_window_is_shared_and_box_assigned_once(self):
        rt = np.linspace(4.57, 6.06, 100)
        y = np.ones(100)
        peaks = {0: [dict(rt_peak=5.36), dict(rt_peak=5.47)]}
        observed = []

        def render(rt, y, lo, hi, path):
            Path(path).touch()

        def predict(**kwargs):
            paths = sorted(Path(kwargs['images_path']).glob('*.jpeg'))
            observed.extend(paths)
            return [dict(image_path=str(p), boxes=np.array([[0, 0, 1, 1]]),
                         scores=np.array([0.9])) for p in paths]

        with patch.object(massnova, 'render_roi_jpeg', side_effect=render), \
                patch.object(massnova, 'box_to_rt_range', return_value=(5.29, 5.48, 0, 0)):
            massnova.validate_with_model(peaks, {0: rt}, {0: y}, 'unused.onnx', predictor=predict)
        self.assertEqual(len(observed), 1)
        self.assertEqual(sum(p['validated'] for p in peaks[0]), 1)
        self.assertTrue(peaks[0][0]['validated'])

    def test_identical_bounds_keep_best_score_even_with_different_apexes(self):
        peaks = [dict(rt_min=5.29, rt_max=5.48, rt_peak=5.36, boundary_source='model', peak_score=.84),
                 dict(rt_min=5.2900005, rt_max=5.4800005, rt_peak=5.47, boundary_source='model', peak_score=.90),
                 dict(rt_min=5.30, rt_max=5.49, rt_peak=5.46, boundary_source='model', peak_score=.95)]
        result = massnova._dedup_identical_peak_bounds(peaks)
        self.assertEqual(result, [peaks[1], peaks[2]])

    def test_model_bounds_take_priority_over_signal_bounds(self):
        peaks = [dict(rt_min=1., rt_max=2., boundary_source='signal', peak_score=.99),
                 dict(rt_min=1., rt_max=2., boundary_source='model', peak_score=.81)]
        self.assertEqual(massnova._dedup_identical_peak_bounds(peaks), [peaks[1]])

    def test_finalization_removes_identical_bounds_despite_deep_valley(self):
        rt = np.linspace(0., 1., 11)
        y = np.array([0, 2, 100, 25, 4, 20, 95, 25, 2, 1, 0], dtype=float)
        candidates = [dict(apex_idx=i, rt_peak=float(rt[i]), apex_intensity=float(y[i]),
                           rt_min=.1, rt_max=.8, model_score=score, validated=True)
                      for i, score in ((2, .81), (6, .92))]
        params = massnova._scan_params_from_args(SimpleNamespace())
        peaks = massnova.finalize_channel_peaks(rt, y, candidates, params)
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0]['peak_score'], .92)
        self.assertEqual(peaks[0]['peak_no'], 1)


if __name__ == '__main__':
    unittest.main()
