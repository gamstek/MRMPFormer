import unittest
from types import SimpleNamespace

import numpy as np

from inference.massnova import finalize_channel_peaks, _scan_params_from_args
from inference.massnova_runtime import MassNovaArrayRuntime


class FinalThresholdTests(unittest.TestCase):
    def test_final_model_score_must_strictly_exceed_threshold(self):
        rt = np.linspace(0, 3, 301)
        y = 10 + 10000 * np.exp(-((rt - 1) / .06) ** 2)
        candidate = dict(apex_idx=100, rt_peak=1., apex_intensity=10010.,
                         rt_min=.9, rt_max=1.1, model_score=.5, validated=True)
        params = _scan_params_from_args(SimpleNamespace())
        for threshold, count in ((.499, 1), (.5, 0), (.8, 0)):
            with self.subTest(threshold=threshold):
                peaks = finalize_channel_peaks(rt, y, [candidate], params, threshold=threshold)
                self.assertEqual(len(peaks), count)

    def test_signal_fallback_cannot_bypass_requested_threshold(self):
        rt = np.linspace(1., 3., 401)
        item = dict(uid='signal', x=rt, y=10 + 10000 * np.exp(-((rt - 2) / .06) ** 2))
        predictor = lambda **kwargs: []
        runtime = MassNovaArrayRuntime('unused.onnx', dict(threshold=0., smooth_sigma=0.), predictor=predictor)
        self.assertTrue(runtime.process_items([item])['items'][0]['peaks'])
        runtime = MassNovaArrayRuntime('unused.onnx', dict(threshold=1., smooth_sigma=0.), predictor=predictor)
        result = runtime.process_items([item])['items'][0]
        self.assertEqual(result['peaks'], [])
        self.assertEqual(result['status'], 'alert')
        self.assertEqual(result['alerts'][0]['code'], 'NO_PEAK_FOUND')
