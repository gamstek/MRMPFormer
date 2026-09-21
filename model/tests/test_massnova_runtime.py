import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from inference.massnova_runtime import MassNovaArrayRuntime
from inference.massnova import _scan_params_from_args
from inference.massnova_runtime import DEFAULT_RUNTIME_CONFIG
from inference.onnx_window_predictor import OnnxWindowPredictor


class _NoDetectionPredictor:
    def __call__(self, **kwargs):
        return []


class _CenteredDetectionPredictor:
    def __call__(self, *, images_path, threshold, verbose=False):
        del threshold, verbose
        return [{
            "image_path": str(path),
            "scores": np.array([0.91], dtype=np.float32),
            "boxes": np.array([[160.0, 0.0, 240.0, 300.0]], dtype=np.float32),
        } for path in sorted(Path(images_path).glob("*.jpeg"))]


class _FakeOnnxSession:
    def __init__(self):
        self.inputs = None

    def run(self, output_names, inputs):
        self.inputs = inputs
        batch = inputs["image"].shape[0]
        scores = np.tile(np.array([[0.5, 0.5001]], dtype=np.float32), (batch, 1))
        boxes = np.tile(
            np.array([[[0, 0, 10, 10], [20, 0, 30, 10]]], dtype=np.float32),
            (batch, 1, 1),
        )
        return scores, boxes


def _runtime_config(**overrides):
    config = {
        "threshold": 0.5,
        "smooth_sigma": 0.0,
        "min_chrom_points": 2,
        "min_max_intensity": 0.0,
        "scan_min_snr": 1.0,
        "scan_min_peak_span_points": 2,
    }
    config.update(overrides)
    return config


class MassNovaArrayRuntimeTests(unittest.TestCase):
    def test_deployment_config_matches_embedded_runtime_defaults(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "massnova.json"
        deployment = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(deployment["threshold"], DEFAULT_RUNTIME_CONFIG["threshold"])
        self.assertEqual(deployment["smooth_sigma"], DEFAULT_RUNTIME_CONFIG["smooth_sigma"])
        self.assertEqual(deployment["use_gpu"], DEFAULT_RUNTIME_CONFIG["use_gpu"])
        self.assertEqual(deployment["batch_size"], DEFAULT_RUNTIME_CONFIG["batch_size"])
        self.assertEqual(
            _scan_params_from_args(SimpleNamespace(**deployment)),
            _scan_params_from_args(SimpleNamespace(**DEFAULT_RUNTIME_CONFIG)),
        )

    def test_model_peak_is_ok_and_uses_model_score(self):
        rt = np.linspace(1.0, 3.0, 401)
        y = 10 + 10000 * np.exp(-((rt - 2.0) / 0.06) ** 2)
        runtime = MassNovaArrayRuntime(
            "fixture.onnx", _runtime_config(), predictor=_CenteredDetectionPredictor())

        item = runtime.process_items([{"uid": "model-peak", "x": rt, "y": y}])["items"][0]

        self.assertEqual(item["status"], "ok")
        self.assertEqual(len(item["peaks"]), 1)
        self.assertAlmostEqual(item["peaks"][0]["c"], 0.91, places=5)

    def test_signal_fallback_peaks_are_ok_and_returned_in_array(self):
        rt = np.linspace(1.0, 3.0, 401)
        y = (10 + 10000 * np.exp(-((rt - 1.5) / 0.06) ** 2)
             + 8000 * np.exp(-((rt - 2.5) / 0.06) ** 2))
        runtime = MassNovaArrayRuntime(
            "fixture.onnx", _runtime_config(), predictor=_NoDetectionPredictor())

        result = runtime.process_items([{"uid": "compound-1", "x": rt, "y": y}])

        item = result["items"][0]
        self.assertEqual(item["status"], "ok")
        self.assertEqual(len(item["peaks"]), 2)
        self.assertEqual(item["alerts"], [])
        self.assertTrue(all(0.0 < peak["c"] <= 1.0 for peak in item["peaks"]))

    def test_flat_channel_without_peak_is_alert(self):
        rt = np.linspace(1.0, 2.0, 101)
        runtime = MassNovaArrayRuntime(
            "fixture.onnx", _runtime_config(), predictor=_NoDetectionPredictor())

        item = runtime.process_items([{"uid": "flat", "x": rt, "y": np.ones_like(rt)}])["items"][0]

        self.assertEqual(item["status"], "alert")
        self.assertEqual(item["peaks"], [])
        self.assertEqual(item["alerts"][0]["code"], "NO_PEAK_FOUND")

    def test_qc_failure_remains_alert(self):
        runtime = MassNovaArrayRuntime(
            "fixture.onnx", _runtime_config(min_max_intensity=1000.0),
            predictor=_NoDetectionPredictor())
        rt = np.linspace(1.0, 2.0, 20)

        item = runtime.process_items([{"uid": "low", "x": rt, "y": np.ones_like(rt)}])["items"][0]

        self.assertEqual(item["status"], "alert")
        self.assertEqual(item["alerts"][0]["code"], "CHANNEL_LOW_INTENSITY")


class OnnxWindowPredictorTests(unittest.TestCase):
    def test_uses_raw_rgb_and_strict_score_threshold(self):
        session = _FakeOnnxSession()
        predictor = OnnxWindowPredictor("fixture.onnx", session=session, batch_size=8)
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((12, 16, 3), dtype=np.uint8)
            image[..., 0] = 255
            Image.fromarray(image).save(Path(directory) / "win_0000000.jpeg")
            result = predictor(images_path=directory, threshold=0.5)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["boxes"].shape, (1, 4))
        self.assertEqual(session.inputs["image"].shape, (1, 3, 12, 16))
        self.assertGreater(float(session.inputs["image"][0, 0].mean()), 200.0)
        self.assertEqual(session.inputs["img_size"].tolist(), [16.0, 12.0])


if __name__ == "__main__":
    unittest.main()
