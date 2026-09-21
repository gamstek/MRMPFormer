import importlib.util
import unittest
from pathlib import Path

import numpy as np

from inference.massnova_runtime import MassNovaArrayRuntime


MODEL = Path(__file__).resolve().parents[1] / "checkpoint" / "mrmpformerv2.onnx"


@unittest.skipUnless(importlib.util.find_spec("onnxruntime") and MODEL.is_file(),
                     "real ONNX Runtime/model integration is unavailable")
class MassNovaRuntimeOnnxIntegrationTests(unittest.TestCase):
    def test_real_onnx_returns_all_synthetic_peaks_as_ok(self):
        rt = np.linspace(1.0, 3.0, 401)
        intensity = (
            10
            + 10000 * np.exp(-((rt - 1.5) / 0.06) ** 2)
            + 8000 * np.exp(-((rt - 2.5) / 0.06) ** 2)
        )
        runtime = MassNovaArrayRuntime(MODEL, {
            "threshold": 0.5,
            "smooth_sigma": 0.8,
            "use_gpu": -1,
            "batch_size": 16,
            "min_chrom_points": 10,
            "min_max_intensity": 1000.0,
        })

        item = runtime.process_items([
            {"uid": "real-onnx-double", "x": rt, "y": intensity}
        ])["items"][0]

        self.assertEqual(item["status"], "ok")
        self.assertEqual(len(item["peaks"]), 2)
        self.assertTrue(all(0.0 < peak["c"] <= 1.0 for peak in item["peaks"]))
        self.assertLess(abs(item["peaks"][0]["a"] - 1.5), 0.3)
        self.assertLess(abs(item["peaks"][1]["a"] - 2.5), 0.3)


if __name__ == "__main__":
    unittest.main()
