"""Device-policy regressions without requiring a physical GPU."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from inference.onnx_window_predictor import OnnxWindowPredictor


class DeviceSelectionTests(unittest.TestCase):
    def fake(self, active=('CPUExecutionProvider',)):
        return SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                               preload_dlls=Mock(), InferenceSession=Mock(return_value=SimpleNamespace(get_providers=lambda: list(active))))

    def test_auto_uses_cuda_when_session_activates_it(self):
        ort = self.fake(('CUDAExecutionProvider', 'CPUExecutionProvider'))
        with patch.dict(sys.modules, onnxruntime=ort):
            self.assertTrue(OnnxWindowPredictor('model', use_gpu=0).is_gpu_enabled)
        ort.preload_dlls.assert_called_once_with()

    def test_auto_accepts_ort_cpu_fallback(self):
        with patch.dict(sys.modules, onnxruntime=self.fake()):
            self.assertFalse(OnnxWindowPredictor('model', use_gpu=0).is_gpu_enabled)

    def test_auto_retries_cpu_when_cuda_session_raises(self):
        ort = self.fake()
        ort.InferenceSession.side_effect = [RuntimeError('no driver'), SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider'])]
        with patch.dict(sys.modules, onnxruntime=ort):
            self.assertFalse(OnnxWindowPredictor('model', use_gpu=0).is_gpu_enabled)
        self.assertEqual(ort.InferenceSession.call_args.kwargs['providers'], ['CPUExecutionProvider'])

    def test_required_gpu_rejects_silent_cpu_fallback(self):
        with patch.dict(sys.modules, onnxruntime=self.fake()), self.assertRaisesRegex(RuntimeError, 'fell back'):
            OnnxWindowPredictor('model', use_gpu=1)

    def test_cpu_does_not_preload_cuda(self):
        ort = self.fake()
        with patch.dict(sys.modules, onnxruntime=ort):
            self.assertFalse(OnnxWindowPredictor('model', use_gpu=-1).is_gpu_enabled)
        ort.preload_dlls.assert_not_called()
        self.assertEqual(ort.InferenceSession.call_args.kwargs['providers'], ['CPUExecutionProvider'])


if __name__ == '__main__':
    unittest.main()
