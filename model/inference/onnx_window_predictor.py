# -*- coding: utf-8 -*-
"""Cached ONNX Runtime predictor for MassNova candidate-window images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class OnnxWindowPredictor:
    """Load the exported MRMPFormer ONNX once and reuse one inference session.

    The exported graph already contains ``/255``, ImageNet normalization,
    softmax and cxcywh-to-xyxy conversion.  Candidate enumeration, score
    thresholding, box-to-RT matching and signal fallback remain outside the
    graph and are intentionally handled by :mod:`inference.massnova`.
    """

    def __init__(self, model_path, *, use_gpu=0, batch_size=128, session=None):
        self.model_path = str(model_path)
        self.batch_size = max(1, int(batch_size))
        if session is not None:
            self.session = session
            providers = getattr(session, "get_providers", lambda: [])()
            self.is_gpu_enabled = "CUDAExecutionProvider" in providers
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required by the embedded MassNova runtime"
            ) from exc

        available = set(ort.get_available_providers())
        if int(use_gpu) == 1:
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDAExecutionProvider was requested but is unavailable")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif int(use_gpu) == 0 and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in providers:
                # Load compatible CUDA/cuDNN DLLs from the target machine's search paths.
                ort.preload_dlls()
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception:
            if int(use_gpu) != 0:
                raise
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.is_gpu_enabled = "CUDAExecutionProvider" in self.session.get_providers()
        if int(use_gpu) == 1 and not self.is_gpu_enabled:
            raise RuntimeError("CUDA was required but ONNX Runtime fell back to CPU")

    @staticmethod
    def _load_rgb(path):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        return np.transpose(rgb, (2, 0, 1))

    def __call__(self, *, images_path, threshold, verbose=False):
        paths = sorted(
            path for path in Path(images_path).iterdir()
            if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        results = []
        cutoff = float(threshold)
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start:start + self.batch_size]
            arrays = [self._load_rgb(path) for path in batch_paths]
            if not arrays:
                continue
            shapes = {(array.shape[1], array.shape[2]) for array in arrays}
            if len(shapes) != 1:
                raise ValueError("candidate images in one ONNX batch must share HxW")
            images = np.stack(arrays, axis=0).astype(np.float32, copy=False)
            height, width = images.shape[2], images.shape[3]
            sizes = np.asarray([width, height], dtype=np.float32)
            outputs = self.session.run(
                ["scores", "boxes_xyxy"],
                {"image": images, "img_size": sizes},
            )
            scores_batch = np.asarray(outputs[0])
            boxes_batch = np.asarray(outputs[1])
            for path, scores, boxes in zip(batch_paths, scores_batch, boxes_batch):
                keep = np.asarray(scores) > cutoff  # match predict_utils.py exactly
                if verbose:
                    maximum = float(np.max(scores)) if np.size(scores) else float("nan")
                    print(f"[onnx] {path.name}: max={maximum:.6f}, kept={int(np.sum(keep))}")
                if np.any(keep):
                    results.append({
                        "image_path": str(path),
                        "scores": np.asarray(scores)[keep].astype(np.float32, copy=False),
                        "boxes": np.asarray(boxes)[keep].astype(np.float32, copy=False),
                    })
        return results
