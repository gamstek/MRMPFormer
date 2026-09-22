# -*- coding: utf-8 -*-
"""Stable module-level facade used by the embedded C/Cython bridge."""
from __future__ import annotations

import json
import threading

from inference.massnova_runtime import MassNovaArrayRuntime

_lock = threading.RLock()
_runtime = None


def initialize(model_path, config_json=None):
    """Create one cached runtime and load the ONNX session exactly once."""
    global _runtime
    config = json.loads(config_json) if config_json else {}
    with _lock:
        if _runtime is not None:
            raise RuntimeError("MassNova embedded runtime is already initialized")
        _runtime = MassNovaArrayRuntime(model_path, config)
    return True


def shutdown():
    global _runtime
    with _lock:
        _runtime = None


def process_items(items):
    with _lock:
        if _runtime is None:
            raise RuntimeError("MassNova embedded runtime is not initialized")
        return _runtime.process_items(items)


def process_json(input_json):
    with _lock:
        if _runtime is None:
            raise RuntimeError("MassNova embedded runtime is not initialized")
        return _runtime.process_json(input_json)


def is_gpu_enabled():
    with _lock:
        return bool(_runtime is not None and _runtime.is_gpu_enabled)
