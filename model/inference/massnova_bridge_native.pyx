# cython: language_level=3
"""Compiled import surface for the embedded CPython runtime.

The numerical and post-processing implementation remains in the normal
``inference.massnova`` modules so the CLI and DLL share one source of truth.
"""

from inference import massnova_bridge as _bridge


def initialize(model_path, config_json=None):
    return _bridge.initialize(model_path, config_json)


def shutdown():
    return _bridge.shutdown()


def process_items(items):
    return _bridge.process_items(items)


def process_json(input_json):
    return _bridge.process_json(input_json)


def is_gpu_enabled():
    return _bridge.is_gpu_enabled()
