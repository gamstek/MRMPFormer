# -*- coding: utf-8 -*-
"""Build the Cython module imported by mrmpformer.dll.

Run from ``model/`` inside the release-build environment:
    python tools/build_massnova_bridge.py build_ext --inplace
"""
import os

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise SystemExit("Cython is required to build massnova_bridge_native") from exc


extensions = cythonize(
    [
        # The native facade is the stable module imported by mrmpformer.dll.
        Extension("inference.massnova_bridge_native", ["inference/massnova_bridge_native.pyx"]),
        # Compile the array runtime and shared MassNova core as extension
        # modules too; the release package can omit their .py sources.
        Extension("inference.massnova_runtime", ["inference/massnova_runtime.py"]),
        Extension("inference.massnova", ["inference/massnova.py"]),
        Extension("inference.onnx_window_predictor", ["inference/onnx_window_predictor.py"]),
    ],
    compiler_directives={"language_level": "3"},
    build_dir=os.environ.get("MRMPFORMER_CYTHON_BUILD_DIR", "build/cython_generated"),
)

setup(name="mrmpformer-massnova-bridge", ext_modules=extensions)
