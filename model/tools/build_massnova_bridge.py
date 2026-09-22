# -*- coding: utf-8 -*-
"""Build the Cython modules imported by mrmpformer.dll.

Run from the repository root inside the release-build environment:
    python model/tools/build_massnova_bridge.py

All generated files are written below ``model/build``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise SystemExit("Cython is required to build massnova_bridge_native") from exc


MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_ROOT.parent
BUILD_ROOT = MODEL_ROOT / "build"
PYTHON_BUILD_ROOT = BUILD_ROOT / "python"
GENERATED_ROOT = BUILD_ROOT / "generated"
TEMP_ROOT = BUILD_ROOT / "temp"
EXTENSION_ROOT = PYTHON_BUILD_ROOT / "inference"


def build_layout() -> dict[str, str]:
    return {
        "generated": str(GENERATED_ROOT),
        "temp": str(TEMP_ROOT),
        "python": str(PYTHON_BUILD_ROOT),
        "extensions": str(EXTENSION_ROOT),
    }


def main() -> None:
    if sys.argv[1:] == ["--print-layout"]:
        print(json.dumps(build_layout()))
        return
    if len(sys.argv) != 1:
        raise SystemExit("usage: python model/tools/build_massnova_bridge.py")

    os.chdir(MODEL_ROOT)
    EXTENSION_ROOT.mkdir(parents=True, exist_ok=True)
    (EXTENSION_ROOT / "__init__.py").write_text(
        "from pkgutil import extend_path\n"
        "__path__ = extend_path(__path__, __name__)\n",
        encoding="utf-8",
    )

    extensions = cythonize(
        [
            Extension("inference.massnova_bridge_native", ["inference/massnova_bridge_native.pyx"]),
            Extension("inference.massnova_runtime", ["inference/massnova_runtime.py"]),
            Extension("inference.massnova", ["inference/massnova.py"]),
            Extension("inference.onnx_window_predictor", ["inference/onnx_window_predictor.py"]),
        ],
        compiler_directives={"language_level": "3"},
        build_dir=str(GENERATED_ROOT),
    )

    setup(
        name="mrmpformer-massnova-bridge",
        ext_modules=extensions,
        script_args=[
            "build_ext",
            "--build-lib", str(PYTHON_BUILD_ROOT),
            "--build-temp", str(TEMP_ROOT),
        ],
    )


if __name__ == "__main__":
    main()
