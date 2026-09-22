# Repository Guidelines

## Project Structure & Module Organization

MRMPFormer detects and quantifies LC-MS chromatographic peaks. Current development focuses on targeted, centroided MRM data.

- `model/`: preprocessing, model architectures, training, inference, postprocessing, and evaluation tools. JSON configurations live in `model/configs/`; deployment models live in `model/checkpoint/`.
- `desktop/`: PySide6 application, with UI pages in `pages/`, background tasks in `workers/`, and resources in `assets/`.
- `cpp/`: C++20 inference library; public C ABI in `include/mrmpformer.h`, implementation in `src/`, C examples in `examples/`, and tests in `tests/`.
- `converters/`: vendor-format conversion utilities. `docs/` and `dev_log.md` contain architecture notes and development history.

## Build, Test, and Development Commands

Use Python 3.11 in the repository-root `.venv`, created with uv. Run these from the repository root unless noted:

```powershell
uv venv --python 3.11 --seed .venv
uv pip install -r cpp/requirements-build.txt
.\.venv\Scripts\Activate.ps1
python model/tools/build_massnova_bridge.py
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON
cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release --output-on-failure
```

The C++ build requires CMake 3.20+ and CPython development files. Install `cpp/requirements-build.txt` for the build environment; packaging derives runtime dependencies from `cpp/requirements-runtime.txt`. Intermediate outputs belong in `cpp/build/` and `model/build/`. Run `bash build.sh` in Git Bash to produce the standalone integration package in `build/windows/`.

From `desktop/`, run `python main.py` to launch the app. From `model/`, run:

```powershell
python -m unittest discover -s tests
python -m train --config configs/mrmpformer_special_v2.json
python -m inference.cli --help
```

These run Python tests, start configured training, and show inference options.

## Coding Style & Naming Conventions

Use four-space indentation and preserve surrounding style. Python modules and functions use `snake_case`; classes use `PascalCase`. C++ follows similar naming, with public C functions prefixed `qf_` and types such as `QfConfig`. Preserve ABI compatibility and documented ownership rules. No repository-wide formatter or linter configuration was found; avoid unrelated formatting changes.

## Testing Guidelines

Python tests use `unittest` in `model/tests/test_*.py`. C++ uses assertion-based `test_*.cpp` executables registered with CTest in `cpp/CMakeLists.txt`. Add regression coverage for changed behavior, especially peak boundaries, thresholds, and API ownership. ONNX integration tests require `model/checkpoint/mrmpformerv2.onnx`. No numerical coverage threshold is configured.

Cross-language tests live entirely in root `tests/`. Import 256 real inputs with `uv run tests/import_inputs.py <validation-package-directory>`, then run `uv run tests/run_parity.py` after building the Windows package and Cython extensions. It builds its own native driver and compares source Python, compiled Cython, and the C DLL using the saved inputs. Read `tests/results/report.html` for complete results. Packaging runs this suite automatically. Real input JSON files stay local and are not committed.

## Commit & Pull Request Guidelines

Follow recent scoped messages: `fix(cpp): ...`, `feat(model): ...`, or `docs(cpp): ...`. Keep commits focused. PRs should explain the behavior change, link relevant issues, and report validation commands and results. Include screenshots for UI changes and model/configuration details for inference changes. Keep raw datasets, generated outputs, and local build artifacts out of commits.
