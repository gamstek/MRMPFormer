# MRMPFormer C API

`mrmpformer` is a C++20 shared library with a stable C ABI. Its public header is
[`include/mrmpformer.h`](include/mrmpformer.h). The DLL embeds one CPython
runtime, imports the compiled MassNova bridge once, and keeps one ONNX session
alive from `qf_init` through `qf_shutdown`. The C API and result JSON schema did
not change when the implementation moved to the Python MassNova core.

`QfConfig.threshold` applies both to model detections and to final returned peak
scores: every returned `c` must be strictly greater than the threshold, including
signal fallback peaks. A model peak uses model confidence; a signal peak uses
its signal-rule score. If filtering removes all peaks, the channel returns
`alert` with `NO_PEAK_FOUND`.

For incompatible QuanFormer callers, read [the API migration guide](../docs/CPP_API_DIFFERENCES.md)
before recompiling.

## Build

For a complete Windows integration package, run from the repository root:

```bash
bash build.sh
```

Run in Git Bash on Windows. The root `build.sh` is a short launcher for
`cpp/tools/build_windows_package.py` using `.venv/Scripts/python.exe`.
Run `bash build.sh --install-deps`
to create the environment with uv if missing and install dependencies.
Use `--skip-build` to package existing build outputs and rerun integration validation.

Give the software team only `build/windows/`; see
[Windows integration instructions](../docs/WINDOWS_INTEGRATION.md).
Intermediate C++ outputs stay in `cpp/build/`; Cython outputs stay in
`model/build/` (`generated/`, `temp/`, and `python/inference/*.pyd`).
The package command builds both, runs CTest, bundles the private runtime,
then validates a relocated copy using the real C DLL with developer paths removed.
It also runs 256 Python/C inference parity cases; any failure prevents publication
of the new package.

### Python/C inference comparison

After building the package, run from the repository root:

```powershell
uv run tests/run_parity.py
```

The suite contains 256 real MRM chromatograms from four stratified validation sets
in test3_cpp_validation_1d4ba0f, preserving original x/y arrays. Each case compares source Python,
the compiled Cython modules in `model/build/python`, and the C DLL. Both batch
and single calls are compared pairwise (1536 comparisons). The native
driver runs in a separate process using the packaged DLL, Cython modules and
private Python runtime. The reference explicitly requires source `.py` modules;
a separate Python process verifies it loads all four compiled `.pyd` modules
from `model/build/python` when running the third implementation.

Assertions check UID, status, alerts, peak count/order, and every `a`/`b`/`c` value.
Numeric tolerances are `atol=1e-12`, `rtol=1e-10`; NaN and infinity fail.
Python batch calls match C batch calls, and Python single calls match C single
calls: different ONNX batch shapes can produce small float32 score differences.
At least 50 cases must produce peaks, preventing an empty-output pass.

The 256 editable inputs are saved in `tests/inputs/`, with an index in its README.
The runner reads these files without regenerating them and builds its native driver
independently in `tests/build/`. Source files live in root `tests/`.
Both reference modes, C results, process logs and `report.json` are saved
under `tests/results/`; open `tests/results/README.md` for a per-case table and links.
See [the test guide](../tests/README.md). Reports and the package manifest stay in
`tests/results/`; development and test artifacts are excluded from `build/windows/`.
Use `--package`, `--runner`, and `--output` to select other paths. These tests
verify CPU implementation parity on synthetic data, not accuracy on labeled
LC-MS samples or GPU equivalence.

Use a release Python environment containing NumPy, SciPy, Pillow, Matplotlib,
pandas and ONNX Runtime. Cython and a C/C++ compiler are build-time
requirements. `pyopenms` is not required by the DLL array-input path.

```powershell
python -m pip install -r cpp/requirements-build.txt
```

Build the compiled Python core and DLL (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python model/tools/build_massnova_bridge.py
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON -DPython3_EXECUTABLE="$((Get-Command python).Source)"
cmake --build cpp/build --config Release
```

The single package bundles ONNX Runtime with both CPU and CUDA providers,
obtained via `onnxruntime-gpu` on the build machine. End users do not need pip.
GPU acceleration uses the target machine's compatible NVIDIA driver and CUDA 12/cuDNN 9
DLLs (available on its DLL search path). Otherwise it falls back to CPU. Set
`QfConfig.use_gpu = 0` to try CUDA then fall back to CPU, `1` to require CUDA,
or `-1` to force CPU. The default is `0`; `qf_is_gpu_enabled()` reports the
actual selection. Device checks are recorded in `tests/results/device_verification.json`.

Linux example:

```bash
python model/tools/build_massnova_bridge.py
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON -DPython3_EXECUTABLE="$(command -v python)"
cmake --build cpp/build --config Release
ctest --test-dir cpp/build --output-on-failure
```

The pure-C samples are normal build targets:

```powershell
cmake --build cpp/build --config Release --target batch_example single_example
ctest --test-dir cpp/build -C Release --output-on-failure
```

`batch_example.c` submits a JSON file; `single_example.c` sets every
`QfCompoundInput` member. The batch example defaults to automatic device selection;
its optional third argument is `-1` (CPU), `0` (automatic), or `1` (required GPU).

## Deployment

The end user does not need a separately installed Python environment. Ship a
private runtime beside the DLL. The package command creates:

```text
build/windows/
  mrmpformer.dll
  mrmpformer.h
  mrmpformer.lib
  python311.dll
  python311.zip
  python311._pth             # private paths relative to python311.dll
  mrmpformerv2.onnx
  README_INTEGRATION.md
  python/
    DLLs/                      # standard-library native extensions
    inference/                 # compiled .pyd modules + two_round_detection.py
    utils/                     # MassNova runtime helper modules
    preprocessing/             # masked_roi_generator.py used by boundary helpers
    Lib/site-packages/         # numpy/scipy/matplotlib/Pillow/onnxruntime, etc.
```

The DLL adds its own directory, `python/`, and `python/Lib/site-packages/` to
`sys.path`. `MRMPFORMER_PYTHON_PATH` is available as a development override.
Pass the deployed ONNX path through `QfConfig.model_path` as before.
Development CTest configures the selected interpreter automatically. The
packaged `python311._pth` removes the need for developer environment variables.
The array-input runtime does not import `pyopenms`; it is only needed by the
separate Python mzML input front end.

## C API lifecycle and ownership

1. Zero/initialize `QfConfig` with `qf_default_config`, set `model_path`, then
   call `qf_init` once per active lifecycle. Only one initialization may be
   active; reinitialization after a successful `qf_shutdown` is supported.
2. Submit `qf_process(json_path, &task_id)` or
   `qf_process_single(&input, &task_id)`, then wait/query/cancel by task ID.
3. On successful completion, use `qf_get_result_json` or `qf_get_result_path`.
4. Release every non-NULL string returned by `qf_get_result_json` with
   `qf_free`, then call `qf_shutdown` outside callbacks.

The library copies configuration strings during `qf_init`, and copies every
string and array in `QfCompoundInput` before `qf_process_single` returns.
`qf_get_result_path` copies its path into the caller buffer; the result file
remains on disk unless the caller removes it. `qf_get_error` is thread-local
and describes the most recent failing call on that thread.

`qf_set_callback` installs one process-global callback.  It runs synchronously
when a task becomes terminal—normally on a worker, but a caller that cancels a
pending task can also trigger it. Keep callbacks short, synchronize shared
state yourself, and do not call `qf_shutdown` from one. A callback may inspect
the terminal task (for example with `qf_query`); its `result_json_path`
argument is a borrowed library pointer valid until `qf_shutdown`. Copy it if
it is needed after shutdown.

## JSON result interpretation

Each item has `uid`, `status`, `peaks`, and `alerts`. `status="ok"` means the
shared Python pipeline produced at least one accepted final peak. This includes
model peaks and signal-rule fallback peaks. Every accepted peak is returned in
`peaks[]`; `c` is the model softmax score for a model peak or the auditable
SNR/point-support score for a signal peak. `alert` means QC failed or no final
peak survived. `CHANNEL_LOW_INTENSITY` and `NO_PEAK_FOUND` retain their prior
meanings. Python assembles this JSON and C++ writes/returns it unchanged.

The ONNX graph contains image scaling, ImageNet normalization, MRMPFormer,
softmax and box conversion only. Candidate enumeration, thresholding,
box-to-RT assignment, signal refinement, SNR/area gating, deduplication,
scoring, JSON assembly and `status` are part of the compiled Python core.

The returned per-peak `c` is the peak's own score. The DLL does not calculate
the business comparison confidence or warning level based on traditional-peak
RT, area and paired-transition support. Those values require the traditional
software result and remain the backend's responsibility.

## Errors

Every non-`QF_OK` call should be followed immediately by `qf_get_error` on the
same thread. `QF_ERR_INVALID_PARAM` is a bad argument/configuration or an
operation prohibited in its current context; `QF_ERR_NOT_INITIALIZED` and
`QF_ERR_ALREADY_INITIALIZED` describe lifecycle misuse; `QF_ERR_FILE_NOT_FOUND`,
`QF_ERR_FILE_READ`, and `QF_ERR_FILE_WRITE` describe filesystem failures;
`QF_ERR_JSON_PARSE` rejects the whole submitted JSON schema; `QF_ERR_MODEL_LOAD`
means ONNX Runtime could not load/validate the model or required provider;
`QF_ERR_TASK_NOT_FOUND`, `QF_ERR_TASK_CANCELLED`, and `QF_ERR_TIMEOUT` describe
task lookup/wait/result state; `QF_ERR_NO_WORKER` means task workers could not
start; `QF_ERR_INTERNAL` is an unexpected library failure.  The stable numeric
values are defined in the public header.
