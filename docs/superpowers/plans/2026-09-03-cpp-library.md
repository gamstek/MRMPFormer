# MRMPFormer C++ Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++20 shared library around `mrmpformerv2.onnx` with the new chromatogram JSON contract, asynchronous C API, item-level status/alerts, and an old-versus-new interface guide.

**Architecture:** A thin exported C ABI owns a process-global task service. Each worker validates JSON, performs signal QC, renders an RGB ROI, runs the two-input/three-output ONNX model, refines or falls back to signal-derived peak boundaries, and serializes a deterministic result. Internal modules remain independently testable; ONNX Runtime is isolated behind `OnnxInference`.

**Tech Stack:** C++20, CMake 3.20+, CTest, ONNX Runtime C++ API, nlohmann/json, stb image codec, Windows/Linux shared libraries.

**Spec:** `docs/superpowers/specs/2026-09-03-cpp-library-design.md`

## Global Constraints

- Target repository is MRMPFormer; create the library below `cpp/`.
- Runtime Python is forbidden; Python is used only to export and verify the ONNX model.
- The new JSON and C ABI intentionally do not preserve QuanFormer Lib v1.3.2 compatibility.
- `mrmpformerv2.onnx` consumes raw RGB `[0,255]` plus `(W,H)` and already performs normalization.
- Default signal QC is `min_chrom_points=10` and `min_max_intensity=1000` after smoothing.
- All filesystem construction uses `std::filesystem`; do not hard-code developer paths.
- Update `dev_log.md` after implementation and testing.

---

### Task 1: Build Skeleton and Public ABI

**Files:**
- Create: `cpp/CMakeLists.txt`
- Create: `cpp/include/mrmpformer.h`
- Create: `cpp/src/api.cpp`
- Create: `cpp/tests/test_header_abi.cpp`
- Create: `cpp/cmake/FindOnnxRuntime.cmake`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `QfError`, `QfTaskStatus`, `QfConfig`, `QfCompoundInput`, `QfCallback`, and all `qf_*` declarations.
- `qf_process_single(const QfCompoundInput*, int64_t*)` is the only array-based submission entry point.

- [ ] **Step 1: Write the ABI compile test**

Create a test that includes only `mrmpformer.h`, initializes `QfConfig` and `QfCompoundInput` with zero initialization, assigns every field, and binds every exported function to an exactly typed function pointer. Add x64 `static_assert` checks for structure sizes and `offsetof` values after compiling once to establish the declared layout.

```cpp
QfError (*submit)(const QfCompoundInput*, int64_t*) = &qf_process_single;
const char* (*version)(void) = &qf_version;
static_assert(std::is_standard_layout_v<QfConfig>);
static_assert(std::is_standard_layout_v<QfCompoundInput>);
```

- [ ] **Step 2: Configure and confirm the test fails**

Run:

```powershell
$mrmpOrtRoot = (Resolve-Path '..\..\QuanFormer_Lib_v1.3.2_python_aligned_20260612\QuanFormer_Lib_v1.3.2\libquanformer_dev\third_party\onnxruntime').Path
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON -DONNXRUNTIME_ROOT="$mrmpOrtRoot"
cmake --build cpp/build --config Release --target test_header_abi
```

Expected: configuration or compilation fails because the public header and target do not exist.

- [ ] **Step 3: Define the new public ABI and CMake targets**

Declare explicit enum values, export macros, config defaults documentation, callback ownership, and this input structure:

```c
typedef struct {
    const char* uid;
    const char* name;
    const char* channel;
    double mzq1;
    double mzq3;
    float smooth_sigma;
    const double* rt;
    const double* intensity;
    int32_t n_points;
} QfCompoundInput;
```

Define `mrmpformer` as a shared target, expose `cpp/include`, require C++20, enable `/utf-8` under MSVC, and register tests through CTest. `FindOnnxRuntime.cmake` must accept `ONNXRUNTIME_ROOT` and report missing include/library paths precisely.

- [ ] **Step 4: Build and run the ABI test**

Run:

```powershell
cmake --build cpp/build --config Release --target test_header_abi
ctest --test-dir cpp/build -C Release -R header_abi --output-on-failure
```

Expected: one test passes.

- [ ] **Step 5: Commit the ABI skeleton**

```powershell
git add cpp/CMakeLists.txt cpp/cmake/FindOnnxRuntime.cmake cpp/include/mrmpformer.h cpp/src/api.cpp cpp/tests/test_header_abi.cpp .gitignore
git commit -m "feat(cpp): define MRMPFormer C API"
```

### Task 2: New JSON Domain Model and Strict Validation

**Files:**
- Create: `cpp/src/json_protocol.h`
- Create: `cpp/src/json_protocol.cpp`
- Create: `cpp/tests/test_json_protocol.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Produces: `ParseResult parse_input_json(std::string_view)` and `std::string generate_output_json(const TaskResult&)`.
- Produces internal `CompoundData`, `PeakResult`, `Alert`, `CompoundResult`, and `TaskResult` types.

- [ ] **Step 1: Write failing parser tests**

Cover a valid item and individual rejection of missing `uid`, `channel`, `mzq1`, `mzq3`, `x`, or `y`; reject legacy-only `mz`; reject unequal arrays, fewer than two points, non-finite values, and non-increasing RT.

```cpp
auto parsed = parse_input_json(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1.0,1.1],"y":[2.0,3.0]}]})");
assert(parsed.ok);
assert(parsed.items[0].channel == "qual");
assert(parsed.items[0].mzq3 == 99.0);
```

- [ ] **Step 2: Run the protocol test and observe failure**

Run `ctest --test-dir cpp/build -C Release -R json_protocol --output-on-failure`.

Expected: target fails to compile because `json_protocol.h` is absent.

- [ ] **Step 3: Implement strict parsing and deterministic serialization**

Return a field-qualified error such as `items[2].mzq3 is required and must be numeric`. Preserve input item order. Serialize every item with `uid`, `status`, `peaks`, and `alerts`; serialize alert `detail` only when populated. Use two-space indentation.

- [ ] **Step 4: Add golden output assertions**

Assert all three shapes:

```json
{"uid":"ok","status":"ok","peaks":[{"a":1.0,"b":1.2,"c":0.95}],"alerts":[]}
{"uid":"review","status":"review","peaks":[{"a":2.0,"b":2.2,"c":0.60}],"alerts":[{"level":"review","code":"SIGNAL_FALLBACK"}]}
{"uid":"bad","status":"alert","peaks":[],"alerts":[{"level":"fail","code":"CHANNEL_LOW_INTENSITY"}]}
```

- [ ] **Step 5: Run and commit**

Run the protocol CTest and expect all cases to pass, then commit:

```powershell
git add cpp/src/json_protocol.* cpp/tests/test_json_protocol.cpp cpp/CMakeLists.txt
git commit -m "feat(cpp): add strict JSON protocol"
```

### Task 3: Signal QC, ROI Rendering, and Fallback

**Files:**
- Create: `cpp/src/signal_processing.h`
- Create: `cpp/src/signal_processing.cpp`
- Create: `cpp/src/roi_renderer.h`
- Create: `cpp/src/roi_renderer.cpp`
- Create: `cpp/tests/test_signal_processing.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Produces: `SignalQc inspect_signal(const CompoundData&, const QfConfig&)`.
- Produces: `RoiImage render_roi(const CompoundData&, float sigma)` with RGB bytes and exact RT bounds.
- Produces: `std::optional<PeakResult> find_signal_fallback(...)`.

- [ ] **Step 1: Write failing QC and fallback tests**

Use fixed synthetic arrays. Assert 9 points triggers `CHANNEL_LOW_INTENSITY`; 10 points with maximum 999.9 triggers it; 10 points with maximum 1000 passes. For `[0,1,4,9,4,1,0]`, assert fallback encloses the dominant apex, has `a < b`, and returns a bounded signal score in `[0,1]`.

- [ ] **Step 2: Run the signal test and observe failure**

Run `ctest --test-dir cpp/build -C Release -R signal_processing --output-on-failure`.

Expected: missing signal-processing symbols.

- [ ] **Step 3: Implement QC and deterministic fallback**

Smooth using a finite Gaussian kernel with nearest-edge padding when sigma is positive. QC uses the smoothed maximum. Fallback chooses the global maximum, walks left and right to the nearest local minima, uses array endpoints when no local minimum exists, and rejects zero-width intervals. Define the signal score as `clamp((apex-baseline)/max(apex, epsilon), 0, 1)` where baseline is the mean of the two boundary intensities.

- [ ] **Step 4: Implement ROI rendering**

Render a white-background RGB plot with fixed margins and a blue polyline, no text/font dependency, defaulting to 400×300. Return raw interleaved RGB bytes and the RT range used for X-coordinate mapping. Tests assert deterministic dimensions, exact byte count `width*height*3`, and unchanged input arrays.

- [ ] **Step 5: Run and commit**

Run the signal and ROI tests, expect all pass, then commit:

```powershell
git add cpp/src/signal_processing.* cpp/src/roi_renderer.* cpp/tests/test_signal_processing.cpp cpp/CMakeLists.txt
git commit -m "feat(cpp): add signal QC and ROI processing"
```

### Task 4: New ONNX Runtime Adapter

**Files:**
- Create: `cpp/src/onnx_inference.h`
- Create: `cpp/src/onnx_inference.cpp`
- Create: `cpp/tests/test_onnx_inference.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Consumes: `RoiImage` RGB data and dimensions.
- Produces: `bool load_model(path, config)`, `infer_batch(images, threshold)`, and `is_gpu_enabled()`.
- Produces `Detection {score, x1, y1, x2, y2, cx, cy, width, height}`.

- [ ] **Step 1: Write the failing metadata test**

Load `model/checkpoint/mrmpformerv2.onnx` and assert inputs `image`, `img_size` and outputs `scores`, `boxes_xyxy`, `boxes_norm`. Assert all are float tensors and validate their ranks.

- [ ] **Step 2: Run and observe failure**

Run `ctest --test-dir cpp/build -C Release -R onnx_inference --output-on-failure`.

Expected: missing adapter implementation.

- [ ] **Step 3: Implement session creation and contract validation**

Support CPU, automatic CUDA fallback, and forced CUDA failure. Create `image` as contiguous float NCHW from raw RGB values without division, normalization, or softmax. Supply one `img_size={width,height}` tensor per model invocation as required by the exported graph. Read scores and both box outputs by name, not ordinal position.

- [ ] **Step 4: Implement batch inference and box validation**

Filter `score < threshold`, non-finite scores, inverted boxes, and boxes with no image intersection. Clamp accepted pixel boxes to image bounds. Keep ONNX query order to make ties deterministic.

- [ ] **Step 5: Verify real CPU inference**

Run the metadata test plus a generated ROI through the checked-in ONNX. Assert three queries are returned before threshold filtering, accepted boxes are finite, and `boxes_xyxy` agrees with `boxes_norm` converted using image dimensions within `1e-3` pixels.

- [ ] **Step 6: Commit**

```powershell
git add cpp/src/onnx_inference.* cpp/tests/test_onnx_inference.cpp cpp/CMakeLists.txt
git commit -m "feat(cpp): integrate MRMPFormer ONNX runtime"
```

### Task 5: Task Manager and End-to-End C API

**Files:**
- Create: `cpp/src/task_manager.h`
- Create: `cpp/src/task_manager.cpp`
- Modify: `cpp/src/api.cpp`
- Create: `cpp/tests/test_api.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Consumes all modules from Tasks 2–4.
- Produces the complete exported behavior declared in `mrmpformer.h`.

- [ ] **Step 1: Write failing lifecycle and validation tests**

Assert calls before initialization return `QF_ERR_NOT_INITIALIZED`; null config/input/output pointers return `QF_ERR_INVALID_PARAM`; duplicate initialization returns `QF_ERR_ALREADY_INITIALIZED`; missing model returns `QF_ERR_FILE_NOT_FOUND`; shutdown succeeds exactly once.

- [ ] **Step 2: Run and observe failure**

Run `ctest --test-dir cpp/build -C Release -R api --output-on-failure`.

Expected: API stubs fail assertions.

- [ ] **Step 3: Implement configuration ownership and task state**

Deep-copy all configuration strings. Store tasks behind a mutex and condition variable, allocate monotonically increasing positive IDs, and model `pending/running/success/failed/cancelled`. Make `qf_get_error()` thread-local. Join worker threads during shutdown.

- [ ] **Step 4: Implement batch and single submission**

`qf_process()` reads and validates the complete JSON before queueing. `qf_process_single()` copies all caller-owned strings and arrays before returning, then feeds the same internal pipeline without creating a temporary input JSON file.

- [ ] **Step 5: Implement item processing and statuses**

For QC failure, emit `alert` with `CHANNEL_LOW_INTENSITY` and `{n_points,max_intensity}`. For accepted model boxes, map X pixels through the renderer's RT bounds, refine to local signal boundaries, emit `ok`, and leave alerts empty. If no usable model box remains, call `find_signal_fallback`, place its interval in `peaks`, emit `review`, and repeat `a/b/c` in `SIGNAL_FALLBACK`. If fallback also fails, emit `alert` with `NO_PEAK_FOUND`.

- [ ] **Step 6: Implement callbacks and result ownership**

Invoke callbacks outside internal locks. Keep result files valid until shutdown. Allocate `qf_get_result_json()` with `malloc` and release only through `qf_free()`. Return `QF_ERR_TIMEOUT` without cancelling the task.

- [ ] **Step 7: Run mixed end-to-end tests**

Submit a three-item JSON fixture covering a normal model result, a deterministic fallback signal, and low intensity. Assert item order, exact statuses/codes, progress reaches `1.0`, callback fires once, result-path JSON equals result-string JSON, and cancellation reaches a terminal state.

- [ ] **Step 8: Commit**

```powershell
git add cpp/src/api.cpp cpp/src/task_manager.* cpp/tests/test_api.cpp cpp/CMakeLists.txt
git commit -m "feat(cpp): implement asynchronous inference API"
```

### Task 6: C Examples and Interface Difference Document

**Files:**
- Create: `cpp/examples/batch_example.c`
- Create: `cpp/examples/single_example.c`
- Create: `docs/CPP_API_DIFFERENCES.md`
- Create: `cpp/README.md`
- Modify: `README.md`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- Documents and exercises the exact ABI delivered by Tasks 1 and 5.

- [ ] **Step 1: Add examples as build targets**

The batch example initializes CPU mode, submits a new-format JSON file, waits, retrieves JSON, frees it, and shuts down. The single example populates every `QfCompoundInput` field. Compile both as C sources and link them against the C++ shared library through CMake.

- [ ] **Step 2: Write the interface difference document**

Include a side-by-side table covering:

- Library/model/version and public header location.
- Old `mz` versus new `mzq1/mzq3/channel` input.
- Old positional `qf_process_single` versus new `QfCompoundInput*`.
- Old one-input normalized ONNX versus new `image + img_size` raw-pixel ONNX.
- Old `pred_logits + pred_boxes` versus new `scores + boxes_xyxy + boxes_norm`.
- Old item output `uid/peaks` versus new `uid/status/peaks/alerts`.
- Removed ABI guarantees and required migration actions.

Show complete old and new JSON examples and a concise C migration example. Explicitly state that `status/alerts` are produced by the C++ QC/post-processing layer, not the neural network.

- [ ] **Step 3: Document build and deployment commands**

Document `ONNXRUNTIME_ROOT`, CPU and CUDA CMake configuration, runtime DLL/SO placement, model location, test commands, callback threading, memory ownership, and the meaning of all errors and alerts.

- [ ] **Step 4: Build examples and validate documentation references**

Run:

```powershell
cmake --build cpp/build --config Release --target batch_example single_example
rg -n "mzq1|mzq3|SIGNAL_FALLBACK|CHANNEL_LOW_INTENSITY|scores|boxes_xyxy" cpp/README.md docs/CPP_API_DIFFERENCES.md README.md
```

Expected: both examples compile and every new contract term appears in the documentation.

- [ ] **Step 5: Commit**

```powershell
git add cpp/examples cpp/README.md docs/CPP_API_DIFFERENCES.md README.md cpp/CMakeLists.txt
git commit -m "docs(cpp): add API migration guide and examples"
```

### Task 7: Full Verification and Development Log

**Files:**
- Modify: `dev_log.md`
- Modify as failures require: files introduced in Tasks 1–6

**Interfaces:**
- Verifies the complete library and records the delivered result.

- [ ] **Step 1: Run the clean CPU build and complete CTest suite**

```powershell
$mrmpOrtRoot = (Resolve-Path '..\..\QuanFormer_Lib_v1.3.2_python_aligned_20260612\QuanFormer_Lib_v1.3.2\libquanformer_dev\third_party\onnxruntime').Path
cmake -S cpp -B cpp/build-clean -DBUILD_TESTS=ON -DONNXRUNTIME_ROOT="$mrmpOrtRoot"
cmake --build cpp/build-clean --config Release
ctest --test-dir cpp/build-clean -C Release --output-on-failure
```

Expected: configuration, build, and every test pass from a fresh directory.

- [ ] **Step 2: Run ONNX numerical verification in the project environment**

From `model/`, run externally in the configured `gamstekpeaking` environment:

```powershell
python -m tools.export_onnx --checkpoint checkpoint/mrmpformerv2.pth --out checkpoint/mrmpformerv2.onnx
```

Expected: Torch/ONNX checks pass at both exported dynamic image sizes with the script's stated tolerances.

- [ ] **Step 3: Audit exported symbols and dependencies**

On Windows use `dumpbin /exports mrmpformer.dll`; on Linux use `nm -D --defined-only libmrmpformer.so`. Confirm every declared `qf_*` symbol exists once and no test-only symbol is exported.

- [ ] **Step 4: Update the development log**

Under `### 2026-09-03`, add concise entries for C API generation, ONNX Runtime integration, tests, and documentation following the repository's required `<类型>(<作用域>): <简要描述>` format.

- [ ] **Step 5: Re-run verification after final edits**

Repeat the clean build, full CTest command, documentation `rg`, and `git diff --check`. Record actual test counts and any skipped CUDA validation in the handoff.

- [ ] **Step 6: Commit final verification changes**

```powershell
git add dev_log.md cpp docs/CPP_API_DIFFERENCES.md README.md
git commit -m "test(cpp): verify MRMPFormer library delivery"
```
