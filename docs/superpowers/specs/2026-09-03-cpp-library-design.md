# MRMPFormer C++ Library Design

## Goal

Add a self-contained C++20 shared library to MRMPFormer, following the asynchronous task API pattern of QuanFormer Lib while intentionally adopting the new, incompatible JSON contract and `mrmpformerv2.onnx` runtime interface.

The library accepts chromatogram arrays, renders ROI images, invokes ONNX Runtime, refines detected peak boundaries, and returns structured JSON. Python is not required at runtime.

## Layout

Create `cpp/` with:

- `include/mrmpformer.h`: public C ABI.
- `src/`: API, task manager, JSON parser, ROI renderer, ONNX Runtime adapter, signal fallback, peak post-processing, and alert serialization.
- `tests/`: parser/serializer, ABI, ONNX smoke, API lifecycle, and end-to-end tests.
- `examples/`: C callers for JSON batch and single-item processing.
- `third_party/`: documented locations for ONNX Runtime, nlohmann/json, and stb headers; do not duplicate model weights.
- `CMakeLists.txt` and `README.md`: CPU/CUDA build and deployment instructions.

The model path remains configurable and defaults are never tied to a developer machine.

## Public C API

The API remains an asynchronous, process-global library with init, shutdown, submit, wait, query, cancel, callback, result retrieval, error, version, and GPU-status functions. Compatibility with the old QuanFormer ABI is not required.

Batch submission continues to use `qf_process(const char*, int64_t*)`. Single submission changes to:

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

QF_API QfError qf_process_single(
    const QfCompoundInput* input,
    int64_t* out_task_id);
```

`QfConfig` gains `min_chrom_points` (default 10) and `min_max_intensity` (default 1000). Version and ABI tests pin all public enum values, sizes, and field offsets.

## Input Contract

Each `items[]` entry requires `uid`, `channel`, `mzq1`, `mzq3`, `x`, and `y`; `name` and `smooth_sigma` are optional. `x` and `y` must be finite, equally sized arrays with strictly increasing `x`. `channel` must be non-empty. The parser rejects the complete task on a malformed required field rather than silently skipping entries. The legacy `mz` field is not accepted.

## ONNX Runtime Contract

Use `model/checkpoint/mrmpformerv2.onnx` as the reference model. The adapter supplies:

- `image`: float32 `[B,3,H,W]`, RGB values in `[0,255]`; normalization is embedded in ONNX.
- `img_size`: float32 `[2]`, `(W,H)`.

It reads:

- `scores`: float32 `[B,Q]`.
- `boxes_xyxy`: float32 `[B,Q,4]` in pixels.
- `boxes_norm`: float32 `[B,Q,4]`, used only for diagnostics.

The adapter discovers dimensions and validates tensor names/types at initialization. C++ must not apply ImageNet normalization or softmax again. ROI images may retain the established 400×300 layout because the model exports dynamic height and width.

## Processing and Output

For each valid channel:

1. Apply optional Gaussian smoothing and render the ROI.
2. Reject fewer than `min_chrom_points` or maximum smoothed intensity below `min_max_intensity`.
3. Run batched ONNX inference and retain boxes meeting `threshold`.
4. Map pixel X coordinates to retention time, refine boundaries against the original signal, and retain deterministic peak order.
5. If the model returns no usable box but the signal passes QC, construct one fallback interval around the dominant local maximum using neighboring valleys.

Every output item contains `uid`, `status`, `peaks`, and `alerts`:

- Model-derived peaks: `status="ok"`, empty alerts.
- Signal fallback: write the fallback interval to `peaks`, set `status="review"`, and add `SIGNAL_FALLBACK`. The same interval and its signal-derived confidence are repeated in the alert for auditability; the confidence must never be described as model confidence.
- Low intensity or too few points: `status="alert"`, `CHANNEL_LOW_INTENSITY`, with `n_points` and `max_intensity` detail.
- Other per-item failures: `status="alert"` with a stable machine-readable code.

One bad item does not fail the entire task after schema validation. Task status is `success` when processing completed and item-level outcomes are available.

## Testing and Delivery

Parser tests cover required fields, invalid arrays, and optional values. ONNX tests verify exact tensor metadata and run CPU inference. Golden serializer tests pin all three item statuses. End-to-end tests cover mixed `ok`, `review`, and `alert` items, cancellation, callbacks, memory ownership, and repeated initialization errors. CUDA verification is conditional on provider availability.

The implementation will update root documentation and `dev_log.md`. Generated build products are excluded from Git; the existing ONNX remains managed by Git LFS.
