# QuanFormer-to-MRMPFormer C API migration

MRMPFormer 0.1.0 is a new, incompatible C ABI over a C++20 implementation.
Use [`cpp/include/mrmpformer.h`](../cpp/include/mrmpformer.h) as the only
source for declarations.  It uses the checked-in
`model/checkpoint/mrmpformerv2.onnx`; do not combine the new header with an old
QuanFormer binary or model.

## Contract changes

| Area | Old QuanFormer contract | New MRMPFormer contract | Required migration |
|---|---|---|---|
| Library/model/version | QuanFormer library and its model; old public header `quanformer.h` | MRMPFormer 0.1.0, `mrmpformerv2.onnx`, public header `cpp/include/mrmpformer.h` | Link only the matching MRMPFormer library/header/model triplet. Recompile; binary compatibility is not provided. |
| Compound identity | One `mz` value; channel identity was absent | Required non-empty `channel`, plus required finite `mzq1` and `mzq3` | Split precursor/product masses and supply a stable transition channel string. |
| Single submission | `qf_process_single(const double* rt, const double* intensity, int32_t n_points, double mz, const char* uid, int64_t* out_task_id)` | `qf_process_single(const QfCompoundInput* input, int64_t* out_task_id)` | Populate all `QfCompoundInput` fields, including `name`, `channel`, `mzq1`, `mzq3`, and `smooth_sigma`. |
| Batch JSON | Legacy item fields including `mz` | Each item requires `uid`, `channel`, `mzq1`, `mzq3`, `x`, and `y`; `name`/`smooth_sigma` are optional | Rewrite JSON and reject `mz`; arrays must be finite, equally sized, at least two points, and strictly increasing in `x`. |
| ONNX input | One normalized image input | `image` float32 `[B,3,H,W]` with raw RGB values `[0,255]`, plus `img_size` float32 `[2]` as `(W,H)` | Pass both tensors. Do not ImageNet-normalize or divide pixels again; normalization is in ONNX. |
| ONNX output | `pred_logits` and `pred_boxes` | `scores` `[B,Q]`, `boxes_xyxy` `[B,Q,4]` pixels, `boxes_norm` `[B,Q,4]` normalized `(cx,cy,w,h)` | Read named outputs, filter `scores`, use pixel boxes for RT mapping, and reserve `boxes_norm` for diagnostics. Do not apply softmax again. |
| Item result | `uid` and `peaks` | `uid`, `status`, `peaks`, `alerts` | Consume status and machine-readable alerts; preserve results even for item-level QC failures. |
| ABI promises | Positional single-call signature and old struct/layout compatibility | No old ABI/layout/function compatibility guarantee | Rebuild all clients, replace copied old headers, and validate the migration examples before deployment. |

`status` and `alerts` are produced by the **C++ QC/post-processing layer, not
by the neural network**.  In particular, a `SIGNAL_FALLBACK` confidence is
signal-derived and must never be presented as model confidence.

## Complete JSON migration

Old input (not accepted by MRMPFormer):

```json
{
  "items": [
    {
      "uid": "caffeine-quant",
      "mz": 195.0877,
      "x": [0.0, 0.1, 0.2, 0.3],
      "y": [100.0, 1200.0, 6000.0, 300.0]
    }
  ]
}
```

New input:

```json
{
  "items": [
    {
      "uid": "caffeine-quant",
      "name": "Caffeine",
      "channel": "195.0877>138.0550",
      "mzq1": 195.0877,
      "mzq3": 138.0550,
      "smooth_sigma": 0.0,
      "x": [0.0, 0.1, 0.2, 0.3],
      "y": [100.0, 1200.0, 6000.0, 300.0]
    }
  ]
}
```

New result examples:

```json
{
  "items": [
    {"uid":"caffeine-quant","status":"ok","peaks":[{"a":0.1,"b":0.3,"c":0.998}],"alerts":[]},
    {"uid":"fallback","status":"review","peaks":[{"a":1.0,"b":1.3,"c":0.71}],"alerts":[{"level":"review","code":"SIGNAL_FALLBACK","detail":{"a":1.0,"b":1.3,"c":0.71}}]},
    {"uid":"weak","status":"alert","peaks":[],"alerts":[{"level":"fail","code":"CHANNEL_LOW_INTENSITY","detail":{"n_points":8.0,"max_intensity":99.0}}]}
  ]
}
```

## Concise C single-item migration

```c
/* Old: qf_process_single(rt, intensity, n_points, mz, uid, &task_id); */
QfCompoundInput input = {
    .uid = "caffeine-quant",
    .name = "Caffeine",
    .channel = "195.0877>138.0550",
    .mzq1 = 195.0877,
    .mzq3 = 138.0550,
    .smooth_sigma = 0.0f,
    .rt = rt,
    .intensity = intensity,
    .n_points = n_points,
};
QfError error = qf_process_single(&input, &task_id);
```

The library copies strings and arrays before this call returns. It also copies
configuration strings during `qf_init`. In contrast, `qf_get_result_json`
allocates a string: release it with `qf_free`, never `free`. `qf_get_result_path`
copies its path into the caller buffer, and the result file remains on disk
unless the caller removes it. The callback's `const char* result_json_path` is
borrowed from the library and valid until `qf_shutdown`; copy it before
shutdown if it must be retained.

## Statuses, alerts, and errors

| Result field | Meaning and client action |
|---|---|
| `status="ok"` | Accepted model-derived peak(s); inspect `peaks[].a/b/c`. |
| `status="review"` / `SIGNAL_FALLBACK` | No usable model box; inspect/approve the signal fallback. Alert detail repeats `a`, `b`, `c`. |
| `status="alert"` / `CHANNEL_LOW_INTENSITY` | QC rejected too few points or a low smoothed maximum. Detail supplies `n_points` and `max_intensity`; correct input/QC settings. |
| `status="alert"` / `NO_PEAK_FOUND` | QC passed but neither model nor fallback produced a peak; review the chromatogram. |

API errors are returned as `QfError`; call `qf_get_error` immediately on the
same thread for detail. `QF_ERR_INVALID_PARAM` means invalid arguments/config
or an invalid call context; `QF_ERR_NOT_INITIALIZED`/`QF_ERR_ALREADY_INITIALIZED`
are lifecycle errors; `QF_ERR_FILE_NOT_FOUND`, `QF_ERR_FILE_READ`, and
`QF_ERR_FILE_WRITE` are filesystem errors; `QF_ERR_JSON_PARSE` rejects the
batch schema; `QF_ERR_MODEL_LOAD` rejects model/provider loading;
`QF_ERR_TASK_NOT_FOUND`, `QF_ERR_TASK_CANCELLED`, and `QF_ERR_TIMEOUT` report
task state; `QF_ERR_NO_WORKER` reports worker startup failure; and
`QF_ERR_INTERNAL` is an unexpected internal failure.

## Build, deployment, and callbacks

Use the Python 3.11 environment containing `cpp/requirements-runtime.txt`.
Build intermediates remain in `cpp/build/` and `model/build/`. Generate the
complete Windows CPU/CUDA unified package with:

```powershell
bash build.sh
```

Deploy the entire `build/windows/` package, including its private Python runtime
and `python/` dependencies. Set `model_path` explicitly. See
[Windows integration](WINDOWS_INTEGRATION.md) for copying, linking and testing.

There is one process-global callback. It executes synchronously at terminal
state, usually on a worker but potentially on the cancelling caller. Keep it
brief, synchronize shared data, and do not call `qf_shutdown` within it. Its
`const char* result_json_path` argument is a borrowed library pointer valid
until `qf_shutdown`; copy it for use after shutdown.
