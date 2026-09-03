# MRMPFormer C API

`mrmpformer` is a C++20 shared library with a C ABI.  Its public header is
[`include/mrmpformer.h`](include/mrmpformer.h); applications written in C must
include that header and link the produced shared library.  The checked-in model
is [`../model/checkpoint/mrmpformerv2.onnx`](../model/checkpoint/mrmpformerv2.onnx).

For incompatible QuanFormer callers, read [the API migration guide](../docs/CPP_API_DIFFERENCES.md)
before recompiling.

## Build

Set `ONNXRUNTIME_ROOT` to an ONNX Runtime package containing `include/` and
`lib/`.  On Windows the package must provide `lib/onnxruntime.lib`; on Linux it
must provide `lib/libonnxruntime.so`.

CPU build (PowerShell):

```powershell
$env:ONNXRUNTIME_ROOT = 'C:\path\to\onnxruntime-win-x64-<version>'
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON
cmake --build cpp/build --config Release
```

CUDA build: use the matching **GPU** ONNX Runtime package for
`ONNXRUNTIME_ROOT`, then run the same configure/build commands.  GPU use is a
runtime request, not a separate MRMPFormer CMake option: set
`QfConfig.use_gpu = 0` to try CUDA then fall back to CPU, or `1` to require
CUDA.  Set it to `-1` to force CPU.

Linux example:

```bash
export ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-<version>
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON
cmake --build cpp/build --config Release
ctest --test-dir cpp/build --output-on-failure
```

The pure-C samples are normal build targets:

```powershell
cmake --build cpp/build --config Release --target batch_example single_example
.\cpp\build\Release\batch_example.exe .\model\checkpoint\mrmpformerv2.onnx .\input.json
.\cpp\build\Release\single_example.exe .\model\checkpoint\mrmpformerv2.onnx
ctest --test-dir cpp/build -C Release --output-on-failure
```

`batch_example.c` submits a JSON file; `single_example.c` sets every
`QfCompoundInput` member.  They intentionally use `use_gpu = -1` so they can
be run with a CPU ONNX Runtime package.

## Deployment

Place the application, `mrmpformer.dll` (Windows) or `libmrmpformer.so`
(Linux), and the matching ONNX Runtime dynamic library in the loader search
path.  The simplest layouts are the same directory on Windows
(`mrmpformer.dll`, `onnxruntime.dll`) and either the same directory with an
appropriate RPATH or a directory named in `LD_LIBRARY_PATH` on Linux
(`libmrmpformer.so`, `libonnxruntime.so`).  Copy any provider DLLs required by
the selected ONNX Runtime GPU package as well.  Deploy
`mrmpformerv2.onnx` with the application or pass its explicit path through
`QfConfig.model_path`; the library never assumes a developer-machine path.

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

Each item has `uid`, `status`, `peaks`, and `alerts`. `status="ok"` means one
or more model-derived peaks; `review` is a signal fallback and includes
`SIGNAL_FALLBACK`; `alert` has no accepted peak. `CHANNEL_LOW_INTENSITY` means
the channel had too few RT points or insufficient smoothed maximum intensity;
`NO_PEAK_FOUND` means both model detections and the signal fallback failed.
`status` and alerts are emitted by C++ QC/post-processing, never directly by
the ONNX network.  See the migration guide for the JSON and ONNX contracts.

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
