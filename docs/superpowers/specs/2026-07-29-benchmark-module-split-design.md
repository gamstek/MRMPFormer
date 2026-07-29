# Benchmark 模块拆分设计

> 日期：2026-07-29 | 状态：待审批

## 背景

`model/tools/benchmark/run_pipeline_timing_benchmark.py` 单文件约 750 行，混杂了 GPU 采样、统计计算、格式化输出、CLI 编排四种职责。需拆分为 4 个独立模块，放在 `model/tools/benchmark/` 下。

## 目标

- 4 个模块，职责单一，依赖单向无环
- 保持原有功能完全不变（命令行接口、输出格式、日志文件均兼容）
- aggregate 层改为纯计算（不直接写文件/print），由 runner 编排输出

## 模块设计

### 1. `sampler.py` — GPU 显存后台采样

**职责：** 读取 GPU 显存已用/总量，后台线程定时采样，汇总统计。

**从原文件迁移：**

| 符号 | 类型 | 说明 |
|------|------|------|
| `GPU_VRAM_SAMPLE_INTERVAL_SEC` | 常量 | 采样间隔 0.5s |
| `_read_gpu_vram_mb(device_index)` | 函数 | 优先 pynvml，fallback nvidia-smi |
| `_RunGpuVramSampler` | 类 → `GpuVramSampler` | 去掉下划线，作为公开类 |

**对外接口：**

```python
class GpuVramSampler:
    def __init__(self, device_index=0): ...
    def start(self) -> bool: ...
    def stop_and_stats(self) -> dict | None: ...
```

**依赖：** `threading`, `subprocess`, `sys`, `statistics`；可选 `pynvml`

---

### 2. `aggregate.py` — 统计计算层（纯函数，无副作用）

**职责：** JSONL 读取、描述性统计、跨记录聚合。

**从原文件迁移：**

| 符号 | 说明 |
|------|------|
| `_load_jsonl(path)` | 读取 JSONL 为 list[dict] |
| `_load_records_from_benchmark_dir(benchmark_dir)` | 收集各次 run 的记录 |
| `_stat_summary(values)` | 返回 {n, mean, median, stdev, min, max} |
| `_collect_values(records, getter)` | 从记录列表提取数值列表 |
| `_mean_of(records, getter)` | 对 getter 结果取均值 |
| `_mean_resource(records, section, key, stage_name)` | 按阶段取资源均值 |
| `KEY_METRIC_SPECS` | 关键指标定义（label, getter, unit）列表 |

**关键改动：`aggregate_records` 去副作用化**

原函数约 120 行，内部直接 `print()`、写文件、调 report 函数。重构后：

```python
def aggregate_records(records) -> dict:
    """返回结构化聚合结果，不写任何文件，不 print"""
    return {
        "n_runs": len(records),
        "n_cpu": os.cpu_count() or 1,
        "total_ms_stats": _stat_summary(total_ms_list),
        "overall_resource": {...},
        "stage_stats": {...},
        "per_sample_stats": {...},
        "gpu_vram_stats": {...},
        "key_metric_rows": [...],   # 供 report 写 CSV
        "detail_rows": [...],
    }
```

**依赖：** `json`, `statistics`, `pathlib`, `os`

---

### 3. `report.py` — 格式化 & 文件输出

**职责：** 所有 `_fmt_*` / `_format_*` 函数、日志块生成、summary 日志/CSV 写入。

**从原文件迁移：**

| 类别 | 符号 |
|------|------|
| 基础格式化 | `_fmt_ms`, `_fmt_elapsed_from_ms`, `_fmt_mb` |
| 资源/GPU 格式化 | `_format_gpu_vram_line`, `_format_single_run_resource_line`, `_format_overall_resource_avg`, `_format_stage_resource_avg`, `_format_stage_resource_pipeline_style` |
| 指标格式化 | `_format_key_metric_value` |
| 日志块 | `_format_benchmark_run_log_block` |
| 汇总节 | `_append_key_metrics_section`, `_append_gpu_vram_benchmark_section`, `_append_pipeline_timing_avg_table` |
| 文件 I/O | `_init_benchmark_runs_log`, `_append_benchmark_run_log`, `write_benchmark_runs_log`, `_resolve_runs_log_path` |
| 常量 | `BENCHMARK_RUNS_LOG_NAME`, `BENCHMARK_SUMMARY_LOG_NAME` |

**新增对外接口（替代原 `aggregate_records` 中的副作用部分）：**

```python
def write_summary_report(agg_result: dict, summary_out_dir: Path, 
                         summary_log_path=None, runs_log_path=None, 
                         benchmark_dir=None, total_runs=None) -> None:
    """接收 aggregate_records 返回的结构化数据，写入所有日志/CSV"""
```

**依赖：** `aggregate`（仅引用 `_stat_summary`, `_collect_values`），`pathlib`, `csv`, `datetime`

---

### 4. `runner.py` — CLI 入口 & 运行编排

**职责：** 命令行参数解析、N 次 subprocess 循环、编排 aggregate → report 全流程。

**从原文件迁移：**

| 符号 | 说明 |
|------|------|
| `ROOT` | 项目根目录（`Path(__file__).resolve().parents[1]` → 需调整为 `parents[2]`） |
| `DEFAULT_OUTPUT_DIR` | 默认输出目录 |
| `_default_main_argv()` | 默认 P3 命令行 |
| `_parse_main_output_dir(argv)` | 提取 `--output_dir` |
| `_set_main_output_dir(argv, out_dir)` | 替换 `--output_dir` |
| `main()` | 主函数（重构编排逻辑） |

**`main()` 重构后的流程：**

```python
def main():
    args = parse_args()
    # ... 初始化 benchmark_dir, summary_out_dir ...
    
    all_records = []
    for i in range(1, args.runs + 1):
        gpu = GpuVramSampler(0)
        gpu.start()
        proc = subprocess.run(cmd)
        gpu_vram = gpu.stop_and_stats()
        
        rows = _load_jsonl(jsonl)
        rec = rows[-1]
        rec["gpu_vram"] = gpu_vram
        all_records.append(rec)
        
        # 逐次写 runs_log
        block = _format_benchmark_run_log_block(i, args.runs, rec, n_cpu)
        _append_benchmark_run_log(runs_log_path, block)
    
    # 聚合 + 输出
    agg = aggregate_records(all_records)
    write_summary_report(agg, summary_out_dir, ...)
```

**依赖：** `sampler`, `aggregate`, `report`, `argparse`, `subprocess`, `time`, `pathlib`, `os`

---

## 依赖关系图

```
runner.py ──→ sampler.py    (GpuVramSampler)
runner.py ──→ aggregate.py  (_load_jsonl, aggregate_records)
runner.py ──→ report.py     (_format_benchmark_run_log_block, _append_benchmark_run_log,
                              _init_benchmark_runs_log, write_summary_report, _resolve_runs_log_path)
report.py ──→ aggregate.py  (_stat_summary, _collect_values)
sampler.py  (无内部依赖)
```

无环。所有导入方向：`runner → {sampler, aggregate, report}`, `report → aggregate`。

## 文件结构

```
model/tools/benchmark/
├── __init__.py          (可选，空文件)
├── sampler.py           (~70 行)
├── aggregate.py         (~180 行)
├── report.py            (~400 行)
└── runner.py            (~180 行)
```

原 `run_pipeline_timing_benchmark.py` 删除或保留为兼容性 re-export。

## 不变量

- CLI 参数、默认值、输出格式 **完全不变**
- `benchmark_runs.log` / `benchmark_summary.log` 格式不变
- `benchmark_key_metrics.csv` / `benchmark_summary_detail.csv` / `all_runs.jsonl` 格式不变
- `--aggregate-only` 模式行为不变
- 函数内部逻辑逐行迁移，不修改业务逻辑

## 风险与注意

1. **`ROOT` 路径**：原文件用 `parents[1]` 定位 `model/`，拆到 `benchmark/` 子目录后需改为 `parents[2]`
2. **私有函数暴露**：原大量 `_` 前缀函数在模块间引用时需去掉下划线或保持模块内私有
3. **`aggregate_records` 重构**：去副作用是最大改动点，需仔细保留所有输出格式
