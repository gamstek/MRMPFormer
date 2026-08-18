# -*- coding: utf-8 -*-
"""
重复运行 inference.cli --mode pipeline，汇总耗时与资源占用。

用法:
  python -m <包名>.benchmark.runner --runs 20
  python -m <包名>.benchmark.runner --aggregate-only --benchmark-dir "D:/.../timing_benchmark_20"
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .sampler import GpuVramSampler
from .aggregate import (
    collect_values,
    load_jsonl,
    load_records_from_benchmark_dir,
    mean_of,
    mean_resource,
    stat_summary,
)
from .report import (
    fmt_elapsed_from_ms,
    fmt_key_metric_value,
    fmt_mb,
    fmt_ms,
    write_benchmark_summary_csvs,
    write_merged_jsonl,
)

# 项目根目录
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 默认值（示例，实际使用需通过参数覆盖）
# 原路径: D:\pycharm\QuanFormer-main\粮食局数据\results\MULT
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "results" / "MULT")

# 关键指标定义
KEY_METRIC_SPECS = [
    ("总耗时", lambda r: r.get("total_ms"), "time_ms"),
    ("CPU 进程均值", lambda r: (r.get("overall_resource") or {}).get("cpu_avg"), "cpu_pct"),
    ("CPU 进程峰值", lambda r: (r.get("overall_resource") or {}).get("cpu_max"), "cpu_pct"),
    ("内存 RSS 均值", lambda r: (r.get("overall_resource") or {}).get("rss_mb_avg"), "mb"),
    ("内存 RSS 峰值", lambda r: (r.get("overall_resource") or {}).get("rss_mb_max"), "mb"),
    ("进程占系统内存 均值", lambda r: (r.get("overall_resource") or {}).get("proc_mem_pct_avg"), "pct"),
    ("进程占系统内存 峰值", lambda r: (r.get("overall_resource") or {}).get("proc_mem_pct_max"), "pct"),
    ("系统内存占用 均值", lambda r: (r.get("overall_resource") or {}).get("sys_mem_pct_avg"), "pct"),
    ("系统内存占用 峰值", lambda r: (r.get("overall_resource") or {}).get("sys_mem_pct_max"), "pct"),
    ("显存已用", lambda r: (r.get("gpu_vram") or {}).get("used_mb_avg"), "mb"),
    ("显存总量", lambda r: (r.get("gpu_vram") or {}).get("total_mb_avg"), "mb"),
]


def _default_main_argv():
    """默认 pipeline 参数（示例，需用户通过 -- 覆盖）。"""
    return [
        "--mode", "pipeline",
        "--mzml", "",
        "--model", str(_REPO_ROOT / "checkpoint" / "quanformer.pth"),
        "--output_dir", DEFAULT_OUTPUT_DIR,
        "--threshold", "0.90",
        "--smooth_sigma", "0.0",
        "--snr_min", "3.0",
        "--snr_gaussian_sigma", "0.8",
        "--snr_min_noise_points", "5",
        "--plot",
        "--post_plot_sigma", "0.8",
        "--post_plot_dir_name", "refined_plots",
    ]


def _parse_main_output_dir(main_argv):
    for i, a in enumerate(main_argv):
        if a == "--output_dir" and i + 1 < len(main_argv):
            return main_argv[i + 1]
    return None


def _set_main_output_dir(main_argv, out_dir):
    argv = list(main_argv)
    for i, a in enumerate(argv):
        if a == "--output_dir" and i + 1 < len(argv):
            argv[i + 1] = out_dir
            return argv
    argv.extend(["--output_dir", out_dir])
    return argv


def _append_gpu_vram_section(lines, records, csv_rows):
    n = len(records)
    used_avg = collect_values(records, lambda r: (r.get("gpu_vram") or {}).get("used_mb_avg"))
    used_max = collect_values(records, lambda r: (r.get("gpu_vram") or {}).get("used_mb_max"))
    total_avg = collect_values(records, lambda r: (r.get("gpu_vram") or {}).get("total_mb_avg"))
    pct_avg = collect_values(records, lambda r: (r.get("gpu_vram") or {}).get("pct_avg"))
    pct_max = collect_values(records, lambda r: (r.get("gpu_vram") or {}).get("pct_max"))
    if not used_avg:
        lines.append("")
        lines.append("【GPU 显存】无数据")
        return
    lines.append("")
    lines.append("=" * 72)
    lines.append("【GPU 显存】%d 次运行" % n)
    lines.append("=" * 72)
    specs = [
        ("显存已用(各次采样均值)", used_avg, "mb"),
        ("显存已用(各次采样峰值)", used_max, "mb"),
        ("显存总量", total_avg, "mb"),
        ("显存占用率(各次均值)", pct_avg, "pct"),
        ("显存占用率(各次峰值)", pct_max, "pct"),
    ]
    for label, vals, unit in specs:
        st = stat_summary(vals)
        if st:
            csv_rows.append({"metric": label, "unit": unit, **st})


def _append_key_metrics_section(lines, records, csv_rows):
    n_runs = len(records)
    lines.append("")
    lines.append("=" * 72)
    lines.append("【关键指标】%d 次运行" % n_runs)
    lines.append("=" * 72)
    for label, getter, unit in KEY_METRIC_SPECS:
        vals = collect_values(records, getter)
        st = stat_summary(vals)
        if st:
            csv_rows.append({"metric": label, "unit": unit, **st})


def _append_pipeline_timing_avg_table(lines, records, mean_total_ms, n_runs):
    stage_keys = sorted({k for r in records for k in r.get("stage_ms", {})})
    lines.append("")
    lines.append("=" * 72)
    lines.append("[PIPELINE TIMING 平均] %d 次运行" % n_runs)
    lines.append("=" * 72)
    if mean_total_ms > 0:
        lines.append("总耗时: %s" % fmt_elapsed_from_ms(mean_total_ms))
    lines.append("-" * 72)
    for sk in stage_keys:
        vals = [r["stage_ms"][sk] for r in records if sk in r.get("stage_ms", {})]
        st = stat_summary(vals)
        if st:
            mean_ms = st["mean"]
            pct = (100.0 * mean_ms / mean_total_ms) if mean_total_ms > 0 else 0.0
            lines.append("%-28s %32s %7.1f%%" % (sk, fmt_elapsed_from_ms(mean_ms), pct))
    lines.append("=" * 72)


def _format_overall_resource_avg(records, n_cpu):
    cpu_avg = mean_resource(records, "overall_resource", "cpu_avg")
    cpu_max = mean_resource(records, "overall_resource", "cpu_max")
    rss_avg = mean_resource(records, "overall_resource", "rss_mb_avg")
    rss_max = mean_resource(records, "overall_resource", "rss_mb_max")
    parts = []
    if cpu_avg is not None:
        parts.append("CPU %.1f%%~%.1f%%（逻辑核 %d）" % (cpu_avg, cpu_max or cpu_avg, n_cpu))
    if rss_avg is not None:
        parts.append("RSS %s~%s" % (fmt_mb(rss_avg), fmt_mb(rss_max or rss_avg)))
    return "；".join(parts) if parts else "—"


def aggregate_records(records, summary_out_dir, total_runs=None):
    if not records:
        print("[WARN] 无有效记录，跳过汇总")
        return

    summary_out_dir = Path(summary_out_dir)
    summary_out_dir.mkdir(parents=True, exist_ok=True)
    n_runs = len(records)
    n_cpu = os.cpu_count() or 1

    total_ms_list = [r["total_ms"] for r in records if "total_ms" in r]
    ts = stat_summary(total_ms_list)
    mean_total_ms = ts.get("mean", 0.0) if ts else 0.0

    lines = []
    lines.append("=" * 72)
    lines.append("[BENCHMARK SUMMARY] %d 次运行" % n_runs)
    lines.append("生成时间: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 72)
    if ts:
        lines.append("总耗时(平均): %s (median %s stdev %.0f ms)" % (
            fmt_elapsed_from_ms(ts["mean"]), fmt_ms(ts["median"]), ts["stdev"]))
    lines.append("资源: %s" % _format_overall_resource_avg(records, n_cpu))

    key_csv_rows = []
    _append_key_metrics_section(lines, records, key_csv_rows)
    _append_gpu_vram_section(lines, records, key_csv_rows)
    _append_pipeline_timing_avg_table(lines, records, mean_total_ms, n_runs)

    text = "\n".join(lines) + "\n"
    print(text)

    summary_log = summary_out_dir / "benchmark_summary.log"
    summary_log.write_text(text, encoding="utf-8")
    print("[INFO] 汇总日志: %s" % summary_log)

    write_benchmark_summary_csvs(summary_out_dir, key_csv_rows, [])
    write_merged_jsonl(summary_out_dir, records)
    print("[INFO] 汇总输出目录: %s" % summary_out_dir.resolve())


def main():
    ap = argparse.ArgumentParser(description="重复运行全流程并统计 TIMING")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--benchmark-dir", type=str, default=None)
    ap.add_argument("--summary-output-dir", type=str, default=None)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--aggregate-only", action="store_true", help="仅从已有记录重新汇总")
    ap.add_argument("main_argv", nargs=argparse.REMAINDER, help="传给 inference.cli 的参数")
    args = ap.parse_args()

    main_argv = args.main_argv
    if main_argv and main_argv[0] == "--":
        main_argv = main_argv[1:]
    if not main_argv:
        main_argv = _default_main_argv()

    base_out = _parse_main_output_dir(main_argv) or DEFAULT_OUTPUT_DIR
    main_argv = _set_main_output_dir(main_argv, base_out)

    summary_out_dir = Path(args.summary_output_dir) if args.summary_output_dir else None
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else Path(base_out) / ("timing_benchmark_%d" % args.runs)
    if not args.aggregate_only:
        benchmark_dir.mkdir(parents=True, exist_ok=True)
    if summary_out_dir is None:
        summary_out_dir = benchmark_dir

    if args.aggregate_only:
        all_records = load_records_from_benchmark_dir(benchmark_dir)
        if not all_records:
            print("[ERROR] 未在 %s 找到记录" % benchmark_dir, file=sys.stderr)
            return 1
        aggregate_records(all_records, summary_out_dir)
        return 0

    print("[INFO] benchmark 目录: %s" % benchmark_dir)
    print("[INFO] 计划运行 %d 次" % args.runs)

    gpu_cls = GpuVramSampler
    probe = gpu_cls(0)
    if probe._available:
        print("[INFO] GPU 显存监测可用")

    all_records = []
    # 推理统一入口：python -m inference.cli（在 model/ 目录下运行）
    cli_module = "inference.cli"

    for i in range(1, int(args.runs) + 1):
        run_out = benchmark_dir / ("run_%03d" % i)
        run_argv = _set_main_output_dir(main_argv, str(run_out))
        cmd = [args.python, "-m", cli_module] + run_argv
        print("\n[BENCHMARK] 第 %d/%d 次" % (i, args.runs))

        gpu = gpu_cls(0)
        gpu.start()
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        wall = time.perf_counter() - t0
        gpu_vram = gpu.stop_and_stats()

        if proc.returncode != 0:
            print("[ERROR] 第 %d 次失败，退出码 %d" % (i, proc.returncode), file=sys.stderr)
            return proc.returncode

        jsonl = run_out / "pipeline_timing_runs.jsonl"
        rows = load_jsonl(jsonl)
        if rows:
            rec = rows[-1]
            rec["benchmark_wall_seconds"] = wall
            rec["benchmark_run_dir"] = str(run_out)
            if gpu_vram:
                rec["gpu_vram"] = gpu_vram
            all_records.append(rec)
        else:
            print("[WARN] 未找到 %s" % jsonl)

    aggregate_records(all_records, summary_out_dir, total_runs=args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
