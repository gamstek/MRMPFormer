# -*- coding: utf-8 -*-
"""格式化函数、CSV/JSONL 报告生成、人类可读摘要。"""

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


def fmt_ms(ms: float) -> str:
    if ms >= 100.0:
        return "%.0f ms (%.2f s)" % (ms, ms / 1000.0)
    return "%.2f ms" % ms


def fmt_elapsed_from_ms(ms: float) -> str:
    s = float(ms) / 1000.0
    if s < 60.0:
        return "%.2f s (%.2f ms)" % (s, ms)
    m, sec = divmod(s, 60.0)
    if m < 60.0:
        return "%dm %.1fs (%.0f ms)" % (int(m), sec, ms)
    h, rem = divmod(m, 60.0)
    return "%dh %dm %.0fs (%.0f ms)" % (int(h), int(rem), sec, ms)


def fmt_mb(mb: float) -> str:
    if mb >= 1024.0:
        return "%.2f GB" % (mb / 1024.0)
    return "%.1f MB" % mb


def fmt_gpu_vram_line(vram_stats: dict) -> str:
    if not vram_stats:
        return ""
    return "显存 已用/总量 均值 %s / %s（占用 %.1f%%）；峰值 已用/总量 %s / %s（占用 %.1f%%）" % (
        fmt_mb(vram_stats.get("used_mb_avg", 0)),
        fmt_mb(vram_stats.get("total_mb_avg", 0)),
        vram_stats.get("pct_avg", 0),
        fmt_mb(vram_stats.get("used_mb_max", 0)),
        fmt_mb(vram_stats.get("total_mb_max", vram_stats.get("total_mb_avg", 0))),
        vram_stats.get("pct_max", 0),
    )


def fmt_key_metric_value(unit: str, value: float) -> str:
    if value is None:
        return "—"
    if unit == "time_ms":
        return fmt_elapsed_from_ms(value)
    if unit == "cpu_pct":
        return "%.1f%%" % value
    if unit == "mb":
        return fmt_mb(value)
    if unit == "pct":
        return "%.1f%%" % value
    return "%.4g" % value


def write_benchmark_summary_csvs(summary_out_dir: Path, key_csv_rows: list, detail_rows: list):
    """写入关键指标和明细 CSV。"""
    key_csv_path = summary_out_dir / "benchmark_key_metrics.csv"
    with open(key_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["metric", "unit", "n", "mean", "min", "max", "median", "stdev"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in key_csv_rows:
            w.writerow(row)
    print("[INFO] 关键指标 CSV: %s" % key_csv_path)

    legacy_csv = summary_out_dir / "benchmark_summary.csv"
    with open(legacy_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in key_csv_rows:
            w.writerow(row)
    print("[INFO] 汇总 CSV: %s" % legacy_csv)

    csv_path = summary_out_dir / "benchmark_summary_detail.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["metric", "n", "mean", "median", "stdev", "min", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in detail_rows:
            w.writerow(row)
    print("[INFO] 明细 CSV: %s" % csv_path)


def write_merged_jsonl(summary_out_dir: Path, records: list):
    """写入合并的 JSONL。"""
    merged_jsonl = summary_out_dir / "all_runs.jsonl"
    with open(merged_jsonl, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            r2 = dict(r)
            r2["benchmark_run_index"] = i
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    print("[INFO] 合并 JSONL: %s" % merged_jsonl.resolve())
