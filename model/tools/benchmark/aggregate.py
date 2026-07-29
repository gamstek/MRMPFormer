# -*- coding: utf-8 -*-
"""JSONL/CSV 结果读取、统计聚合、多轮运行结果合并。"""

import json
import statistics
from pathlib import Path
from typing import Callable, List


def load_jsonl(path: Path) -> list:
    """读取 JSONL 文件，返回记录列表。"""
    if not path.is_file():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stat_summary(values: list) -> dict:
    """计算均值、中位数、标准差、min、max。"""
    if not values:
        return {}
    if len(values) == 1:
        return {"n": 1, "mean": values[0], "median": values[0], "stdev": 0.0, "min": values[0], "max": values[0]}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def collect_values(records: list, getter: Callable) -> list:
    """从记录列表中提取数值。"""
    vals = []
    for r in records:
        v = getter(r)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return vals


def mean_of(records: list, getter: Callable) -> float:
    """计算 getter 提取值的算术平均。"""
    vals = collect_values(records, getter)
    if not vals:
        return None
    return statistics.mean(vals)


def mean_resource(records: list, section: str, key: str, stage_name: str = None) -> float:
    """从记录的 resource 段提取特定 key 的平均值。"""
    def getter(r):
        if stage_name is not None:
            block = (r.get(section) or {}).get(stage_name) or {}
        else:
            block = r.get(section) or {}
        return block.get(key)
    return mean_of(records, getter)


def load_records_from_benchmark_dir(benchmark_dir: Path) -> list:
    """从 benchmark 目录读取各次 pipeline_timing_runs.jsonl。"""
    benchmark_dir = Path(benchmark_dir)
    merged = benchmark_dir / "all_runs.jsonl"
    if merged.is_file():
        rows = load_jsonl(merged)
        if rows:
            return rows
    records = []
    for jsonl in sorted(benchmark_dir.glob("run_*/pipeline_timing_runs.jsonl")):
        rows = load_jsonl(jsonl)
        if rows:
            records.append(rows[-1])
    return records
