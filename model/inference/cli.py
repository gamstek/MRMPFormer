# -*- coding: utf-8 -*-
"""
MRMPFormer 统一推理入口（3 种模式；roi / pipeline 均支持单文件与目录递归扫描）。

用法（须在 model/ 目录下运行）:
  # 仅 ROI 生成（单文件或目录递归；每个 mzML 输出到 <output_dir>/<文件名stem>/）
  python -m inference.cli --mode roi --model checkpoint/quanformer.pth --mzml ../data/sample.mzML
  python -m inference.cli --mode roi --model checkpoint/quanformer.pth --batch_dir ../data/mzML_dir

  # 对已有 ROI 目录批量预测+积分
  python -m inference.cli --mode batch_dir --model checkpoint/quanformer.pth --batch_dir xic-roi-batch

  # 完整管线（ROI → 预测 → SNR 筛选 → 精修，单文件或目录递归）
  python -m inference.cli --mode pipeline --model checkpoint/quanformer.pth --batch_dir ../data/mzML_dir

  # 目录递归时不同子目录出现同名 mzML：输出目录自动加路径前缀（如 子目录A__样品1）避免覆盖
"""
import argparse
import json
import os
import sys

# 解决 Windows 下 PyTorch(libomp.dll) 与 numpy/MKL(libiomp5md.dll) 的 OpenMP 运行时冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _format_elapsed_ms(seconds: float) -> str:
    """Milliseconds only."""
    ms = float(max(0.0, seconds)) * 1000.0
    if ms >= 100.0:
        return "%.0f ms" % ms
    return "%.2f ms" % ms


def _format_elapsed(seconds: float) -> str:
    """Human-readable duration with milliseconds for console logs."""
    s = float(max(0.0, seconds))
    if s < 60.0:
        base = "%.2f s" % s
    else:
        m, sec = divmod(s, 60.0)
        if m < 60.0:
            base = "%dm %.1fs" % (int(m), sec)
        else:
            h, rem = divmod(m, 60.0)
            base = "%dh %dm %.0fs" % (int(h), int(rem), sec)
    return "%s (%s)" % (base, _format_elapsed_ms(s))


def _format_mb(mb: float) -> str:
    if not np.isfinite(mb):
        return "—"
    if mb >= 1024.0:
        return "%.2f GB" % (mb / 1024.0)
    return "%.1f MB" % mb


class _PipelineResourceMonitor:
    """Background CPU / memory sampling for pipeline modes (psutil preferred)."""

    INTERVAL_SEC = 0.5

    def __init__(self):
        self._lock = threading.Lock()
        self._samples = []  # (perf_counter, dict)
        self._thread = None
        self._stop_evt = threading.Event()
        self._enabled = False
        self._backend = "none"
        self._psutil = None
        self._proc = None
        self._n_cpu = os.cpu_count() or 1
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(interval=None)
            self._enabled = True
            self._backend = "psutil"
            return
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t),
                    ]

                self._pmc_type = PROCESS_MEMORY_COUNTERS_EX
                self._enabled = True
                self._backend = "win32"
            except Exception:
                pass

    def _read_sample(self) -> dict:
        out = {
            "cpu_pct": float("nan"),
            "rss_mb": float("nan"),
            "proc_mem_pct": float("nan"),
            "sys_mem_pct": float("nan"),
        }
        if self._backend == "psutil" and self._proc is not None:
            out["cpu_pct"] = float(self._proc.cpu_percent(interval=None))
            mi = self._proc.memory_info()
            out["rss_mb"] = float(mi.rss) / (1024.0 ** 2)
            out["proc_mem_pct"] = float(self._proc.memory_percent())
            out["sys_mem_pct"] = float(self._psutil.virtual_memory().percent)
        elif self._backend == "win32":
            try:
                import ctypes

                pmc = self._pmc_type()
                pmc.cb = ctypes.sizeof(pmc)
                hproc = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(hproc, ctypes.byref(pmc), pmc.cb):
                    out["rss_mb"] = float(pmc.WorkingSetSize) / (1024.0 ** 2)
            except Exception:
                pass
        return out

    def start(self) -> None:
        if not self._enabled:
            return
        self._stop_evt.clear()
        with self._lock:
            self._samples = []
        self._read_sample()
        self._thread = threading.Thread(target=self._loop, name="pipeline-resource-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_evt.wait(self.INTERVAL_SEC):
            ts = time.perf_counter()
            snap = self._read_sample()
            with self._lock:
                self._samples.append((ts, snap))

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._samples.append((time.perf_counter(), self._read_sample()))

    @staticmethod
    def _stats_from_snaps(snaps: list) -> dict:
        if not snaps:
            return {}
        cpus = [s["cpu_pct"] for _, s in snaps if np.isfinite(s.get("cpu_pct", np.nan))]
        rss = [s["rss_mb"] for _, s in snaps if np.isfinite(s.get("rss_mb", np.nan))]
        proc_mem = [s["proc_mem_pct"] for _, s in snaps if np.isfinite(s.get("proc_mem_pct", np.nan))]
        sys_mem = [s["sys_mem_pct"] for _, s in snaps if np.isfinite(s.get("sys_mem_pct", np.nan))]
        out = {}
        if cpus:
            out["cpu_avg"] = float(np.mean(cpus))
            out["cpu_max"] = float(np.max(cpus))
        if rss:
            out["rss_mb_avg"] = float(np.mean(rss))
            out["rss_mb_max"] = float(np.max(rss))
        if proc_mem:
            out["proc_mem_pct_avg"] = float(np.mean(proc_mem))
            out["proc_mem_pct_max"] = float(np.max(proc_mem))
        if sys_mem:
            out["sys_mem_pct_avg"] = float(np.mean(sys_mem))
            out["sys_mem_pct_max"] = float(np.max(sys_mem))
        return out

    def stats_for_interval(self, t0: float, t1: float) -> dict:
        with self._lock:
            snaps = [(t, s) for t, s in self._samples if t0 <= t <= t1]
        return self._stats_from_snaps(snaps)

    def stats_for_merged_intervals(self, intervals: list) -> dict:
        if not intervals:
            return {}
        with self._lock:
            snaps = []
            for t0, t1 in intervals:
                snaps.extend([(t, s) for t, s in self._samples if t0 <= t <= t1])
            snaps.sort(key=lambda x: x[0])
        return self._stats_from_snaps(snaps)

    def overall_stats(self, t0: float, t1: float) -> dict:
        return self.stats_for_interval(t0, t1)

    def backend_name(self) -> str:
        return self._backend


def _print_mzml_roi_stats_summary(mzml_roi_stats: list) -> None:
    """Print per-mzML ROI image counts after testXIC stage."""
    if not mzml_roi_stats:
        return
    total_roi = sum(int(s.get("n_roi_images", 0) or 0) for s in mzml_roi_stats)
    total_chrom = sum(int(s.get("n_chromatograms", 0) or 0) for s in mzml_roi_stats)
    total_excl = sum(int(s.get("n_qc_excluded", 0) or 0) for s in mzml_roi_stats)
    print(
        "[PIPELINE ROI] %d 个 mzML：读取色谱合计 %d 条，生成 ROI 图合计 %d 张（QC 剔除 %d 条）"
        % (len(mzml_roi_stats), total_chrom, total_roi, total_excl)
    )
    if len(mzml_roi_stats) <= 20:
        print("%-28s %10s %10s %10s" % ("mzML 样品", "色谱条数", "ROI图", "QC剔除"))
        for s in mzml_roi_stats:
            label = s.get("stem") or s.get("mzml") or "?"
            print(
                "%-28s %10d %10d %10d"
                % (
                    label,
                    int(s.get("n_chromatograms", 0) or 0),
                    int(s.get("n_roi_images", 0) or 0),
                    int(s.get("n_qc_excluded", 0) or 0),
                )
            )
    else:
        avg_roi = total_roi / len(mzml_roi_stats) if mzml_roi_stats else 0.0
        print("（%d 个样品，省略逐条；平均每样品 ROI 图 %.1f 张）" % (len(mzml_roi_stats), avg_roi))


def _print_resource_stats_block(title: str, stats: dict, indent: str = "") -> None:
    if not stats:
        print("%s%s: 无采样数据" % (indent, title))
        return
    parts = []
    if "cpu_avg" in stats:
        parts.append(
            "CPU 进程均值/峰值 %.1f%% / %.1f%%（相对单核，逻辑核数 %d）"
            % (stats["cpu_avg"], stats["cpu_max"], os.cpu_count() or 1)
        )
    if "rss_mb_avg" in stats:
        parts.append(
            "内存 RSS 均值/峰值 %s / %s"
            % (_format_mb(stats["rss_mb_avg"]), _format_mb(stats["rss_mb_max"]))
        )
    if "proc_mem_pct_avg" in stats:
        parts.append(
            "进程占系统内存 均值/峰值 %.1f%% / %.1f%%"
            % (stats["proc_mem_pct_avg"], stats["proc_mem_pct_max"])
        )
    if "sys_mem_pct_avg" in stats:
        parts.append(
            "系统内存占用 均值/峰值 %.1f%% / %.1f%%"
            % (stats["sys_mem_pct_avg"], stats["sys_mem_pct_max"])
        )
    print("%s%s: %s" % (indent, title, "；".join(parts) if parts else "—"))


def _build_pipeline_timing_report(
    mode_label: str,
    n_samples: int,
    stage_seconds: dict,
    per_sample_seconds: list,
    total_seconds: float,
    resource_monitor: _PipelineResourceMonitor = None,
    stage_intervals: dict = None,
    pipeline_t0: float = None,
    mzml_roi_stats: list = None,
) -> tuple:
    """Build human-readable timing lines and a JSON-serializable record."""
    from datetime import datetime

    lines = []
    stage_resource = {}
    overall_stats = None

    def add(line: str = "") -> None:
        lines.append(line)

    add("")
    add("=" * 60)
    add("[PIPELINE TIMING] %s" % mode_label)
    add("=" * 60)
    add("时间戳: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    add("样品数: %d" % int(n_samples))
    total_roi = 0
    if mzml_roi_stats:
        total_roi = sum(int(s.get("n_roi_images", 0) or 0) for s in mzml_roi_stats)
        add("ROI 图合计: %d 张（来自 %d 个 mzML）" % (total_roi, len(mzml_roi_stats)))
    add("总耗时: %s" % _format_elapsed(total_seconds))
    if resource_monitor is not None:
        if resource_monitor.backend_name() == "none":
            add(
                "资源监控: 未启用（可执行 pip install psutil 以统计 CPU/内存；Windows 下仍尝试仅统计内存 RSS）"
            )
        else:
            add("资源监控后端: %s" % resource_monitor.backend_name())
            if pipeline_t0 is not None:
                overall_stats = resource_monitor.overall_stats(pipeline_t0, time.perf_counter())
                parts = []
                if "cpu_avg" in overall_stats:
                    parts.append(
                        "CPU 进程均值/峰值 %.1f%% / %.1f%%（相对单核，逻辑核数 %d）"
                        % (overall_stats["cpu_avg"], overall_stats["cpu_max"], os.cpu_count() or 1)
                    )
                if "rss_mb_avg" in overall_stats:
                    parts.append(
                        "内存 RSS 均值/峰值 %s / %s"
                        % (_format_mb(overall_stats["rss_mb_avg"]), _format_mb(overall_stats["rss_mb_max"]))
                    )
                if "proc_mem_pct_avg" in overall_stats:
                    parts.append(
                        "进程占系统内存 均值/峰值 %.1f%% / %.1f%%"
                        % (overall_stats["proc_mem_pct_avg"], overall_stats["proc_mem_pct_max"])
                    )
                if "sys_mem_pct_avg" in overall_stats:
                    parts.append(
                        "系统内存占用 均值/峰值 %.1f%% / %.1f%%"
                        % (overall_stats["sys_mem_pct_avg"], overall_stats["sys_mem_pct_max"])
                    )
                add("全流程合计: %s" % ("；".join(parts) if parts else "—"))
    add("-" * 72)
    add("%-28s %32s %8s" % ("阶段", "耗时", "占比"))
    add("-" * 72)
    for name, sec in stage_seconds.items():
        pct = (100.0 * sec / total_seconds) if total_seconds > 0 else 0.0
        add("%-28s %32s %7.1f%%" % (name, _format_elapsed(sec), pct))
        st = None
        if resource_monitor is not None and stage_intervals and name in stage_intervals:
            iv = stage_intervals[name]
            if isinstance(iv, list):
                st = resource_monitor.stats_for_merged_intervals(iv)
            else:
                st = resource_monitor.stats_for_interval(iv[0], iv[1])
        if st:
            stage_resource[name] = dict(st)
            res_parts = []
            if "cpu_avg" in st:
                res_parts.append("CPU %.1f%%~%.1f%%" % (st["cpu_avg"], st["cpu_max"]))
            if "rss_mb_max" in st:
                res_parts.append("RSS峰值 %s" % _format_mb(st["rss_mb_max"]))
            if "sys_mem_pct_max" in st:
                res_parts.append("系统内存峰值 %.1f%%" % st["sys_mem_pct_max"])
            if res_parts:
                add("                             (%s)" % "，".join(res_parts))
    if per_sample_seconds:
        snr_sum = sum(p.get("snr", 0.0) for p in per_sample_seconds)
        post_sum = sum(p.get("post", 0.0) for p in per_sample_seconds)
        sample_sum = sum(p.get("total", 0.0) for p in per_sample_seconds)
        add("-" * 60)
        add(
            "按样品 SNR+后处理: 合计 %s (SNR %s + post %s)"
            % (_format_elapsed(sample_sum), _format_elapsed(snr_sum), _format_elapsed(post_sum))
        )
        if len(per_sample_seconds) <= 20:
            add("-" * 60)
            add("%-24s %22s %22s %22s" % ("样品", "SNR", "post", "合计"))
            for p in per_sample_seconds:
                add(
                    "%-24s %22s %22s %22s"
                    % (
                        p.get("stem", "?"),
                        _format_elapsed(p.get("snr", 0.0)),
                        _format_elapsed(p.get("post", 0.0)),
                        _format_elapsed(p.get("total", 0.0)),
                    )
                )
        else:
            add(
                "（%d 个样品，省略逐条；单样品平均 SNR %s, post %s）"
                % (
                    len(per_sample_seconds),
                    _format_elapsed(snr_sum / len(per_sample_seconds)),
                    _format_elapsed(post_sum / len(per_sample_seconds)),
                )
            )
    add("=" * 60)
    add("")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode_label,
        "n_samples": int(n_samples),
        "total_seconds": float(total_seconds),
        "total_ms": float(total_seconds) * 1000.0,
        "total_roi_images": int(total_roi),
        "stage_seconds": {k: float(v) for k, v in stage_seconds.items()},
        "stage_ms": {k: float(v) * 1000.0 for k, v in stage_seconds.items()},
        "stage_resource": stage_resource,
        "overall_resource": overall_stats,
        "resource_backend": resource_monitor.backend_name() if resource_monitor else None,
        "mzml_roi_stats": mzml_roi_stats or [],
        "per_sample_seconds": [
            {
                "stem": p.get("stem"),
                "snr_seconds": float(p.get("snr", 0.0)),
                "post_seconds": float(p.get("post", 0.0)),
                "total_seconds": float(p.get("total", 0.0)),
                "snr_ms": float(p.get("snr", 0.0)) * 1000.0,
                "post_ms": float(p.get("post", 0.0)) * 1000.0,
                "total_ms": float(p.get("total", 0.0)) * 1000.0,
            }
            for p in (per_sample_seconds or [])
        ],
    }
    return lines, record


def _write_pipeline_timing_logs(log_dir: Path, lines: list, record: dict) -> None:
    """Append timing summary to pipeline_timing.log and pipeline_timing_runs.jsonl."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    text_path = log_dir / "pipeline_timing.log"
    jsonl_path = log_dir / "pipeline_timing_runs.jsonl"
    block = "\n".join(lines)
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(block)
        if not block.endswith("\n"):
            f.write("\n")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("[INFO] 全流程 TIMING 已写入: %s" % text_path)
    print("[INFO] 结构化记录已追加: %s" % jsonl_path)


def _print_pipeline_timing_summary(
    mode_label: str,
    n_samples: int,
    stage_seconds: dict,
    per_sample_seconds: list,
    total_seconds: float,
    resource_monitor: _PipelineResourceMonitor = None,
    stage_intervals: dict = None,
    pipeline_t0: float = None,
    mzml_roi_stats: list = None,
    log_dir: Path = None,
) -> dict:
    """Print pipeline stage timings to stdout; optionally write logs. Returns JSON record."""
    lines, record = _build_pipeline_timing_report(
        mode_label=mode_label,
        n_samples=n_samples,
        stage_seconds=stage_seconds,
        per_sample_seconds=per_sample_seconds,
        total_seconds=total_seconds,
        resource_monitor=resource_monitor,
        stage_intervals=stage_intervals,
        pipeline_t0=pipeline_t0,
        mzml_roi_stats=mzml_roi_stats,
    )
    for line in lines:
        if line:
            print(line)
        else:
            print()
    if log_dir is not None:
        _write_pipeline_timing_logs(log_dir, lines, record)
    return record


def _collect_mzml_inputs(mzml_arg, batch_dir_arg):
    """收集输入 mzML：--mzml 单文件/目录，或 --batch_dir 目录（递归含子目录）。

    返回 [(Path, key), ...]；key 默认为文件名 stem，目录递归下不同子目录出现同名 stem 时，
    key 改为相对扫描根目录的路径展平（如 ``子目录A__样品1``），避免输出目录互相覆盖。
    """
    if mzml_arg and batch_dir_arg:
        print("[ERROR] --mzml 与 --batch_dir 不可同时提供", file=sys.stderr)
        sys.exit(1)
    if mzml_arg:
        p = Path(mzml_arg)
        if p.is_file():
            if p.suffix.lower() != ".mzml":
                print("[ERROR] --mzml 需为 .mzML 文件或包含 mzML 的目录: %s" % mzml_arg, file=sys.stderr)
                sys.exit(1)
            return [(p.resolve(), p.stem)]
        if not p.is_dir():
            print("[ERROR] --mzml 路径不存在: %s" % mzml_arg, file=sys.stderr)
            sys.exit(1)
        scan_root = p
    elif batch_dir_arg:
        scan_root = Path(batch_dir_arg)
        if not scan_root.is_dir():
            print("[ERROR] --batch_dir 需为目录: %s" % batch_dir_arg, file=sys.stderr)
            sys.exit(1)
    else:
        print("[ERROR] 需提供 --mzml（单文件或目录）或 --batch_dir（目录）", file=sys.stderr)
        sys.exit(1)

    files = sorted(set(scan_root.rglob("*.mzml")) | set(scan_root.rglob("*.mzML")))
    if not files:
        print("[ERROR] 未找到 .mzML/.mzml 文件（含子目录递归）: %s" % scan_root, file=sys.stderr)
        sys.exit(1)
    stem_counts = Counter(f.stem for f in files)
    inputs = []
    for f in files:
        if stem_counts[f.stem] > 1:
            key = "__".join(f.relative_to(scan_root).with_suffix("").parts)
            print("[INFO] 同名 mzML 自动加路径前缀避免输出覆盖: %s -> %s/" % (f, key))
        else:
            key = f.stem
        inputs.append((f.resolve(), key))
    return inputs


def main_cli():
    parser = argparse.ArgumentParser(description="MRMPFormer 统一推理入口（roi / batch_dir / pipeline）")
    parser.add_argument("--mode", type=str, default="pipeline",
                        choices=["roi", "batch_dir", "pipeline"],
                        help=(
                            "roi=仅ROI生成(单文件或目录递归); batch_dir=对已有ROI目录批量预测+积分; "
                            "pipeline=完整流水线（ROI->预测->SNR筛选->post_newtest，单文件或目录递归）"
                        ))
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径 (.pth)；也可由 --config 提供")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--integration_method", type=str, default="linear",
                        choices=["linear", "raw", "external_baseline"])
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出根目录；roi 默认 xic-roi-batch/，pipeline 默认 results/full_pipeline/")
    parser.add_argument("--mzml", type=str,
                        help="[roi/pipeline] 单个 mzML 文件路径，或包含 mzML 的目录（递归扫描）")
    parser.add_argument("--batch_dir", type=str,
                        help="[roi/pipeline] mzML 目录（递归扫描）；[batch_dir] testXIC 输出目录")
    parser.add_argument("--plot", action="store_true", help="生成预测框标注图")

    # ===== Pipeline (QC + post_newtest + SNR) 对齐流程图：低强度/少点数剔除 → 预测 → 框修正/二次峰 → SNR =====
    parser.add_argument("--standard_refs_csv", type=str, default=None, help="[testXIC] 已弃用：ROI 以各通道 XIC 最高峰 RT 居中，不再读取；传入时仅打印提示")
    parser.add_argument(
        "--pipeline_min_max_intensity",
        type=float,
        default=1000.0,
        help="[QC] 整条 XIC（与 testXIC 相同 smooth_sigma 平滑后）最大强度低于此值的通道不生成 ROI、不参与预测；0=关闭",
    )
    parser.add_argument(
        "--pipeline_min_chrom_points",
        type=int,
        default=10,
        help="[QC] 单条 chromatogram RT 点数少于此值则剔除；0=关闭",
    )
    parser.add_argument("--snr_min", type=float, default=3.0, help="[SNR筛选] 框外SNR阈值，单位同 mzml_box_outside_snr_pipeline")
    parser.add_argument("--snr_gaussian_sigma", type=float, default=0.8, help="[SNR筛选] mzML 强度高斯平滑 sigma")
    parser.add_argument("--snr_min_noise_points", type=int, default=5, help="[SNR筛选] 框外噪声至少点数")

    parser.add_argument("--post_output_name", type=str, default="prediction_refined.csv", help="[post_newtest] 输出CSV名")
    parser.add_argument("--post_small_peak_rt_tol", type=float, default=0.25)
    parser.add_argument(
        "--post_min_secondary_ratio",
        type=float,
        default=0.04,
        help="[post] 次峰相对主峰动态最小比例（与 post_newtest --min_secondary_ratio 一致）",
    )
    parser.add_argument(
        "--post_noise_barrier_ratio",
        type=float,
        default=0.45,
        help="[post] 噪声阻碍系数，略降有利于弱次峰通过 sec_min_h",
    )
    parser.add_argument(
        "--post_secondary_roi_global_gate_relax_frac",
        type=float,
        default=0.055,
        help="[post] ROI 次峰全局门槛放宽系数，与 post_newtest --secondary_roi_global_gate_relax_frac 一致",
    )
    parser.add_argument(
        "--post_edge_max_span_min",
        type=float,
        default=0.24,
        help="[post] 峰顶单侧估计截停阈值时的最大 RT 跨度(min)，略收紧默认",
    )
    parser.add_argument(
        "--post_edge_noise_percentile",
        type=float,
        default=55.0,
        help="[post] 单侧低噪声分位数，越高→截停阈值越高→边界外推越短（默认 55，抑制区间外扩）",
    )
    parser.add_argument("--post_small_boundary_pad", type=float, default=0.08)
    parser.add_argument(
        "--post_boundary_posterior_lookahead",
        type=int,
        default=0,
        help="[post] 边界外推后验窗口点数；0=仅首点阈值，通常比 5 更少外扩",
    )
    parser.add_argument(
        "--post_boundary_posterior_mean_scale",
        type=float,
        default=1.25,
        help="[post] 后验均值相对阈值的倍数上限（lookahead>0 时生效）",
    )
    parser.add_argument(
        "--post_disable_valley_fallback",
        action="store_true",
        help="默认启用谷值回退（与常用 post_newtest 命令一致）；传入此项则关闭",
    )
    parser.add_argument("--post_disable_lr_repredict_on_small_fail", action="store_true")

    parser.add_argument("--post_min_confidence", type=float, default=0.99)
    parser.add_argument("--post_min_snr", type=float, default=3.0)
    parser.add_argument("--post_small_noise_window_half", type=float, default=0.30)
    parser.add_argument("--post_main_boundary_noise_percentile", type=float, default=20.0)
    parser.add_argument("--post_plot_sigma", type=float, default=0.8)
    parser.add_argument("--post_plot_dir_name", type=str, default="refined_plots")
    parser.add_argument(
        "--post_edge_noise_stop_mode",
        type=str,
        default="roi_bottom_decile_mean",
        choices=["roi_bottom_decile_mean", "stable_tail_mean", "low_percentile"],
        help="[post] 边框阈值：roi_bottom_decile_mean=全ROI最低10%%强度均值",
    )
    parser.add_argument(
        "--post_edge_flat_triplet_step_frac",
        type=float,
        default=0.010,
        help="[post] 三连微降早停（相对峰高）；0 关闭",
    )
    parser.add_argument(
        "--post_refine_width_max_expand_vs_pred",
        type=float,
        default=1.08,
        help="[post] 上限：修正框宽≤原始预测宽×倍数（不强行扩框）",
    )
    parser.add_argument(
        "--post_refine_width_max_frac_of_roi",
        type=float,
        default=0.45,
        help="[post] 上限：修正框宽≤ROI窗口×比例",
    )
    parser.add_argument("--post_enable_small_peak_rt_gate",
        action="store_true",
        help="[post] 启用小峰相对主峰的 RT 门控；不显式传入则关闭（允许多峰不按 RT 限制）",
    )

    # ==================== 输出控制 ====================
    parser.add_argument(
        "--no_timing",
        action="store_true",
        help="[pipeline] 不写 pipeline_timing.log / pipeline_timing_runs.jsonl（终端仍打印计时汇总）",
    )
    parser.add_argument(
        "--save_snr_jpeg",
        action="store_true",
        help="[SNR筛选] 生成 筛选保留/筛选剔除/ 下的红框标注 jpeg（默认关闭，省磁盘）",
    )

    # ==================== 日志级别 ====================
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示 INFO 级别日志（默认仅显示 WARNING 及以上）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="仅显示 ERROR 级别日志",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件路径（作为默认参数，CLI 可覆盖；参数外置，仿 train.py --config）",
    )

    # 参数配置外置：手动提取 --config（避免 parse_known_args 触发 required 校验），
    # 加载 JSON 作为默认值，CLI 参数仍可覆盖
    _cfg_path = None
    for _i, _tok in enumerate(sys.argv[1:]):
        if _tok == "--config" and _i + 1 < len(sys.argv[1:]):
            _cfg_path = sys.argv[2 + _i]
            break
        if _tok.startswith("--config="):
            _cfg_path = _tok.split("=", 1)[1]
            break
    if _cfg_path:
        with open(_cfg_path, encoding="utf-8") as _f:
            _cfg = json.load(_f)
        _cfg.pop("config", None)
        _cfg = {_k: _v for _k, _v in _cfg.items() if not _k.startswith("_")}  # 过滤 _comment_* 注释键
        parser.set_defaults(**_cfg)
        print(f"[INFO] 已加载推理配置: {_cfg_path}")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model 必填（命令行或 --config 提供）")

    # ---- 配置运行时日志过滤 ----
    from framework.util.logutil import configure_log_level, install_filter

    if args.quiet:
        configure_log_level("ERROR")
    elif args.verbose:
        configure_log_level("INFO")
    # 否则保持默认 WARNING（抑制 [INFO] 行）
    install_filter()

    if args.mode == "pipeline":
        from .predictor import main as newtest_main
        from postprocessing.snr_filter import run as snr_pipeline_run
        from postprocessing import peak_refinement
        from utils.torch_device import resolve_torch_device

        from preprocessing.xic_extraction import extract_xic_with_pyopenms

        print("=" * 60)
        resolve_torch_device(verbose=True)
        print("=" * 60)

        pipeline_t0 = time.perf_counter()
        stage_seconds = {}
        stage_intervals = {}
        per_sample_seconds = []
        resource_monitor = _PipelineResourceMonitor()
        resource_monitor.start()
        print(
            "[INFO] 全流程计时与资源监控已启用，运行结束后输出 PIPELINE TIMING 汇总（含 CPU/内存）；"
            "建议 pip install psutil 以获得完整 CPU 与系统内存统计"
        )

        # base output layout (one run per invocation)
        base_out = Path(args.output_dir) if args.output_dir else Path("results/full_pipeline")
        base_out.mkdir(parents=True, exist_ok=True)
        roi_root = base_out / "xic-roi-batch"
        pred_root = base_out / "batch_predictions"
        snr_root = base_out / "snr_filtered"
        roi_root.mkdir(parents=True, exist_ok=True)
        pred_root.mkdir(parents=True, exist_ok=True)
        snr_root.mkdir(parents=True, exist_ok=True)

        if args.standard_refs_csv:
            print(
                "[INFO] --standard_refs_csv 已弃用：mzML ROI 以各通道色谱最高峰（smooth_sigma 平滑后）RT 为中心。"
            )

        # 1) Collect input mzML files (single file, or directory scanned recursively)
        mzml_inputs = _collect_mzml_inputs(args.mzml, args.batch_dir)
        mzml_files = [p for p, _ in mzml_inputs]

        # ---- 简明启动信息 ----
        print("=" * 64)
        print(f"MRMPFormer 推理 | {args.mode}")
        print("-" * 64)
        print(f"模型   : {args.model} | 置信度阈值 {args.threshold} | 平滑 sigma {args.smooth_sigma}")
        if len(mzml_files) == 1:
            print(f"输入   : {mzml_files[0].name}")
        else:
            print(f"输入   : {len(mzml_files)} 个 mzML ({args.mzml or args.batch_dir}，含子目录递归)")
        print(f"输出   : {base_out}/")
        print(f"QC     : 强度>={args.pipeline_min_max_intensity:g} | 点数>={args.pipeline_min_chrom_points}"
              f" | SNR>={args.snr_min:g}")
        print("=" * 64)

        # 2) Generate ROI (testXIC)
        def _pipeline_qc_kwargs():
            return dict(
                min_chrom_points=int(max(0, args.pipeline_min_chrom_points)),
                min_max_intensity=float(max(0.0, args.pipeline_min_max_intensity)),
            )

        t_roi = time.perf_counter()
        mzml_roi_stats = []
        qc_kw = _pipeline_qc_kwargs()
        for mzml_path, key in mzml_inputs:
            out_dir = roi_root / key
            out_dir.mkdir(parents=True, exist_ok=True)
            st = extract_xic_with_pyopenms(
                str(mzml_path),
                str(out_dir),
                smooth_sigma=args.smooth_sigma,
                **qc_kw,
            )
            if st:
                mzml_roi_stats.append({"stem": key, **st})
        _print_mzml_roi_stats_summary(mzml_roi_stats)
        t_roi_end = time.perf_counter()
        stage_seconds["1_ROI生成(testXIC)"] = t_roi_end - t_roi
        stage_intervals["1_ROI生成(testXIC)"] = (t_roi, t_roi_end)

        # 3) Run MRMPFormer (newtest) in batch mode（batch_dir 下逐子目录预测，单样品同样适用）
        integration_method = getattr(args, "integration_method", "linear")
        pred_basename = (
            "prediction.csv"
            if integration_method == "linear"
            else f"prediction_{integration_method}.csv"
        )
        a = argparse.Namespace(
            images_path=None,
            batch_dir=str(roi_root),
            batch_output=str(pred_root),
            model=args.model,
            feature=None,
            prediction_output=str(pred_root / pred_basename),
            threshold=args.threshold,
            plot=bool(args.plot),
            plot_dir="predicted_plots",
            baseline_correction=False,
            integration_method=integration_method,
            baseline_json=None,
            verbose=False,
        )
        t_pred = time.perf_counter()
        newtest_main(a)
        t_pred_end = time.perf_counter()
        stage_seconds["2_模型预测(newtest)"] = t_pred_end - t_pred
        stage_intervals["2_模型预测(newtest)"] = (t_pred, t_pred_end)

        # 4) Per-sample: SNR filter -> post_newtest
        snr_intervals = []
        post_intervals = []

        def _count_csv_rows(p):
            try:
                return len(pd.read_csv(p))
            except Exception:
                return -1

        _roi_by_stem = {str(s.get("stem")): s for s in (mzml_roi_stats or [])}

        for mzml_path, key in mzml_inputs:
            stem = key
            sample_t0 = time.perf_counter()
            pred_csv = pred_root / stem / pred_basename
            roi_windows_csv = roi_root / stem / "roi_windows.csv"
            if not pred_csv.is_file():
                print(f"[WARN] Skip {stem}: missing prediction csv: {pred_csv}")
                continue
            if not roi_windows_csv.is_file():
                print(f"[WARN] Skip {stem}: missing roi_windows.csv: {roi_windows_csv}")
                continue

            sample_snr_parent = snr_root / stem
            sample_snr_parent.mkdir(parents=True, exist_ok=True)

            t_snr = time.perf_counter()
            snr_pipeline_run(
                mzml_path=str(mzml_path),
                prediction_csv=str(pred_csv),
                output_dir=str(sample_snr_parent),
                min_snr_eff=float(args.snr_min),
                min_snr_cli=float(args.snr_min),
                roi_windows_csv=str(roi_windows_csv),
                smooth_sigma=float(args.snr_gaussian_sigma),
                min_noise_pts=int(args.snr_min_noise_points),
                min_chrom_points=int(max(0, args.pipeline_min_chrom_points)),
                min_chrom_max_intensity=float(max(0.0, args.pipeline_min_max_intensity)),
                save_jpeg=bool(args.save_snr_jpeg),
            )
            snr_sec = time.perf_counter() - t_snr
            snr_intervals.append((t_snr, time.perf_counter()))

            # locate SNR_box_<thr>/ directory generated by mzml_box_outside_snr_pipeline.py
            t = float(args.snr_min)
            if t < 0:
                snr_run_dir_name = "SNR_box_all"
            else:
                if abs(t - round(t)) < 1e-9:
                    snr_run_dir_name = f"SNR_box_{int(round(t))}"
                else:
                    snr_run_dir_name = f"SNR_box_{(t):.10g}"

            snr_run_dir = sample_snr_parent / snr_run_dir_name
            refined_root_dir = snr_run_dir  # post_newtest writes inside the same folder
            if not (refined_root_dir / "prediction.csv").is_file():
                print(f"[WARN] Skip post_newtest for {stem}: missing {refined_root_dir/'prediction.csv'}")
                per_sample_seconds.append(
                    {"stem": stem, "snr": snr_sec, "post": 0.0, "total": time.perf_counter() - sample_t0}
                )
                continue

            post_parser = peak_refinement.build_parser()
            xic_dir_for_post = str(roi_root / stem) if (roi_root / stem).is_dir() else ""
            post_cli = [
                "post_newtest",
                "--results_dir",
                str(refined_root_dir),
                "--xic_dir",
                xic_dir_for_post,
                "--output_name",
                str(args.post_output_name),
                "--small_peak_rt_tol",
                str(args.post_small_peak_rt_tol),
                "--min_confidence",
                str(args.post_min_confidence),
                "--min_snr",
                str(args.post_min_snr),
                "--min_secondary_ratio",
                str(args.post_min_secondary_ratio),
                "--noise_barrier_ratio",
                str(args.post_noise_barrier_ratio),
                "--secondary_roi_global_gate_relax_frac",
                str(args.post_secondary_roi_global_gate_relax_frac),
                "--small_noise_window_half",
                str(args.post_small_noise_window_half),
                "--main_boundary_noise_percentile",
                str(args.post_main_boundary_noise_percentile),
                "--edge_max_span_min",
                str(args.post_edge_max_span_min),
                "--edge_noise_percentile",
                str(args.post_edge_noise_percentile),
                "--small_boundary_pad",
                str(args.post_small_boundary_pad),
                "--boundary_posterior_lookahead",
                str(args.post_boundary_posterior_lookahead),
                "--boundary_posterior_mean_scale",
                str(args.post_boundary_posterior_mean_scale),
                "--edge_noise_stop_mode",
                str(args.post_edge_noise_stop_mode),
                "--refine_width_max_expand_vs_pred",
                str(args.post_refine_width_max_expand_vs_pred),
                "--refine_width_max_frac_of_roi",
                str(args.post_refine_width_max_frac_of_roi),
                "--edge_flat_triplet_step_frac",
                str(args.post_edge_flat_triplet_step_frac),
            ]
            if bool(args.plot):
                post_cli.extend(
                    [
                        "--plot",
                        "--plot_sigma",
                        str(args.post_plot_sigma),
                        "--plot_dir_name",
                        str(args.post_plot_dir_name),
                    ]
                )
            if not args.post_disable_valley_fallback:
                post_cli.append("--enable_valley_fallback")
            if args.post_disable_lr_repredict_on_small_fail:
                post_cli.append("--disable_lr_repredict_on_small_fail")
            if bool(getattr(args, "post_enable_small_peak_rt_gate", False)):
                post_cli.append("--enable_small_peak_rt_gate")

            post_args = post_parser.parse_args(post_cli)
            t_post = time.perf_counter()
            peak_refinement.run_post_newtest(post_args)
            post_sec = time.perf_counter() - t_post
            post_intervals.append((t_post, time.perf_counter()))
            per_sample_seconds.append(
                {
                    "stem": stem,
                    "snr": snr_sec,
                    "post": post_sec,
                    "total": time.perf_counter() - sample_t0,
                }
            )
            # ---- 每样品结论行 ----
            _n_roi = int(_roi_by_stem.get(stem, {}).get("n_roi_images", -1))
            print(f"[样品] {stem}: ROI {_n_roi} 张 | 检出 {_count_csv_rows(pred_csv)}"
                  f" | SNR 保留 {_count_csv_rows(refined_root_dir / 'prediction.csv')}"
                  f" | 精修输出 {_count_csv_rows(refined_root_dir / args.post_output_name)}"
                  f" | {per_sample_seconds[-1]['total']:.1f}s")

        stage_seconds["3_SNR筛选(全部样品)"] = sum(p.get("snr", 0.0) for p in per_sample_seconds)
        stage_seconds["4_框修正post_newtest(全部)"] = sum(p.get("post", 0.0) for p in per_sample_seconds)
        if snr_intervals:
            stage_intervals["3_SNR筛选(全部样品)"] = snr_intervals
        if post_intervals:
            stage_intervals["4_框修正post_newtest(全部)"] = post_intervals
        accounted = sum(stage_seconds.values())
        resource_monitor.stop()
        total_sec = time.perf_counter() - pipeline_t0
        if total_sec > accounted:
            stage_seconds["5_其它(跳过/间隙)"] = total_sec - accounted
        print("=" * 64)
        print(f"[推理完成] {len(mzml_files)} 个样品 | 总耗时 {_format_elapsed(total_sec)} | 输出 {base_out}/")
        print("=" * 64)
        _print_pipeline_timing_summary(
            mode_label=str(args.mode),
            n_samples=len(mzml_files),
            stage_seconds=stage_seconds,
            per_sample_seconds=per_sample_seconds,
            total_seconds=total_sec,
            resource_monitor=resource_monitor,
            stage_intervals=stage_intervals,
            pipeline_t0=pipeline_t0,
            mzml_roi_stats=mzml_roi_stats,
            log_dir=None if bool(args.no_timing) else base_out,
        )
        return

    if args.mode == "roi":
        from preprocessing.xic_extraction import extract_xic_with_pyopenms, generate_prediction_plots
        mzml_inputs = _collect_mzml_inputs(args.mzml, args.batch_dir)
        out_base = Path(args.output_dir) if args.output_dir else Path("xic-roi-batch")
        for mzml_path, key in mzml_inputs:
            out_dir = out_base / key
            out_dir.mkdir(parents=True, exist_ok=True)
            extract_xic_with_pyopenms(str(mzml_path), str(out_dir), smooth_sigma=args.smooth_sigma)
        if args.model and args.plot:
            for _, key in mzml_inputs:
                generate_prediction_plots(str(out_base / key), args.model, args.threshold)
        return

    if args.mode == "batch_dir":
        from .predictor import main as newtest_main
        import argparse as ap
        if not args.batch_dir or not os.path.isdir(args.batch_dir):
            print("[ERROR] --batch_dir 必填且需为目录", file=sys.stderr)
            sys.exit(1)
        a = ap.Namespace(images_path=None, batch_dir=args.batch_dir, batch_output=args.output_dir or "results/batch_predictions",
                         model=args.model, feature=None, prediction_output="results/prediction.csv",
                         threshold=args.threshold, plot=args.plot, plot_dir="predicted_plots",
                         baseline_correction=False, integration_method="linear", baseline_json=None, verbose=False)
        newtest_main(a)
        return


if __name__ == "__main__":
    main_cli()
