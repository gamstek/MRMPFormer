# -*- coding: utf-8 -*-
"""
MRMPFormer 统一推理入口（3 种模式；roi / pipeline 均支持单文件与目录递归扫描）。

ROI 生成一律标注驱动（B 范式，与训练数据生成同一路径）：roi / pipeline 模式必须提供
--labels，仅标注命中通道生成 ROI、窗口中心=标注 rt；不再支持 apex（最高强度点）通道驱动。

用法（须在 model/ 目录下运行）:
  # 仅 ROI 生成（单文件或目录递归；每个 mzML 输出到 <output_dir>/<文件名stem>/）；无需 --model
  python -m inference.cli --mode roi --mzml ../data/test/mzml/sample.mzML --labels ../data/label/<实验>.xlsx
  python -m inference.cli --mode roi --batch_dir ../data/test/mzml --labels ../data/label/<实验>.xlsx

  # 想看预测框标注：先 roi 生成 ROI，再 roi2inference --plot 对已有 ROI 目录画图
  python -m inference.cli --mode roi --batch_dir ../data/test/mzml --labels ../data/label/<实验>.xlsx
  python -m inference.cli --mode roi2inference --model checkpoint/quanformer.pth --batch_dir ../output/inference/xic-roi-batch --plot

  # 对已有 ROI 目录批量预测+积分
  python -m inference.cli --mode roi2inference --model checkpoint/quanformer.pth --batch_dir ../output/inference/xic-roi-batch

  # 完整管线（ROI → 预测 → SNR 筛选 → 精修，单文件或目录递归）
  python -m inference.cli --mode pipeline --model checkpoint/quanformer.pth --batch_dir ../data/test/mzml --labels ../data/label/<实验>.xlsx

  # 目录递归时不同子目录出现同名 mzML：输出目录自动加路径前缀（如 子目录A__样本1）避免覆盖
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
        print("%-28s %10s %10s %10s" % ("mzML 样本", "色谱条数", "ROI图", "QC剔除"))
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
        print("（%d 个样本，省略逐条；平均每样本 ROI 图 %.1f 张）" % (len(mzml_roi_stats), avg_roi))


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
    add("样本数: %d" % int(n_samples))
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
            "按样本 SNR+后处理: 合计 %s (SNR %s + post %s)"
            % (_format_elapsed(sample_sum), _format_elapsed(snr_sum), _format_elapsed(post_sum))
        )
        if len(per_sample_seconds) <= 20:
            add("-" * 60)
            add("%-24s %22s %22s %22s" % ("样本", "SNR", "post", "合计"))
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
                "（%d 个样本，省略逐条；单样本平均 SNR %s, post %s）"
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
        "stage_schema": "v2",  # 2026-08-19：阶段键名去 legacy（testXIC→xic_extraction 等）；v1 为旧键名
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


def _resolve_exp_name(args) -> str:
    """确定实验名（Step 7）：--exp_name 优先；缺省回退：单 mzML → 文件名 stem；
    目录输入 → 目录名；再兜底 UTC 时间戳。"""
    if getattr(args, "exp_name", None):
        return str(args.exp_name).strip()
    import datetime
    for _a in (getattr(args, "mzml", None), getattr(args, "batch_dir", None)):
        if _a:
            _p = Path(_a)
            if _p.is_file():
                return _p.stem
            if _p.is_dir():
                return _p.name
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _collect_mzml_inputs(mzml_arg, batch_dir_arg):
    """收集输入 mzML：--mzml 单文件/目录，或 --batch_dir 目录（递归含子目录）。

    返回 [(Path, key), ...]；key 默认为文件名 stem，目录递归下不同子目录出现同名 stem 时，
    key 改为相对扫描根目录的路径展平（如 ``子目录A__样本1``），避免输出目录互相覆盖。
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


def _prepare_label_driven_roi(labels_path, qc_label_rt_tol):
    """ROI 标注驱动（B 范式）必经准备：解析标注 xlsx + RT 一致性 QC。

    返回 (labels, label_qc_rows, exclude_native_ids, n_excl, n_review)。"""
    from preprocessing.coco_annotation import parse_labels_xlsx, label_key
    from preprocessing.label_qc import check_label_rt_consistency

    labels = parse_labels_xlsx(labels_path)
    print(f"[INFO] 标注 xlsx: {len(labels)} 行")
    label_qc_rows, _exclude_keys = check_label_rt_consistency(labels, tol=qc_label_rt_tol)
    exclude_native_ids = {}
    for r in label_qc_rows:
        if r.get("action") == "excluded":
            kid = label_key(r.get("compound"), r.get("channel"))
            if kid:
                exclude_native_ids.setdefault(kid, "label_rt_" + str(r.get("check_type", "")))
    n_excl = len(exclude_native_ids)
    n_review = sum(1 for r in label_qc_rows if r.get("suggest_review"))
    print(f"[INFO] 标注 QC: 检查 {len(label_qc_rows)} 项，剔除通道 {n_excl} 个"
          f"（{n_review} 项需人工复核，详见 QC 表）")
    return labels, label_qc_rows, exclude_native_ids, n_excl, n_review


def _group_labels_by_sample(labels):
    """按 sample_id 分组（保持出现顺序），返回 ({sid: [rows]}, [sid, ...])。"""
    _by, _order = {}, []
    for rec in labels:
        _sid = (rec.get("sample_id") or "").strip()
        if _sid not in _by:
            _by[_sid] = []
            _order.append(_sid)
        _by[_sid].append(rec)
    return _by, _order


def _pick_sample_labels(labels, groups, key, mzml_name, mzml_idx):
    """多样本标注 → 当前 mzML 对应行（与 coco_annotation 同规则；匹配失败回退全部标注）。"""
    _by, _order = groups
    if key in _by:
        return _by[key]
    if mzml_name in _by:
        return _by[mzml_name]
    if len(_order) == 1:
        return _by[_order[0]]
    if mzml_idx < len(_order):
        return _by[_order[mzml_idx]]
    print(f"[WARN] mzML「{key}」未匹配到标注样本({list(_order)})，回退用全部标注")
    return labels


_QC_POST_GATE_COLS = [
    "image", "compound_name", "mz", "q3", "rt_min", "rt_max", "rt_peak",
    "score_ai", "main_score_ai", "main_snr", "main_interval_width",
    "main_conf_composite", "main_conf_final", "small_conf_final", "final_conf_best",
    "need_manual_review", "keep_small_by_standard", "small_rt_gate_pass",
    "small_skew_gate_pass", "small_ai_gate_pass", "gate_ok_for_adjustment",
    "has_secondary_gate", "main_refine_applied", "small_recover_trigger",
]


def _merge_qc_csvs(glob_result, out_path):
    """按 glob 汇总多个同构 QC csv（各样本子目录），加 stem 列；返回 (文件数, 合并行数)。"""
    frames = []
    for p in glob_result:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] 读取 QC 表失败: {p}: {e}")
            continue
        df.insert(0, "stem", p.parent.name)
        frames.append(df)
    if not frames:
        return 0, 0
    merged = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    return len(frames), len(merged)


def _emit_qc5_sample(refined_csv, sample_dir):
    """从样本的 prediction_refined.csv 抽门控列子集，落盘 qc5_refined_<样本名>.csv（防线5 样本级 QC）。

    返回写盘路径（无可用门控列时仍保留首列 + 尽力列）。
    """
    try:
        df = pd.read_csv(refined_csv)
    except Exception as e:
        print(f"[WARN] 读取精修输出失败 {refined_csv}: {e}")
        return None
    if df.empty:
        return None
    keep = [c for c in _QC_POST_GATE_COLS if c in df.columns]
    df_sub = df[keep].copy() if keep else df.iloc[:, :1].copy()
    out_path = sample_dir / ("qc5_refined_%s.csv" % sample_dir.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def _collect_qc_tables(base_out, qc_root, label_qc_rows=None, n_label_excl=0, n_label_review=0,
                       post_output_name="prediction_refined.csv"):
    """pipeline 结束后统一汇总各环节 QC 表到 ../output/QC/<run_name>/。

    P2 覆盖：qc1_label_rt.csv（标注 RT 一致性，调用方已先写入）+ qc2_roi.csv + qc_summary.md。
    P3 覆盖：qc3_threshold.csv（② 预测阈值丢弃）+ qc4_snr.csv（④ SNR 逐框）
    + qc5_refined.csv（⑤ 精修门控列，样本级 qc5_refined_<样本名>.csv 合并）+ qc_summary.md 增补。
    """
    import datetime

    qc_root.mkdir(parents=True, exist_ok=True)

    # 1) ROI 通道级剔除：汇总各样本 pipeline_qc_excluded.csv → qc2_roi.csv（防线2）
    n_sample_excl, n_roi_excl = _merge_qc_csvs(
        sorted((base_out / "xic_roi").glob("*/pipeline_qc_excluded.csv")),
        qc_root / "qc2_roi.csv",
    )
    if n_roi_excl:
        print(f"[INFO] QC 汇总: ROI 通道级剔除 -> {qc_root / 'qc2_roi.csv'}（{n_roi_excl} 行）")

    # 2) 预测阈值丢弃：汇总各样本 qc3_threshold_*.csv → qc3_threshold.csv（防线3）
    n_pred_files, n_pred_imgs = _merge_qc_csvs(
        sorted((base_out / "predictions_model").glob("*/qc3_threshold_*.csv")),
        qc_root / "qc3_threshold.csv",
    )
    n_pred_dropped = 0
    if n_pred_imgs:
        pred_merged = pd.read_csv(qc_root / "qc3_threshold.csv")
        n_pred_dropped = int(pred_merged.get("n_dropped", pd.Series(dtype=int)).sum())
        print(f"[INFO] QC 汇总: 预测阈值 -> {qc_root / 'qc3_threshold.csv'}（{n_pred_imgs} 图，丢弃 {n_pred_dropped} 框）")

    # 3) SNR 框级：汇总各样本 qc4_snr_*.csv → qc4_snr.csv（防线4）
    n_snr_files, n_snr_rows = _merge_qc_csvs(
        sorted((base_out / "prediction_refined").glob("*/qc4_snr_*.csv")),
        qc_root / "qc4_snr.csv",
    )
    n_snr_passed = 0
    if n_snr_rows:
        snr_merged = pd.read_csv(qc_root / "qc4_snr.csv")
        n_snr_passed = int(snr_merged.get("passed_snr_threshold", pd.Series(dtype=int)).sum())
        print(f"[INFO] QC 汇总: SNR 框级 -> {qc_root / 'qc4_snr.csv'}（{n_snr_rows} 框，通过 {n_snr_passed}）")

    # 4) 精修框级：汇总各样本 qc5_refined_*.csv → qc5_refined.csv（防线5，样本级由每样本 post 结束后落盘）
    n_post_files, n_post_rows, n_post_review = 0, 0, 0
    post_frames = []
    for p in sorted((base_out / "prediction_refined").glob("*/qc5_refined_*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] 读取精修 QC 失败: {p}: {e}")
            continue
        df.insert(0, "stem", p.parent.name)
        post_frames.append(df)
        n_post_files += 1
        n_post_rows += len(df)
        if "need_manual_review" in df.columns:
            n_post_review += int(df["need_manual_review"].fillna(0).sum())
    if post_frames:
        post_merged = pd.concat(post_frames, ignore_index=True)
        post_path = qc_root / "qc5_refined.csv"
        post_merged.to_csv(post_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] QC 汇总: 精修框级 -> {post_path}（{n_post_rows} 行，需人工复核 {n_post_review}）")

    # 5) qc_summary.md
    lines = [
        "# QC 汇总 — %s" % qc_root.name,
        "",
        "- 生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "- 来源: pipeline 推理（output_dir=%s）" % base_out,
        "",
        "## 1. 标注 RT 一致性（qc1_label_rt.csv）",
        "- 检查项: %d" % len(label_qc_rows or []),
        "- 剔除通道: %d" % n_label_excl,
        "- 需人工复核: %d" % n_label_review,
        "",
        "## 2. ROI 通道级剔除（qc2_roi.csv）",
        "- 涉及样本数: %d" % n_sample_excl,
        "- 剔除条目: %d 行" % n_roi_excl,
        "",
        "## 3. 预测阈值（qc3_threshold.csv）",
        "- 涉及样本数: %d" % n_pred_files,
        "- ROI 图: %d" % n_pred_imgs,
        "- 阈值丢弃框: %d" % n_pred_dropped,
        "",
        "## 4. SNR 框级（qc4_snr.csv）",
        "- 涉及样本数: %d" % n_snr_files,
        "- 总框: %d" % n_snr_rows,
        "- 通过 SNR: %d" % n_snr_passed,
        "- 剔除: %d" % (n_snr_rows - n_snr_passed),
        "",
        "## 5. 精修框级（qc5_refined.csv）",
        "- 涉及样本数: %d" % n_post_files,
        "- 总行: %d" % n_post_rows,
        "- 需人工复核（final_conf 低于阈值）: %d" % n_post_review,
        "",
        "## 6. 人工复核清单（标注 RT 一致性）",
    ]
    if (qc_root / "qc2_roi.csv").is_file():
        reason_counts = pd.read_csv(qc_root / "qc2_roi.csv")["reason"].value_counts()
        _reason_line = "- reason 分布: " + ", ".join(f"{k}={v}" for k, v in reason_counts.items())
        _idx3 = lines.index("## 3. 预测阈值（qc3_threshold.csv）")
        if lines[_idx3 - 1] == "":
            lines.insert(_idx3 - 1, _reason_line)
        else:
            lines.insert(_idx3, _reason_line)
    review_rows = [r for r in (label_qc_rows or []) if r.get("suggest_review")]
    if review_rows:
        lines += ["| 检查类型 | 样本 | 化合物 | 通道 | RT(min) | 组中位 | 极差(min) |", "|---|---|---|---|---|---|---|"]
        for r in review_rows:
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r.get("check_type", ""), r.get("sample_id", ""), r.get("compound", ""),
                r.get("channel", ""), r.get("rt", ""), r.get("group_median", ""),
                r.get("rt_range", "")))
    else:
        lines.append("（无，未触发人工复核）")
    lines += [
        "",
        "## 7. 说明",
        "- 各环节 QC 表：标注 RT 一致性 / ROI 通道级 / 预测阈值 / SNR 框级 / 精修框级（P2+P3 已全部接入）。",
    ]
    summary_path = qc_root / "qc_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] QC 汇总: 报告 -> {summary_path}")

    # ---- 全防线人工预警 qc_alert.md（覆盖阶段①的防线1-only 版本）----
    _write_full_qc_alert(
        base_out=base_out, qc_root=qc_root,
        label_qc_rows=label_qc_rows,
        n_roi_excl=n_roi_excl, n_pred_dropped=n_pred_dropped,
        n_snr_rows=n_snr_rows, n_snr_passed=n_snr_passed,
        n_post_review=n_post_review,
    )


def _write_full_qc_alert(base_out, qc_root, label_qc_rows=None, n_roi_excl=0,
                         n_pred_dropped=0, n_snr_rows=0, n_snr_passed=0, n_post_review=0):
    """生成 qc_alert.md：汇总全部 QC 防线的人工预警（防线1-5），供人工一站式复核。

    防线1 明细来自标注 RT 一致性检查行；防线5 明细来自 qc5_refined.csv 的
    need_manual_review=1 行；防线2/3/4 给出剔除/丢弃统计与关注点提示。
    """
    import datetime

    excl1 = [r for r in (label_qc_rows or []) if r.get("action") == "excluded"]
    rev1 = [r for r in (label_qc_rows or []) if r.get("suggest_review")]
    n_alert_total = len(excl1) + int(n_roi_excl > 0) + n_post_review

    lines = [
        "# QC 人工预警报告（全防线汇总）",
        "",
        "- 来源: pipeline 推理（output_dir=%s）" % base_out,
        "- 生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "- 防线: ①标注RT ②ROI通道 ③预测阈值 ④SNR框 ⑤精修框",
        "",
        "## 一、防线1 标注 RT 一致性（qc1_label_rt.csv）",
        "- 剔除标注: %d 行 | 需人工复核: %d 项" % (len(excl1), len(rev1)),
        "",
    ]
    if rev1:
        lines += ["| 检查类型 | 样本 | 化合物 | 通道 | RT(min) | 组中位 | 极差(min) | 处置 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rev1:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r.get("check_type", ""), r.get("sample_id", ""), r.get("compound", ""),
                r.get("channel", ""), r.get("rt", ""), r.get("group_median", ""),
                r.get("rt_range", ""), r.get("action", "")))
        lines.append("")
    else:
        lines += ["（无）", ""]

    lines += ["## 二、防线2 ROI 通道级（qc2_roi.csv）",
              "- 剔除条目: %d 行（reason 分布见 qc2_roi.csv / qc_summary.md）" % n_roi_excl,
              "",
              "## 三、防线3 预测阈值（qc3_threshold.csv）",
              "- 阈值丢弃框: %d 个" % n_pred_dropped,
              "- 关注点: 若某图 n_queries 全部被丢弃，多为低置信度/模糊峰，建议抽查对应 ROI。",
              "",
              "## 四、防线4 SNR 框级（qc4_snr.csv）",
              "- 总框: %d | 通过 SNR: %d | 剔除: %d" % (n_snr_rows, n_snr_passed, n_snr_rows - n_snr_passed),
              "- 关注点: 空白/低浓度样品整批剔除属正常；加标样品大量剔除需复核 SNR 阈值设置。",
              "",
              "## 五、防线5 精修框级（qc5_refined.csv）",
              "- 需人工复核（final_conf 低于阈值）: %d 项" % n_post_review,
              "",
    ]
    if n_post_review and (qc_root / "qc5_refined.csv").is_file():
        try:
            q5 = pd.read_csv(qc_root / "qc5_refined.csv")
            q5_rev = q5[q5.get("need_manual_review", pd.Series(dtype=int)).fillna(0) == 1]
            if not q5_rev.empty:
                cols = [c for c in ("stem", "image", "main_rt_peak", "main_score_ai",
                                    "main_conf_final", "final_conf_best", "need_manual_review")
                        if c in q5_rev.columns]
                lines += ["| %s |" % " | ".join(cols), "|" + "---|" * len(cols)]
                for _, r in q5_rev.iterrows():
                    lines.append("| %s |" % " | ".join(str(r.get(c, "")) for c in cols))
                lines.append("")
        except Exception as e:
            print(f"[WARN] 读取 qc5_refined.csv 生成预警明细失败: {e}")
    else:
        lines += ["（无）", ""]

    if n_alert_total == 0:
        lines += ["**结论: 未发现需人工干预的告警。**", ""]
    else:
        lines += ["**结论: 发现 %d 项告警，请按上表逐项复核。**" % n_alert_total, ""]
    lines += ["## 处置指引",
              "- 防线1 剔除行已生效于本轮（不生成 ROI）；复核修正标注后需重建数据并重跑；",
              "- 防线5 需复核峰请人工查看对应 refined_plots 图后决定是否保留；",
              "- 各防线明细与汇总见 QC 根目录 qc1_label_rt / qc2_roi / qc3_threshold / qc4_snr / qc5_refined。",
              ""]
    alert_path = qc_root / "qc_alert.md"
    alert_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] QC 预警(全防线): {alert_path}（告警 {n_alert_total} 项）")


def _write_predictions_model_summary(base_out):
    """需求1：predictions_model 根级汇总——predictions_model_all.csv + predictions_model_report.md（放阶段文件夹内）。"""
    import datetime

    pred_root = base_out / "predictions_model"
    csv_files = sorted(pred_root.glob("*/model_prediction_*.csv")) if pred_root.is_dir() else []
    if not csv_files:
        return
    frames = []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] 读取预测明细失败 {p}: {e}")
            continue
        df.insert(0, "stem", p.parent.name)
        frames.append(df)
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    all_csv = pred_root / "predictions_model_all.csv"
    merged.to_csv(all_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# 模型推理输出汇总（predictions_model）",
        "",
        "- 生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "- 样本数: %d | 检出框总数: %d" % (len(frames), len(merged)),
        "- 明细: %s" % all_csv.name,
        "",
        "| 样本 | 图数 | 检出峰数 | 最高置信度 | 平均SNR | 平均面积 |",
        "|---|---|---|---|---|---|",
    ]
    for stem, g in merged.groupby("stem"):
        n_img = int(g["image"].nunique()) if "image" in g.columns else len(g)
        n_peak = len(g)
        max_sc = float(g["score"].max()) if "score" in g.columns and g["score"].notna().any() else None
        mean_snr = float(g["snr"].mean()) if "snr" in g.columns and g["snr"].notna().any() else None
        mean_area = float(g["area"].mean()) if "area" in g.columns and g["area"].notna().any() else None
        lines.append("| %s | %d | %d | %s | %s | %s |" % (
            stem, n_img, n_peak,
            "%.3f" % max_sc if max_sc is not None else "—",
            "%.4g" % mean_snr if mean_snr is not None else "—",
            "%.6g" % mean_area if mean_area is not None else "—"))
    report = pred_root / "predictions_model_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] 模型输出汇总: {all_csv} / {report}")


def _write_prediction_refined_summary(base_out):
    """需求5：prediction_refined 根级汇总——prediction_refined_all.csv + prediction_refined_report.md（放阶段文件夹内）。"""
    import datetime

    ref_root = base_out / "prediction_refined"
    if not ref_root.is_dir():
        return
    frames = []
    for p in sorted(ref_root.glob("*/prediction_refined_with_area.csv")):
        frames.append((p.parent.name, p))
    if not frames:
        for p in sorted(ref_root.glob("*/prediction_refined.csv")):
            frames.append((p.parent.name, p))
    if not frames:
        return
    all_rows = []
    for stem, p in frames:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] 读取精修明细失败 {p}: {e}")
            continue
        df.insert(0, "stem", stem)
        all_rows.append(df)
    if not all_rows:
        return
    merged = pd.concat(all_rows, ignore_index=True)
    all_csv = ref_root / "prediction_refined_all.csv"
    merged.to_csv(all_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# 修正后输出汇总（prediction_refined）",
        "",
        "- 生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "- 样本数: %d | 精修峰总数: %d" % (len(all_rows), len(merged)),
        "- 明细: %s" % all_csv.name,
        "",
        "| 样本 | 精修峰数 | 含次峰 | 需人工复核 | 主峰平均置信度 | 面积合计 |",
        "|---|---|---|---|---|---|",
    ]
    for stem, g in merged.groupby("stem"):
        n = len(g)
        n_small = int((g.get("small_count", pd.Series(0)).fillna(0) > 0).sum()) if "small_count" in g.columns else 0
        n_rev = int(g.get("need_manual_review", pd.Series(0)).fillna(0).sum()) if "need_manual_review" in g.columns else 0
        mean_sc = float(g["main_score_ai"].mean()) if "main_score_ai" in g.columns and g["main_score_ai"].notna().any() else None
        area_col = "main_area" if "main_area" in g.columns else ("area" if "area" in g.columns else None)
        tot_area = float(g[area_col].sum()) if area_col and g[area_col].notna().any() else None
        lines.append("| %s | %d | %d | %d | %s | %s |" % (
            stem, n, n_small, n_rev,
            "%.3f" % mean_sc if mean_sc is not None else "—",
            "%.6g" % tot_area if tot_area is not None else "—"))
    report = ref_root / "prediction_refined_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] 修正后输出汇总: {all_csv} / {report}")


def main_cli():
    parser = argparse.ArgumentParser(description="MRMPFormer 统一推理入口（roi / roi2inference / pipeline / fullscan）")
    parser.add_argument("--mode", type=str, default="pipeline",
                        choices=["roi", "roi2inference", "pipeline", "fullscan"],
                        help=(
                            "roi=仅ROI生成(单文件或目录递归); roi2inference=对已有ROI目录批量预测+积分; "
                            "pipeline=完整流水线（ROI->预测->SNR筛选->post_newtest，单文件或目录递归）; "
                            "fullscan=整谱 XIC 全峰识别（全部 transition，不依赖标注）"
                        ))
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径 (.pth)；也可由 --config 提供（roi 模式非必填，其余模式必填）")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--integration_method", type=str, default="linear",
                        choices=["linear", "raw", "external_baseline"])
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出根目录（统一写到 ../output/ 下）；null=按模式默认：roi→../output/inference/xic_roi，"
                             "roi2inference→../output/inference/predictions_model，pipeline→../output/inference/<实验模式>_<实验名>"
                             "（实验名由 --exp_name 或自动回退确定）。测试运行建议指定 ../output/test/<名称> 单独存放")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名（用于输出目录 ../output/inference/<实验模式>_<实验名> 与推理报告 "
                             "inference_report_<实验名>.md）；缺省回退：单 mzML→文件名，目录输入→目录名，再兜底 UTC 时间戳")
    parser.add_argument("--mzml", type=str,
                        help="[roi/pipeline] 单个 mzML 文件路径，或包含 mzML 的目录（递归扫描）")
    parser.add_argument("--batch_dir", type=str,
                        help="[roi/pipeline] mzML 目录（递归扫描）；[roi2inference] testXIC 输出目录")
    parser.add_argument("--plot", action="store_true", help="[pipeline/roi2inference] 生成预测可视化图（roi 模式不支持）")
    parser.add_argument("--plot_style", type=str, default="xic", choices=["xic", "roi"],
                        help="--plot 的图型：xic=XIC 曲线+多 query 阴影+信息框（默认，model_plots/）；"
                             "roi=ROI 原图叠红框（predicted_plots/，功能保留）")

    # ===== Pipeline (QC + post_newtest + SNR) 对齐流程图：低强度/少点数剔除 → 预测 → 框修正/二次峰 → SNR =====
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
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="[roi/pipeline] 必填：人工标注 xlsx（data/label/<实验>.xlsx）。ROI 由标注驱动（B 范式，"
             "与训练数据生成同一路径）：仅标注命中通道生成 ROI、窗口中心=标注 rt；"
             "同时开启标注 RT 一致性 QC（--qc_label_rt_tol），结果写入 output/QC/<run_name>/",
    )
    parser.add_argument(
        "--qc_label_rt_tol",
        type=float,
        default=1.0,
        help="[pipeline] 标注 RT 一致性 QC 阈值（min）：跨样本/双离子 rt 极差超此值判疑似实验有误，"
             "剔除涉事通道并警示人工复核；0=关闭。默认 1.0（需 --labels）",
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

    # ==================== fullscan（整谱全峰识别）====================
    parser.add_argument("--scan_baseline_percentile", type=float, default=25.0,
                        help="[fullscan] 基线分位（global_percentile 模式）")
    parser.add_argument("--scan_baseline_mode", type=str, default="global_percentile",
                        choices=["global_percentile", "local_valley"],
                        help="[fullscan] 基线模式")
    parser.add_argument("--scan_min_peak_ratio", type=float, default=0.04,
                        help="[fullscan] 峰高 = baseline + r·dynamic")
    parser.add_argument("--scan_prominence_ratio", type=float, default=0.055,
                        help="[fullscan] find_peaks prominence（相对 dynamic）")
    parser.add_argument("--scan_min_prominence_abs", type=float, default=0.0,
                        help="[fullscan] prominence 绝对下限（>0 时启用，防小峰被全局 dynamic 吞）")
    parser.add_argument("--scan_min_peak_gap_points", type=int, default=3,
                        help="[fullscan] find_peaks 最小点距")
    parser.add_argument("--scan_min_peak_width_min", type=float, default=0.10,
                        help="[fullscan] RT 尺度最小峰宽（distance 按通道中位步长换算）")
    parser.add_argument("--scan_void_time_min", type=float, default=0.5,
                        help="[fullscan] 排除溶剂前沿/柱平衡区（RT < 此值不参与 dynamic 与枚举）")
    parser.add_argument("--scan_max_peaks_per_channel", type=int, default=50,
                        help="[fullscan] 单通道候选上限，超限保留 prominence 前 N")
    parser.add_argument("--scan_init_half_width_min", type=float, default=0.05,
                        help="[fullscan] 精修初始半宽（min）")
    parser.add_argument("--scan_boundary_posterior_lookahead", type=int, default=5,
                        help="[fullscan] 边界后验窗点数")
    parser.add_argument("--scan_boundary_posterior_mean_scale", type=float, default=1.25,
                        help="[fullscan] 边界后验均值倍数")
    parser.add_argument("--scan_edge_noise_stop_mode", type=str, default="stable_tail_mean",
                        choices=["stable_tail_mean", "roi_bottom_decile_mean", "low_percentile"],
                        help="[fullscan] 边界截停阈值：stable_tail_mean=峰侧局部稳定尾噪声（默认）")
    parser.add_argument("--scan_edge_max_span_min", type=float, default=1.0,
                        help="[fullscan] 边界截停阈值估计的最大单侧跨度（min）")
    parser.add_argument("--scan_min_snr", type=float, default=3.0,
                        help="[fullscan] 峰级本地 SNR 门")
    parser.add_argument("--scan_min_peak_span_points", type=int, default=5,
                        help="[fullscan] 峰跨距（baseline 以上连续点数）门")
    parser.add_argument("--scan_min_area", type=float, default=0.0,
                        help="[fullscan] 峰面积门（0=关）")
    parser.add_argument("--scan_window_half_min", type=float, default=1.0,
                        help="[fullscan] 模型验证窗口半宽（与训练一致）")
    parser.add_argument("--keep_windows", action="store_true",
                        help="[fullscan] 保留模型验证窗口 JPEG")
    parser.add_argument("--no_plots", action="store_true",
                        help="[fullscan] 关闭整谱标注图")

    # ==================== 输出控制 ====================
    parser.add_argument(
        "--no_timing",
        action="store_true",
        help="[pipeline] 不写 pipeline_timing.log / pipeline_timing_runs.jsonl（终端仍打印计时汇总）",
    )
    parser.add_argument(
        "--no_report",
        action="store_true",
        help="[pipeline] 跑完后不自动生成推理报告（默认生成 inference_report_<实验名>.md + all.csv + 补算面积）",
    )
    parser.add_argument(
        "--save_snr_jpeg",
        action="store_true",
        help="[SNR筛选] 生成 snr_kept/snr_dropped/ 下的红框标注 jpeg（默认关闭，省磁盘）",
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
    if not args.model and args.mode not in ("roi", "fullscan"):
        parser.error("--model 必填（命令行或 --config 提供；roi 模式仅生成 ROI，无需模型；"
                     "fullscan 模型可选，提供则开启验证）")
    if args.mode in ("roi", "pipeline") and not args.labels:
        parser.error("--labels 必填：ROI 生成只支持标注驱动（B 范式，与训练一致），"
                     "不再支持 apex（最高强度点）通道驱动")
    if args.mode == "fullscan" and args.labels:
        print("[INFO] fullscan 模式不依赖 --labels，已忽略")

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

        from preprocessing.xic_extraction import extract_xic_with_pyopenms

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
        # Step 7：默认输出目录 ../output/inference/<实验模式>_<实验名>
        exp_name = _resolve_exp_name(args)
        if args.output_dir:
            base_out = Path(args.output_dir)
        else:
            base_out = Path("../output/inference") / ("%s_%s" % (args.mode, exp_name))
        base_out.mkdir(parents=True, exist_ok=True)
        run_name = base_out.name  # 各环节 QC 表统一输出 ../output/QC/<run_name>/
        qc_root = Path("../output/QC") / run_name
        roi_root = base_out / "xic_roi"
        pred_root = base_out / "predictions_model"
        snr_root = base_out / "prediction_refined"
        roi_root.mkdir(parents=True, exist_ok=True)
        pred_root.mkdir(parents=True, exist_ok=True)
        snr_root.mkdir(parents=True, exist_ok=True)

        # ---- 标注驱动 ROI（B 范式，必填 --labels）：解析 + RT 一致性 QC + 按样本分组 ----
        exclude_native_ids = None
        labels = None
        label_qc_rows = []
        n_label_excl = 0
        n_label_review = 0
        labels, label_qc_rows, exclude_native_ids, n_label_excl, n_label_review = (
            _prepare_label_driven_roi(args.labels, args.qc_label_rt_tol)
        )
        if label_qc_rows:
            qc_root.mkdir(parents=True, exist_ok=True)
            from preprocessing.label_qc import write_qc_table, write_qc_alert
            write_qc_table(label_qc_rows, qc_root / "qc1_label_rt.csv")
            _n_alert = write_qc_alert(label_qc_rows, qc_root / "qc_alert.md",
                                      source=f"推理管线 {args.mode}",
                                      tol=args.qc_label_rt_tol)
            if _n_alert:
                print(f"[ALERT] QC 预警: {_n_alert} 行标注未通过 RT 一致性检查（已剔除，"
                      f"不生成 ROI），请人工复核 → {qc_root / 'qc_alert.md'}")

        labels_by_sample = _group_labels_by_sample(labels)

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
        for mzml_idx, (mzml_path, key) in enumerate(mzml_inputs):
            out_dir = roi_root / key
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_labels = _pick_sample_labels(
                labels, labels_by_sample, key, mzml_path.name, mzml_idx)
            st = extract_xic_with_pyopenms(
                str(mzml_path),
                str(out_dir),
                smooth_sigma=args.smooth_sigma,
                exclude_native_ids=exclude_native_ids,
                labels=sample_labels,
                **qc_kw,
            )
            if st:
                mzml_roi_stats.append({"stem": key, **st})
        _print_mzml_roi_stats_summary(mzml_roi_stats)
        t_roi_end = time.perf_counter()
        stage_seconds["1_ROI生成(xic_extraction)"] = t_roi_end - t_roi
        stage_intervals["1_ROI生成(xic_extraction)"] = (t_roi, t_roi_end)

        # 3) Run MRMPFormer (newtest) in batch mode（batch_dir 下逐子目录预测，单样本同样适用）
        integration_method = getattr(args, "integration_method", "linear")
        # Step 3：预测输出改名为 model_prediction_<样本名>.csv（pipeline 内逐样本定位）
        method_suffix = "" if integration_method == "linear" else "_%s" % integration_method
        a = argparse.Namespace(
            images_path=None,
            batch_dir=str(roi_root),
            batch_output=str(pred_root),
            model=args.model,
            feature=None,
            prediction_output=None,  # 批量分支不使用该字段：predictor 以 batch_output/<子目录>/<pred_basename> 落盘
            threshold=args.threshold,
            plot=bool(args.plot),
            plot_style=args.plot_style,
            plot_dir="predicted_plots",
            baseline_correction=False,
            integration_method=integration_method,
            baseline_json=None,
            verbose=False,
        )
        t_pred = time.perf_counter()
        newtest_main(a)
        t_pred_end = time.perf_counter()
        stage_seconds["2_模型预测(predictor)"] = t_pred_end - t_pred
        stage_intervals["2_模型预测(predictor)"] = (t_pred, t_pred_end)

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
            # Step 3：样本内 prediction 表 model_prediction_<样本名>.csv（pipeline 批量分支由 predictor 落盘）
            pred_csv = pred_root / stem / f"model_prediction_{stem}{method_suffix}.csv"
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
            # QC（min_chrom_points / min_max_intensity）仅在阶段① ROI 生成生效；
            # 阶段①已把不达标通道整条排除，阶段③只需按 SNR 复核，无需重复下发 QC 参数
            snr_pipeline_run(
                mzml_path=str(mzml_path),
                prediction_csv=str(pred_csv),
                output_dir=str(sample_snr_parent),
                min_snr_eff=float(args.snr_min),
                min_snr_cli=float(args.snr_min),
                roi_windows_csv=str(roi_windows_csv),
                smooth_sigma=float(args.snr_gaussian_sigma),
                min_noise_pts=int(args.snr_min_noise_points),
                save_jpeg=bool(args.save_snr_jpeg),
            )
            snr_sec = time.perf_counter() - t_snr
            snr_intervals.append((t_snr, time.perf_counter()))

            # Step 2：snr_filter 结果直写样本目录（不再有 SNR_box_<thr>/ 子层）
            refined_root_dir = sample_snr_parent
            if not (refined_root_dir / "prediction_snr.csv").is_file():
                print(f"[WARN] Skip post_newtest for {stem}: missing {refined_root_dir/'prediction_snr.csv'}")
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
            # Step 5：防线5 样本级 QC（精修门控列子集）落盘 qc5_refined_<样本名>.csv
            _qc5_path = _emit_qc5_sample(
                refined_root_dir / args.post_output_name, refined_root_dir)
            if _qc5_path is not None:
                print(f"[INFO] 精修 QC(样本级): {_qc5_path}")
            # ---- 每样本结论行 ----
            _n_roi = int(_roi_by_stem.get(stem, {}).get("n_roi_images", -1))
            print(f"[样本] {stem}: ROI {_n_roi} 张 | 检出 {_count_csv_rows(pred_csv)}"
                  f" | SNR 保留 {_count_csv_rows(refined_root_dir / 'prediction_snr.csv')}"
                  f" | 精修输出 {_count_csv_rows(refined_root_dir / args.post_output_name)}"
                  f" | {per_sample_seconds[-1]['total']:.1f}s")

        stage_seconds["3_SNR筛选(全部样本)"] = sum(p.get("snr", 0.0) for p in per_sample_seconds)
        stage_seconds["4_框修正(peak_refinement)"] = sum(p.get("post", 0.0) for p in per_sample_seconds)
        if snr_intervals:
            stage_intervals["3_SNR筛选(全部样本)"] = snr_intervals
        if post_intervals:
            stage_intervals["4_框修正(peak_refinement)"] = post_intervals
        accounted = sum(stage_seconds.values())
        resource_monitor.stop()
        total_sec = time.perf_counter() - pipeline_t0
        if total_sec > accounted:
            stage_seconds["5_其它(跳过/间隙)"] = total_sec - accounted
        print("=" * 64)
        print(f"[推理完成] {len(mzml_files)} 个样本 | 总耗时 {_format_elapsed(total_sec)} | 输出 {base_out}/")
        print("=" * 64)
        _collect_qc_tables(
            base_out,
            qc_root,
            label_qc_rows=label_qc_rows,
            n_label_excl=n_label_excl,
            n_label_review=n_label_review,
            post_output_name=str(args.post_output_name),
        )
        # Step 7b：阶段级根级汇总（predictions_model_all.csv + report；prediction_refined_all.csv + report）
        _write_predictions_model_summary(base_out)
        _write_prediction_refined_summary(base_out)
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

        # ---- 自动生成推理报告（汇总精修结果 + 补算峰面积；Step 7c 改名 inference_report_<实验名>.md + all.csv）----
        if not bool(getattr(args, "no_report", False)):
            try:
                from tools.evaluation.inference_report import generate_for_pipeline
                _rep = generate_for_pipeline(
                    output_dir=str(base_out),
                    exp_name=exp_name,
                    integration_method="snr",
                    do_integrate=True,
                    verbose=True,
                )
                print(f"[INFO] 推理报告已生成: {_rep['report_md']}")
            except Exception as _e:
                print(f"[WARN] 推理报告生成失败（不影响主流程）: {_e}")
        return

    if args.mode == "fullscan":
        from .fullscan import main as fullscan_main

        fullscan_main(args)
        return

    if args.mode == "roi":
        from preprocessing.xic_extraction import extract_xic_with_pyopenms
        # 标注驱动（B 范式）：--labels 必填（参数校验已强制），与训练/pipeline 同一路径
        labels, _lqc, exclude_native_ids, _n_excl, _n_rev = _prepare_label_driven_roi(
            args.labels, args.qc_label_rt_tol)
        groups = _group_labels_by_sample(labels)
        mzml_inputs = _collect_mzml_inputs(args.mzml, args.batch_dir)
        out_base = Path(args.output_dir) if args.output_dir else Path("../output/inference/xic_roi")
        for mzml_idx, (mzml_path, key) in enumerate(mzml_inputs):
            out_dir = out_base / key
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_labels = _pick_sample_labels(
                labels, groups, key, mzml_path.name, mzml_idx)
            extract_xic_with_pyopenms(
                str(mzml_path), str(out_dir), smooth_sigma=args.smooth_sigma,
                exclude_native_ids=exclude_native_ids, labels=sample_labels)
        return

    if args.mode == "roi2inference":
        from .predictor import main as newtest_main
        import argparse as ap
        if not args.batch_dir or not os.path.isdir(args.batch_dir):
            print("[ERROR] --batch_dir 必填且需为目录", file=sys.stderr)
            sys.exit(1)
        a = ap.Namespace(
            images_path=None,
            batch_dir=args.batch_dir,
            batch_output=args.output_dir or "../output/inference/predictions_model",
            model=args.model,
            feature=None,
            prediction_output=None,  # 批量分支不使用该字段：predictor 以 batch_output/<子目录>/prediction[_<方法>].csv 落盘
            threshold=args.threshold,
            plot=args.plot,
            plot_style=args.plot_style,
            plot_dir="predicted_plots",
            baseline_correction=False,
            integration_method=args.integration_method,  # 透传 CLI 参数，不再硬编码 linear
            baseline_json=None,
            verbose=False,
        )
        newtest_main(a)
        return


if __name__ == "__main__":
    main_cli()
