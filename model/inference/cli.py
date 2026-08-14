# -*- coding: utf-8 -*-
"""
MRMPFormer 统一推理入口：合并 testXIC + newtest，支持单张图模式。

单张图模式：前端一张一张输入输出
- 输入：一张图对应 (rt[], intensity[])，可选基线 (x[], y[])，无需 mz/q3
- 输出：{detections: [{x1,x2,score,area,rt_min,rt_max,...}, ...]}
  一张图可能有两个及以上置信度窗口，detections 为检测框列表（按 score 降序）

用法:
  # 单张图模式（JSON 输入输出）
  python main.py --mode single --model checkpoint/checkpoint0029.pth --input example_single_input.json
  echo '{"rt":[1,2,3],"intensity":[100,500,200]}' | python main.py --mode single --model x.pth

  # 目录下所有 JSON 逐张处理；每 JSON 一个子目录；--plot 时所有预测图在 batch_dir/predicted_plots_all/，文件名与源 JSON 对应
  python main.py --mode batch_json_dir --model checkpoint/checkpoint0029.pth --batch_dir truedata/2026318 --plot

  # 单张图 Python API
  from main import process_single_image
  result = process_single_image(rt=[1,2,3], intensity=[100,500,200], model_path="x.pth")
"""
import argparse
import json
import os
import sys

# 解决 Windows 下 PyTorch(libomp.dll) 与 numpy/MKL(libiomp5md.dll) 的 OpenMP 运行时冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent  # model/ 目录


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

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def _generate_single_roi(output_dir, mz, rt, intensity, q3=None, expected_rt=None, smooth_sigma=0.0):
    """
    单张图：由 (mz, rt[], intensity[]) 生成 ROI 图像、feature.csv、roi_windows.csv、xic_matrix.npy。
    """
    rt_raw = np.asarray(rt, dtype=np.float64)
    intensity_raw = np.asarray(intensity, dtype=np.float64)
    if rt_raw.size < 2 or intensity_raw.size < 2 or rt_raw.size != intensity_raw.size:
        raise ValueError("rt 与 intensity 长度需≥2且相等")

    if np.nanmax(rt_raw) > 200:
        rt_min = rt_raw / 60.0
    else:
        rt_min = rt_raw.copy()

    if smooth_sigma > 0:
        intensity_raw = gaussian_filter1d(intensity_raw.astype(np.float64), sigma=smooth_sigma)

    apex_idx = np.argmax(intensity_raw)
    rt_apex_min = float(rt_min[apex_idx])
    rt_for_feature = expected_rt if expected_rt is not None else rt_apex_min

    mz_val = float(mz) if mz is not None else np.nan
    q3_val = float(q3) if q3 is not None and not (isinstance(q3, float) and np.isnan(q3)) else np.nan

    window_half_min = 1.0
    rt_start_min = max(rt_for_feature - window_half_min, float(np.nanmin(rt_min)))
    rt_end_min = min(rt_for_feature + window_half_min, float(np.nanmax(rt_min)))
    if rt_end_min <= rt_start_min:
        rt_start_min, rt_end_min = float(np.nanmin(rt_min)), float(np.nanmax(rt_min))

    mask = (rt_min >= rt_start_min) & (rt_min <= rt_end_min)
    plot_rt = rt_min[mask] if np.sum(mask) >= 2 else rt_min
    plot_intensity = intensity_raw[mask] if np.sum(mask) >= 2 else intensity_raw

    from ..preprocessing.xic_extraction import roi_safe_name_base

    safe_name = roi_safe_name_base(1, mz_val, q3_val)
    roi_path = os.path.join(output_dir, f"{safe_name}.jpeg")

    fig = Figure(figsize=(4, 3), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(plot_rt, plot_intensity, color="blue", linewidth=1.5)
    if rt_end_min > rt_start_min:
        ax.set_xlim(rt_start_min, rt_end_min)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas.print_jpeg(roi_path)

    n_pts = max(200, int((rt_min.max() - rt_min.min()) / 0.01) + 1)
    common_rt = np.linspace(rt_min.min(), rt_min.max(), n_pts)
    f_interp = interp1d(rt_min, intensity_raw, kind="linear", bounds_error=False, fill_value=0.0)
    aligned_intensity = f_interp(common_rt)
    xic_full = np.vstack([common_rt, aligned_intensity])
    np.save(os.path.join(output_dir, "xic_matrix.npy"), xic_full)

    pd.DataFrame([{
        "Compound Name": 1, "mz": mz_val, "q3": q3_val,
        "RT": round(float(rt_for_feature), 3)
    }]).to_csv(os.path.join(output_dir, "feature.csv"), index=False)

    pd.DataFrame([{"image": f"{safe_name}.jpeg", "rt_lo": rt_start_min, "rt_hi": rt_end_min}]).to_csv(
        os.path.join(output_dir, "roi_windows.csv"), index=False)

    return roi_path, rt_start_min, rt_end_min, common_rt, aligned_intensity


def process_single_image(
    rt,
    intensity,
    mz=None,
    q3=None,
    expected_rt=None,
    baseline_x=None,
    baseline_y=None,
    model_path=None,
    threshold=0.99,
    integration_method="linear",
    smooth_sigma=0.0,
    output_dir=None,
    keep_temp=False,
    plot=False,
    plot_dir=None,
    plot_save_filename=None,
):
    """
    单张图处理：输入 (rt[], intensity[])，可选基线 (baseline_x[], baseline_y[])，
    返回该图的预测与积分结果。一张输入一张输出，无需 mz/q3 索引。

    Parameters:
        rt: 保留时间数组（分钟；>200 视为秒）
        intensity: 强度数组
        mz: 可选，母离子 m/z（不传则内部用 nan）
        q3: 可选，子离子 m/z
        expected_rt: 可选，预期 RT
        baseline_x, baseline_y: 可选，外部基线曲线（用于 external_baseline 积分）
        model_path: 模型路径
        threshold: 置信度阈值
        integration_method: linear | raw | external_baseline
        smooth_sigma: 高斯平滑
        output_dir: 输出目录，None 则用临时目录
        keep_temp: 是否保留临时文件
        plot_dir: 预测框图保存目录；默认 output_dir/predicted_plots。用于 batch_json_dir 时统一到同一文件夹。
        plot_save_filename: 仅一张 ROI 时有效，指定输出 PNG 文件名（如 chrom_0001_pred.png），保存到 plot_dir。

    Returns:
        dict: {detections: [{x1, x2, score, area, rt_min, rt_max, ...}, ...]}
              一张图可能对应多个置信度窗口，detections 为检测框列表（按 score 降序）
    """
    from utils.io_utils import load_features
    from utils.predict_utils import build_predictor
    from utils.quantify import max_consecutive, AREA_TIME_UNIT_SCALE, integrate_with_external_baseline, integrate_with_baseline_correction_avg
    from utils.roi_rt_mapping import box_to_rt_range
    from utils.roi_quality_params import compute_roi_quality_params

    use_external_baseline = integration_method == "external_baseline" and baseline_x is not None and baseline_y is not None
    if use_external_baseline and (len(baseline_x) < 2 or len(baseline_y) < 2):
        use_external_baseline = False

    if output_dir is None:
        tmp = tempfile.mkdtemp(prefix="mrmpformer_single_")
        output_dir = tmp
        if not keep_temp:
            import atexit
            import shutil
            atexit.register(lambda: shutil.rmtree(tmp, ignore_errors=True))
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    roi_path, _, _, _, _ = _generate_single_roi(output_dir, mz, rt, intensity, q3, expected_rt, smooth_sigma)

    if not model_path or not os.path.exists(model_path):
        return {"error": "model_path required and must exist"}

    plot_dir_effective = None
    plot_out_filenames = None
    if plot and output_dir:
        plot_dir_effective = plot_dir if plot_dir is not None else os.path.join(output_dir, "predicted_plots")
        if plot_save_filename:
            fn = plot_save_filename
            if not fn.lower().endswith(".png"):
                fn = fn + ".png"
            plot_out_filenames = [fn]
    results = build_predictor(
        model_path=model_path,
        images_path=output_dir,
        threshold=threshold,
        plot=plot,
        plot_dir=plot_dir_effective or "predicted_plots",
        verbose=False,
        plot_out_filenames=plot_out_filenames,
    )

    xic_info = load_features(os.path.join(output_dir, "feature.csv"), preserve_order=True)
    xic_full = np.load(os.path.join(output_dir, "xic_matrix.npy"))
    rt_array = xic_full[0, :].astype(np.float64)
    if np.nanmax(rt_array) > 200:
        rt_array = rt_array / 60.0
    intensity_row = xic_full[1, :].astype(np.float64)
    xic_list = [np.vstack([rt_array, intensity_row])]

    roi_windows = {}
    rw_path = os.path.join(output_dir, "roi_windows.csv")
    if os.path.exists(rw_path):
        df_rw = pd.read_csv(rw_path)
        for _, row in df_rw.iterrows():
            roi_windows[str(row["image"]).strip()] = (float(row["rt_lo"]), float(row["rt_hi"]))

    img_path, scores, boxes = "", np.empty((0, 1)), np.empty((0, 4))
    if results:
        r = results[0]
        img_path = r.get("image_path", "")
        scores = np.array(r.get("scores", []), dtype=np.float32)
        boxes = np.array(r.get("boxes", []), dtype=np.float32)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if boxes.ndim == 1 and len(boxes) > 0:
            boxes = boxes.reshape(1, -1)

    true_rt = float(xic_info.loc[0, "RT"])
    mz_val = float(xic_info.loc[0, "mz"])
    q3_val = xic_info.loc[0, "q3"] if "q3" in xic_info.columns else np.nan
    image_name = os.path.basename(roi_path)
    rt_window = roi_windows.get(image_name)

    scale = float(AREA_TIME_UNIT_SCALE)
    base_out = {"detections": []}

    if len(scores) == 0 or len(boxes) == 0:
        return base_out

    # 按 score 降序排列，逐框积分
    order = np.argsort(-scores[:, 0])
    for idx in order:
        score = float(scores[idx, 0])
        x1, y1, x2, y2 = boxes[idx]
        left, right, _, _ = box_to_rt_range(x1, y1, x2, y2, true_rt, rt_array, rt_window=rt_window)

        mask = (rt_array >= left) & (rt_array <= right)
        filter_x = rt_array[mask]
        filter_y = intensity_row[mask]

        det = {
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "score": score,
            "rt_min": left, "rt_max": right,
            "area": 0.0,
            "retention_time": 0.0,
            "intensity_max": 0.0,
            "point_counts": 0,
            "integration_method_used": "raw",
            "snr": None, "noise_std": None, "baseline_slope": None,
            "peak_width_ratio": None, "dynamic_range": None,
        }

        if filter_x.size < 2 or filter_y.size < 2:
            base_out["detections"].append(det)
            continue

        max_intensity = float(np.max(filter_y))
        max_idx = int(np.argmax(filter_y))
        max_x = float(filter_x[max_idx])
        qparams = compute_roi_quality_params(filter_x, filter_y)

        if use_external_baseline:
            bx = np.asarray(baseline_x, dtype=np.float64)
            by = np.asarray(baseline_y, dtype=np.float64)
            area_val = integrate_with_external_baseline(rt_array, intensity_row, left, right, bx, by, scale)
            method_used = "external_baseline"
        elif integration_method == "linear":
            area_val = integrate_with_baseline_correction_avg(rt_array, intensity_row, left, right, scale)
            method_used = "linear"
        else:
            area_val = float(np.trapz(filter_y, filter_x) * scale)
            method_used = "raw"

        point_count = int(max_consecutive(filter_y))

        det.update({
            "area": area_val,
            "retention_time": max_x,
            "intensity_max": max_intensity,
            "point_counts": point_count,
            "integration_method_used": method_used,
            "snr": qparams.get("snr", np.nan),
            "noise_std": qparams.get("noise_std", np.nan),
            "baseline_slope": qparams.get("baseline_slope", np.nan),
            "peak_width_ratio": qparams.get("peak_width_ratio", np.nan),
            "dynamic_range": qparams.get("dynamic_range", np.nan),
        })
        base_out["detections"].append(det)

    return base_out


def main_cli():
    parser = argparse.ArgumentParser(description="MRMPFormer 统一推理入口（合并 testXIC + newtest）")
    parser.add_argument("--mode", type=str, default="single",
                        choices=[
                            "single",
                            "mzml",
                            "batch_mzml",
                            "batch_dir",
                            "batch_json_dir",
                            "pipeline_mzml",
                            "pipeline_batch_mzml",
                        ],
                        help=(
                            "single=单张图; mzml=单mzML; batch_mzml=批量mzML; batch_dir=批量目录; "
                            "batch_json_dir=目录下所有JSON逐张处理; "
                            "pipeline_mzml=完整流水线（ROI->预测->SNR筛选->post_newtest）; "
                            "pipeline_batch_mzml=完整流水线（批量mzML）"
                        ))
    parser.add_argument("--model", type=str, required=True, help="模型路径 (.pth)")
    parser.add_argument("--input", type=str, default="-",
                        help="输入 JSON 文件路径，- 表示 stdin")
    parser.add_argument("--output", type=str, default="-",
                        help="输出 JSON 文件路径，- 表示 stdout")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--integration_method", type=str, default="linear",
                        choices=["linear", "raw", "external_baseline"])
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录；single 模式默认临时目录")
    parser.add_argument("--keep_temp", action="store_true", help="[single] 保留临时文件")
    parser.add_argument("--mzml", type=str, help="[mzml] mzML 文件路径")
    parser.add_argument("--batch_dir", type=str, help="[batch_mzml] mzML 目录; [batch_dir] testXIC 输出目录; [batch_json_dir] JSON 目录")
    parser.add_argument("--plot", action="store_true", help="生成预测框标注图（single/batch_json_dir 时保存到输出目录）")
    parser.add_argument(
        "--batch_plot_dir",
        type=str,
        default=None,
        help="[batch_json_dir] 所有预测图统一目录，默认 <batch_dir>/predicted_plots_all",
    )

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
    parser.add_argument(
        "--post_enable_small_peak_rt_gate",
        action="store_true",
        help="[post] 启用小峰相对主峰的 RT 门控；不显式传入则关闭（允许多峰不按 RT 限制）",
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

    args = parser.parse_args()

    # ---- 配置运行时日志过滤 ----
    from framework.util.logutil import configure_log_level, install_filter

    if args.quiet:
        configure_log_level("ERROR")
    elif args.verbose:
        configure_log_level("INFO")
    # 否则保持默认 WARNING（抑制 [INFO] 行）
    install_filter()

    if args.mode in {"pipeline_mzml", "pipeline_batch_mzml"}:
        from .predictor import main as newtest_main
        from ..postprocessing.snr_filter import run as snr_pipeline_run
        from ..postprocessing import peak_refinement
        from utils.torch_device import resolve_torch_device

        from ..preprocessing.xic_extraction import run_batch_mzml, extract_xic_with_pyopenms

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

        # 1) Collect input mzML files
        if args.mode == "pipeline_mzml":
            if not args.mzml or not os.path.exists(args.mzml):
                print("[ERROR] --mzml must be provided for pipeline_mzml mode", file=sys.stderr)
                sys.exit(1)
            mzml_files = [Path(args.mzml).resolve()]
        else:
            if not args.batch_dir or not os.path.isdir(args.batch_dir):
                print("[ERROR] --batch_dir must be provided for pipeline_batch_mzml mode", file=sys.stderr)
                sys.exit(1)
            batch_path = Path(args.batch_dir)
            mzml_files = sorted(
                list(batch_path.glob("*.mzML")) + list(batch_path.glob("*.mzml"))
            )
            if not mzml_files:
                print(f"[ERROR] No .mzML/.mzml files found under: {batch_path}", file=sys.stderr)
                sys.exit(1)

        # 2) Generate ROI (testXIC)
        def _pipeline_qc_kwargs():
            return dict(
                min_chrom_points=int(max(0, args.pipeline_min_chrom_points)),
                min_max_intensity=float(max(0.0, args.pipeline_min_max_intensity)),
            )

        t_roi = time.perf_counter()
        mzml_roi_stats = []
        if args.mode == "pipeline_batch_mzml":
            mzml_roi_stats = run_batch_mzml(
                batch_dir=str(Path(args.batch_dir).resolve()),
                output_base=str(roi_root),
                smooth_sigma=args.smooth_sigma,
                model_path=None,
                plot_predictions=False,
                threshold=args.threshold,
                **_pipeline_qc_kwargs(),
            ) or []
        else:
            qc_kw = _pipeline_qc_kwargs()
            for mzml_path in mzml_files:
                stem = mzml_path.stem
                out_dir = roi_root / stem
                out_dir.mkdir(parents=True, exist_ok=True)
                st = extract_xic_with_pyopenms(
                    str(mzml_path),
                    str(out_dir),
                    smooth_sigma=args.smooth_sigma,
                    **qc_kw,
                )
                if st:
                    mzml_roi_stats.append({"stem": stem, **st})
        _print_mzml_roi_stats_summary(mzml_roi_stats)
        t_roi_end = time.perf_counter()
        stage_seconds["1_ROI生成(testXIC)"] = t_roi_end - t_roi
        stage_intervals["1_ROI生成(testXIC)"] = (t_roi, t_roi_end)

        # 3) Run MRMPFormer (newtest) in batch mode
        integration_method = getattr(args, "integration_method", "linear")
        pred_basename = (
            "prediction.csv"
            if integration_method == "linear"
            else f"prediction_{integration_method}.csv"
        )

        if args.mode == "pipeline_mzml":
            sample_stem = mzml_files[0].stem
            sample_roi_dir = roi_root / sample_stem
            sample_pred_dir = pred_root / sample_stem
            sample_pred_dir.mkdir(parents=True, exist_ok=True)
            a = argparse.Namespace(
                images_path=str(sample_roi_dir),
                batch_dir=None,
                batch_output=str(pred_root),
                model=args.model,
                feature=None,
                prediction_output=str(sample_pred_dir / pred_basename),
                threshold=args.threshold,
                plot=bool(args.plot),
                plot_dir=str(sample_pred_dir / "predicted_plots"),
                baseline_correction=False,
                integration_method=integration_method,
                baseline_json=None,
                verbose=False,
            )
            print("[INFO] pipeline_mzml 单样品预测: %s" % sample_stem)
        else:
            a = argparse.Namespace(
                images_path=None,
                batch_dir=str(roi_root),
                batch_output=str(pred_root),
                model=args.model,
                feature=None,
                prediction_output=str(pred_root / "prediction.csv"),
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
        for mzml_path in mzml_files:
            stem = mzml_path.stem
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
            log_dir=base_out,
        )
        return

    if args.mode == "mzml":
        from ..preprocessing.xic_extraction import extract_xic_with_pyopenms
        if not args.mzml or not os.path.exists(args.mzml):
            print("[ERROR] --mzml 必填且文件需存在", file=sys.stderr)
            sys.exit(1)
        out_dir = args.output_dir or "xic-roi-batch"
        extract_xic_with_pyopenms(args.mzml, out_dir)
        if args.model and args.plot:
            from ..preprocessing.xic_extraction import generate_prediction_plots
            generate_prediction_plots(out_dir, args.model, args.threshold)
        return
    if args.mode == "batch_mzml":
        from ..preprocessing.xic_extraction import run_batch_mzml
        if not args.batch_dir or not os.path.isdir(args.batch_dir):
            print("[ERROR] --batch_dir 必填且需为目录", file=sys.stderr)
            sys.exit(1)
        run_batch_mzml(args.batch_dir, args.output_dir or "xic-roi-batch", smooth_sigma=args.smooth_sigma,
                       model_path=args.model, plot_predictions=bool(args.plot), threshold=args.threshold)
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

    if args.mode == "batch_json_dir":
        json_dir = Path(args.batch_dir or str(ROOT_DIR / "truedata" / "2026318")).resolve()
        if not json_dir.is_dir():
            print(f"[ERROR] 目录不存在: {json_dir}", file=sys.stderr)
            sys.exit(1)
        # 含子目录：与 extract_json 输出「根目录/源mzML名/chrom_*.json」对齐；仅 glob 顶层会漏文件
        json_files = sorted(
            p
            for p in json_dir.rglob("*.json")
            if not p.stem.endswith("_result")
        )
        if not json_files:
            print(
                f"[WARN] 未找到 JSON 文件: {json_dir}\n"
                f"      提示：若 JSON 在子文件夹（如 ...\\\\20260204-01_3\\\\chrom_0000.json），\n"
                f"      请把 --batch_dir 设为包含这些 .json 的上级目录，或使用默认递归扫描。",
                file=sys.stderr,
            )
            return
        plot_root = Path(args.batch_plot_dir) if args.batch_plot_dir else (json_dir / "predicted_plots_all")
        if args.plot:
            plot_root.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] 处理 {len(json_files)} 个 JSON；预测框图统一保存到: {plot_root}")

        def _safe_plot_stem(stem: str) -> str:
            bad = '<>:"/\\|?*'
            return "".join((c if c not in bad else "_") for c in stem)

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] 跳过 {jf.name}: {e}")
                continue
            rt = data.get("rt", data.get("time", []))
            intensity = data.get("intensity", [])
            if not rt or not intensity:
                print(f"[WARN] 跳过 {jf.name}: 缺少 rt 或 intensity")
                continue
            out_subdir = json_dir / jf.stem
            out_subdir.mkdir(parents=True, exist_ok=True)
            result = process_single_image(
                rt=rt,
                intensity=intensity,
                mz=data.get("mz"),
                q3=data.get("q3"),
                expected_rt=data.get("expected_rt"),
                baseline_x=data.get("baseline_x", data.get("x")),
                baseline_y=data.get("baseline_y", data.get("y")),
                model_path=args.model,
                threshold=args.threshold,
                integration_method=args.integration_method,
                smooth_sigma=args.smooth_sigma,
                output_dir=str(out_subdir),
                keep_temp=True,
                plot=args.plot,
                plot_dir=str(plot_root) if args.plot else None,
                plot_save_filename=(
                    f"{_safe_plot_stem(str(rel.with_suffix('')).replace(os.sep, '_'))}_pred.png"
                    if args.plot
                    else None
                ),
            )

            def _json_default(x):
                if isinstance(x, (np.floating, np.integer)):
                    v = float(x)
                    return None if (isinstance(x, np.floating) and np.isnan(v)) else v
                return str(x)

            out_path = out_subdir / "result.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, default=_json_default))
            plot_info = f"、预测框图见 {plot_root}/" if args.plot else ""
            print(f"[OK] {jf.name} -> {out_subdir.name}/ (含 ROI、result.json{plot_info})")
        return

    if args.mode != "single":
        print("未知 mode", file=sys.stderr)
        sys.exit(1)

    if args.input == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

    mz = data.get("mz")
    rt = data.get("rt", data.get("time", []))
    intensity = data.get("intensity", [])
    q3 = data.get("q3")
    expected_rt = data.get("expected_rt")
    baseline_x = data.get("baseline_x", data.get("x"))
    baseline_y = data.get("baseline_y", data.get("y"))

    if not rt or not intensity:
        result = {"error": "rt, intensity 必填"}
    else:
        result = process_single_image(
            rt=rt,
            intensity=intensity,
            mz=mz,
            q3=q3,
            expected_rt=expected_rt,
            baseline_x=baseline_x,
            baseline_y=baseline_y,
            model_path=args.model,
            threshold=args.threshold,
            integration_method=args.integration_method,
            smooth_sigma=args.smooth_sigma,
            output_dir=args.output_dir,
            keep_temp=args.keep_temp,
            plot=args.plot,
        )

    def _json_default(x):
        if isinstance(x, (np.floating, np.integer)):
            v = float(x)
            return None if (isinstance(x, np.floating) and np.isnan(v)) else v
        return str(x)
    out_str = json.dumps(result, ensure_ascii=False, default=_json_default)
    if args.output == "-":
        print(out_str)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)


if __name__ == "__main__":
    main_cli()
