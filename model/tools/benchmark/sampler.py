# -*- coding: utf-8 -*-
"""GPU 显存采样与资源监控。"""

import statistics
import subprocess
import sys
import threading

GPU_VRAM_SAMPLE_INTERVAL_SEC = 0.5


def read_gpu_vram_mb(device_index: int = 0):
    """读取 GPU 显存已用/总量(MB)。优先 pynvml，否则 nvidia-smi。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = float(mem.used) / (1024.0 ** 2)
        total = float(mem.total) / (1024.0 ** 2)
        pynvml.nvmlShutdown()
        return used, total
    except Exception:
        pass
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i", str(int(device_index)),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            creationflags=flags,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return None, None


class GpuVramSampler:
    """单次 subprocess 运行期间后台采样显存已用/总量。"""

    def __init__(self, device_index: int = 0):
        self.device_index = int(device_index)
        self._lock = threading.Lock()
        self._samples = []
        self._thread = None
        self._stop = threading.Event()
        self._available = False
        used, total = read_gpu_vram_mb(self.device_index)
        self._available = used is not None and total is not None and total > 0

    def start(self) -> bool:
        if not self._available:
            return False
        self._stop.clear()
        with self._lock:
            self._samples = []
        u, t = read_gpu_vram_mb(self.device_index)
        if u is not None and t is not None:
            with self._lock:
                self._samples.append((u, t))
        self._thread = threading.Thread(target=self._loop, name="benchmark-gpu-vram", daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while not self._stop.wait(GPU_VRAM_SAMPLE_INTERVAL_SEC):
            u, t = read_gpu_vram_mb(self.device_index)
            if u is not None and t is not None:
                with self._lock:
                    self._samples.append((u, t))

    def stop_and_stats(self) -> dict:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=3.0)
            self._thread = None
        u, t = read_gpu_vram_mb(self.device_index)
        if u is not None and t is not None:
            with self._lock:
                self._samples.append((u, t))
        with self._lock:
            snaps = list(self._samples)
        if not snaps:
            return None
        used_list = [s[0] for s in snaps]
        total_list = [s[1] for s in snaps]
        pct_list = [100.0 * u / t if t > 0 else 0.0 for u, t in snaps]
        return {
            "used_mb_avg": float(statistics.mean(used_list)),
            "used_mb_max": float(max(used_list)),
            "total_mb_avg": float(statistics.mean(total_list)),
            "total_mb_max": float(max(total_list)),
            "pct_avg": float(statistics.mean(pct_list)),
            "pct_max": float(max(pct_list)),
        }
