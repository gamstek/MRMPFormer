"""
ion_zenith.py — 离子天顶算法后台线程
=====================================
封装 IonZenithWorker(QThread)，遍历 mzML 的 MS1 谱图，
按 m/z 容差分箱聚合，每个离子保留最高强度的观测，输出精简 CSV。

算法核心:
  遍历 MS1 → m/z 在容差内合并 → 保留最高强度 → 按 m/z 排序写入 CSV

依赖: pymzml, numpy, csv, time, pathlib, PySide6.QtCore
"""

import csv
import math
import os
import tempfile
import time
from pathlib import Path
import numpy as np
from PySide6.QtCore import QThread, Signal


class IonZenithWorker(QThread):
    """
    离子天顶算法后台线程。

    读取 mzML 文件，遍历 MS1 谱图，提取每个 m/z 信号顶点（最高强度），
    输出 (m/z, RT, intensity, n_observations) 的 CSV 文件。
    """

    # (scanned: int, total: int) — 已扫描谱图数（total=0 表示未知总数）
    progress = Signal(int, int)
    # (ms1_count: int, total_peaks: int) — MS1 谱图数, 累计扫描峰数
    stats = Signal(int, int)
    # (ion_count: int, elapsed_sec: float, output_path: str) — 任务完成
    finished = Signal(int, float, str)
    # (message: str) — 错误消息
    error = Signal(str)

    def __init__(self, params: dict, parent=None):
        """
        Args:
            params: 参数字典，包含以下键：
                input_mzml (str):      输入 mzML 文件路径（必需）
                output_csv (str):      输出 CSV 路径（必需）
                mz_min (float):        m/z 下限，默认 50.0
                mz_max (float):        m/z 上限，默认 2000.0
                ppm_tol (float):       ppm 容差，默认 10.0
                da_tol (float):        Da 容差，默认 0.01
                intensity_min (float|None): 强度下限，None=不限制
                intensity_max (float|None): 强度上限，None=不限制
                max_spectra (int):     最大扫描谱图数，0=全部，默认 0
                build_index (bool):    是否重建 mzML 索引，默认 False
                show_progress (bool):  是否通过 Signal 发送进度，默认 True
            parent: Qt parent object
        """
        super().__init__(parent)
        self._params = params
        self._cancelled = False  # 预留取消标志（后续版本实现真正的中断逻辑）

    def cancel(self):
        """请求取消当前运行。（当前版本仅设置标志，worker loop 检查此标志）"""
        self._cancelled = True

    def run(self):
        """在线程中执行离子天顶算法。"""
        import pymzml

        p = self._params
        t0 = time.perf_counter()

        # ── 验证必需参数 ──
        input_path = Path(p.get("input_mzml", ""))
        output_path = Path(p.get("output_csv", ""))
        if not input_path.exists():
            self.error.emit(f"输入文件不存在: {input_path}")
            return
        if not output_path.parent.exists():
            self.error.emit(f"输出目录不存在: {output_path.parent}")
            return

        # ── 读取参数（带默认值） ──
        mz_min = float(p.get("mz_min", 50.0))
        mz_max = float(p.get("mz_max", 2000.0))
        ppm_tol = float(p.get("ppm_tol", 10.0))
        da_tol = float(p.get("da_tol", 0.01))
        intensity_min = p.get("intensity_min")  # None or float
        intensity_max = p.get("intensity_max")  # None or float
        max_spectra = int(p.get("max_spectra", 0))
        build_index = bool(p.get("build_index", False))

        if intensity_min is not None:
            intensity_min = float(intensity_min)
        if intensity_max is not None:
            intensity_max = float(intensity_max)

        limit = max_spectra if max_spectra > 0 else None
        # best: mz_key → [mz_center, rt_min, rt_sec, max_intensity, count]
        best = {}
        n_spec = 0      # 总谱图计数
        n_ms1 = 0       # MS1 谱图计数
        n_peaks = 0     # 累计扫描峰总数

        # ── 打开 mzML 文件（容错处理编码异常 + 自动转码） ──
        resolved_path, temp_file = self._resolve_encoding(str(input_path))
        try:
            run = pymzml.run.Reader(resolved_path, build_index_from_scratch=build_index)
        except UnicodeDecodeError:
            if not build_index:
                try:
                    run = pymzml.run.Reader(resolved_path, build_index_from_scratch=True)
                except UnicodeDecodeError:
                    self.error.emit(
                        "mzML 文件编码异常 (非 UTF-8)，已尝试自动转码仍失败。"
                        "请用 ProteoWizard 重新转换为标准 UTF-8 编码的 mzML"
                    )
                    self._cleanup_temp(temp_file)
                    return
            else:
                self.error.emit(
                    "mzML 文件编码异常 (非 UTF-8)，已尝试自动转码仍失败。"
                    "请用 ProteoWizard 重新转换为标准 UTF-8 编码的 mzML"
                )
                self._cleanup_temp(temp_file)
                return
        except Exception as e:
            self.error.emit(f"无法打开 mzML 文件: {e}")
            self._cleanup_temp(temp_file)
            return

        # ── 遍历谱图 ──
        for spectrum in run:
            if self._cancelled:
                break
            n_spec += 1

            # 只处理 MS1
            ms_level = getattr(spectrum, "ms_level", None)
            if ms_level != 1:
                if limit and n_spec >= limit:
                    break
                continue

            # 获取保留时间
            rt_min, rt_sec = self._parse_rt(spectrum)
            if rt_sec is None:
                if limit and n_spec >= limit:
                    break
                continue

            # 获取峰数组 (m/z, intensity)
            arr = self._get_peaks(spectrum)
            if arr is None or arr.size == 0:
                if limit and n_spec >= limit:
                    break
                continue

            n_ms1 += 1
            mz_vals = arr[:, 0]
            int_vals = arr[:, 1]

            # ── 强度 + m/z 过滤 ──
            mask = np.ones(len(mz_vals), dtype=bool)
            mask &= (mz_vals >= mz_min) & (mz_vals <= mz_max)
            mask &= (int_vals > 0)
            if intensity_min is not None:
                mask &= (int_vals >= intensity_min)
            if intensity_max is not None:
                mask &= (int_vals <= intensity_max)

            idx = np.where(mask)[0]
            if len(idx) == 0:
                if limit and n_spec >= limit:
                    break
                continue

            mz_filt = mz_vals[idx]
            int_filt = int_vals[idx]

            # ── 按 m/z 容差聚合：同一 key 只保留最高强度 ──
            for j in range(len(mz_filt)):
                mz = float(mz_filt[j])
                intensity = float(int_filt[j])
                n_peaks += 1

                # 动态容差：取 ppm 和 Da 中较大者
                tol = max(mz * ppm_tol * 1e-6, da_tol)
                key = round(mz / tol) * tol

                if key not in best or intensity > best[key][3]:
                    best[key] = [mz, rt_min, rt_sec, intensity, 1]
                else:
                    # 同一 key：保留最高强度，累加观测次数
                    if intensity > best[key][3]:
                        best[key][0] = mz
                        best[key][1] = rt_min
                        best[key][2] = rt_sec
                        best[key][3] = intensity
                    best[key][4] += 1

            # ── 定期发送进度（每 10 张 MS1 谱图） ──
            if n_ms1 % 10 == 0:
                self.progress.emit(n_spec, limit or 0)
                self.stats.emit(n_ms1, n_peaks)

            if limit and n_spec >= limit:
                break

        # ── 写入 CSV ──
        os.makedirs(output_path.parent, exist_ok=True)
        rows = sorted(best.values(), key=lambda x: x[0])  # 按 m/z 升序

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["mz", "rt_min", "rt_sec", "max_intensity", "n_observations"])
            for row in rows:
                w.writerow(row)

        elapsed = time.perf_counter() - t0
        # ── 清理临时转码文件 ──
        self._cleanup_temp(temp_file)
        # 最终信号：进度到底 + 统计 + 完成
        self.progress.emit(n_spec, n_spec)
        self.stats.emit(n_ms1, n_peaks)
        self.finished.emit(len(rows), elapsed, str(output_path))

    # ================================================================
    # 辅助静态方法
    # ================================================================

    @staticmethod
    def _resolve_encoding(file_path: str) -> tuple:
        """
        尝试将非 UTF-8 编码的 mzML 自动转码为 UTF-8 临时文件。

        依次尝试 UTF-8 → UTF-16 → GBK → Latin-1 → cp1252 解码。
        若原始文件已是合法 UTF-8，直接返回原路径（不创建临时文件）。

        Args:
            file_path: 原始 mzML 文件路径

        Returns:
            (resolved_path: str, temp_file: tempfile.NamedTemporaryFile | None)
            resolved_path — 供 pymzml 使用的文件路径
            temp_file   — 若创建了临时文件则返回其句柄（调用方负责清理），否则 None
        """
        # 快速路径：直接尝试 UTF-8 解码
        try:
            with open(file_path, "rb") as fh:
                raw = fh.read()
            raw.decode("utf-8")
            return file_path, None  # 已是合法 UTF-8，无需转码
        except UnicodeDecodeError:
            pass

        # 尝试常见编码列表
        candidates = ["utf-16", "gbk", "gb2312", "latin-1", "cp1252"]
        decoded = None
        for enc in candidates:
            try:
                decoded = raw.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded is None:
            return file_path, None  # 所有编码均失败，交给 pymzml 处理

        # 写入 UTF-8 临时文件
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".mzML", prefix="ion_zenith_utf8_",
            encoding="utf-8", delete=False,
        )
        try:
            tmp.write(decoded)
            tmp.flush()
            return tmp.name, tmp
        except Exception:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            return file_path, None

    @staticmethod
    def _cleanup_temp(temp_file):
        """安全清理临时转码文件。"""
        if temp_file is None:
            return
        try:
            tmp_path = Path(temp_file.name)
            temp_file.close()
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def _parse_rt(spectrum) -> tuple:
        """
        从 pymzML spectrum 对象提取保留时间。

        Args:
            spectrum: pymzML spectrum 对象

        Returns:
            (rt_min: float | None, rt_sec: float | None)
        """
        st = getattr(spectrum, "scan_time", None)
        if st is None:
            return None, None
        if isinstance(st, (tuple, list)) and len(st) >= 2:
            val, unit = float(st[0]), str(st[1]).lower()
        else:
            val, unit = float(st), ""
        if math.isnan(val) or math.isinf(val):
            return None, None
        if "min" in unit:
            return val, val * 60.0
        # 默认当作秒处理
        return val / 60.0, val

    @staticmethod
    def _get_peaks(spectrum):
        """
        从 pymzML spectrum 提取 (m/z, intensity) 二维 numpy 数组。

        Args:
            spectrum: pymzML spectrum 对象

        Returns:
            np.ndarray (shape=(n,2), dtype=float64) 或 None
        """
        p = getattr(spectrum, "peaks", None)
        if p is not None and len(p) > 0:
            return np.asarray(p, dtype=np.float64)
        mz_arr = getattr(spectrum, "mz", None)
        int_arr = getattr(spectrum, "i", None)
        if mz_arr is not None and int_arr is not None:
            return np.column_stack((
                np.asarray(mz_arr, dtype=np.float64),
                np.asarray(int_arr, dtype=np.float64),
            ))
        return None
