# -*- coding: utf-8 -*-
"""
ion_zenith.py — 离子天顶算法（纯算法 + CLI）
=============================================

遍历 mzML 的 MS1 谱图，按 m/z 容差分箱聚合，每个离子保留最高强度的观测，
输出精简 CSV。

算法核心:
  遍历 MS1 → m/z 在容差内合并 → 保留最高强度 → 按 m/z 排序写入 CSV

与 preprocessing.xic_extraction 的关系:
  - xic_extraction 处理 mzML 的 **chromatogram**（MRM/SRM 已抽好的 XIC）
  - ion_zenith 处理 mzML 的 **MS1 spectrum**（DDA/DIA/Full-scan 散点谱）
  - 两者数据源/参数/输出列均不重叠，互不替代。

依赖: pymzml, numpy, csv, time, pathlib

CLI 用法:
  python -m preprocessing.ion_zenith --input_mzml path/to/file.mzML --output_csv out.csv
  python -m preprocessing.ion_zenith --input_mzml file.mzML --output_csv out.csv \\
      --mz_min 50 --mz_max 2000 --ppm_tol 10 --da_tol 0.01

Python API:
  from preprocessing.ion_zenith import extract_ions_from_ms1
  result = extract_ions_from_ms1(
      mzml_path="file.mzML",
      output_csv="out.csv",
      mz_min=50.0, mz_max=2000.0,
      ppm_tol=10.0, da_tol=0.01,
      on_progress=lambda scanned, total: print(scanned, total),
  )
"""
import argparse
import csv
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent  # model/ 目录


# ============================================================
# 内部辅助函数（自包含，不依赖 Qt）
# ============================================================

def _resolve_encoding(file_path: str) -> tuple:
    """
    尝试将非 UTF-8 编码的 mzML 自动转码为 UTF-8 临时文件。

    依次尝试 UTF-8 → UTF-16 → GBK → Latin-1 → cp1252 解码。
    若原始文件已是合法 UTF-8，直接返回原路径（不创建临时文件）。

    Args:
        file_path: 原始 mzML 文件路径

    Returns:
        (resolved_path: str, temp_file: tempfile.NamedTemporaryFile | None)
    """
    try:
        with open(file_path, "rb") as fh:
            raw = fh.read()
        raw.decode("utf-8")
        return file_path, None  # 已是合法 UTF-8，无需转码
    except UnicodeDecodeError:
        pass

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


def _parse_rt(spectrum) -> tuple:
    """
    从 pymzML spectrum 对象提取保留时间。

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


def _get_peaks(spectrum):
    """
    从 pymzML spectrum 提取 (m/z, intensity) 二维 numpy 数组。

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


# ============================================================
# 主算法
# ============================================================

def extract_ions_from_ms1(
    mzml_path: str,
    output_csv: str,
    mz_min: float = 50.0,
    mz_max: float = 2000.0,
    ppm_tol: float = 10.0,
    da_tol: float = 0.01,
    intensity_min: Optional[float] = None,
    intensity_max: Optional[float] = None,
    max_spectra: int = 0,
    build_index: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
    on_stats: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    遍历 mzML 的 MS1 谱图，按 m/z 容差聚合，每个离子保留最高强度观测，写入 CSV。

    Args:
        mzml_path: 输入 mzML 文件路径
        output_csv: 输出 CSV 路径
        mz_min: m/z 下限（默认 50.0）
        mz_max: m/z 上限（默认 2000.0）
        ppm_tol: ppm 容差（默认 10.0）
        da_tol: Da 容差（默认 0.01）
        intensity_min: 强度下限，None=不限制
        intensity_max: 强度上限，None=不限制
        max_spectra: 最大扫描谱图数，0=全部
        build_index: 是否重建 mzML 索引
        on_progress: 进度回调 (scanned, total)，total=0 表示未知总数
        on_stats: 统计回调 (ms1_count, total_peaks)

    Returns:
        dict: {
            "n_ions": int, "n_ms1": int, "n_peaks": int,
            "n_spectra": int, "elapsed_sec": float, "output_csv": str
        }

    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: pymzml 打开失败
    """
    import pymzml

    t0 = time.perf_counter()

    input_path = Path(mzml_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    output_path = Path(output_csv)
    if not output_path.parent.exists():
        raise FileNotFoundError(f"输出目录不存在: {output_path.parent}")

    limit = max_spectra if max_spectra > 0 else None
    # best: mz_key → [mz_center, rt_min, rt_sec, max_intensity, count]
    best: dict = {}
    n_spec = 0      # 总谱图计数
    n_ms1 = 0       # MS1 谱图计数
    n_peaks = 0     # 累计扫描峰总数

    # ── 打开 mzML 文件（容错处理编码异常 + 自动转码） ──
    resolved_path, temp_file = _resolve_encoding(str(input_path))
    try:
        run = pymzml.run.Reader(resolved_path, build_index_from_scratch=build_index)
    except UnicodeDecodeError:
        if not build_index:
            try:
                run = pymzml.run.Reader(resolved_path, build_index_from_scratch=True)
            except UnicodeDecodeError:
                _cleanup_temp(temp_file)
                raise ValueError(
                    "mzML 文件编码异常 (非 UTF-8)，已尝试自动转码仍失败。"
                    "请用 ProteoWizard 重新转换为标准 UTF-8 编码的 mzML"
                )
        else:
            _cleanup_temp(temp_file)
            raise ValueError(
                "mzML 文件编码异常 (非 UTF-8)，已尝试自动转码仍失败。"
                "请用 ProteoWizard 重新转换为标准 UTF-8 编码的 mzML"
            )
    except Exception as e:
        _cleanup_temp(temp_file)
        raise ValueError(f"无法打开 mzML 文件: {e}")

    # ── 遍历谱图 ──
    for spectrum in run:
        n_spec += 1

        # 只处理 MS1
        ms_level = getattr(spectrum, "ms_level", None)
        if ms_level != 1:
            if limit and n_spec >= limit:
                break
            continue

        # 获取保留时间
        rt_min, rt_sec = _parse_rt(spectrum)
        if rt_sec is None:
            if limit and n_spec >= limit:
                break
            continue

        # 获取峰数组 (m/z, intensity)
        arr = _get_peaks(spectrum)
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
            if on_progress is not None:
                on_progress(n_spec, limit or 0)
            if on_stats is not None:
                on_stats(n_ms1, n_peaks)

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
    _cleanup_temp(temp_file)

    # 最终回调
    if on_progress is not None:
        on_progress(n_spec, n_spec)
    if on_stats is not None:
        on_stats(n_ms1, n_peaks)

    return {
        "n_ions": len(rows),
        "n_ms1": n_ms1,
        "n_peaks": n_peaks,
        "n_spectra": n_spec,
        "elapsed_sec": elapsed,
        "output_csv": str(output_path),
    }


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="离子天顶算法：遍历 mzML MS1 谱图 → 按 m/z 聚合 → 输出 CSV"
    )
    parser.add_argument("--input_mzml", "--mzml", required=True,
                        help="输入 mzML 文件路径")
    parser.add_argument("--output_csv", required=True,
                        help="输出 CSV 路径")
    parser.add_argument("--mz_min", type=float, default=50.0,
                        help="m/z 下限 (默认 50.0)")
    parser.add_argument("--mz_max", type=float, default=2000.0,
                        help="m/z 上限 (默认 2000.0)")
    parser.add_argument("--ppm_tol", type=float, default=10.0,
                        help="ppm 容差 (默认 10.0)")
    parser.add_argument("--da_tol", type=float, default=0.01,
                        help="Da 容差 (默认 0.01)")
    parser.add_argument("--intensity_min", type=float, default=None,
                        help="强度下限，不传则不限制")
    parser.add_argument("--intensity_max", type=float, default=None,
                        help="强度上限，不传则不限制")
    parser.add_argument("--max_spectra", type=int, default=0,
                        help="最大扫描谱图数，0=全部 (默认 0)")
    parser.add_argument("--build_index", action="store_true",
                        help="重建 mzML 索引（mzML 缺少索引时使用）")
    args = parser.parse_args()

    def _progress(scanned: int, total: int):
        if total > 0:
            print(f"[进度] {scanned}/{total} 谱图", flush=True)
        else:
            print(f"[进度] 已扫描 {scanned} 谱图", flush=True)

    def _stats(ms1_count: int, peaks: int):
        print(f"[统计] MS1: {ms1_count} | 累计峰数: {peaks}", flush=True)

    try:
        result = extract_ions_from_ms1(
            mzml_path=args.input_mzml,
            output_csv=args.output_csv,
            mz_min=args.mz_min,
            mz_max=args.mz_max,
            ppm_tol=args.ppm_tol,
            da_tol=args.da_tol,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            max_spectra=args.max_spectra,
            build_index=args.build_index,
            on_progress=_progress,
            on_stats=_stats,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"[DONE] 离子天顶算法完成")
    print(f"  离子数: {result['n_ions']}")
    print(f"  MS1 谱图数: {result['n_ms1']}")
    print(f"  累计扫描峰数: {result['n_peaks']}")
    print(f"  总谱图数: {result['n_spectra']}")
    print(f"  耗时: {result['elapsed_sec']:.2f}s")
    print(f"  输出: {result['output_csv']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
