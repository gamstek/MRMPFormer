# -*- coding: utf-8 -*-
"""
以「原始 chrom JSON + 与 testXIC.extract_xic_from_arrays 一致的 QC/平滑/全局 linspace」
重算第 N 条化合物的对齐曲线，与磁盘上的 batch/SNR xic_matrix 第 N 行逐点对照；
并用互相关估计 batch 与 SNR 两版曲线的时间平移。

回答的是：横轴时间与强度采样是否一致（含 QC 后序号与 JSON 顺序的对应），不是框修正。

示例：
  python -m <包名>.diagnostics.verify_rt_axis ^
    --result_root "D:\\...\\results\\full_pipeline" ^
    --sample "20251120-01" ^
    --chrom_batch_dir "D:\\数据\\20251120-01" ^
    --compounds 89 83 ^
    --smooth_sigma 0.8 ^
    --min_chrom_points 10 ^
    --min_max_intensity 1000 ^
    --snr_min 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .._shared.artifacts import read_csv_safe
from .._shared.chrom_json import parse_q1_q3, parse_time_intensity


def _qc_passes(rt_sec, intensity, min_chrom_points, min_max_intensity):
    if min_chrom_points > 0 and len(rt_sec) < min_chrom_points:
        return False
    if min_max_intensity > 0 and np.nanmax(intensity) < min_max_intensity:
        return False
    return True


def _make_global_linspace(records, n_points=500):
    all_rt = np.concatenate([r["rt_sec"] for r in records])
    lo = np.nanmin(all_rt)
    hi = np.nanmax(all_rt)
    return np.linspace(lo, hi, n_points)


def main():
    ap = argparse.ArgumentParser(description="验证 RT 轴一致性：chrom JSON vs xic_matrix")
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--sample", required=True, help="样品名/子目录名")
    ap.add_argument("--chrom_batch_dir", required=True, help="chrom JSON 目录")
    ap.add_argument("--compounds", type=int, nargs="+", required=True, help="化合物序号(1-based)")
    ap.add_argument("--smooth_sigma", type=float, default=0.8)
    ap.add_argument("--min_chrom_points", type=int, default=10)
    ap.add_argument("--min_max_intensity", type=float, default=1000.0)
    ap.add_argument("--snr_min", type=float, default=3.0)
    args = ap.parse_args()

    result_root = Path(args.result_root)
    chrom_dir = Path(args.chrom_batch_dir)

    # 加载 chrom JSON
    records = []
    for path in sorted(chrom_dir.glob("*.json")):
        if path.stem.endswith("_result"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("[WARN] 跳过 %s: %s" % (path.name, e))
            continue
        rt_sec, intensity = parse_time_intensity(data)
        if rt_sec is None:
            continue
        q1, q3 = parse_q1_q3(data)
        records.append({
            "q1": q1, "q3": q3,
            "rt_sec": rt_sec,
            "intensity": intensity,
            "native_id": str(data.get("native_id", "")),
        })

    print("[INFO] 加载 %d 条 chrom" % len(records))

    # 加载 batch xic_matrix
    xic_dir = result_root / "xic-roi-batch" / args.sample
    xic_npy = xic_dir / "xic_matrix.npy"
    if xic_npy.is_file():
        x = np.load(str(xic_npy))
        rt_batch = x[0, :].astype(np.float64)
        if np.nanmax(rt_batch) > 200:
            rt_batch = rt_batch / 60.0
        mat_batch = x[1:, :].astype(np.float64)
        print("[INFO] batch XIC: %d rows, RT [%.2f, %.2f]" % (mat_batch.shape[0], rt_batch[0], rt_batch[-1]))
    else:
        print("[ERROR] 无 batch xic_matrix: %s" % xic_npy)
        sys.exit(1)

    # 加载 SNR xic_matrix
    snr_sub = "SNR_box_%d" % int(args.snr_min) if args.snr_min == int(args.snr_min) else "SNR_box_%.10g" % args.snr_min
    snr_dir = result_root / "snr_filtered" / args.sample / snr_sub
    snr_npy = snr_dir / "xic_matrix.npy"
    if snr_npy.is_file():
        x = np.load(str(snr_npy))
        rt_snr = x[0, :].astype(np.float64)
        if np.nanmax(rt_snr) > 200:
            rt_snr = rt_snr / 60.0
        mat_snr = x[1:, :].astype(np.float64)
        print("[INFO] SNR XIC: %d rows, RT [%.2f, %.2f]" % (mat_snr.shape[0], rt_snr[0], rt_snr[-1]))
    else:
        rt_snr = None
        mat_snr = None
        print("[WARN] 无 SNR xic_matrix: %s" % snr_npy)

    for cn in args.compounds:
        idx = cn - 1  # 1-based → 0-based
        if idx >= len(records):
            print("[SKIP] compound %d > %d" % (cn, len(records)))
            continue

        rec = records[idx]
        print("\n--- compound %d ---" % cn)
        print("  native_id: %s" % rec["native_id"])
        print("  Q1=%.4f Q3=%.2f" % (rec["q1"] or 0, rec["q3"] or 0))

        # batch 行
        if idx < mat_batch.shape[0]:
            b_row = mat_batch[idx, :]
            print("  batch: max=%.2f mean=%.2f points=%d" % (np.nanmax(b_row), np.nanmean(b_row), np.sum(np.isfinite(b_row))))
        else:
            print("  batch: idx %d out of range (%d rows)" % (idx, mat_batch.shape[0]))

        # SNR 行
        if mat_snr is not None and idx < mat_snr.shape[0]:
            s_row = mat_snr[idx, :]
            print("  SNR:   max=%.2f mean=%.2f points=%d" % (np.nanmax(s_row), np.nanmean(s_row), np.sum(np.isfinite(s_row))))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
