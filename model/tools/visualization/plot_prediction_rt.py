# -*- coding: utf-8 -*-
"""
将 newtest 输出的 prediction.csv 中「已映射到 RT」的 rt_min/rt_max 画在原始 xic 上，
风格对齐 run_unified_peak_workflow._plot_refined_predictions（refined_plots：蓝线 + 绿色 Main interval）。

不依赖 testXIC/pyopenms。可选高斯平滑（默认 sigma=0.8）。

示例：
  python -m <包名>.visualization.plot_prediction_rt ^
    --prediction_csv "D:\\...\\result\\batch_predictions\\json\\prediction.csv" ^
    --output_dir "D:\\...\\result\\batch_predictions\\json\\prediction_rt_plots"

  # 若未放在默认相对路径，可显式指定：
    --xic_matrix "D:\\...\\result\\xic-roi-batch\\json\\xic_matrix.npy" ^
    --roi_windows_csv "D:\\...\\result\\xic-roi-batch\\json\\roi_windows.csv"
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .._shared.artifacts import (
    image_to_row_index,
    load_roi_map,
    read_csv_safe,
    resolve_rt_window,
    safe_float,
)


def _matplotlib_cjk_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_one(
    row_idx: int,
    row: pd.Series,
    rt_full: np.ndarray,
    mat: np.ndarray,
    roi_map: Dict[str, Tuple[float, float]],
    output_dir: Path,
    smooth_sigma: float = 0.8,
):
    _matplotlib_cjk_font()
    img = str(row.get("image", "")).strip()
    y_raw = mat[row_idx, :].astype(np.float64)

    if smooth_sigma > 0:
        y = gaussian_filter1d(y_raw, sigma=smooth_sigma)
    else:
        y = y_raw

    rw = resolve_rt_window(roi_map, img)
    rt_lo, rt_hi = rw[0] if rw[0] else (np.nanmin(rt_full), np.nanmax(rt_full))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rt_full, y, color="blue", linewidth=0.8, label="XIC (smoothed)" if smooth_sigma > 0 else "XIC (raw)")
    ax.axvspan(rt_lo, rt_hi, color="green", alpha=0.15, label="Main interval")

    rt_min_p = safe_float(row.get("rt_min"), np.nan)
    rt_max_p = safe_float(row.get("rt_max"), np.nan)
    if np.isfinite(rt_min_p) and np.isfinite(rt_max_p):
        ax.axvline(rt_min_p, color="red", linestyle="--", linewidth=0.8, label="rt_min")
        ax.axvline(rt_max_p, color="red", linestyle="--", linewidth=0.8, label="rt_max")

    cn = row.get("compound_name", "")
    ax.set_title("Row %d  compound_name=%s" % (row_idx + 1, cn))
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=7)
    fig.tight_layout()

    stem = Path(img).stem if img else "row_%d" % (row_idx + 1)
    out = output_dir / ("%s_pred_rt.png" % stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="绘制 prediction 中 RT 映射到 XIC 的图")
    ap.add_argument("--prediction_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--xic_matrix", default=None)
    ap.add_argument("--roi_windows_csv", default=None)
    ap.add_argument("--smooth_sigma", type=float, default=0.8)
    args = ap.parse_args()

    pred_path = Path(args.prediction_csv)
    if not pred_path.is_file():
        print("[ERROR] prediction_csv 不存在: %s" % pred_path)
        sys.exit(1)

    df = read_csv_safe(pred_path)

    # 自动推断 xic_matrix 和 roi_windows_csv 路径
    xic_path = Path(args.xic_matrix) if args.xic_matrix else pred_path.parent.parent.parent / "xic-roi-batch" / pred_path.parent.name / "xic_matrix.npy"
    roi_path = Path(args.roi_windows_csv) if args.roi_windows_csv else xic_path.parent / "roi_windows.csv"

    if not xic_path.is_file():
        print("[ERROR] xic_matrix 不存在: %s" % xic_path)
        sys.exit(1)

    x = np.load(str(xic_path))
    rt_full = x[0, :].astype(np.float64)
    if np.nanmax(rt_full) > 200:
        rt_full = rt_full / 60.0
    mat = x[1:, :].astype(np.float64)

    roi_map = load_roi_map(roi_path) if roi_path.is_file() else {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for i, row in df.iterrows():
        img = str(row.get("image", ""))
        row_idx = image_to_row_index(img, row.get("compound_name"))
        if row_idx is None or row_idx >= mat.shape[0]:
            continue
        try:
            plot_one(row_idx, row, rt_full, mat, roi_map, out_dir, args.smooth_sigma)
            n_done += 1
        except Exception as e:
            print("[WARN] row %d: %s" % (i, e))

    print("[DONE] 绘制 %d 张图 → %s" % (n_done, out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
