# -*- coding: utf-8 -*-
"""
根据 refined_plots 下的结果图（*_refined.png），从同批次的 xic_matrix.npy 重绘「原始」XIC。

使用同一套数据：xic 第 0 行为 RT，第 1 行起为各化合物强度。

示例：
  python -m <包名>.visualization.plot_refined_xic ^
    --refined_png "D:\\...\\SNR_box_3\\refined_plots\\30_mz384.2000_q3247.1000_snr27.8987_refined.png"

  # 时间窗：roi=roi_windows 全窗；refined=主峰±1分钟；full=整条 XIC
  python -m <包名>.visualization.plot_refined_xic --refined_png ... --window refined
"""
import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .._shared.artifacts import (
    find_row_for_refined_png,
    image_to_row_index,
    load_roi_map,
    locate_roi_csv,
    locate_xic_npy,
    read_csv_safe,
    resolve_rt_window,
    safe_float,
)


def _matplotlib_cjk_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_xic_from_refined_png(
    refined_png: Path,
    xic_matrix: Optional[Path],
    roi_windows_csv: Optional[Path],
    pred_refined_csv: Optional[Path],
    output_png: Optional[Path],
    window: str,
    smooth_sigma: float,
):
    refined_png = refined_png.resolve()
    if not refined_png.is_file():
        raise FileNotFoundError(f"找不到 refined 图: {refined_png}")

    snr_dir = refined_png.parent.parent
    pr = pred_refined_csv or (snr_dir / "prediction_refined.csv")
    if not pr.is_file():
        raise FileNotFoundError(f"需要 prediction_refined.csv: {pr}")

    df = read_csv_safe(pr)
    row = find_row_for_refined_png(df, refined_png)
    if row is None:
        raise RuntimeError(f"在 {pr} 中未找到与 refined 图对应的行")

    image_name = str(row.get("image", "")).strip()
    row_idx = image_to_row_index(image_name, row.get("compound_name"))
    if row_idx is None:
        raise RuntimeError(f"无法从 image={image_name!r} 解析 XIC 行索引。")

    xic_path = xic_matrix or locate_xic_npy(snr_dir)
    if xic_path is None or not xic_path.is_file():
        raise FileNotFoundError(f"找不到 xic_matrix.npy。请用 --xic_matrix 指定。")

    x = np.load(str(xic_path))
    rt_full = x[0, :].astype(np.float64)
    if np.nanmax(rt_full) > 200:
        rt_full = rt_full / 60.0
    mat = x[1:, :].astype(np.float64)

    if row_idx >= mat.shape[0]:
        raise IndexError(f"行索引 {row_idx} 超出 XIC 矩阵范围 (共 {mat.shape[0]} 行)")

    y_raw = mat[row_idx, :].astype(np.float64)
    if smooth_sigma > 0:
        y = gaussian_filter1d(y_raw, sigma=smooth_sigma)
    else:
        y = y_raw

    roi_csv = roi_windows_csv or locate_roi_csv(snr_dir, None)
    roi_map = load_roi_map(roi_csv) if roi_csv.is_file() else {}
    rw, _ = resolve_rt_window(roi_map, image_name)

    _matplotlib_cjk_font()
    fig, ax = plt.subplots(figsize=(10, 4))

    if window == "full":
        mask = slice(None)
        title_extra = "full XIC"
    elif window == "roi" and rw:
        lo, hi = rw
        mask = (rt_full >= lo - 0.5) & (rt_full <= hi + 0.5)
        title_extra = "ROI window [%.2f, %.2f] min" % (lo, hi)
    elif window == "refined":
        main_rt = safe_float(row.get("main_rt_peak"), np.nan)
        if np.isfinite(main_rt):
            mask = (rt_full >= main_rt - 1.0) & (rt_full <= main_rt + 1.0)
            title_extra = "main_rt=%.2f ± 1 min" % main_rt
        else:
            mask = slice(None)
            title_extra = "full (no main_rt)"
    else:
        mask = slice(None)
        title_extra = "full"

    ax.plot(rt_full[mask], y[mask], color="blue", linewidth=0.8)
    if rw:
        lo, hi = rw
        ax.axvspan(lo, hi, color="green", alpha=0.15, label="ROI window")
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("Intensity")
    ax.set_title("%s — %s" % (Path(image_name).stem if image_name else "?", title_extra))
    if rw:
        ax.legend(fontsize=7)
    fig.tight_layout()

    out = output_png or (refined_png.parent / ("%s_xic.png" % Path(image_name).stem))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print("[OK] %s" % out)


def main():
    ap = argparse.ArgumentParser(description="从 refined PNG 重绘原始 XIC")
    ap.add_argument("--refined_png", required=True, help="refined 图路径")
    ap.add_argument("--xic_matrix", default=None)
    ap.add_argument("--roi_windows_csv", default=None)
    ap.add_argument("--pred_refined_csv", default=None)
    ap.add_argument("--output_png", default=None)
    ap.add_argument("--window", default="roi", choices=("roi", "refined", "full"))
    ap.add_argument("--smooth_sigma", type=float, default=0.0)
    args = ap.parse_args()

    plot_xic_from_refined_png(
        refined_png=Path(args.refined_png),
        xic_matrix=Path(args.xic_matrix) if args.xic_matrix else None,
        roi_windows_csv=Path(args.roi_windows_csv) if args.roi_windows_csv else None,
        pred_refined_csv=Path(args.pred_refined_csv) if args.pred_refined_csv else None,
        output_png=Path(args.output_png) if args.output_png else None,
        window=args.window,
        smooth_sigma=args.smooth_sigma,
    )


if __name__ == "__main__":
    raise SystemExit(main())
