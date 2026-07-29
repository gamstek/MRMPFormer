# -*- coding: utf-8 -*-
"""
检测「ROI 图 模型框(像素 x) → RT」是否与 prediction.csv 中 rt_min/rt_max 一致，
并在提供 xic_matrix.npy 时检查 ROI 时间窗内谱峰顶与框在 RT 上是否严重错位。

用法示例：
  python -m <包名>.diagnostics.check_box_rt_mapping ^
    --prediction_csv "D:\\...\\batch_predictions\\json\\prediction.csv" ^
    --roi_windows_csv "D:\\...\\xic-roi-batch\\json\\roi_windows.csv" ^
    --xic_matrix "D:\\...\\xic-roi-batch\\json\\xic_matrix.npy"

SNR 之后的目录（image 可能为 筛选保留/xxx.jpeg）请指向同一次运行写入的 roi_windows.csv：
  ...\\snr_filtered\\json\\SNR_box_3\\roi_windows.csv
  ...\\snr_filtered\\json\\SNR_box_3\\prediction.csv
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .._shared.artifacts import (
    load_roi_map,
    read_csv_safe,
    resolve_rt_window,
    safe_float,
)

# 副本外依赖：仅在此处集中引用
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.roi_rt_mapping import box_to_rt_range


def _apex_in_window(rt: np.ndarray, y: np.ndarray, lo: float, hi: float):
    m = (rt >= lo) & (rt <= hi) & np.isfinite(y)
    if not np.any(m):
        return float("nan"), float("nan")
    yi = y[m]
    ri = rt[m]
    j = int(np.argmax(yi))
    return float(ri[j]), float(yi[j])


def check_box_rt(
    pred_csv: Path,
    roi_csv: Path,
    xic_npy: Optional[Path] = None,
):
    df = read_csv_safe(pred_csv)
    roi_map = load_roi_map(roi_csv)

    n_ok = 0
    n_mismatch = 0
    n_no_window = 0

    for i, row in df.iterrows():
        img = str(row.get("image", "")).strip()
        rt_min_pred = safe_float(row.get("rt_min"), np.nan)
        rt_max_pred = safe_float(row.get("rt_max"), np.nan)

        window, note = resolve_rt_window(roi_map, img)
        if window is None:
            n_no_window += 1
            continue

        rt_lo, rt_hi = window
        if abs(rt_min_pred - rt_lo) < 0.001 and abs(rt_max_pred - rt_hi) < 0.001:
            n_ok += 1
        else:
            n_mismatch += 1
            if n_mismatch <= 10:
                print(
                    "[MISMATCH] row %d image=%s pred=[%.4f,%.4f] roi=[%.4f,%.4f]"
                    % (i, img, rt_min_pred, rt_max_pred, rt_lo, rt_hi)
                )

    print("[RESULT] OK=%d MISMATCH=%d NO_WINDOW=%d" % (n_ok, n_mismatch, n_no_window))

    if xic_npy and xic_npy.is_file():
        x = np.load(str(xic_npy))
        rt_full = x[0, :].astype(np.float64)
        if np.nanmax(rt_full) > 200:
            rt_full = rt_full / 60.0
        mat = x[1:, :].astype(np.float64)
        print("[INFO] XIC shape: %s, RT range [%.2f, %.2f] min" % (mat.shape, rt_full[0], rt_full[-1]))


def main():
    ap = argparse.ArgumentParser(description="检查 ROI 框 → RT 映射一致性")
    ap.add_argument("--prediction_csv", required=True)
    ap.add_argument("--roi_windows_csv", required=True)
    ap.add_argument("--xic_matrix", default=None)
    args = ap.parse_args()

    pred = Path(args.prediction_csv)
    roi = Path(args.roi_windows_csv)
    xic = Path(args.xic_matrix) if args.xic_matrix else None

    if not pred.is_file():
        print("[ERROR] prediction_csv 不存在: %s" % pred)
        sys.exit(1)
    if not roi.is_file():
        print("[ERROR] roi_windows_csv 不存在: %s" % roi)
        sys.exit(1)

    check_box_rt(pred, roi, xic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
