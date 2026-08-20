# -*- coding: utf-8 -*-
"""
针对「refined_plots 里峰与绿色 Main interval 对不上」等案例，逐步对照：
  Batch newtest → SNR 输出 → prediction_refined / post_newtest

用于定位偏差来自：compound/xic 行错位、ROI 窗不一致、SNR 重写 compound_name、
post_newtest 收框/双峰劈分、以及「用 SNR 的 roi_windows 重算像素→RT 是否与表里 rt 一致」等。

示例（89 / 83）：
  python -m <包名>.diagnostics.trace_refined_case ^
    --result_root "D:\\...\\result" ^
    --stems 89_mznan 83_mznan ^
    --snr_subdir "snr_filtered/json/SNR_box_0"
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .._shared.artifacts import load_roi_map, read_csv_safe

# 副本外依赖
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.roi_rt_mapping import box_to_rt_range


def _norm_img(s: str) -> str:
    return str(s).strip().replace("\\", "/")


def _apex_in_window(rt: np.ndarray, y: np.ndarray, lo: float, hi: float):
    m = (rt >= lo) & (rt <= hi) & np.isfinite(y)
    if not np.any(m):
        return float("nan"), float("nan")
    yi = y[m]
    ri = rt[m]
    j = int(np.argmax(yi))
    return float(ri[j]), float(yi[j])


def _find_rows(df: pd.DataFrame, stem: str):
    """
    stem 如 89_mznan：只匹配「基名」为 89_mznan 或 89_mznan_snr... 的行，
    避免 '89_mznan' 误命中 '1089_mznan'（子串包含）。
    """
    stem_l = stem.lower()
    out = []
    for i, r in df.iterrows():
        img = _norm_img(r.get("image", ""))
        bn = os.path.basename(img)
        pst = Path(bn).stem.lower()
        if pst == stem_l or pst.startswith(stem_l + "_"):
            out.append((i, r))
    return out


def _load_xic(path: Path):
    x = np.load(str(path))
    rt = x[0, :].astype(np.float64)
    if np.nanmax(rt) > 200:
        rt = rt / 60.0
    mat = x[1:, :].astype(np.float64)
    return rt, mat


def _roi_map_from_df(df: pd.DataFrame):
    m = {}
    for _, r in df.iterrows():
        key = _norm_img(r.get("image", ""))
        m[key] = (float(r["rt_lo"]), float(r["rt_hi"]))
    return m


def main():
    ap = argparse.ArgumentParser(description="诊断 refined 案例：逐层对照数据来源")
    ap.add_argument("--result_root", required=True, help="pipeline result 根目录")
    ap.add_argument("--stems", nargs="+", required=True, help="要诊断的 case stem，如 89_mznan")
    ap.add_argument("--snr_subdir", default="snr_filtered/json/SNR_box_0", help="SNR 子目录相对路径")
    args = ap.parse_args()

    root = Path(args.result_root)
    snr_base = root / args.snr_subdir

    pred_ref = snr_base / "prediction_refined.csv"
    pred_snr = snr_base / "prediction_snr.csv"
    if not pred_snr.is_file():
        pred_snr = snr_base / "prediction.csv"  # 兼容旧版 SNR 输出名
    roi_snr = snr_base / "roi_windows.csv"
    xic_npy = snr_base / "xic_matrix.npy"

    if pred_ref.is_file():
        df_ref = read_csv_safe(pred_ref)
        print("[INFO] prediction_refined: %d 行" % len(df_ref))
    else:
        df_ref = None
        print("[WARN] 无 prediction_refined.csv")

    if pred_snr.is_file():
        df_snr = read_csv_safe(pred_snr)
        print("[INFO] prediction (SNR): %d 行" % len(df_snr))
    else:
        df_snr = None

    roi_map = load_roi_map(roi_snr) if roi_snr.is_file() else {}
    print("[INFO] roi_map 条目: %d" % len(roi_map))

    rt_full = None
    mat = None
    if xic_npy.is_file():
        rt_full, mat = _load_xic(xic_npy)
        print("[INFO] XIC: %d rows, RT [%.2f, %.2f] min" % (mat.shape[0], rt_full[0], rt_full[-1]))

    for stem in args.stems:
        print("\n--- stem=%s ---" % stem)
        if df_ref is not None:
            rows = _find_rows(df_ref, stem)
            print("  refined 命中 %d 行" % len(rows))
            for idx, r in rows[:3]:
                print("    row %d: image=%s main_rt=%s" % (idx, r.get("image", ""), r.get("main_rt_peak", "")))

        if df_snr is not None:
            rows = _find_rows(df_snr, stem)
            print("  SNR 命中 %d 行" % len(rows))
            for idx, r in rows[:3]:
                img = _norm_img(r.get("image", ""))
                w = roi_map.get(img) or roi_map.get(os.path.basename(img))
                print("    row %d: image=%s roi_window=%s" % (idx, img, w))


if __name__ == "__main__":
    raise SystemExit(main())
