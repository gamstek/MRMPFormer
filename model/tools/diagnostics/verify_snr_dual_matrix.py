# -*- coding: utf-8 -*-
"""
验证：四象限矩阵（全局RMS vs 局部RMS）+ 局部峰峰辅助，能否区分空白假阳性与混标真峰。

对 prediction_refined 每个精修行（main_rt_min/max 框），在 ROI xic_matrix 上统一计算：
  - snr_global_rms : 噪声=框外全部点（ROI 内全局），基线=噪声均值，噪声量=σ（RMS）
  - snr_local_rms  : 噪声=框两侧紧邻点（与算法B同范围，max(3,0.2*框内点数)），基线=均值，噪声量=σ
  - snr_local_pp   : 算法 B（compute_snr_outside_box，峰峰定义，现有报告口径）
  - ratio          : snr_local_rms / snr_global_rms（同统计量，无结构时≈1）

四象限（阈值 3.0）：
  (全局≥3, 局部≥3) -> 保留      (全局<3, 局部<3) -> 纯噪声剔除
  (全局<3, 局部≥3) -> 多峰复核  (全局≥3, 局部<3) -> 当前可疑剔除

只读：不改任何产物；结果打印 + 写 verify_snr_dual_matrix.csv。
用法（model/ 下）：python -m tools.diagnostics.verify_snr_dual_matrix
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # model/
sys.path.insert(0, str(ROOT))

from postprocessing.peak_refinement import _load_optional_csv
from utils.xic_peak_utils import compute_snr_outside_box, get_noise_regions_outside_box

BASE = Path(__file__).resolve().parents[3] / "output" / "test" / "verify_step5" / "prediction_refined"

BLANK_SAMPLES = ["空白1", "空白1-2", "空白1-3"]
MIX_SAMPLES = ["农残混标", "农残混标-2", "农残混标-3"]

# 报告 WARN 中列出的空白假阳性（compound-离子）
BLANK_FP = [
    ("空白1", "杀扑磷-2"), ("空白1", "杀扑磷-2"), ("空白1", "杀扑磷-2"),
    ("空白1-2", "涕灭威-2"), ("空白1-2", "涕灭威-2"), ("空白1-2", "杀扑磷-2"),
    ("空白1-3", "涕灭威-1"), ("空白1-3", "涕灭威-2"), ("空白1-3", "涕灭威-2"),
    ("空白1-3", "杀扑磷-2"),
]

_IMG_RE = re.compile(r"_(\d+)-(.+?)-([12])(?:_snr[\d.e+-]*)?\.(?:jpe?g|png)$", re.I)


def _compound_ion_from_image(img):
    m = _IMG_RE.search(str(img or "").replace("\\", "/"))
    if m:
        return "%s-%s" % (m.group(2), m.group(3))
    return str(img or "")


def _rt_axis(xic):
    rt = xic[0, :].astype(np.float64)
    if np.nanmax(rt) > 200:
        rt = rt / 60.0
    return rt


def _resolve_row(image_name, feature_df, n_rows, mz=None, q3=None, compound_name=None):
    """
    样本目录 xic_matrix.npy 是 snr_filter 压缩后的（行索引=kept 顺序，1-based）。
    优先用精修行的 compound_name（即压缩索引）；其次图片前缀 N_mz；再 feature.csv 匹配。
    """
    try:
        cn = float(compound_name)
        if np.isfinite(cn) and cn >= 1:
            idx = int(cn) - 1
            if 0 <= idx < n_rows:
                return idx
    except (TypeError, ValueError):
        pass
    stem = Path(str(image_name).strip()).stem
    m = re.match(r"^(\d+)_mz", stem, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_rows:
            return idx
    if feature_df is not None and not feature_df.empty and mz is not None:
        mz_r = feature_df["mz"].apply(lambda v: round(float(v), 4) if pd.notna(v) else np.nan)
        q3_r = feature_df["q3"].apply(lambda v: round(float(v), 2) if pd.notna(v) else np.nan)
        if pd.notna(q3):
            mm = (mz_r == round(float(mz), 4)) & (q3_r == round(float(q3), 2))
        else:
            mm = mz_r == round(float(mz), 4)
        if mm.any():
            return int(np.flatnonzero(mm.to_numpy())[0])
    return None


def compute_snrs(rt, y, lo, hi, min_noise_pts=5):
    """同一精修框计算三种 SNR。返回 (snr_global_rms, snr_local_rms, snr_local_pp, ratio)。"""
    nan = float("nan")
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return nan, nan, nan, nan
    yy = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    m_in = (rt >= lo) & (rt <= hi)
    if np.sum(m_in) < 3:
        return nan, nan, nan, nan
    peak = float(np.max(yy[m_in]))

    # --- 全局 RMS（ROI 框外全部点） ---
    yn = yy[~m_in]
    if yn.size >= min_noise_pts:
        mu = float(np.mean(yn))
        sig = max(0.0, peak - mu)
        sigma = float(np.sqrt(np.mean((yn - mu) ** 2)))
        g = sig / sigma if sigma > 1e-15 else nan
    else:
        g = nan

    # --- 局部（框两侧紧邻，与算法 B 同范围） ---
    left, right = get_noise_regions_outside_box(rt, yy, lo, hi, min_points=3, frac=0.2)
    ln = np.concatenate([np.maximum(left.astype(np.float64), 0.0),
                         np.maximum(right.astype(np.float64), 0.0)])
    if ln.size >= min_noise_pts:
        mu_l = float(np.mean(ln))
        sig_l = max(0.0, peak - mu_l)
        sigma_l = float(np.sqrt(np.mean((ln - mu_l) ** 2)))
        l = sig_l / sigma_l if sigma_l > 1e-15 else nan
    else:
        l = nan

    # --- 局部峰峰（现有算法 B，报告口径） ---
    pp = compute_snr_outside_box(rt, yy, lo, hi)

    ratio = l / g if (np.isfinite(l) and np.isfinite(g) and g > 0) else nan
    return g, l, pp, ratio


def quadrant(g, l, thr=3.0):
    if not (np.isfinite(g) and np.isfinite(l)):
        return "无效(NaN)"
    g_ok = g >= thr
    l_ok = l >= thr
    if g_ok and l_ok:
        return "保留"
    if (not g_ok) and (not l_ok):
        return "纯噪声剔除"
    if (not g_ok) and l_ok:
        return "多峰复核"
    return "当前可疑剔除"


def collect(samples, label, out_rows):
    for sample in samples:
        sdir = BASE / sample
        refined = _load_optional_csv(sdir / "prediction_refined.csv")
        area_csv = _load_optional_csv(sdir / "prediction_refined_with_area.csv")
        snr_csv = _load_optional_csv(sdir / "prediction_snr.csv")
        if refined is None or refined.empty:
            print(f"[WARN] {sample}: 无 prediction_refined.csv")
            continue
        xic = np.load(str(sdir / "xic_matrix.npy"))
        rt = _rt_axis(xic)
        inten_mat = xic[1:, :].astype(np.float64)
        feature_df = _load_optional_csv(sdir / "feature.csv")
        snr_map = {}
        if snr_csv is not None and not snr_csv.empty and "image" in snr_csv.columns:
            snr_map = dict(zip(snr_csv["image"].astype(str).str.strip(),
                               snr_csv.get("snr_outside_box", np.nan)))
        area_map = {}
        if area_csv is not None and not area_csv.empty and "image" in area_csv.columns:
            area_map = dict(zip(area_csv["image"].astype(str).str.strip(),
                                area_csv.get("main_area", np.nan)))
        for _, r in refined.iterrows():
            lo = r.get("main_rt_min")
            hi = r.get("main_rt_max")
            try:
                lo_f, hi_f = float(lo), float(hi)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(lo_f) and np.isfinite(hi_f) and hi_f > lo_f):
                continue
            img = str(r.get("image", "")).strip()
            idx = _resolve_row(img, feature_df, int(inten_mat.shape[0]),
                               mz=r.get("mz"), q3=r.get("q3"),
                               compound_name=r.get("compound_name"))
            if idx is None or not (0 <= idx < inten_mat.shape[0]):
                continue
            y = np.maximum(inten_mat[idx, :].astype(np.float64), 0.0)
            g, l, pp, ratio = compute_snrs(rt, y, lo_f, hi_f)
            comp = _compound_ion_from_image(img)
            main_snr = r.get("main_snr")
            main_area = area_map.get(img, r.get("main_area", np.nan))
            try:
                main_area_f = float(main_area)
            except (TypeError, ValueError):
                main_area_f = np.nan
            out_rows.append({
                "sample": sample, "label": label, "compound_ion": comp,
                "image": Path(img).stem, "main_rt_peak": r.get("main_rt_peak"),
                "main_height": r.get("main_height"), "score": r.get("main_score_ai"),
                "main_snr_stage4": main_snr,
                "snr_outside_box_stage3": snr_map.get(img, np.nan),
                "snr_global_rms": g, "snr_local_rms": l, "snr_local_pp": pp,
                "ratio_local_over_global": ratio, "quadrant": quadrant(g, l),
                "main_area": main_area_f,
                "is_detected": bool(np.isfinite(main_area_f) and main_area_f > 0),
                "is_blank_fp": (sample, comp) in BLANK_FP,
            })


def main():
    out_rows = []
    collect(BLANK_SAMPLES, "blank", out_rows)
    collect(MIX_SAMPLES, "mix", out_rows)
    df = pd.DataFrame(out_rows)
    if df.empty:
        print("[ERROR] 无数据")
        return 1

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    show = df[["sample", "label", "compound_ion", "main_rt_peak", "main_height",
               "score", "main_snr_stage4", "snr_outside_box_stage3",
               "snr_global_rms", "snr_local_rms", "snr_local_pp",
               "ratio_local_over_global", "quadrant", "is_blank_fp"]].copy()
    print("\n===== 明细（空白的 10 个 FP 标记 is_blank_fp=True）=====")
    print(show.to_string(index=False))

    # 汇总
    print("\n===== 汇总：四象限分布 =====")
    piv = df.groupby(["label", "quadrant"]).size().unstack(fill_value=0)
    print(piv.to_string())

    print("\n===== 关键检查 =====")
    blank_fp = df[df["is_blank_fp"]]
    blank_all = df[df["label"] == "blank"]
    mix = df[df["label"] == "mix"]
    keep_quad = {"保留"}
    reject_quad = {"纯噪声剔除", "当前可疑剔除"}

    n_fp = len(blank_fp)
    n_fp_kept = int(blank_fp["quadrant"].isin(keep_quad).sum()) if n_fp else 0
    n_fp_rejected = int(blank_fp["quadrant"].isin(reject_quad).sum()) if n_fp else 0
    n_fp_review = n_fp - n_fp_kept - n_fp_rejected
    print(f"空白 FP（WARN 10 个）: 共 {n_fp} | 保留 {n_fp_kept} | 剔除 {n_fp_rejected} | 多峰复核 {n_fp_review}")
    n_b = len(blank_all)
    print(f"空白全部精修行: 共 {n_b} | 落入保留象限 {int(blank_all['quadrant'].isin(keep_quad).sum())} | "
          f"落入剔除象限 {int(blank_all['quadrant'].isin(reject_quad).sum())}")
    n_m = len(mix)
    mix_det = mix[mix["is_detected"]]
    n_md = len(mix_det)
    n_m_kept = int(mix_det["quadrant"].isin(keep_quad).sum()) if n_md else 0
    n_m_rej = int(mix_det["quadrant"].isin(reject_quad).sum()) if n_md else 0
    print(f"混标检出真峰(main_area>0): 共 {n_md} | 保留 {n_m_kept} ({n_m_kept / max(n_md, 1) * 100:.0f}%) | "
          f"误剔 {n_m_rej} | 多峰复核 {n_md - n_m_kept - n_m_rej}")
    if n_m - n_md > 0:
        print(f"混标未检出行(main_area<=0): 共 {n_m - n_md}（不计入真峰统计）")

    # 同化合物 空白 vs 混标
    print("\n===== 同化合物对比（空白 FP 化合物在混标中的表现）=====")
    fp_comps = sorted({c for _, c in BLANK_FP})
    for c in fp_comps:
        in_blank = df[(df["label"] == "blank") & (df["compound_ion"] == c)]
        in_mix = df[(df["label"] == "mix") & (df["compound_ion"] == c)]
        if in_blank.empty and in_mix.empty:
            continue
        bq = in_blank["quadrant"].value_counts().to_dict()
        mq = in_mix["quadrant"].value_counts().to_dict()
        print(f"  {c}: 空白象限={bq} | 混标象限={mq} | "
              f"混标局部RMS中位={in_mix['snr_local_rms'].median():.1f} 全局RMS中位={in_mix['snr_global_rms'].median():.1f}")

    # 判别力 gap 分析：找三种 SNR 的完美分离区间
    print("\n===== 判别力（空白 vs 混标真峰 的数值鸿沟）=====")
    blank_all = df[df["label"] == "blank"]
    mix_all = df[df["label"] == "mix"]
    for col in ["snr_global_rms", "snr_local_rms", "snr_local_pp", "main_snr_stage4"]:
        if col not in df.columns:
            continue
        bv = blank_all[col].dropna()
        mv = mix_all[col].dropna()
        if bv.empty or mv.empty:
            continue
        b_max, m_min = float(bv.max()), float(mv.min())
        sep = "可完美分离" if m_min > b_max else "有重叠"
        print(f"  {col:20s}: 空白最大={b_max:9.2f} | 混标真峰最小={m_min:9.2f} | 完美分离阈值区间=[{b_max:.1f},{m_min:.1f}] -> {sep}")

    out_csv = BASE / ".." / "verify_snr_dual_matrix.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 结果已写: {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
