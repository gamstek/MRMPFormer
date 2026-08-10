# -*- coding: utf-8 -*-
"""
结合 prediction.csv（CNN/Transformer 检测框）与同一样本的 mzML，在整条 XIC 上：
  - 从 mzML 读出色谱后，对强度做 **一维高斯平滑**（`scipy.ndimage.gaussian_filter1d`）；命令行用 **`--gaussian_sigma`**（或 **`--smooth_sigma`**）指定 sigma，默认 0.8；设为 **0** 表示不平滑；
  - 将预测框在 x 方向映射到保留时间区间 [rt_box_lo, rt_box_hi]；
  - **框外**所有数据点视为噪声区，估计均值与 RMS；
  - **框内**信号取 max(强度) − 噪声区均值，SNR = 信号 / RMS_noise（均基于平滑后的强度）。

输出（均写在 --output_dir 下自动创建的 SNR_box_<阈值>/ 内）：
  - box_outside_snr_report.csv：每条 prediction 行的原始 SNR 计算结果；
  - prediction.csv：仅保留 SNR ≥ 阈值 的行（列与输入 prediction 对齐，并追加 snr_outside_box 等）；
  - feature.csv、roi_windows.csv、xic_matrix.npy：仅针对保留行，与项目惯例一致；
  - 筛选保留/、筛选剔除/：带红色预测框的 ROI jpeg（文件名含 SNR）。

依赖：pyopenms、numpy、pandas、scipy、matplotlib

示例：
  python mzml_box_outside_snr_pipeline.py ^
    --mzml "D:\\data\\run.mzML" ^
    --prediction_csv "D:\\out\\prediction.csv" ^
    --roi_windows_csv "D:\\out\\roi_windows.csv" ^
    --output_dir "D:\\results\\box_snr" ^
    --min_snr 5 ^
    --gaussian_sigma 0.8

roi_windows.csv 可选；若缺省则用 old_rt/retention_time 与 XIC 轴按 roi_rt_mapping 推算 ±1 min 窗口。
不修改项目内其它已有脚本。
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pyopenms import MSExperiment, MzMLFile
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from utils.roi_rt_mapping import (
    ROI_IMAGE_HEIGHT_PX,
    ROI_IMAGE_WIDTH_PX,
    box_x_to_rt_minutes,
    rt_window_bounds_minutes,
)


def _native_id_to_str(native_id: Any) -> str:
    if native_id is None:
        return ""
    if isinstance(native_id, bytes):
        return native_id.decode("utf-8", errors="replace")
    return str(native_id)


def _parse_q1_q3_from_text(native_id_text: str) -> Tuple[Optional[float], Optional[float]]:
    """从 native_id 文本解析 Q1/Q3（兼容无 Q1=/Q3= 的厂商格式）。"""
    text = str(native_id_text)
    q1 = q3 = None
    for pat in (r"Q1=([\d\.]+)", r"q1=([\d\.]+)", r"precursor[=:_ ]([\d\.]+)"):
        m1 = re.search(pat, text)
        if m1:
            q1 = float(m1.group(1))
            break
    for pat in (r"Q3=([\d\.]+)", r"q3=([\d\.]+)", r"product[=:_ ]([\d\.]+)"):
        m3 = re.search(pat, text)
        if m3:
            q3 = float(m3.group(1))
            break
    return q1, q3


def _q1_q3_from_chrom_metadata(chrom) -> Tuple[Optional[float], Optional[float]]:
    """与 testmzml / pyopenms 一致：优先用 chromatogram 的 precursor / product m/z。"""
    q1 = q3 = None
    try:
        mz_pre = float(chrom.getPrecursor().getMZ())
        if np.isfinite(mz_pre) and mz_pre > 0:
            q1 = mz_pre
    except Exception:
        pass
    try:
        mz_pro = float(chrom.getProduct().getMZ())
        if np.isfinite(mz_pro) and mz_pro > 0:
            q3 = mz_pro
    except Exception:
        pass
    return q1, q3


def _extract_q1_q3(chrom, native_id_text: str) -> Tuple[Optional[float], Optional[float]]:
    q1, q3 = _parse_q1_q3_from_text(native_id_text)
    mq1, mq3 = _q1_q3_from_chrom_metadata(chrom)
    if q1 is None:
        q1 = mq1
    if q3 is None:
        q3 = mq3
    return q1, q3


def _collect_chroms(mzml_path: str, smooth_sigma: float) -> List[dict]:
    from utils.mzml_load import load_ms_experiment

    exp = load_ms_experiment(mzml_path)
    chromatograms = exp.getChromatograms()
    if len(chromatograms) == 0:
        raise ValueError("mzML 中无 chromatogram。")
    seen = set()
    out: List[dict] = []
    for i, chrom in enumerate(chromatograms):
        rt_sec = np.array([p.getRT() for p in chrom], dtype=np.float64)
        intensity = np.array([p.getIntensity() for p in chrom], dtype=np.float64)
        if rt_sec.size == 0:
            continue
        native_id_raw = chrom.getNativeID()
        native_id = _native_id_to_str(native_id_raw)
        q1, q3 = _extract_q1_q3(chrom, native_id)
        key = (round(q1, 4) if q1 is not None else None, round(q3, 2) if q3 is not None else None)
        if key in seen:
            continue
        seen.add(key)
        if smooth_sigma and smooth_sigma > 0:
            intensity = gaussian_filter1d(intensity, sigma=smooth_sigma)
        out.append(
            {
                "chrom_order": len(out),
                "source_index": i,
                "native_id": native_id,
                "q1": q1,
                "q3": q3,
                "rt_sec": rt_sec,
                "intensity": intensity.copy(),
            }
        )
    return out


def _chrom_lookup_by_mzq3(chroms: List[dict]) -> Dict[Tuple, dict]:
    m: Dict[Tuple, dict] = {}
    for r in chroms:
        k = (
            round(r["q1"], 4) if r["q1"] is not None else None,
            round(r["q3"], 2) if r["q3"] is not None else None,
        )
        m[k] = r
    return m


def _read_df_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    return pd.read_csv(path)


def _load_roi_windows(path: Optional[Path]) -> Dict[str, Tuple[float, float]]:
    if not path or not path.is_file():
        return {}
    try:
        df = _read_df_csv(path)
        if "image" not in df.columns or "rt_lo" not in df.columns or "rt_hi" not in df.columns:
            return {}
        return {
            str(row["image"]).strip(): (float(row["rt_lo"]), float(row["rt_hi"]))
            for _, row in df.iterrows()
        }
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def _true_rt_from_row(row: pd.Series) -> float:
    for k in ("retention_time", "old_rt", "RT"):
        if k in row.index and pd.notna(row.get(k)):
            try:
                return float(row[k])
            except (TypeError, ValueError):
                pass
    return 0.0


def snr_outside_prediction_box(
    rt_sec: np.ndarray,
    intensity: np.ndarray,
    x1: float,
    x2: float,
    rt_lo: float,
    rt_hi: float,
    min_noise_pts: int = 5,
) -> Tuple[float, float, float, float, float]:
    """
    框外噪声 RMS，框内峰高相对框外均值的超出为信号。
    返回 (snr, noise_rms, signal, mu_noise, rt_box_lo)  最后一个为 rt_box_lo 便于调试；实际返回五元组带上 rt_box_hi
    返回 (snr, noise_rms, signal, mu_noise, n_noise_pts)
    """
    rt_sec = np.asarray(rt_sec, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if rt_hi <= rt_lo or not np.isfinite(x1) or not np.isfinite(x2):
        return float("nan"), float("nan"), float("nan"), float("nan"), 0.0
    xa, xb = float(x1), float(x2)
    if xa > xb:
        xa, xb = xb, xa
    t_lo = box_x_to_rt_minutes(xa, rt_lo, rt_hi)
    t_hi = box_x_to_rt_minutes(xb, rt_lo, rt_hi)
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    # RT 用分钟，与 rt_sec/60 一致
    rtm = rt_sec / 60.0
    noise_m = (rtm < t_lo) | (rtm > t_hi)
    inside_m = (rtm >= t_lo) & (rtm <= t_hi)
    if np.sum(noise_m) < min_noise_pts or np.sum(inside_m) < 1:
        return float("nan"), float("nan"), float("nan"), float("nan"), float(np.sum(noise_m))
    yn = y[noise_m]
    mu_n = float(np.mean(yn))
    resid = yn - mu_n
    n_rms = float(np.sqrt(np.mean(resid**2)))
    if n_rms <= 1e-15:
        n_rms = 1e-15
    peak_in = float(np.max(y[inside_m]))
    sig = max(0.0, peak_in - mu_n)
    snr = sig / n_rms
    return snr, n_rms, sig, mu_n, float(np.sum(noise_m))


def _snr_suffix(v: float) -> str:
    if not np.isfinite(v):
        return "_snrNA"
    x = float(v)
    if abs(x - round(x)) < 1e-9 * max(1.0, abs(x)):
        return "_snr%d" % int(round(x))
    return "_snr%s" % ("%.6g" % x)


def _safe_jpeg_name_from_prediction_row(row: pd.Series, snr: float) -> str:
    """用于保存图：原 image 列主名 + snr。"""
    img = str(row.get("image", "row")).strip() or "row"
    stem = Path(img).stem
    bad = '<>:"/\\|?*'
    for c in bad:
        stem = stem.replace(c, "_")
    return "%s%s.jpeg" % (stem, _snr_suffix(snr))


def _pixel_y_to_intensity(py: float, y_min: float, y_max: float) -> float:
    """模型框 y（0 顶、300 底）→ 与 roi_rt_mapping 一致的强度轴。"""
    py = float(np.clip(py, 0.0, ROI_IMAGE_HEIGHT_PX))
    t = (ROI_IMAGE_HEIGHT_PX - py) / ROI_IMAGE_HEIGHT_PX
    return float(y_min + t * (y_max - y_min))


def save_roi_jpeg_with_box(
    dest_path: str,
    rt_sec: np.ndarray,
    intensity: np.ndarray,
    row: pd.Series,
    rt_win_lo: float,
    rt_win_hi: float,
    snr_display: Optional[float] = None,
) -> None:
    """与 testXIC 相同的 ±1 min ROI 图，并绘制 prediction 中的红框与置信度。"""
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    rt_sec = np.asarray(rt_sec, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    max_idx = int(np.argmax(intensity))
    rt_apex_min = float(rt_sec[max_idx] / 60.0)
    wh = 1.0
    rt_start_sec = max((rt_apex_min - wh) * 60.0, float(rt_sec[0]))
    rt_end_sec = min((rt_apex_min + wh) * 60.0, float(rt_sec[-1]))
    mask = (rt_sec >= rt_start_sec) & (rt_sec <= rt_end_sec)
    plot_rt_sec = rt_sec[mask]
    plot_intensity = intensity[mask]

    y_min = float(np.min(plot_intensity)) if plot_intensity.size else 0.0
    y_max = float(np.max(plot_intensity)) if plot_intensity.size else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    fig = Figure(figsize=(4, 3), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(plot_rt_sec / 60.0, plot_intensity, color="blue", linewidth=1.5)
    if rt_end_sec > rt_start_sec and plot_rt_sec.size > 0:
        ax.set_xlim(rt_start_sec / 60.0, rt_end_sec / 60.0)
    ax.set_ylim(y_min, y_max)

    try:
        x1 = float(row["box_x1"])
        y1 = float(row["box_y1"])
        x2 = float(row["box_x2"])
        y2 = float(row["box_y2"])
    except (TypeError, ValueError, KeyError):
        x1 = y1 = x2 = y2 = float("nan")

    if (
        rt_win_hi > rt_win_lo
        and np.isfinite(x1)
        and np.isfinite(x2)
        and np.isfinite(y1)
        and np.isfinite(y2)
    ):
        xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
        ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
        rt_left = box_x_to_rt_minutes(xa, rt_win_lo, rt_win_hi)
        rt_right = box_x_to_rt_minutes(xb, rt_win_lo, rt_win_hi)
        if rt_left > rt_right:
            rt_left, rt_right = rt_right, rt_left
        iy_lo = _pixel_y_to_intensity(yb, y_min, y_max)
        iy_hi = _pixel_y_to_intensity(ya, y_min, y_max)
        rect = Rectangle(
            (rt_left, min(iy_lo, iy_hi)),
            max(rt_right - rt_left, 1e-6),
            max(abs(iy_hi - iy_lo), 1e-6),
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        sc = row.get("score")
        if sc is not None and pd.notna(sc):
            try:
                sf = float(sc)
                if np.isfinite(sf):
                    ax.text(
                        rt_left,
                        max(iy_lo, iy_hi),
                        "%.3f" % sf,
                        color="white",
                        fontsize=8,
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="red", edgecolor="none", alpha=0.85),
                    )
            except (TypeError, ValueError):
                pass

    # 实际 / 积分 RT 竖线 + 左上角信息（筛选保留与筛选剔除共用）
    tr_min = float(_true_rt_from_row(row))
    rt_plot_lo = float(rt_start_sec / 60.0)
    rt_plot_hi = float(rt_end_sec / 60.0)
    if np.isfinite(tr_min) and rt_plot_lo <= tr_min <= rt_plot_hi:
        ax.axvline(tr_min, color="#c0392b", linestyle="--", linewidth=1.6, zorder=6)

    try:
        cn = row.get("compound_name", "")
        mz_v = row.get("mz", "")
        q3_v = row.get("q3", "")
        imx = float(row["intensity_max"]) if "intensity_max" in row.index and pd.notna(row.get("intensity_max")) else float("nan")
        pts = float(row["point_counts"]) if "point_counts" in row.index and pd.notna(row.get("point_counts")) else float("nan")
        snr_v = snr_display
        if snr_v is None or (isinstance(snr_v, float) and not np.isfinite(snr_v)):
            if "snr_outside_box" in row.index and pd.notna(row.get("snr_outside_box")):
                snr_v = float(row["snr_outside_box"])
            elif "snr" in row.index and pd.notna(row.get("snr")):
                snr_v = float(row["snr"])
            else:
                snr_v = float("nan")
        imx_s = "%.6g" % imx if np.isfinite(imx) else "—"
        snr_s = "%.4g" % snr_v if np.isfinite(snr_v) else "—"
        pts_s = "%d" % int(pts) if np.isfinite(pts) else "—"
        tr_s = "%.4f" % tr_min if np.isfinite(tr_min) else "—"
        info = (
            "积分RT: %s min\n初步匹配化合物: #%s mz=%s q3=%s\n响应(强度max): %s\n信噪比: %s\n扫描点数: %s"
            % (tr_s, cn, mz_v, q3_v, imx_s, snr_s, pts_s)
        )
        fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.06)
        fig.text(
            0.02,
            0.99,
            info,
            transform=fig.transFigure,
            va="top",
            ha="left",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.75", alpha=0.9),
        )
    except Exception:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.print_jpeg(dest_path)


def _snr_box_run_dir_name(min_snr_cli: float) -> str:
    if min_snr_cli < 0:
        return "SNR_box_all"
    t = float(min_snr_cli)
    if abs(t - round(t)) < 1e-9:
        return "SNR_box_%d" % int(round(t))
    return "SNR_box_%s" % ("%.10g" % t)


def _df_to_csv_safe(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        alt = path.with_name("%s_%s%s" % (path.stem, time.strftime("%Y%m%d_%H%M%S"), path.suffix))
        print("[WARN] 无法写入 %s，已改为 %s" % (path, alt))
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        return alt


def _npy_save_safe(arr: np.ndarray, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(path, arr)
        return path
    except PermissionError:
        alt = path.with_name("%s_%s%s" % (path.stem, time.strftime("%Y%m%d_%H%M%S"), path.suffix))
        np.save(alt, arr)
        return alt


def _match_chrom(
    row: pd.Series,
    by_mz: Dict[Tuple, dict],
    chroms: List[dict],
) -> Optional[dict]:
    mz = row.get("mz")
    q3 = row.get("q3")
    try:
        if mz is not None and not (isinstance(mz, float) and np.isnan(mz)):
            mz_f = float(mz)
        else:
            mz_f = None
        if q3 is not None and not (isinstance(q3, float) and np.isnan(q3)):
            q3_f = float(q3)
        else:
            q3_f = None
    except (TypeError, ValueError):
        mz_f = q3_f = None
    k = (
        round(mz_f, 4) if mz_f is not None else None,
        round(q3_f, 2) if q3_f is not None else None,
    )
    if k in by_mz:
        return by_mz[k]
    cname = row.get("compound_name")
    if cname is not None and not (isinstance(cname, float) and np.isnan(cname)):
        try:
            idx = int(float(cname)) - 1
            if 0 <= idx < len(chroms):
                return chroms[idx]
        except (TypeError, ValueError):
            pass
    return None


def run(
    mzml_path: str,
    prediction_csv: str,
    output_dir: str,
    min_snr_eff: float,
    min_snr_cli: float,
    roi_windows_csv: Optional[str],
    smooth_sigma: float,
    min_noise_pts: int,
    min_chrom_points: int = 0,
    min_chrom_max_intensity: float = 0.0,
) -> int:
    parent = Path(output_dir).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / _snr_box_run_dir_name(min_snr_cli)
    run_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] 输出根目录: %s" % run_dir)

    pred_path = Path(prediction_csv)
    if not pred_path.is_file():
        print("[ERROR] 找不到 prediction.csv: %s" % pred_path)
        return 1
    df_pred = _read_df_csv(pred_path)
    required = {"box_x1", "box_y1", "box_x2", "box_y2", "image"}
    if not required.issubset(df_pred.columns):
        print("[ERROR] prediction.csv 缺少列: %s" % (required - set(df_pred.columns)))
        return 1

    roi_map = _load_roi_windows(Path(roi_windows_csv) if roi_windows_csv else None)

    chroms = _collect_chroms(mzml_path, smooth_sigma)
    if not chroms:
        print("[ERROR] 未解析到任何色谱。")
        return 1
    print(
        "[INFO] 高斯平滑 sigma = %s（gaussian_filter1d；0 为不平滑）"
        % smooth_sigma
    )
    by_mz = _chrom_lookup_by_mzq3(chroms)

    dir_keep = run_dir / "筛选保留"
    dir_drop = run_dir / "筛选剔除"
    dir_keep.mkdir(parents=True, exist_ok=True)
    dir_drop.mkdir(parents=True, exist_ok=True)

    report_rows: List[Dict[str, Any]] = []
    passed_records: List[Tuple[dict, pd.Series, float, str]] = []

    for idx, row in df_pred.iterrows():
        if "excluded" in df_pred.columns and row.get("excluded") == 1:
            continue
        ch = _match_chrom(row, by_mz, chroms)
        img_key = str(row.get("image", "")).strip()
        rt_axis = ch["rt_sec"] / 60.0 if ch is not None else None
        if img_key in roi_map:
            rt_lo, rt_hi = roi_map[img_key]
        else:
            if ch is None:
                tr = _true_rt_from_row(row)
                rt_lo, rt_hi = 0.0, 1.0
            else:
                tr = _true_rt_from_row(row)
                rt_lo, rt_hi = rt_window_bounds_minutes(tr, rt_axis)

        try:
            x1 = float(row["box_x1"])
            x2 = float(row["box_x2"])
        except (TypeError, ValueError):
            x1 = x2 = float("nan")

        if ch is None:
            snr = n_rms = sig = mu_n = n_n = float("nan")
        else:
            snr, n_rms, sig, mu_n, n_n = snr_outside_prediction_box(
                ch["rt_sec"],
                ch["intensity"],
                x1,
                x2,
                rt_lo,
                rt_hi,
                min_noise_pts=min_noise_pts,
            )

        fname = _safe_jpeg_name_from_prediction_row(row, snr)
        point_ok = True
        inten_ok = True
        if ch is not None:
            if min_chrom_points > 0 and int(ch["rt_sec"].size) < int(min_chrom_points):
                point_ok = False
            if min_chrom_max_intensity > 0.0:
                mx_ch = float(np.max(ch["intensity"]))
                if mx_ch < float(min_chrom_max_intensity):
                    inten_ok = False
        ok = (
            ch is not None
            and np.isfinite(snr)
            and snr >= min_snr_eff
            and point_ok
            and inten_ok
        )
        if ch is not None:
            out_img = (dir_keep if ok else dir_drop) / fname
            save_roi_jpeg_with_box(
                str(out_img),
                ch["rt_sec"],
                ch["intensity"],
                row,
                rt_lo,
                rt_hi,
                snr_display=float(snr),
            )
        if ok:
            passed_records.append((ch, row, snr, fname))

        rec = {k: row[k] for k in df_pred.columns if k in row.index}
        rec["snr_outside_box"] = snr
        rec["noise_rms_outside_box"] = n_rms
        rec["signal_outside_mean"] = sig
        rec["noise_mean_outside_box"] = mu_n
        rec["n_noise_points"] = n_n
        rec["rt_window_lo_used"] = rt_lo
        rec["rt_window_hi_used"] = rt_hi
        rec["passed_snr_threshold"] = bool(ok)
        rec["passed_min_chrom_points"] = bool(point_ok)
        rec["passed_min_chrom_max_intensity"] = bool(inten_ok)
        rec["pred_row_index"] = idx
        report_rows.append(rec)

    df_report = pd.DataFrame(report_rows)
    rep_path = _df_to_csv_safe(df_report, run_dir / "box_outside_snr_report.csv")
    print("[INFO] SNR 报告: %s （%d 行）" % (rep_path, len(df_report)))

    if not passed_records:
        print("[WARN] 无通过 SNR 阈值的行，不生成 prediction.csv / xic_matrix。")
        return 0

    kept_rows: List[Dict[str, Any]] = []
    features: List[dict] = []
    roi_w: List[dict] = []
    intensity_matrix: List[np.ndarray] = []

    # Build a unified RT axis across all passed records.
    # If we only use the first record's RT axis, other records might fall outside
    # and interpolation (fill_value=0.0) turns many xic_matrix rows into all-zeros,
    # which makes downstream plots look "empty".
    rt_mins = [ch["rt_sec"] / 60.0 for (ch, _, _, _) in passed_records]
    global_min = float(min(float(rt.min()) for rt in rt_mins if rt.size > 0))
    global_max = float(max(float(rt.max()) for rt in rt_mins if rt.size > 0))

    steps: List[float] = []
    for rt in rt_mins:
        if rt.size < 3:
            continue
        rts = np.sort(rt.astype(np.float64))
        diffs = np.diff(rts)
        diffs = diffs[diffs > 0]
        if diffs.size > 0:
            steps.append(float(np.median(diffs)))
    step = float(np.median(steps)) if steps else 0.01
    if not np.isfinite(step) or step <= 0:
        step = 0.01

    n_pts = int(np.ceil((global_max - global_min) / step)) + 1
    n_pts = max(n_pts, 50)
    n_pts = min(n_pts, 50000)
    common_rt = np.linspace(global_min, global_max, n_pts).astype(np.float64)

    for j, (ch, row, snr, fname) in enumerate(passed_records, start=1):
        rt_sec = ch["rt_sec"]
        intensity = ch["intensity"]
        q1, q3 = ch["q1"], ch["q3"]
        max_idx = int(np.argmax(intensity))
        rt_apex_min = float(rt_sec[max_idx] / 60.0)

        out_row = {k: row[k] for k in df_pred.columns if k in row.index}
        out_row["image"] = fname
        # Important: xic_matrix.npy/features are compacted by `j` (1-based) below.
        # prediction.csv must use the same compacted index; otherwise later mapping from
        # (image/compound_name) -> xic_matrix row index will be wrong and plots become blank.
        if "compound_name" in out_row:
            out_row["compound_name"] = int(j)
        else:
            out_row["compound_name"] = int(j)
        out_row["snr_outside_box"] = snr
        kept_rows.append(out_row)

        features.append(
            {
                "Compound Name": j,
                "mz": q1 if q1 is not None else np.nan,
                "q3": q3 if q3 is not None else np.nan,
                "RT": round(rt_apex_min, 3),
            }
        )

        rt_min = rt_sec / 60.0
        if len(rt_min) == len(common_rt) and np.allclose(rt_min, common_rt, atol=1e-3):
            aligned = intensity
        else:
            f = interp1d(rt_min, intensity, bounds_error=False, fill_value=0.0)
            aligned = f(common_rt)
        intensity_matrix.append(aligned)

        wh = 1.0
        rt_start_sec = max((rt_apex_min - wh) * 60.0, float(rt_sec[0]))
        rt_end_sec = min((rt_apex_min + wh) * 60.0, float(rt_sec[-1]))
        rel = Path("筛选保留") / fname
        roi_w.append(
            {
                "image": rel.as_posix(),
                "rt_lo": rt_start_sec / 60.0,
                "rt_hi": rt_end_sec / 60.0,
            }
        )

    _df_to_csv_safe(pd.DataFrame(kept_rows), run_dir / "prediction.csv")
    _df_to_csv_safe(pd.DataFrame(features), run_dir / "feature.csv")
    _df_to_csv_safe(pd.DataFrame(roi_w), run_dir / "roi_windows.csv")
    arr = np.array(intensity_matrix)
    xic_full = np.vstack([common_rt, arr])
    _npy_save_safe(xic_full, run_dir / "xic_matrix.npy")
    print("[INFO] 已写入 prediction.csv / feature.csv / roi_windows.csv / xic_matrix.npy")
    print("[DONE] 筛选保留: %s ，筛选剔除: %s" % (dir_keep, dir_drop))
    return 0


def main():
    ap = argparse.ArgumentParser(description="mzML + prediction 框外噪声 SNR 筛选管线")
    ap.add_argument("--mzml", required=True)
    ap.add_argument("--prediction_csv", required=True)
    ap.add_argument("--roi_windows_csv", default="", help="可选；需含 image, rt_lo, rt_hi")
    ap.add_argument("--output_dir", required=True, help="父目录；其下创建 SNR_box_<阈值>/")
    ap.add_argument("--min_snr", "--snr_threshold", dest="min_snr", type=float, default=3.0)
    ap.add_argument(
        "--gaussian_sigma",
        "--smooth_sigma",
        dest="smooth_sigma",
        type=float,
        default=0.8,
        metavar="SIGMA",
        help="mzML 强度一维高斯平滑 sigma（scipy gaussian_filter1d）；0 为不平滑；SNR 与 XIC/图均基于此数据",
    )
    ap.add_argument("--min_noise_points", type=int, default=5, help="框外至少多少点参与噪声估计")
    ap.add_argument(
        "--min_chrom_points",
        type=int,
        default=0,
        help="整条 XIC 至少包含的数据点数；0 表示不启用（与 ROI 阶段 QC 一致时可设相同值）",
    )
    ap.add_argument(
        "--min_chrom_max_intensity",
        type=float,
        default=0.0,
        help="整条 XIC（平滑后）最大强度下限；0 表示不启用",
    )
    args = ap.parse_args()
    cli = float(args.min_snr)
    eff = float("-inf") if cli < 0 else cli
    sys.exit(
        run(
            args.mzml,
            args.prediction_csv,
            args.output_dir,
            eff,
            cli,
            args.roi_windows_csv.strip() or None,
            args.smooth_sigma,
            int(args.min_noise_points),
            int(args.min_chrom_points),
            float(args.min_chrom_max_intensity),
        )
    )


if __name__ == "__main__":
    main()
