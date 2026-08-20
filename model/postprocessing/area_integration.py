"""
预测框积分模块：从 newtest 抽出的积分逻辑，接口与输出位置不变。
基线处理改为考虑信噪比（SNR）：估计噪声后对低 SNR 段做保守积分（噪声门限），减少把噪声当信号积分。

修正结果积分（--from_refined）：
  读取 post_newtest 输出的 prediction_refined.csv（main_rt_min/max 等），
  在 xic-roi-batch 的 xic_matrix.npy 上用本文件同一套 SNR/peak_adaptive/linear/raw 算法积分，
  不调用模型。默认写出 prediction_refined_with_area.csv。
"""
import os
import re
import numpy as np
import pandas as pd

from utils.roi_rt_mapping import box_to_rt_range
from utils.integrate_peak_adaptive import integrate_peak_adaptive

# 与 utils.quantify 一致
AREA_TIME_UNIT_SCALE = 60.0

# 信噪比相关（已放宽：更少区间触发门限、门限更软，面积更接近线性基线，相对误差通常更小）
NOISE_PERCENTILE = 25.0       # 用该分位数估计基线水平
SNR_THRESHOLD = 0.8           # SNR 低于此值才启用噪声门限（原 3.0 偏严，易削峰/得 0 面积）
NOISE_GATE_K = 1.0           # 校正后强度低于 k*noise_std 视为噪声；k 越小保留越多（原 2.0）


def _estimate_noise_and_snr(y):
    """
    从积分段 y 估计噪声与信噪比。
    返回 (noise_std, snr, baseline_level)。
    - 基线水平用低分位数近似；噪声用低强度区标准差。
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 0.0)
    if y.size < 3:
        return 0.0, 0.0, 0.0
    baseline_level = float(np.percentile(y, NOISE_PERCENTILE))
    peak = float(np.max(y))
    signal = peak - baseline_level
    low_mask = y <= np.percentile(y, 50)
    noise_region = y[low_mask]
    noise_std = float(np.std(noise_region)) if np.sum(low_mask) > 2 else float(np.std(y))
    if noise_std <= 0:
        noise_std = 1e-10
    snr = signal / noise_std
    return noise_std, snr, baseline_level


def integrate_with_snr_baseline(x, y, scale=AREA_TIME_UNIT_SCALE):
    """
    带信噪比考虑的基线校正积分：
    1. 用窗口两端点做线性基线，corrected = max(0, y - baseline)。
    2. 估计噪声 noise_std 与 SNR；若 SNR < 阈值，将 corrected 中低于 NOISE_GATE_K*noise_std 的部分置 0（噪声门限），再积分。

    回退：若门限后面积为 0，但无门限的线性基线面积 > 0，则改用线性基线结果。
    原因：同一窗口内若有两峰或框较宽，端点连线可能抬高基线，噪声按整段估计会偏大，
    门限会把本应保留的峰面积全削成 0；此时与 newtest 线性基线一致更稳。
    """
    if x.size < 2 or y.size < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y_left = float(y[0])
    y_right = float(y[-1])
    baseline = y_left + (y_right - y_left) * (x - x[0]) / (x[-1] - x[0])
    corrected_linear = np.maximum(0.0, y - baseline)
    area_linear = float(np.trapz(corrected_linear, x) * scale)

    corrected = corrected_linear.copy()
    noise_std, snr, _ = _estimate_noise_and_snr(y)
    if snr < SNR_THRESHOLD and noise_std > 0:
        gate = NOISE_GATE_K * noise_std
        corrected = np.where(corrected >= gate, corrected, 0.0)

    raw = np.trapz(corrected, x)
    area_gated = float(raw * scale)

    # 门限后面积被削成 0，但线性基线仍有正面积 → 退回线性基线（避免“框已检出却 area=0”）
    if area_gated <= 0.0 and area_linear > 0.0:
        return area_linear
    return area_gated


def max_consecutive(arr):
    """连续非零长度（与 utils.quantify 一致）。"""
    if arr is None or arr.size == 0:
        return 0
    greater_than_zero = arr > 0
    diff = np.diff(greater_than_zero.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if greater_than_zero[0]:
        starts = np.insert(starts, 0, 0)
    if greater_than_zero[-1]:
        ends = np.append(ends, len(arr))
    if len(starts) == len(ends) == 0:
        return 0
    return int(np.max(ends - starts))


def _safe_float(val, default=np.nan):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        if isinstance(val, str) and not val.strip():
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _image_name_to_xic_index(image_name):
    """从 ROI 文件名解析 XIC 行索引：N_mz* -> N-1（0-based）。"""
    stem = os.path.splitext(str(image_name))[0] if "." in str(image_name) else str(image_name)
    m = re.match(r"^(\d+)_mz", stem, re.IGNORECASE)
    return int(m.group(1)) - 1 if m else None


# post_newtest 修正表各峰 RT 列（与 run_unified_peak_workflow 一致）
REFINED_PEAK_SPECS = [
    ("main", "main_rt_min", "main_rt_max"),
    ("small", "small_rt_min", "small_rt_max"),
    ("small2", "small2_rt_min", "small2_rt_max"),
    ("small3", "small3_rt_min", "small3_rt_max"),
]


def _integrate_area_on_segment(
    filter_x,
    filter_y,
    baseline_correction=True,
    integration_method="snr",
    scale=None,
):
    """
    对已有 RT 区间的 XIC 片段做积分（与 integrate_each_predicted_box 内逻辑一致）。
    返回 (area, retention_time, intensity_max, point_counts)。
    """
    if scale is None:
        scale = float(AREA_TIME_UNIT_SCALE)
    fx = np.asarray(filter_x, dtype=np.float64)
    fy = np.asarray(filter_y, dtype=np.float64)
    if fx.size == 0 or fy.size == 0:
        return 0.0, 0.0, 0.0, 0
    max_intensity = float(np.max(fy))
    max_index = int(np.argmax(fy))
    max_x = float(fx[max_index])
    if baseline_correction and integration_method != "raw":
        if integration_method == "peak_adaptive":
            area_val = integrate_peak_adaptive(fx, fy, scale)
        elif integration_method == "linear":
            from utils.quantify import integrate_with_baseline_correction

            area_val = integrate_with_baseline_correction(fx, fy, scale)
        else:
            area_val = integrate_with_snr_baseline(fx, fy, scale)
    else:
        area_val = float(np.trapz(fy, fx) * scale)
    point_count = int(max_consecutive(fy))
    return area_val, max_x, max_intensity, point_count


def _build_xic_list_from_roi_dir(roi_dir):
    """从 roi_dir/xic_matrix.npy 构建 xic_list（分钟制），与 run_snr_single 一致。"""
    xic_npy_path = os.path.join(roi_dir, "xic_matrix.npy")
    if not os.path.isfile(xic_npy_path):
        return None
    xic_full = np.load(xic_npy_path)
    rt_array = np.asarray(xic_full[0, :], dtype=np.float64)
    if np.nanmax(rt_array) > 200:
        rt_array = rt_array / 60.0
    intensity_matrix = xic_full[1:, :]
    return [
        np.vstack([rt_array, intensity_matrix[i, :]])
        for i in range(intensity_matrix.shape[0])
    ]


def integrate_from_refined_dataframe(
    df,
    xic_list,
    baseline_correction=True,
    integration_method="snr",
):
    """
    对 prediction_refined.csv 每行按 main/small/small2/small3 的 RT 区间积分，
    在原列基础上增加 {tag}_area、{tag}_retention_time 等；area 列等于 main_area（兼容下游 R²）。
    """
    rows_out = []
    scale = float(AREA_TIME_UNIT_SCALE)
    for _, row in df.iterrows():
        out = row.to_dict()
        image_name = str(row.get("image", ""))
        xic_idx = _image_name_to_xic_index(image_name)
        if xic_idx is None or xic_idx < 0 or xic_idx >= len(xic_list):
            for tag, _, _ in REFINED_PEAK_SPECS:
                out[f"{tag}_area"] = 0.0
                out[f"{tag}_retention_time"] = np.nan
                out[f"{tag}_intensity_max"] = 0.0
                out[f"{tag}_point_counts"] = 0
            out["area"] = 0.0
            rows_out.append(out)
            continue

        rt = xic_list[xic_idx][0]
        intensity = xic_list[xic_idx][1]
        for tag, col_min, col_max in REFINED_PEAK_SPECS:
            rt_min = _safe_float(row.get(col_min), np.nan)
            rt_max = _safe_float(row.get(col_max), np.nan)
            if not (np.isfinite(rt_min) and np.isfinite(rt_max) and rt_max > rt_min):
                out[f"{tag}_area"] = 0.0
                out[f"{tag}_retention_time"] = np.nan
                out[f"{tag}_intensity_max"] = 0.0
                out[f"{tag}_point_counts"] = 0
                continue
            mask = (rt >= rt_min) & (rt <= rt_max)
            filter_x = rt[mask]
            filter_y = intensity[mask]
            area_val, max_x, max_intensity, point_count = _integrate_area_on_segment(
                filter_x,
                filter_y,
                baseline_correction=baseline_correction,
                integration_method=integration_method,
                scale=scale,
            )
            out[f"{tag}_area"] = area_val
            out[f"{tag}_retention_time"] = max_x
            out[f"{tag}_intensity_max"] = max_intensity
            out[f"{tag}_point_counts"] = point_count

        out["area"] = out.get("main_area", 0.0)
        out["retention_time"] = out.get("main_retention_time", np.nan)
        out["intensity_max"] = out.get("main_intensity_max", 0.0)
        out["point_counts"] = out.get("main_point_counts", 0)
        out["rt_min"] = _safe_float(row.get("main_rt_min"), np.nan)
        out["rt_max"] = _safe_float(row.get("main_rt_max"), np.nan)
        out["integration_method_used"] = integration_method
        rows_out.append(out)
    return pd.DataFrame(rows_out)


def integrate_from_rt_intervals_df(
    df,
    xic_list,
    baseline_correction=True,
    integration_method="snr",
):
    """对含 rt_min/rt_max 的表（如 prediction.csv）逐行用 SNR 等算法重算 area。"""
    rows_out = []
    scale = float(AREA_TIME_UNIT_SCALE)
    for _, row in df.iterrows():
        out = row.to_dict()
        image_name = str(row.get("image", ""))
        rt_min = _safe_float(row.get("rt_min"), np.nan)
        rt_max = _safe_float(row.get("rt_max"), np.nan)
        xic_idx = _image_name_to_xic_index(image_name)
        if xic_idx is None or xic_idx < 0 or xic_idx >= len(xic_list):
            out["area"] = out.get("area", 0.0)
            rows_out.append(out)
            continue
        if not (np.isfinite(rt_min) and np.isfinite(rt_max) and rt_max > rt_min):
            out["area"] = 0.0
            rows_out.append(out)
            continue
        rt = xic_list[xic_idx][0]
        intensity = xic_list[xic_idx][1]
        mask = (rt >= rt_min) & (rt <= rt_max)
        filter_x = rt[mask]
        filter_y = intensity[mask]
        area_val, max_x, max_intensity, point_count = _integrate_area_on_segment(
            filter_x,
            filter_y,
            baseline_correction=baseline_correction,
            integration_method=integration_method,
            scale=scale,
        )
        out["area"] = area_val
        out["retention_time"] = max_x
        out["intensity_max"] = max_intensity
        out["point_counts"] = point_count
        out["integration_method_used"] = integration_method
        rows_out.append(out)
    return pd.DataFrame(rows_out)


def _resolve_roi_dir(sample_dir, input_root):
    """修正 CSV 所在目录与 XIC 目录可能不同：优先 input_root/<样本名>，否则 sample_dir 自带 xic。"""
    sample_dir = os.path.abspath(sample_dir)
    if input_root:
        cand = os.path.join(os.path.abspath(input_root), os.path.basename(sample_dir))
        if os.path.isfile(os.path.join(cand, "xic_matrix.npy")):
            return cand
    if os.path.isfile(os.path.join(sample_dir, "xic_matrix.npy")):
        return sample_dir
    return sample_dir


def _find_refined_csv(sample_dir, refined_name, refined_csv_explicit=None):
    if refined_csv_explicit and os.path.isfile(refined_csv_explicit):
        return refined_csv_explicit
    for base in (sample_dir, os.path.join(sample_dir, "predicted_plots")):
        p = os.path.join(base, refined_name)
        if os.path.isfile(p):
            return p
    return os.path.join(sample_dir, refined_name)


def run_refined_integrate_single(args, sample_dir, prediction_output, refined_csv_path=None):
    """
    对 post 修正结果积分：读 prediction_refined.csv，XIC 来自 roi 目录，不跑模型。
    """
    import time
    from pathlib import Path

    refined_name = getattr(args, "refined_input", "prediction_refined.csv")
    refined_path = _find_refined_csv(
        sample_dir,
        refined_name,
        refined_csv_path or getattr(args, "refined_csv", None),
    )
    if not os.path.isfile(refined_path):
        print(f"[ERROR] Refined prediction not found: {refined_path}")
        return False

    roi_dir = _resolve_roi_dir(sample_dir, getattr(args, "input_root", None))
    xic_list = _build_xic_list_from_roi_dir(roi_dir)
    if not xic_list:
        print(f"[ERROR] xic_matrix.npy not found under: {roi_dir}")
        return False

    print(f"[INFO] Integrating refined predictions (no model): {refined_path}")
    print(f"[INFO] XIC directory: {roi_dir}")
    df = pd.read_csv(refined_path)
    baseline_corr = getattr(args, "baseline_correction", True)
    integration_method = getattr(args, "integration_method", "snr")

    if "main_rt_min" in df.columns:
        df_out = integrate_from_refined_dataframe(
            df,
            xic_list,
            baseline_correction=baseline_corr,
            integration_method=integration_method,
        )
    elif "rt_min" in df.columns and "rt_max" in df.columns:
        print("[INFO] Input has rt_min/rt_max (flat format), using interval integration.")
        df_out = integrate_from_rt_intervals_df(
            df,
            xic_list,
            baseline_correction=baseline_corr,
            integration_method=integration_method,
        )
    else:
        print("[ERROR] CSV must contain main_rt_min/main_rt_max or rt_min/rt_max columns.")
        return False

    pred_output_dir = os.path.dirname(prediction_output)
    if pred_output_dir:
        Path(pred_output_dir).mkdir(parents=True, exist_ok=True)

    max_retries = 3
    retry_delay = 1.0
    saved = False
    for attempt in range(max_retries):
        try:
            df_out.to_csv(prediction_output, index=False, encoding="utf-8-sig")
            print(f"[INFO] Refined integration saved to: {prediction_output}")
            saved = True
            break
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"[WARN] File locked, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                fallback_dir = os.path.dirname(prediction_output) or "."
                base = os.path.splitext(os.path.basename(prediction_output))[0]
                fallback_path = os.path.join(
                    fallback_dir, f"{base}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                )
                try:
                    df_out.to_csv(fallback_path, index=False, encoding="utf-8-sig")
                    print(f"[INFO] Saved to fallback: {fallback_path}")
                    saved = True
                except Exception as e2:
                    print(f"[ERROR] Save failed: {e2}")
    return saved


def _expected_roi_name(xic_idx_0based, mz, q3=None, native_id=""):
    from utils.mzml_chromatogram_ids import roi_image_stem

    n = xic_idx_0based + 1
    return roi_image_stem(n, mz, q3, native_id or "") + ".jpeg"


def _load_roi_windows(images_path):
    """若存在 roi_windows.csv（testXIC 生成），返回 { image_name: (rt_lo, rt_hi) }，否则返回 {}。"""
    path = os.path.join(images_path, "roi_windows.csv")
    if not os.path.isfile(path):
        return {}
    try:
        df = pd.read_csv(path)
        if "image" not in df.columns or "rt_lo" not in df.columns or "rt_hi" not in df.columns:
            return {}
        return {
            str(row["image"]).strip(): (float(row["rt_lo"]), float(row["rt_hi"]))
            for _, row in df.iterrows()
        }
    except Exception:
        return {}


def integrate_each_predicted_box(xic_list, prediction, xic_info, baseline_correction=True,
                                 integration_method="snr", roi_windows=None):
    """
    为每个预测框计算积分信息，返回可直接写入 prediction.csv 的明细行。
    接口与 newtest._integrate_each_predicted_box 一致；输出格式与写入位置由调用方（newtest）保持不变。
    roi_windows: 可选 dict，image_name -> (rt_lo, rt_hi)，来自 roi_windows.csv，使像素→RT 与 ROI 图一致。
    baseline_correction=True 时使用基线校正；False 时仅梯形积分乘 scale。
    integration_method: "snr" | "peak_adaptive" | "linear" | "raw"
      - snr: 带信噪比考虑的线性基线（默认）
      - peak_adaptive: 谷-谷基线，适用于双峰/宽峰（如 548_mz413）
      - linear: 纯线性端点基线
      - raw: 无基线，原始梯形积分
    """
    rows = []
    scale = float(AREA_TIME_UNIT_SCALE)
    if roi_windows is None:
        roi_windows = {}

    for i, (img_path, scores, boxes) in enumerate(prediction):
        if i >= len(xic_info) or i >= len(xic_list):
            continue

        compound_name = xic_info.loc[i, 'Compound Name']
        mz = xic_info.loc[i, 'mz']
        q3_val = xic_info.loc[i, 'q3'] if 'q3' in xic_info.columns else np.nan
        true_rt = xic_info.loc[i, 'RT']
        nid_val = ""
        if "native_id" in xic_info.columns:
            raw_n = xic_info.loc[i, "native_id"]
            if pd.notna(raw_n):
                nid_val = str(raw_n).strip()
        rt = xic_list[i][0]
        intensity = xic_list[i][1]

        if len(scores) == 0 or len(boxes) == 0:
            image_name = (
                os.path.basename(img_path)
                if img_path
                else _expected_roi_name(i, mz, q3_val, nid_val)
            )
            rows.append({
                'image': image_name,
                'image_path': img_path if img_path else "",
                'compound_name': compound_name,
                'mz': float(mz),
                'q3': float(q3_val) if pd.notna(q3_val) else np.nan,
                'native_id': nid_val,
                'old_rt': float(true_rt),
                'box_x1': np.nan,
                'box_y1': np.nan,
                'box_x2': np.nan,
                'box_y2': np.nan,
                'score': 0.0,
                'rt_min': 0.0,
                'rt_max': 0.0,
                'retention_time': 0.0,
                'intensity_max': 0.0,
                'area': 0.0,
                'point_counts': 0
            })
            continue

        for j in range(min(len(scores), len(boxes))):
            x1, y1, x2, y2 = boxes[j]
            score = float(scores[j][0])

            # 优先用 roi_windows.csv 的窗口（与 ROI 图 set_xlim 一致），否则 true_rt±1min
            image_name = os.path.basename(img_path)
            rt_window = roi_windows.get(image_name)
            left, right, _, _ = box_to_rt_range(x1, y1, x2, y2, true_rt, rt, rt_window=rt_window)

            mask = (rt >= left) & (rt <= right)
            filter_x = rt[mask]
            filter_y = intensity[mask]

            if filter_x.size == 0 or filter_y.size == 0:
                rows.append({
                    'image': os.path.basename(img_path),
                    'image_path': img_path,
                    'compound_name': compound_name,
                    'mz': float(mz),
                    'q3': float(q3_val) if pd.notna(q3_val) else np.nan,
                    'native_id': nid_val,
                    'old_rt': float(true_rt),
                    'box_x1': float(x1),
                    'box_y1': float(y1),
                    'box_x2': float(x2),
                    'box_y2': float(y2),
                    'score': score,
                    'rt_min': left,
                    'rt_max': right,
                    'retention_time': 0.0,
                    'intensity_max': 0.0,
                    'area': 0.0,
                    'point_counts': 0
                })
                continue

            area_val, max_x, max_intensity, point_count = _integrate_area_on_segment(
                filter_x,
                filter_y,
                baseline_correction=baseline_correction,
                integration_method=integration_method,
                scale=scale,
            )

            rows.append({
                'image': os.path.basename(img_path),
                'image_path': img_path,
                'compound_name': compound_name,
                'mz': float(mz),
                'q3': float(q3_val) if pd.notna(q3_val) else np.nan,
                'native_id': nid_val,
                'old_rt': float(true_rt),
                'box_x1': float(x1),
                'box_y1': float(y1),
                'box_x2': float(x2),
                'box_y2': float(y2),
                'score': score,
                'rt_min': left,
                'rt_max': right,
                'retention_time': max_x,
                'intensity_max': max_intensity,
                'area': area_val,
                'point_counts': point_count
            })

    return rows


# -----------------------------------------------------------------------------
# 命令行入口：与 newtest 单目录/批量流程一致，但积分使用 SNR 基线（integrate_each_predicted_box）。
# 用法见文件末尾 argparse 说明。
# -----------------------------------------------------------------------------
def run_snr_single(args, images_path, prediction_output, plot_dir):
    """
    与 newtest.run_single 相同流程，仅积分步骤使用 integrate_each_predicted_box（SNR 基线）。
    若 args.from_refined 为真，则走 run_refined_integrate_single（不跑模型）。
    """
    if getattr(args, "from_refined", False):
        refined_csv = getattr(args, "refined_csv", None)
        return run_refined_integrate_single(
            args, images_path, prediction_output, refined_csv_path=refined_csv
        )

    import time
    from pathlib import Path

    # 延迟导入，避免循环依赖
    from inference.predictor import _adapt_prediction_for_quantify
    from utils.io_utils import load_features
    from utils.predict_utils import build_predictor

    feature_path = args.feature
    if feature_path is None:
        feature_path = os.path.join(images_path, "feature.csv")
    elif os.path.isdir(feature_path):
        feature_path = os.path.join(feature_path, "feature.csv")
    if not os.path.exists(feature_path):
        print(f"[ERROR] feature.csv not found: {feature_path}")
        return False

    xic_info = load_features(feature_path, preserve_order=True)
    if not os.path.exists(images_path):
        print(f"[ERROR] Images path not found: {images_path}")
        return False

    print(f"[INFO]  Running MRMPFormer model (SNR integration mode)...")
    results = build_predictor(
        model_path=args.model,
        images_path=images_path,
        threshold=args.threshold,
        plot=args.plot,
        plot_dir=plot_dir,
        verbose=args.verbose,
    )
    print(f"[INFO] Model prediction completed. Detected peaks in {len(results)} images.")

    print("[INFO] Performing SNR-aware quantification...")
    xic_npy_path = os.path.join(images_path, "xic_matrix.npy")
    if not os.path.exists(xic_npy_path):
        print(f"[ERROR] xic_matrix.npy not found: {xic_npy_path}")
        return False

    xic_full = np.load(xic_npy_path)
    rt_array = xic_full[0, :]
    intensity_matrix = xic_full[1:, :]
    if np.nanmax(rt_array) > 200:
        print("[WARN] Detected second-scale RT axis, converting to minutes.")
        rt_array = rt_array / 60.0

    xic_list = [
        np.vstack([rt_array, intensity_matrix[i, :]])
        for i in range(intensity_matrix.shape[0])
    ]
    aligned_len = min(len(xic_info), len(xic_list))
    if aligned_len < len(xic_info) or aligned_len < len(xic_list):
        print(f"[WARN] Feature/XIC count mismatch, truncating to {aligned_len}.")
        xic_info = xic_info.iloc[:aligned_len].reset_index(drop=True)
        xic_list = xic_list[:aligned_len]

    xic_count = len(xic_list)
    prediction_for_quantify = _adapt_prediction_for_quantify(
        results, xic_count=xic_count, xic_info=xic_info
    )

    pred_output_dir = os.path.dirname(prediction_output)
    if pred_output_dir:
        Path(pred_output_dir).mkdir(parents=True, exist_ok=True)

    baseline_corr = getattr(args, "baseline_correction", True)
    integration_method = getattr(args, "integration_method", "snr")
    roi_windows = _load_roi_windows(images_path)
    if roi_windows:
        print(f"[INFO] 使用 roi_windows.csv 做像素→RT 映射（共 {len(roi_windows)} 条）")
    prediction_rows = integrate_each_predicted_box(
        xic_list, prediction_for_quantify, xic_info,
        baseline_correction=baseline_corr,
        integration_method=integration_method,
        roi_windows=roi_windows,
    )
    df_prediction = pd.DataFrame(prediction_rows)

    if "q3" in df_prediction.columns:
        before = len(df_prediction)
        df_prediction = df_prediction.loc[
            df_prediction.groupby(["mz", "q3"], dropna=False)["area"].idxmax()
        ].reset_index(drop=True)
        if len(df_prediction) < before:
            print(f"[INFO] 按 (mz, q3) 去重: {before} -> {len(df_prediction)} 行")

    max_retries = 3
    retry_delay = 1.0
    out_path = prediction_output
    saved = False
    for attempt in range(max_retries):
        try:
            df_prediction.to_csv(out_path, index=False)
            print(f"[INFO] SNR integration saved to: {out_path}")
            saved = True
            break
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"[WARN] File locked, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                fallback_dir = os.path.dirname(out_path) or "."
                base = os.path.splitext(os.path.basename(out_path))[0]
                fallback_path = os.path.join(fallback_dir, f"{base}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                try:
                    df_prediction.to_csv(fallback_path, index=False)
                    print(f"[INFO] Saved to fallback: {fallback_path}")
                    saved = True
                except Exception as e2:
                    print(f"[ERROR] Save failed: {e2}")
    return saved


def main_cli():
    import argparse
    import time
    import multiprocessing
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="integrate_prediction: 模型预测+SNR 积分，或 --from_refined 对 prediction_refined.csv 仅积分（算法不变）"
    )
    parser.add_argument("--feature", type=str, default=None, help="feature.csv 路径，默认 <images_path>/feature.csv")
    parser.add_argument("--images_path", type=str, default=None, help="单样本：ROI/XIC 目录；--from_refined 时为含修正 CSV 的目录")
    parser.add_argument("--batch_dir", type=str, default=None, help="批量：子目录各跑一遍")
    parser.add_argument("--batch_output", type=str, default="../output/inference/batch_predictions_snr", help="批量输出根目录")
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="--from_refined 批量时 XIC 根目录（如 xic-roi-batch），子目录名与 batch_dir 一致",
    )
    parser.add_argument("--model", type=str, default=None, help="模型 .pth（--from_refined 时不需要）")
    parser.add_argument(
        "--from_refined",
        action="store_true",
        help="对 post 修正结果积分：读 prediction_refined.csv，不重新预测",
    )
    parser.add_argument(
        "--refined_input",
        type=str,
        default="prediction_refined.csv",
        help="修正结果文件名（在各样本目录或 predicted_plots/ 下查找）",
    )
    parser.add_argument(
        "--refined_csv",
        type=str,
        default=None,
        help="单样本时修正 CSV 的完整路径（覆盖 --refined_input 搜索）",
    )
    parser.add_argument(
        "--refined_output_name",
        type=str,
        default="prediction_refined_with_area.csv",
        help="--from_refined 时输出文件名（写在样本目录或 batch_output/<name>/）",
    )
    parser.add_argument("--prediction_output", type=str, default="../output/inference/prediction_snr.csv", help="单目录时输出 CSV 路径")
    parser.add_argument("--plot_dir", type=str, default="predicted_plots_snr", help="预测可视化输出目录")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--baseline_correction",
        action="store_true",
        default=True,
        help="使用 SNR 基线积分（默认开启）；加 --no_baseline_correction 则仅梯形积分",
    )
    parser.add_argument("--no_baseline_correction", action="store_true", help="关闭基线校正，仅 raw trapz")
    parser.add_argument(
        "--integration_method", type=str, default="snr",
        choices=["snr", "peak_adaptive", "linear", "raw"],
        help="积分方法: snr=SNR基线(默认), peak_adaptive=谷-谷基线(双峰/宽峰), linear=线性端点, raw=无基线",
    )
    args = parser.parse_args()
    if args.no_baseline_correction:
        args.baseline_correction = False

    if not args.batch_dir and not args.images_path:
        parser.error("请指定 --images_path 或 --batch_dir")
    if not args.from_refined and not args.model:
        parser.error("请指定 --model，或使用 --from_refined 对修正结果积分（无需模型）")

    cpu_cores = multiprocessing.cpu_count()
    print("=" * 60)
    mode = "refined-only" if args.from_refined else "model+SNR"
    print(f"[INFO] integrate_prediction ({mode})  CPU cores:", cpu_cores)
    start_time = time.time()

    if args.batch_dir:
        batch_path = Path(args.batch_dir)
        if not batch_path.exists():
            print(f"[ERROR] Batch directory not found: {batch_path}")
            return
        output_base = Path(args.batch_output)
        output_base.mkdir(parents=True, exist_ok=True)
        subdirs = sorted([d for d in batch_path.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"[WARN] No subdirectories in {batch_path}")
            return
        for i, subdir in enumerate(subdirs):
            print("=" * 60)
            print(f"[BATCH {i+1}/{len(subdirs)}] {subdir.name}")
            if args.from_refined:
                pred_out = subdir / args.refined_output_name
                if args.batch_output:
                    out_parent = output_base / subdir.name
                    out_parent.mkdir(parents=True, exist_ok=True)
                    pred_out = out_parent / args.refined_output_name
                plot_dir = str(subdir / "predicted_plots")
            else:
                pred_out = output_base / subdir.name / "prediction.csv"
                plot_dir = output_base / subdir.name / "predicted_plots"
            pred_out.parent.mkdir(parents=True, exist_ok=True)
            run_snr_single(args, str(subdir), str(pred_out), str(plot_dir))
        print(f"[DONE] Batch finished in {time.time() - start_time:.2f} s")
    else:
        if args.from_refined and args.prediction_output == "../output/inference/prediction_snr.csv":
            args.prediction_output = os.path.join(
                args.images_path, args.refined_output_name
            )
        run_snr_single(args, args.images_path, args.prediction_output, args.plot_dir)
        print(f"[DONE] Total time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main_cli()
