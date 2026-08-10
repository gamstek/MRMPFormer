import argparse
import io
import os
import sys
import time                      # 计时
import multiprocessing           # 获取 CPU 核心数
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import re

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# from utils.io_utils import export_results  # 暂停：按需求仅保留 prediction.csv 输出
from utils.io_utils import load_features
from utils.predict_utils import build_predictor
from utils.quantify import max_consecutive, AREA_TIME_UNIT_SCALE, integrate_with_baseline_correction_avg, integrate_with_baseline_minval_noise_right, integrate_with_external_baseline, get_baseline_endpoint_heights, get_baseline_minval_noise_right
from utils.roi_rt_mapping import box_to_rt_range, rt_to_pixel_x, intensity_to_pixel_y, rt_window_bounds_minutes, ROI_IMAGE_WIDTH_PX, ROI_IMAGE_HEIGHT_PX
from utils.integrate_peak_adaptive import integrate_peak_adaptive
from utils.roi_quality_params import compute_roi_quality_params
from utils.adaptive_integration import select_integration_method
from testXIC import roi_safe_name_base


def _to_result_tuple(res):
    if isinstance(res, tuple):
        img_path, scores, boxes = res
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
    else:
        img_path = res.get('image_path', '')
        boxes = np.array(res.get('boxes', []), dtype=np.float32)
        scores = np.array(res.get('scores', []), dtype=np.float32)

    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    elif scores.size == 0:
        scores = np.empty((0, 1), dtype=np.float32)

    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
    elif boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    n = min(len(scores), len(boxes))
    if n == 0:
        return img_path, np.empty((0, 1), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    scores = scores[:n]
    boxes = boxes[:n]
    return img_path, scores, boxes


def _compound_label_from_feature(xic_info, i):
    if "native_id" in xic_info.columns:
        nid = xic_info.loc[i, "native_id"]
        if pd.notna(nid) and str(nid).strip():
            return str(nid).strip()
    return xic_info.loc[i, "Compound Name"]


def _expected_roi_name(xic_idx_0based, compound_name, mz, q3=np.nan, native_id=None):
    """无检测时用于 prediction.csv 的 image 列占位，与 testXIC.roi_safe_name_base 命名一致。"""
    n = xic_idx_0based + 1
    label = native_id if native_id else compound_name
    if label is not None and str(label).strip().isdigit():
        label = native_id or None
    return f"{roi_safe_name_base(n, mz, q3, compound_name=label)}.jpeg"


def _parse_roi_basename(basename):
    """
    从 ROI 文件名解析对齐方式。
    - 若匹配 N_mz*（如 3_mz142.0000.jpeg）-> 返回 ('index', None, N-1)
    - 若匹配 Q1_xxx 或 Q1_xxx_Q3_yyy（testXIC 默认）-> 返回 ('q1q3', q1_float, q3_float_or_None)
    - 若匹配 roi_opencv_001 这类序号名（test plot 生成）-> 返回 ('index', None, N-1)
    - 否则返回 (None, None, None)
    """
    stem = (os.path.splitext(basename)[0] if "." in basename else basename).strip()
    # 规则1: N_mz...
    m1 = re.match(r"^(\d+)_mz", stem, re.IGNORECASE)
    if m1:
        return ("index", None, int(m1.group(1)) - 1)
    # 规则2: Q1_xxx 或 Q1_xxx_Q3_yyy（testXIC 生成）
    m2 = re.search(r"Q1[_\-]?([\d.]+)", stem, re.IGNORECASE)
    if m2:
        try:
            q1 = float(m2.group(1))
        except ValueError:
            return (None, None, None)
        m3 = re.search(r"Q3[_\-]?([\d.]+)", stem, re.IGNORECASE)
        q3 = float(m3.group(1)) if m3 else None
        return ("q1q3", q1, q3)

    # 规则3: roi_opencv_001 这类序号名（对应 XIC 的第 N 行）
    m3 = re.match(r"^roi[_\-]?opencv[_\-]?(\d+)$", stem, re.IGNORECASE)
    if m3:
        return ("index", None, int(m3.group(1)) - 1)
    return (None, None, None)


def _adapt_prediction_for_quantify(results, xic_count, xic_info=None):
    """
    将检测结果对齐到 XIC 行索引，并且每个 XIC 只保留最高置信度框。
    对齐规则（二选一）：
    1) 文件名前缀 N_mz* -> XIC 第 N 行（0-based 为 N-1）；
    2) 文件名 Q1_xxx_Q3_yyy（testXIC 默认）-> 按 feature 表 (mz, q3) 匹配到行索引。
    未检出的图像不会出现在 results 中，故 aligned[i] 可能为空 -> 该行 area 等为 0。
    """
    aligned = [
        ("", np.empty((0, 1), dtype=np.float32), np.empty((0, 4), dtype=np.float32))
        for _ in range(xic_count)
    ]
    mz_tol = 0.01
    q3_tol = 0.15

    for res in results:
        img_path, scores, boxes = _to_result_tuple(res)
        if len(scores) == 0:
            continue
        image_name = os.path.basename(str(img_path)).strip()
        kind, v1, v2 = _parse_roi_basename(image_name)

        if kind == "index":
            xic_idx = v2
        elif kind == "q1q3" and xic_info is not None:
            q1, q3 = v1, v2
            mz_col = pd.to_numeric(xic_info["mz"], errors="coerce")
            mask = np.abs(mz_col - q1) <= mz_tol
            if "q3" in xic_info.columns:
                q3_col = pd.to_numeric(xic_info["q3"], errors="coerce")
                if q3 is not None:
                    mask &= (q3_col.notna() & (np.abs(q3_col - q3) <= q3_tol))
                else:
                    mask &= q3_col.isna()
            cand = np.where(mask)[0]
            xic_idx = int(cand[0]) if len(cand) > 0 else -1
        else:
            # Fallback: handle roi_opencv_XXX even if _parse_roi_basename missed.
            if kind is None:
                m3 = re.match(r"^roi[_\-]?opencv[_\-]?(\d+)$", os.path.splitext(image_name)[0].strip(), re.IGNORECASE)
                if m3:
                    xic_idx = int(m3.group(1)) - 1
                else:
                    xic_idx = -1
            else:
                xic_idx = -1

        if xic_idx < 0 or xic_idx >= xic_count:
            continue

        top_idx = int(np.argmax(scores[:, 0]))
        top_score = scores[top_idx:top_idx + 1]
        top_box = boxes[top_idx:top_idx + 1]

        prev_path, prev_score, _ = aligned[xic_idx]
        if len(prev_score) == 0 or float(top_score[0][0]) > float(prev_score[0][0]):
            aligned[xic_idx] = (img_path, top_score, top_box)

    return aligned


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


def _load_external_baselines(baseline_json_path, mz_tol=0.01, q3_tol=0.15):
    """
    加载外部基线 JSON，返回 {(mz, q3): (x_array, y_array)} 供匹配使用。
    JSON 格式：list of {mz_name, x, y, q3?} 或 dict keyed by "mz_name" or "mz_name_q3"
    """
    import json
    path = Path(baseline_json_path).resolve()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    items = data if isinstance(data, list) else data.get("baselines", data.get("compounds", []))
    for item in items:
        if isinstance(item, dict):
            mz = item.get("mz_name", item.get("mz"))
            q3 = item.get("q3", np.nan)
            x = item.get("x", [])
            y = item.get("y", [])
        else:
            continue
        if mz is None or x is None or y is None:
            continue
        try:
            mz_val = float(mz)
        except (TypeError, ValueError):
            continue
        q3_val = float(q3) if pd.notna(q3) and q3 is not None else np.nan
        key = (round(mz_val, 4), round(q3_val, 2) if not np.isnan(q3_val) else None)
        result[key] = (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return result


def _match_baseline_for_compound(mz, q3, baseline_dict, mz_tol=0.01, q3_tol=0.15):
    """按 (mz, q3) 容差匹配化合物对应的外部基线，返回 (x, y) 或 None。优先匹配 q3 一致的。"""
    mz_val = float(mz) if pd.notna(mz) else np.nan
    q3_val = float(q3) if pd.notna(q3) else np.nan
    exact_match = None
    mz_only_match = None
    for (bmz, bq3), (bx, by) in baseline_dict.items():
        if abs(mz_val - bmz) <= mz_tol:
            if bq3 is None or (isinstance(bq3, float) and np.isnan(bq3)):
                mz_only_match = (bx, by)
            elif pd.notna(q3_val) and abs(q3_val - bq3) <= q3_tol:
                exact_match = (bx, by)
    return exact_match if exact_match is not None else mz_only_match


def _integrate_each_predicted_box(xic_list, prediction, xic_info, baseline_correction=False,
                                   integration_method="linear", roi_windows=None, external_baselines=None):
    """
    为每个预测框计算积分信息，返回可直接写入 prediction.csv 的明细行。
    说明：积分上下限由预测框 x1/x2 决定，映射逻辑与 quantify() 一致。
    roi_windows: 可选 dict，image_name -> (rt_lo, rt_hi)，来自 roi_windows.csv，使像素→RT 与 ROI 图一致。
    integration_method: "linear" | "raw" | "peak_adaptive" | "adaptive"
      - linear: 起止点基线积分（默认）
      - raw: 无基线，原始梯形积分
      - peak_adaptive: 谷-谷基线，适用于双峰/宽峰
      - adaptive: 按 SNR、baseline_slope、peak_width_ratio 自适应选择
    """
    rows = []
    scale = float(AREA_TIME_UNIT_SCALE)
    if roi_windows is None:
        roi_windows = {}

    for i, (img_path, scores, boxes) in enumerate(prediction):
        if i >= len(xic_info) or i >= len(xic_list):
            continue

        compound_name = _compound_label_from_feature(xic_info, i)
        mz = xic_info.loc[i, 'mz']
        q3_val = xic_info.loc[i, 'q3'] if 'q3' in xic_info.columns else np.nan
        true_rt = xic_info.loc[i, 'RT']
        rt = xic_list[i][0]
        intensity = xic_list[i][1]
        native_id = (
            str(xic_info.loc[i, "native_id"]).strip()
            if "native_id" in xic_info.columns and pd.notna(xic_info.loc[i, "native_id"])
            else None
        )

        if len(scores) == 0 or len(boxes) == 0:
            image_name = os.path.basename(img_path) if img_path else _expected_roi_name(
                i, compound_name, mz, q3_val, native_id=native_id
            )
            rows.append({
                'image': image_name,
                'image_path': img_path if img_path else "",
                'compound_name': compound_name,
                'mz': float(mz),
                'q3': float(q3_val) if pd.notna(q3_val) else np.nan,
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
                'point_counts': 0,
                'snr': np.nan, 'noise_std': np.nan, 'baseline_slope': np.nan,
                'peak_width_ratio': np.nan, 'dynamic_range': np.nan,
            })
            continue

        for j in range(min(len(scores), len(boxes))):
            x1, y1, x2, y2 = boxes[j]
            score = float(scores[j][0])

            # 像素 → RT：优先用 roi_windows.csv 的窗口（与 ROI 图 set_xlim 一致），否则用 true_rt±1min 裁剪
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
                    'point_counts': 0,
                    'snr': np.nan, 'noise_std': np.nan, 'baseline_slope': np.nan,
                    'peak_width_ratio': np.nan, 'dynamic_range': np.nan,
                })
                continue

            max_intensity = float(np.max(filter_y))
            max_index = int(np.argmax(filter_y))
            max_x = float(filter_x[max_index])
            qparams = compute_roi_quality_params(filter_x, filter_y)
            method_used = integration_method
            if integration_method == "adaptive":
                method_used = select_integration_method(qparams)
            if integration_method == "external_baseline" and external_baselines:
                bl = _match_baseline_for_compound(mz, q3_val, external_baselines)
                if bl is not None:
                    bx, by = bl
                    area_val = integrate_with_external_baseline(rt, intensity, left, right, bx, by, scale)
                    method_used = "external_baseline"
                else:
                    raw_area = np.trapz(filter_y, filter_x)
                    area_val = float(raw_area * scale)
                    method_used = "raw"
            elif baseline_correction:
                if method_used == "peak_adaptive":
                    area_val = integrate_peak_adaptive(filter_x, filter_y, scale)
                elif method_used == "minval_noise_right":
                    area_val = integrate_with_baseline_minval_noise_right(rt, intensity, left, right, scale)
                elif method_used == "raw":
                    area_val = float(np.trapz(filter_y, filter_x) * scale)
                else:
                    area_val = integrate_with_baseline_correction_avg(rt, intensity, left, right, scale)
            else:
                raw_area = np.trapz(filter_y, filter_x)
                area_val = float(raw_area * scale)
            point_count = int(max_consecutive(filter_y))

            rows.append({
                'image': os.path.basename(img_path),
                'image_path': img_path,
                'compound_name': compound_name,
                'mz': float(mz),
                'q3': float(q3_val) if pd.notna(q3_val) else np.nan,
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
                'point_counts': point_count,
                'snr': qparams["snr"],
                'noise_std': qparams["noise_std"],
                'baseline_slope': qparams["baseline_slope"],
                'peak_width_ratio': qparams["peak_width_ratio"],
                'dynamic_range': qparams["dynamic_range"],
                'integration_method_used': method_used,
            })

    return rows


def _plot_predictions_with_baseline(results, df_prediction, xic_list, xic_info, roi_windows,
                                    plot_dir, baseline_correction, integration_method,
                                    external_baselines=None):
    """
    绘制预测图；若使用线性基线积分，在图上叠加基线。
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    os.makedirs(plot_dir, exist_ok=True)
    if df_prediction.empty or "image" not in df_prediction.columns:
        pred_by_image = pd.DataFrame()
    else:
        pred_by_image = df_prediction.set_index("image", drop=False)
    n_mz_pattern = re.compile(r"^(\d+)_mz", re.IGNORECASE)

    for res in results:
        img_path = res["image_path"]
        img = Image.open(img_path).convert("RGB")
        image_name = os.path.basename(img_path)
        row = pred_by_image.loc[image_name] if (len(pred_by_image) > 0 and image_name in pred_by_image.index) else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(img)
        for box, score in zip(res["boxes"], res["scores"]):
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, f"{score:.2f}", color="white", backgroundcolor="red")

        # 若启用基线积分或外部基线，叠加基线
        draw_baseline = baseline_correction or (integration_method == "external_baseline" and external_baselines)
        if draw_baseline and row is not None:
            method_used = row.get("integration_method_used", integration_method)
            rt_min = float(row["rt_min"])
            rt_max = float(row["rt_max"])
            m = n_mz_pattern.match(os.path.splitext(image_name)[0])
            xic_idx = int(m.group(1)) - 1 if m else 0
            if 0 <= xic_idx < len(xic_list):
                rt = xic_list[xic_idx][0]
                intensity = xic_list[xic_idx][1]
                mz = xic_info.loc[xic_idx, "mz"] if "mz" in xic_info.columns else np.nan
                q3 = xic_info.loc[xic_idx, "q3"] if "q3" in xic_info.columns else np.nan
                true_rt = float(xic_info.loc[xic_idx, "RT"]) if xic_idx < len(xic_info) else (rt_min + rt_max) / 2
                rt_lo, rt_hi = roi_windows.get(image_name, rt_window_bounds_minutes(true_rt, rt))
                mask_full = (rt >= rt_lo) & (rt <= rt_hi)
                full_x = rt[mask_full]
                full_y = intensity[mask_full]
                mask_box = (rt >= rt_min) & (rt <= rt_max)
                filter_x = rt[mask_box]
                filter_y = intensity[mask_box]
                if filter_x.size >= 2 and full_y.size > 0:
                    y_min = float(np.min(full_y))
                    y_max = float(np.max(full_y))
                    baseline = None
                    label = "linear"
                    if str(method_used) == "external_baseline" and external_baselines:
                        bl = _match_baseline_for_compound(mz, q3, external_baselines)
                        if bl is not None:
                            from scipy.interpolate import interp1d
                            bx, by = bl
                            if bx.size >= 2 and by.size >= 2:
                                f_bl = interp1d(bx, by, kind="linear", bounds_error=False,
                                               fill_value=(by[0], by[-1]))
                                baseline = f_bl(filter_x)
                                label = "外部基线"
                    elif str(method_used) == "linear":
                        y_left_avg, y_right_avg = get_baseline_endpoint_heights(rt, intensity, rt_min, rt_max)
                        if y_left_avg is None:
                            y_left_avg = float(filter_y[0])
                        if y_right_avg is None:
                            y_right_avg = float(filter_y[-1])
                        baseline = y_left_avg + (y_right_avg - y_left_avg) * (filter_x - filter_x[0]) / (filter_x[-1] - filter_x[0])
                        label = "线性基线"
                    elif str(method_used) == "minval_noise_right":
                        baseline = get_baseline_minval_noise_right(rt, intensity, rt_min, rt_max, filter_x)
                        label = "谷点-噪声基线"
                    if baseline is not None and y_max > y_min:
                        px = np.array([rt_to_pixel_x(t, rt_lo, rt_hi) for t in filter_x])
                        py = np.array([intensity_to_pixel_y(b, y_min, y_max) for b in baseline])
                        ax.plot(px, py, "r--", linewidth=2, label=label)

        img_w, img_h = img.size
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.axis("off")
        out_path = os.path.join(plot_dir, image_name.replace(".jpeg", "_pred.png").replace(".jpg", "_pred.png"))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close(fig)
        buf.seek(0)
        with open(out_path, "wb") as f:
            f.write(buf.read())
        print(f"[INFO] Saved prediction plot: {out_path}")


def _feature_csv_has_compounds(feature_path):
    """feature.csv 是否含至少一行有效化合物（空文件或仅表头 → False）。"""
    if not os.path.isfile(feature_path) or os.path.getsize(feature_path) == 0:
        return False
    try:
        data = pd.read_csv(feature_path, encoding_errors="ignore")
    except pd.errors.EmptyDataError:
        return False
    if data.empty or "Compound Name" not in data.columns:
        return False
    key = data[["Compound Name", "mz", "RT"]].dropna(how="all") if "mz" in data.columns and "RT" in data.columns else data
    return len(key) > 0


def run_single(args, images_path, prediction_output, plot_dir):
    """
    对单个 images_path 执行预测与积分，结果保存到 prediction_output 和 plot_dir。
    返回 True 表示成功，False 表示失败。
    """
    # 加载 feature.csv
    feature_path = args.feature
    if feature_path is None:
        feature_path = os.path.join(images_path, "feature.csv")
    elif os.path.isdir(feature_path):
        feature_path = os.path.join(feature_path, "feature.csv")
    if not os.path.exists(feature_path):
        print(f"[ERROR] feature.csv not found: {feature_path}")
        return False

    if not _feature_csv_has_compounds(feature_path):
        qc_hint = os.path.join(images_path, "pipeline_qc_excluded.csv")
        print(
            "[WARN] Skip prediction: feature.csv has no compounds (%s). "
            "Often all channels were removed by pipeline QC (e.g. wash/blank below "
            "--pipeline_min_max_intensity).%s"
            % (
                feature_path,
                " See %s" % qc_hint if os.path.isfile(qc_hint) else "",
            )
        )
        return True

    xic_info = load_features(feature_path, preserve_order=True)

    if not os.path.exists(images_path):
        print(f"[ERROR] Images path not found: {images_path}")
        return False

    # 运行模型预测（plot 延后到积分后，以便叠加基线）
    print(f"[INFO]  Running MRMPFormer model...")
    model_images_path = images_path
    smoothed_tmp_dir = None
    sigma = float(getattr(args, "predict_smooth_sigma", 0.0) or 0.0)
    if sigma > 0:
        from PIL import Image, ImageFilter
        smoothed_tmp_dir = tempfile.mkdtemp(prefix=f"roi_smooth_sigma{sigma}_")
        src_dir = Path(images_path)
        for p in src_dir.iterdir():
            if not p.is_file():
                continue
            dst = Path(smoothed_tmp_dir) / p.name
            suffix = p.suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                try:
                    img = Image.open(p).convert("RGB")
                    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
                    img.save(dst)
                except Exception:
                    shutil.copy2(p, dst)
            else:
                shutil.copy2(p, dst)
        model_images_path = smoothed_tmp_dir
        print(f"[INFO] 预测输入高斯平滑已启用: sigma={sigma}")

    results = build_predictor(
        model_path=args.model,
        images_path=model_images_path,
        threshold=args.threshold,
        plot=False,
        plot_dir=plot_dir,
        verbose=args.verbose
    )

    print(f"[INFO] ✅ Model prediction completed. Detected peaks in {len(results)} images.")

    # 定量积分
    print("[INFO] Performing quantification...")
    xic_npy_path = os.path.join(images_path, "xic_matrix.npy")
    if not os.path.exists(xic_npy_path):
        print(f"[ERROR] xic_matrix.npy not found: {xic_npy_path}")
        print("[ERROR] Please run testXIC.py first to generate XIC matrix.")
        return False
    
    xic_full = np.load(xic_npy_path)
    rt_array = xic_full[0, :]  # RT in minutes (如果检测到秒制会自动转换)
    intensity_matrix = xic_full[1:, :]  # (N, S)
    
    # 与原项目一致：quantify 期待 rt 为分钟。对历史秒制数据做兜底转换。
    if np.nanmax(rt_array) > 200:
        print("[WARN] Detected second-scale RT axis, converting to minutes for quantify compatibility.")
        rt_array = rt_array / 60.0

    # 2. 重构 xic_list: list of [rt (min), intensity] arrays
    xic_list = [
        np.vstack([rt_array, intensity_matrix[i, :]])
        for i in range(intensity_matrix.shape[0])
    ]
    print(f"[INFO] ✅ Prepared xic_list with {len(xic_list)} compounds")
    
    # 按最小长度截断，避免越界
    aligned_len = min(len(xic_info), len(xic_list))
    if aligned_len < len(xic_info) or aligned_len < len(xic_list):
        print(f"[WARN] Feature/XIC count mismatch: features={len(xic_info)}, xic={len(xic_list)}. Truncating to {aligned_len}.")
        xic_info = xic_info.iloc[:aligned_len].reset_index(drop=True)
        xic_list = xic_list[:aligned_len]

    # 3. 预测输出适配为 quantify 所需格式：
    #    按图像名对齐到 XIC 行（支持 N_mz* 或 Q1_xxx_Q3_yyy），每个 XIC 仅保留最高置信度框
    xic_count = len(xic_list)
    prediction_for_quantify = _adapt_prediction_for_quantify(
        results, xic_count=xic_count, xic_info=xic_info
    )
    # 对齐校验：统计成功匹配的 XIC 数
    n_filled = sum(1 for i in range(xic_count) if len(prediction_for_quantify[i][1]) > 0)
    if n_filled < len(results) and len(results) > 0:
        print(f"[INFO] 对齐: {len(results)} 张有检测的图像 -> {n_filled} 个 XIC 槽已填充（同名 N 已去重或部分未匹配）")
    if n_filled < xic_count and xic_count <= 1000:
        empty = [i + 1 for i in range(xic_count) if len(prediction_for_quantify[i][1]) == 0]
        if len(empty) <= 20:
            print(f"[INFO] area=0 的 compound 行号（1-based）: {empty}")
        else:
            print(f"[INFO] area=0 的 compound 行号（前20个）: {empty[:20]} ... 共 {len(empty)} 个")

    # 4. （已按需求停用）原本会调用 quantify 并输出 area.csv
    # area = quantify(xic_list, prediction_for_quantify, xic_info)
    # print(f"[INFO] ✅ Quantification completed: {len(area)} records")

    # 5. 仅输出 prediction.csv（逐框积分明细）
    pred_output_dir = os.path.dirname(prediction_output)
    if pred_output_dir:
        Path(pred_output_dir).mkdir(parents=True, exist_ok=True)

    roi_windows = _load_roi_windows(images_path)
    if roi_windows:
        print(f"[INFO] 使用 roi_windows.csv 做像素→RT 映射（共 {len(roi_windows)} 条）")

    integration_method = getattr(args, "integration_method", "linear")
    external_baselines = {}  # 仅当 integration_method=external_baseline 时使用
    if integration_method == "external_baseline":
        baseline_json = getattr(args, "baseline_json", None)
        if baseline_json:
            external_baselines = _load_external_baselines(baseline_json)
            print(f"[INFO] 外部基线模式: 已加载 {len(external_baselines)} 条基线")
        else:
            print("[WARN] external_baseline 需指定 --baseline_json，回退为 raw 积分")

    # 重试机制：处理文件占用问题
    max_retries = 3
    retry_delay = 1.0  # 秒
    
    prediction_rows = _integrate_each_predicted_box(
        xic_list, prediction_for_quantify, xic_info,
        baseline_correction=getattr(args, "baseline_correction", False),
        integration_method=integration_method,
        roi_windows=roi_windows,
        external_baselines=external_baselines if external_baselines else None,
    )
    df_prediction = pd.DataFrame(prediction_rows)

    # 每个 (mz, q3) 只保留面积最大的一行，保证一个母离子最多对应两条子离子（Quantifier+Qualifier）
    if "q3" in df_prediction.columns:
        before = len(df_prediction)
        df_prediction = df_prediction.loc[
            df_prediction.groupby(["mz", "q3"], dropna=False)["area"].idxmax()
        ].reset_index(drop=True)
        if len(df_prediction) < before:
            print(f"[INFO] 按 (mz, q3) 去重: {before} -> {len(df_prediction)} 行（一母离子仅保留至多两条子离子）")

    # 统计 area=0 的行（对应「该 m/z 的 ROI 未检出峰」或「图像名未匹配 N_mz」）
    zero_area = (df_prediction["area"] == 0) | (df_prediction["area"].isna())
    n_zero = zero_area.sum()
    if n_zero > 0:
        print(f"[INFO] prediction.csv 中 area=0 的行数: {n_zero}（共 {len(df_prediction)} 行）")
        print("[INFO] 原因: 该化合物对应 ROI 图像上模型未检出置信度 > threshold 的峰，或图像名不符合 N_mz*.jpeg 导致未匹配。")

    # 绘图：若启用基线积分或外部基线，在图上叠加基线
    if args.plot and len(results) > 0:
        _plot_predictions_with_baseline(
            results, df_prediction, xic_list, xic_info, roi_windows,
            plot_dir,
            baseline_correction=getattr(args, "baseline_correction", False),
            integration_method=getattr(args, "integration_method", "linear"),
            external_baselines=external_baselines if external_baselines else None,
        )

    out_path = prediction_output
    saved = False
    for attempt in range(max_retries):
        try:
            df_prediction.to_csv(out_path, index=False)
            print(f"[INFO] ✅ Per-box prediction integration saved to: {out_path}")
            saved = True
            break
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"[WARN] ⚠️ File locked, retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # 原路径被占用时写入带时间戳的备用文件，避免结果丢失
                fallback_dir = os.path.dirname(out_path) or "."
                base = os.path.splitext(os.path.basename(out_path))[0]
                fallback_path = os.path.join(fallback_dir, f"{base}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                try:
                    df_prediction.to_csv(fallback_path, index=False)
                    print(f"[ERROR] ❌ Could not write to {out_path} (Permission denied).")
                    print(f"[INFO] ✅ Results saved to fallback: {fallback_path}")
                    saved = True
                except Exception as e2:
                    print(f"[ERROR] ❌ Failed to save after {max_retries} attempts: {e}")
                    print(f"[ERROR] Fallback path also failed: {e2}")
                    print(f"[INFO] You can manually export: df_prediction.to_csv('{out_path}', index=False)")
    if not saved:
        print(f"[WARN] Prediction table was not saved to disk.")

    if smoothed_tmp_dir and (not bool(getattr(args, "keep_smoothed_inputs", False))):
        shutil.rmtree(smoothed_tmp_dir, ignore_errors=True)

    return True


def main(args):
    # ====== 设备与 CPU 信息、计时开始 ======
    from utils.torch_device import resolve_torch_device

    cpu_cores = multiprocessing.cpu_count()
    print("=" * 60)
    resolve_torch_device(verbose=True)
    print("[INFO]  CPU 逻辑核心数: {}".format(cpu_cores))
    start_time = time.time()
    # ===================================

    print("[INFO] Starting MRMPFormer - Simplified Mode")
    print("=" * 60)

    if args.batch_dir:
        # 批量模式：处理 batch_dir 下所有子目录
        batch_path = Path(args.batch_dir)
        if not batch_path.exists():
            print(f"[ERROR] Batch directory not found: {batch_path}")
            return
        output_base = Path(args.batch_output)
        output_base.mkdir(parents=True, exist_ok=True)

        subdirs = sorted([d for d in batch_path.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"[WARN] No subdirectories found in {batch_path}")
            return

        print(f"[INFO] Batch mode: {len(subdirs)} subdir(s) in {batch_path}")
        integration_method = getattr(args, "integration_method", "linear")
        pred_basename = f"prediction_{integration_method}.csv" if integration_method != "linear" else "prediction.csv"
        for i, subdir in enumerate(subdirs):
            print("=" * 60)
            print(f"[BATCH {i+1}/{len(subdirs)}] {subdir.name}")
            print("=" * 60)
            pred_out = output_base / subdir.name / pred_basename
            plot_dir = output_base / subdir.name / "predicted_plots"
            pred_out.parent.mkdir(parents=True, exist_ok=True)
            feature_csv = subdir / "feature.csv"
            if not _feature_csv_has_compounds(str(feature_csv)):
                print(
                    "[WARN] Skip [BATCH %d/%d] %s: no compounds in feature.csv (QC excluded all channels?)."
                    % (i + 1, len(subdirs), subdir.name)
                )
                continue
            run_single(args, str(subdir), str(pred_out), str(plot_dir))

        # 结束计时
        elapsed = time.time() - start_time
        print("=" * 60)
        print(f"[✅ BATCH DONE] {len(subdirs)} subdir(s) processed in {elapsed:.2f} s")
        print("=" * 60)
    else:
        # 单目录模式：非 linear（默认）时另存为 prediction_{method}.csv，不覆盖原有
        integration_method = getattr(args, "integration_method", "linear")
        if integration_method != "linear":
            base = Path(args.prediction_output)
            pred_out = str(base.parent / f"{base.stem}_{integration_method}{base.suffix}")
            print(f"[INFO] 积分方法 {integration_method}，结果另存为: {pred_out}")
        else:
            pred_out = args.prediction_output
        plot_dir = args.plot_dir
        run_single(args, args.images_path, pred_out, plot_dir)

        # 结束计时
        elapsed = time.time() - start_time
        print("[INFO]  总运行时间: {:.2f} 秒 ({:.2f} 分钟)".format(elapsed, elapsed / 60.0))
    print("=" * 60)
    print("[✅ DONE] MRMPFormer simplified run completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MRMPFormer - Simplified Mode (Direct feature.csv + ROI images)"
    )

    # 必需参数（--feature 默认与 main.py 一致：取 ROI 目录下的 feature.csv）
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Path to feature.csv (default: <images_path>/feature.csv, same as main.py new_data_path)"
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default=None,
        help="Path to directory containing ROI images (required in single mode; ignored when --batch_dir is set)"
    )
    parser.add_argument(
        "--batch_dir",
        type=str,
        default=None,
        help="Batch mode: directory containing subdirs (e.g. xic-roi-batch); each subdir = one input set, output to batch_output/<subdir_name>/"
    )
    parser.add_argument(
        "--batch_output",
        type=str,
        default="results/batch_predictions",
        help="Base output dir for batch mode; each subdir's results -> batch_output/<subdir_name>/prediction.csv and predicted_plots/"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)"
    )

    # 可选参数
    parser.add_argument(
        "--output",
        type=str,
        default="results/area.csv",
        help="(disabled) Output quantification CSV path"
    )
    parser.add_argument(
        "--prediction_output",
        type=str,
        default="results/prediction.csv",
        help="Output CSV path for per-box prediction integration details"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Confidence threshold for predictions (default: 0.99)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate prediction visualization plots"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="predicted_plots",
        help="Directory to save prediction plots with bounding boxes"
    )
    parser.add_argument(
        "--predict_smooth_sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma for ROI images before model inference (0=off)"
    )
    parser.add_argument(
        "--keep_smoothed_inputs",
        action="store_true",
        help="Keep temporary smoothed ROI folder when --predict_smooth_sigma > 0"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image DEBUG logs during prediction (slower)"
    )
    parser.add_argument(
        "--baseline_correction",
        action="store_true",
        help="Apply baseline correction to integration (reduces overestimation from wide windows)"
    )
    parser.add_argument(
        "--integration_method",
        type=str,
        default="linear",
        choices=["linear", "peak_adaptive", "raw", "adaptive", "minval_noise_right", "external_baseline"],
        help="积分方法: linear=起止点基线(默认), raw=无基线, peak_adaptive=谷-谷, minval_noise_right=谷-噪声, external_baseline=外部(mz,x[],y[])基线",
    )
    parser.add_argument(
        "--baseline_json",
        type=str,
        default=None,
        help="外部基线 JSON 路径，用于 integration_method=external_baseline。格式: [{mz_name, x, y, q3?}, ...]",
    )

    args = parser.parse_args()

    if not args.batch_dir and not args.images_path:
        parser.error("Either --images_path (single mode) or --batch_dir (batch mode) is required.")
    if args.batch_dir and args.images_path:
        print("[INFO] --batch_dir is set; --images_path will be ignored.")
    main(args)