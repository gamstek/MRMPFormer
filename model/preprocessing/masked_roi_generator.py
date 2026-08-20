# -*- coding: utf-8 -*-
"""
将 prediction.csv 识别出的主峰区域删除，替换为基线/噪声，生成用于测试其余小峰识别的掩蔽图像。
与 testXIC 一致：从 XIC 取最高点，高斯光滑后提取 ROI 图像。

流程：
1. 读取 batch_predictions 各子目录的 prediction.csv、xic_matrix.npy、roi_windows.csv
2. 对每条预测：取主峰区域 [rt_min, rt_max] 的 XIC 强度
3. 按 --mask_method 替换主峰区：random_noise | linear_interp | baseline_interp
4. 高斯光滑后按 roi_windows 提取 ROI 图，保存到输出目录

用法:
  python generate_masked_roi_for_small_peak_test.py --batch_predictions results/batch_predictions --images_root xic-roi-batch --output_dir results/batch_predictions_masked --seed 42
  python generate_masked_roi_for_small_peak_test.py ... --mask_method linear_interp   # 线性插值连接前后基线
  python generate_masked_roi_for_small_peak_test.py ... --mask_method baseline_interp # 真实基线曲线插值（适合重叠峰）
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

ROI_IMAGE_WIDTH_PX = 400.0
ROI_IMAGE_HEIGHT_PX = 300.0


def load_roi_windows(subdir):
    p = Path(subdir) / "roi_windows.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "image" not in df.columns or "rt_lo" not in df.columns or "rt_hi" not in df.columns:
        return {}
    return {str(row["image"]).strip(): (float(row["rt_lo"]), float(row["rt_hi"]))
            for _, row in df.iterrows()}


def estimate_baseline_noise(rt, intensity, rt_min, rt_max, margin_min=0.1):
    """从峰前、峰后区域估计基线噪声的均值和标准差。"""
    rt = np.asarray(rt, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    mask_left = (rt >= rt_min - margin_min) & (rt < rt_min)
    mask_right = (rt > rt_max) & (rt <= rt_max + margin_min)
    vals = []
    if np.any(mask_left):
        vals.extend(intensity[mask_left])
    if np.any(mask_right):
        vals.extend(intensity[mask_right])
    if len(vals) < 2:
        mask_out = (rt < rt_min) | (rt > rt_max)
        if np.sum(mask_out) >= 2:
            vals = intensity[mask_out]
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size < 2:
        return float(np.mean(intensity)), float(np.std(intensity)) if np.std(intensity) > 0 else 1.0
    return float(np.mean(vals)), max(float(np.std(vals)), 1e-6)


def get_last_25pct_noise_values(rt, intensity, rt_lo, rt_hi, frac=0.25):
    """
    取图像后 frac（默认25%）RT 区间的强度值，用于随机采样填充。
    后25% = rt 在 [rt_hi - frac*(rt_hi-rt_lo), rt_hi] 内的点。
    """
    rt = np.asarray(rt, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    rt_range = rt_hi - rt_lo
    rt_cut = rt_hi - frac * rt_range
    mask = (rt >= rt_cut) & (rt <= rt_hi)
    vals = intensity[mask]
    return vals


def _replace_peak_random_noise(rt, intensity, rt_min, rt_max, rt_lo=None, rt_hi=None, use_last_25pct=True):
    """
    主峰区替换为噪声。use_last_25pct=True 时从图像后25%随机点采样；否则用峰前峰后基线估计。
    """
    mask_peak = (rt >= rt_min) & (rt <= rt_max)
    n_peak = int(np.sum(mask_peak))
    if n_peak <= 0:
        return

    if use_last_25pct and rt_lo is not None and rt_hi is not None:
        vals = get_last_25pct_noise_values(rt, intensity, rt_lo, rt_hi, frac=0.25)
        if vals.size >= 1:
            fill_vals = np.random.choice(vals, size=n_peak, replace=True)
            intensity[mask_peak] = np.maximum(fill_vals, 0.0)
            return

    mean_noise, std_noise = estimate_baseline_noise(rt, intensity, rt_min, rt_max)
    noise = np.random.normal(mean_noise, std_noise, n_peak)
    noise = np.maximum(noise, 0.0)
    intensity[mask_peak] = noise


def _replace_peak_linear_interp(rt, intensity, rt_min, rt_max):
    """主峰区用线性插值连接峰前、峰后基线端点。"""
    mask_peak = (rt >= rt_min) & (rt <= rt_max)
    if not np.any(mask_peak):
        return
    idx_left = np.where(rt < rt_min)[0]
    idx_right = np.where(rt > rt_max)[0]
    y_left = float(np.mean(intensity[idx_left[-5:]])) if len(idx_left) >= 1 else 0.0
    y_right = float(np.mean(intensity[idx_right[:5]])) if len(idx_right) >= 1 else 0.0
    rt_peak = rt[mask_peak]
    t = (rt_peak - rt_min) / max(rt_max - rt_min, 1e-9)
    intensity[mask_peak] = y_left * (1 - t) + y_right * t
    intensity[mask_peak] = np.maximum(intensity[mask_peak], 0.0)


def _replace_peak_baseline_interp(rt, intensity, rt_min, rt_max):
    """主峰区用真实基线区曲线插值（适合重叠峰，保留基线漂移）。"""
    mask_baseline = (rt < rt_min) | (rt > rt_max)
    if np.sum(mask_baseline) < 2:
        _replace_peak_linear_interp(rt, intensity, rt_min, rt_max)
        return
    rt_base = rt[mask_baseline]
    int_base = intensity[mask_baseline]
    order = np.argsort(rt_base)
    rt_base = rt_base[order]
    int_base = int_base[order]
    mask_peak = (rt >= rt_min) & (rt <= rt_max)
    rt_peak = rt[mask_peak]
    interp_vals = np.interp(rt_peak, rt_base, int_base)
    intensity[mask_peak] = np.maximum(interp_vals, 0.0)


def mask_main_peak_and_redraw(rt, intensity, rt_min, rt_max, rt_lo, rt_hi, output_path, mask_method="random_noise", smooth_sigma=1.0, use_last_25pct_noise=True):
    """
    将 [rt_min, rt_max] 区间的强度按 mask_method 替换，高斯光滑后按 roi_windows 提取 ROI 图（与 testXIC 一致）。
    mask_method: random_noise | linear_interp | baseline_interp
    use_last_25pct_noise: random_noise 时从图像后25%随机点采样
    返回修改后的 intensity（用于保存 xic_matrix）。
    """
    rt = np.asarray(rt, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64).copy()
    mask_peak = (rt >= rt_min) & (rt <= rt_max)
    if np.any(mask_peak):
        if mask_method == "linear_interp":
            _replace_peak_linear_interp(rt, intensity, rt_min, rt_max)
        elif mask_method == "baseline_interp":
            _replace_peak_baseline_interp(rt, intensity, rt_min, rt_max)
        else:
            _replace_peak_random_noise(rt, intensity, rt_min, rt_max, rt_lo, rt_hi, use_last_25pct=use_last_25pct_noise)

    plot_intensity = intensity.copy()
    if smooth_sigma > 0 and plot_intensity.size >= 3:
        plot_intensity = gaussian_filter1d(plot_intensity, sigma=smooth_sigma)

    mask_plot = (rt >= rt_lo) & (rt <= rt_hi)
    if np.sum(mask_plot) < 2:
        plot_rt, plot_intensity_roi = rt, plot_intensity
    else:
        plot_rt = rt[mask_plot]
        plot_intensity_roi = plot_intensity[mask_plot]

    fig = Figure(figsize=(4, 3), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(plot_rt, plot_intensity_roi, color="blue", linewidth=1.5)
    ax.set_xlim(rt_lo, rt_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.print_jpeg(output_path)
    return intensity


def main():
    parser = argparse.ArgumentParser(
        description="生成主峰掩蔽的 ROI 图像，用于测试小峰识别"
    )
    parser.add_argument(
        "--batch_predictions",
        type=str,
        default="../output/inference/batch_predictions",
        help="batch_predictions 根目录",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="../output/inference/xic-roi-batch",
        help="ROI 图像所在根目录（子目录名与 batch_predictions 子目录对应）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output/inference/batch_predictions_masked",
        help="掩蔽图像输出根目录",
    )
    parser.add_argument(
        "--prediction_csv",
        type=str,
        default="prediction.csv",
        help="预测 CSV 文件名",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（保证可重复）",
    )
    parser.add_argument(
        "--mask_method",
        type=str,
        choices=["random_noise", "linear_interp", "baseline_interp"],
        default="random_noise",
        help="主峰区替换方式: random_noise=基线噪声随机采样; linear_interp=线性插值; baseline_interp=真实基线曲线插值",
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.0,
        help="高斯光滑 sigma（与 testXIC 一致，0 表示不平滑）",
    )
    parser.add_argument(
        "--no_last_25pct_noise",
        action="store_true",
        help="random_noise 时不用图像后25%%随机点，改用峰前峰后基线估计",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    batch_path = Path(args.batch_predictions).resolve()
    images_root = Path(args.images_root).resolve()
    output_root = Path(args.output_dir).resolve()

    if not batch_path.is_dir():
        print(f"[ERROR] batch_predictions 不存在: {batch_path}")
        return

    subdirs = sorted([d for d in batch_path.iterdir() if d.is_dir()])
    total = 0
    for subdir in subdirs:
        conc_name = subdir.name
        pred_path = subdir / args.prediction_csv
        xic_path = subdir / "xic_matrix.npy"
        if not pred_path.exists() or not xic_path.exists():
            print(f"[WARN] 跳过 {conc_name}: 缺少 {args.prediction_csv} 或 xic_matrix.npy")
            continue

        img_dir = images_root / conc_name
        if not img_dir.is_dir():
            img_dir = subdir
        out_dir = output_root / conc_name
        out_dir.mkdir(parents=True, exist_ok=True)

        roi_windows = load_roi_windows(subdir)
        if not roi_windows:
            roi_windows = load_roi_windows(str(img_dir))
        if not roi_windows:
            roi_windows = load_roi_windows(str(images_root / conc_name))

        xic_full = np.load(str(xic_path))
        xic_masked = xic_full.copy()
        rt_row = xic_full[0, :].astype(np.float64)
        if np.nanmax(rt_row) > 200:
            rt_row = rt_row / 60.0

        df = pd.read_csv(pred_path)
        for _, row in df.iterrows():
            image_name = str(row.get("image", "")).strip()
            if not image_name:
                continue
            compound_name = row.get("compound_name", 0)
            rt_min = row.get("rt_min")
            rt_max = row.get("rt_max")
            if pd.isna(rt_min) or pd.isna(rt_max):
                continue
            rt_min, rt_max = float(rt_min), float(rt_max)
            if rt_min >= rt_max:
                continue

            idx = int(compound_name) - 1 if pd.notna(compound_name) else 0
            if idx < 0 or idx + 1 >= xic_full.shape[0]:
                continue
            intensity = xic_masked[idx + 1, :].astype(np.float64).copy()

            rt_lo, rt_hi = roi_windows.get(image_name, (rt_row.min(), rt_row.max()))
            if isinstance(rt_lo, (list, tuple)):
                rt_lo, rt_hi = rt_lo[0], rt_hi[1]
            rt_lo, rt_hi = float(rt_lo), float(rt_hi)

            out_path = out_dir / image_name
            intensity_masked = mask_main_peak_and_redraw(
                rt_row, intensity, rt_min, rt_max, rt_lo, rt_hi, str(out_path),
                mask_method=args.mask_method,
                smooth_sigma=args.smooth_sigma,
                use_last_25pct_noise=not args.no_last_25pct_noise,
            )
            xic_masked[idx + 1, :] = intensity_masked
            total += 1

        np.save(str(out_dir / "xic_matrix.npy"), xic_masked)
        for fname in ("feature.csv", "roi_windows.csv"):
            src = subdir / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
        print(f"[OK] {conc_name}: 已生成掩蔽图像到 {out_dir}")

    print(f"[DONE] 共生成 {total} 张掩蔽图像")


if __name__ == "__main__":
    main()
