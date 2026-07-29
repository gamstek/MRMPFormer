"""
从多浓度标品的 CSV 提取浓度-面积数据，绘制标准曲线（线性拟合），计算 R²。

不存在的图像和积分为 0 的数据不参与标曲拟合和计算。

用法:
  python build_standard_curves.py --dir "D:\pycharm\QuanFormer-main\results\batch_predictions"
  python build_standard_curves.py --dir "D:\pycharm\QuanFormer-main\results\compare" --csv_pattern "compare_*.csv" --recursive --no_plots
  python build_standard_curves.py --dir "path/to/folder" --filename "my_result.csv" --recursive
  python build_standard_curves.py --dir "path/to/folder" --csv_pattern "prediction*.csv"

输出:
  - r2_summary.csv: 每个物质的 R²、斜率、截距、点数、r2_5pts（R²<0.995 且 n>=7 时用剔除 2 离群后的 5 点 R²）
  - standard_curves/: 每个物质的浓度-面积拟合曲线图
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_concentration(text):
    """解析浓度字符串如 10ppb/0.5ppm，返回 (数值, 单位)。"""
    if text is None:
        return None, None
    text = str(text).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ppb|ppm|ng|ug|mg)", text, flags=re.IGNORECASE)
    if not m:
        return None, None
    return float(m.group(1)), m.group(2).lower()


def concentration_to_numeric(text):
    """将浓度转为数值（用于拟合），统一到 ppb 量纲。"""
    val, unit = parse_concentration(text)
    if val is None:
        return np.nan
    if unit == "ppb":
        return val
    if unit == "ppm":
        return val * 1000  # ppm -> ppb
    if unit in ("ng", "ug", "mg"):
        return val  # 保持原值，用户需保证单位一致
    return val


def infer_concentration_from_path(csv_path):
    """从 CSV 路径或父目录名或文件名推断浓度。"""
    p = Path(csv_path)
    # 父目录名如 20240330_PES_CMA_加标-STD_10ppb_青菜
    parent_name = p.parent.name
    conc = concentration_to_numeric(parent_name)
    if not np.isnan(conc):
        return conc
    # 文件名如 prediction_10ppb.csv 或 compare_10ppb.csv
    stem = p.stem
    conc = concentration_to_numeric(stem)
    if not np.isnan(conc):
        return conc
    return np.nan


def collect_compound_area_per_file(csv_path, concentration, area_column_override=None):
    """
    从单个 CSV 提取每个化合物 (mz, q3) 的面积。
    支持两种格式：
    1) prediction 格式：列 mz, q3, area
    2) compare 格式（加标 vs auto）：列 precursor_mz, fragment_mz, manual_area（或 auto_area）
    积分为 0 的数据不参与。
    返回 [(mz, q3, area), ...]
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # compare_*.csv 格式：precursor_mz, fragment_mz, manual_area, auto_area
    # 默认用 auto_area 评估软件线性；manual_area 会令不同软件得到相同 R²
    if "precursor_mz" in df.columns and ("manual_area" in df.columns or "auto_area" in df.columns):
        if area_column_override:
            area_col = area_column_override
        else:
            area_col = "auto_area" if "auto_area" in df.columns else "manual_area"
        df["mz"] = pd.to_numeric(df["precursor_mz"], errors="coerce")
        df["q3"] = pd.to_numeric(df["fragment_mz"], errors="coerce") if "fragment_mz" in df.columns else np.nan
        df["area"] = pd.to_numeric(df[area_col], errors="coerce")
        df = df.dropna(subset=["mz", "area"])
        df = df[df["area"] > 0]
        if df.empty:
            return []
        rows = []
        for _, r in df.iterrows():
            rows.append((float(r["mz"]), float(r["q3"]) if pd.notna(r["q3"]) else np.nan, float(r["area"])))
        return rows
    # prediction 格式：mz, q3, area
    if "area" not in df.columns or "mz" not in df.columns:
        return []
    df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df.dropna(subset=["mz", "area"])
    df = df[df["area"] > 0]
    if df.empty:
        return []
    q3_col = "q3" if "q3" in df.columns else None
    if q3_col:
        df["q3"] = pd.to_numeric(df["q3"], errors="coerce")
        best = df.loc[df.groupby(["mz", "q3"], dropna=False)["area"].idxmax()]
    else:
        df["q3"] = np.nan
        best = df.loc[df.groupby("mz")["area"].idxmax()]
    rows = []
    for _, r in best.iterrows():
        mz = float(r["mz"])
        q3 = float(r["q3"]) if pd.notna(r["q3"]) else np.nan
        area = float(r["area"])
        rows.append((mz, q3, area))
    return rows


def build_compound_data(csv_paths, area_column_override=None):
    """
    汇总所有 CSV 的浓度-面积数据。
    返回: {(mz, q3): [(conc, area), ...], ...}
    """
    compound_data = {}
    for csv_path in csv_paths:
        conc = infer_concentration_from_path(csv_path)
        if np.isnan(conc):
            continue
        rows = collect_compound_area_per_file(csv_path, conc, area_column_override)
        for mz, q3, area in rows:
            key = (round(mz, 4), round(q3, 4) if not np.isnan(q3) else np.nan)
            if key not in compound_data:
                compound_data[key] = []
            compound_data[key].append((conc, area))
    return compound_data


R2_THRESHOLD = 0.995  # R² 低于此值时，用剔除 2 离群后的 5 点重算 R²


def linear_fit_r2(x, y):
    """线性拟合 y = k*x + b，返回 (k, b, r2)。排除 x<=0 及非有限值。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    k, b = np.polyfit(x, y, 1)
    y_pred = k * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(k), float(b), float(r2)


def r2_5pts_after_remove_2_outliers(concs, areas):
    """
    剔除 2 个残差最大的点后，用剩余 5 点拟合并返回 R²。
    若点数 < 7 返回 np.nan。
    """
    concs = np.asarray(concs, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    mask = np.isfinite(concs) & np.isfinite(areas) & (concs > 0)
    concs, areas = concs[mask], areas[mask]
    if len(concs) < 7:
        return np.nan
    k, b, _ = linear_fit_r2(concs, areas)
    if not np.isfinite(k):
        return np.nan
    pred = k * concs + b
    res = np.abs(areas - pred)
    idx_remove = np.argsort(res)[-2:]
    keep = np.ones(len(concs), dtype=bool)
    keep[idx_remove] = False
    _, _, r2_5 = linear_fit_r2(concs[keep], areas[keep])
    return float(r2_5)


def main():
    parser = argparse.ArgumentParser(
        description="从多浓度 prediction.csv 绘制标准曲线，计算 R²"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="包含 prediction.csv 的文件夹（可含子目录）",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="指定要参与计算的 CSV 文件名（如 my_result.csv）；填写后只按该文件名检索并计算，忽略 --csv_pattern",
    )
    parser.add_argument(
        "--csv_pattern",
        type=str,
        default="prediction*.csv",
        help="CSV 文件名模式（未填 --filename 时生效），默认 prediction*.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录中的 CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认与 --dir 相同",
    )
    parser.add_argument(
        "--curves_dir",
        type=str,
        default="standard_curves",
        help="曲线图子目录名，默认 standard_curves",
    )
    parser.add_argument(
        "--ppb_only",
        action="store_true",
        help="仅使用路径中含 ppb 的 CSV（如 STD_10ppb_青菜），排除 ug/kg 等",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="不生成曲线图，仅输出 r2_summary.csv",
    )
    parser.add_argument(
        "--area_column",
        type=str,
        default=None,
        choices=["auto_area", "manual_area"],
        help="compare 格式时用哪列作面积：auto_area=软件积分(默认，评估软件线性), manual_area=标准面积",
    )
    parser.add_argument(
        "--r2_threshold",
        type=float,
        default=R2_THRESHOLD,
        help=f"R² 低于此值时用 5 点重算 r2_5pts（默认 {R2_THRESHOLD}）",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir).resolve()
    if not base_dir.is_dir():
        print(f"[ERROR] 目录不存在: {base_dir}")
        return

    output_dir = Path(args.output_dir).resolve() if args.output_dir else base_dir
    curves_dir = output_dir / args.curves_dir
    curves_dir.mkdir(parents=True, exist_ok=True)

    # 若用户填写了 --filename，只按该文件名检索
    if args.filename:
        args.csv_pattern = args.filename.strip()
        print(f"[INFO] 按指定文件名计算: {args.csv_pattern}")

    # 收集 CSV：支持递归、多种模式与回退
    def gather_csv(root, pattern):
        if args.recursive:
            out = sorted(Path(root).rglob(pattern))
        else:
            root = Path(root)
            out = sorted(root.glob(pattern))
            for d in root.iterdir():
                if d.is_dir():
                    out.extend(d.glob(pattern))
            out = sorted(set(out))
        return [f for f in out if f.is_file()]

    csv_files = gather_csv(base_dir, args.csv_pattern)
    # 未指定 --filename 时才做默认/回退检索
    if not csv_files and args.recursive and not args.filename:
        if "*" not in args.csv_pattern:
            csv_files = gather_csv(base_dir, "prediction*.csv")
        if not csv_files and args.csv_pattern != "prediction.csv":
            csv_files = gather_csv(base_dir, "prediction.csv")
        if not csv_files and base_dir.parent != base_dir:
            csv_files = gather_csv(base_dir.parent, "prediction*.csv")
            if csv_files:
                print(f"[INFO] 当前目录未找到 CSV，已从上级目录检索: {base_dir.parent}")
    if args.ppb_only:
        csv_files = [f for f in csv_files if "ppb" in str(f).lower()]
    if not csv_files:
        if args.filename:
            print(f"[ERROR] 未找到文件: {args.csv_pattern}（--dir 下及递归子目录）")
        else:
            print(f"[ERROR] 未找到匹配 '{args.csv_pattern}' 的 CSV 文件（已尝试回退）")
        print(f"[INFO] 可指定 --filename 使用自定义文件名，或将 --dir 指向含该 CSV 的目录")
        return

    print(f"[INFO] 找到 {len(csv_files)} 个 CSV 文件")

    compound_data = build_compound_data(csv_files, area_column_override=args.area_column)
    if not compound_data:
        print("[ERROR] 未提取到有效浓度-面积数据")
        return

    # 线性拟合与 R²
    results = []
    for (mz, q3), points in compound_data.items():
        concs = [p[0] for p in points]
        areas = [p[1] for p in points]
        k, b, r2 = linear_fit_r2(concs, areas)
        compound_id = f"mz{mz:.4f}_q3{q3:.4f}" if not np.isnan(q3) else f"mz{mz:.4f}"
        r2_5pts = np.nan
        if (np.isnan(r2) or r2 < args.r2_threshold) and len(points) >= 7:
            r2_5pts = r2_5pts_after_remove_2_outliers(concs, areas)
        else:
            r2_5pts = r2
        results.append({
            "compound_id": compound_id,
            "mz": mz,
            "q3": q3 if not np.isnan(q3) else "",
            "slope": k,
            "intercept": b,
            "r2": r2,
            "n_points": len(points),
            "r2_5pts": r2_5pts,
        })

    df_r2 = pd.DataFrame(results)
    df_r2 = df_r2.sort_values("r2", ascending=False)
    r2_path = output_dir / "r2_summary.csv"
    df_r2.to_csv(r2_path, index=False, encoding="utf-8-sig")
    n_recalc = ((df_r2["r2"] < args.r2_threshold) | df_r2["r2"].isna()) & (df_r2["n_points"] >= 7)
    if n_recalc.any():
        print(f"[INFO] R² < {args.r2_threshold} 且 n>=7 的物质: {n_recalc.sum()} 个，已计算 5 点 R² 填入 r2_5pts 列")
    print(f"[OK] R2 summary saved: {r2_path}")

    # 绘制每个物质的标准曲线
    if args.no_plots:
        print("[INFO] Skipping plots (--no_plots)")
    else:
        for (mz, q3), points in compound_data.items():
            concs = np.array([p[0] for p in points])
            areas = np.array([p[1] for p in points])
            k, b, r2 = linear_fit_r2(concs, areas)
            compound_id = f"mz{mz:.4f}_q3{q3:.4f}" if not np.isnan(q3) else f"mz{mz:.4f}"

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(concs, areas, color="steelblue", s=40, label="Data")
            if len(concs) >= 2 and np.isfinite(k):
                x_fit = np.linspace(concs.min(), concs.max(), 100)
                y_fit = k * x_fit + b
                ax.plot(x_fit, y_fit, "r-", lw=2, label=f"Fit R2={r2:.4f}")
            ax.set_xlabel("Concentration (ppb)", fontsize=11)
            ax.set_ylabel("Area", fontsize=11)
            ax.set_title(f"Standard curve - {compound_id}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = curves_dir / f"{compound_id}.png"
            plt.savefig(out_path, dpi=120)
            plt.close()

        print(f"[OK] Standard curves saved to: {curves_dir} ({len(compound_data)} plots)")
    print(f"[INFO] Valid fits (R2 not NaN): {df_r2['r2'].notna().sum()}")


if __name__ == "__main__":
    main()
