# -*- coding: utf-8 -*-
"""
从 integrate_prediction.py 批量输出目录读取各浓度子文件夹中的 prediction.csv，
按 (mz, q3) 对齐同一物质，用浓度(ppb)-面积 线性拟合计算 R²。

用法（示例）:
  python calc_r2_integrate_prediction.py ^
    --data_root "D:\\pycharm\\QuanFormer-main\\truedata\\qingcai\\out\\integrate_prediction_snr" ^
    --output "D:\\pycharm\\QuanFormer-main\\truedata\\qingcai\\out\\integrate_prediction_snr\\r2_by_compound.csv"

浓度从子文件夹名中解析，形如: ..._5ppb_... 或 ..._12.5ppb_...（不区分大小写）。

运行结束后会在输出 CSV 同目录写入 r2_gt_0.995_summary.txt，并在终端打印：
大于 0.995 的个数、总 transition 数、占比。

判定规则（与 build_standard_curves 一致）：
  - 浓度点数 ≥7：用「剔除 2 个离群点后、剩余 5 点线性拟合」的 R²（列 r2_5pts_remove2outliers）> 0.995 则计为合格；
  - 否则：用全点线性 R²（列 r2）> 0.995 则计为合格。
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_standard_curves import linear_fit_r2, r2_5pts_after_remove_2_outliers


def parse_conc_ppb(folder_name: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*ppb", str(folder_name), flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


def load_long_table(data_root: Path, area_column: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    读取两种输入格式并拼成长表，列含 concentration_ppb、sample_folder、area：
      1) integrate_prediction_snr: 每个子目录一个 prediction.csv
      2) integrate_simple_results: 根目录下多个 *_area_simple.csv
    """
    frames = []  # type: List[pd.DataFrame]
    skipped = []  # type: List[str]
    for sub in sorted(data_root.iterdir()):
        if sub.is_dir():
            pred = sub / "prediction.csv"
            if not pred.is_file():
                skipped.append("%s: 无 prediction.csv" % sub.name)
                continue
            conc = parse_conc_ppb(sub.name)
            if conc is None:
                skipped.append("%s: 文件夹名中未匹配到 ppb 浓度" % sub.name)
                continue
            pred_s = str(pred)
            df = None
            for enc in ("utf-8-sig", "utf-8", "gbk"):
                try:
                    df = pd.read_csv(pred_s, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                try:
                    df = pd.read_csv(pred_s)
                except Exception as e:
                    skipped.append("%s: 读取失败 %s" % (sub.name, e))
                    continue
            if "mz" not in df.columns:
                skipped.append("%s: 缺少 mz 列" % sub.name)
                continue
            if area_column not in df.columns:
                if "area" in df.columns:
                    df["area"] = pd.to_numeric(df["area"], errors="coerce")
                else:
                    skipped.append("%s: 缺少面积列 %s" % (sub.name, area_column))
                    continue
            else:
                df["area"] = pd.to_numeric(df[area_column], errors="coerce")
            if "q3" not in df.columns:
                df["q3"] = np.nan
            df = df.copy()
            df["concentration_ppb"] = float(conc)
            df["sample_folder"] = sub.name
            frames.append(df)
        elif sub.is_file() and sub.suffix.lower() == ".csv" and sub.name.endswith("_area_simple.csv"):
            conc = parse_conc_ppb(sub.name)
            if conc is None:
                skipped.append("%s: 文件名中未匹配到 ppb 浓度" % sub.name)
                continue
            df = None
            for enc in ("utf-8-sig", "utf-8", "gbk"):
                try:
                    df = pd.read_csv(str(sub), encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                try:
                    df = pd.read_csv(str(sub))
                except Exception as e:
                    skipped.append("%s: 读取失败 %s" % (sub.name, e))
                    continue
            if "mz" not in df.columns:
                skipped.append("%s: 缺少 mz 列" % sub.name)
                continue
            if area_column not in df.columns:
                skipped.append("%s: 缺少面积列 %s" % (sub.name, area_column))
                continue
            df = df.copy()
            df["area"] = pd.to_numeric(df[area_column], errors="coerce")
            if "q3" not in df.columns:
                df["q3"] = np.nan
            df["concentration_ppb"] = float(conc)
            df["sample_folder"] = sub.stem.replace("_area_simple", "")
            frames.append(df)
    if not frames:
        return pd.DataFrame(), skipped
    return pd.concat(frames, ignore_index=True), skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="integrate_prediction_snr 多浓度结果按物质计算 R²")
    ap.add_argument(
        "--data_root",
        type=str,
        default=str(ROOT / "truedata" / "qingcai" / "out" / "integrate_prediction_snr"),
        help="integrate_prediction 的 batch_output 根目录（其下每个浓度一个子文件夹）",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 CSV；默认写到 data_root/r2_by_compound.csv",
    )
    ap.add_argument(
        "--min_points",
        type=int,
        default=2,
        help="参与拟合的浓度点数下限，低于此则 R² 记为 nan（默认 2）",
    )
    ap.add_argument(
        "--area_column",
        type=str,
        default="area",
        help="面积列名。integrate_prediction 用 area；integrate_simple_results 可用 area_linear_baseline 或 area_raw_trapz",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        print(f"[ERROR] 目录不存在: {data_root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).expanduser().resolve() if args.output else data_root / "r2_by_compound.csv"

    long_df, skipped = load_long_table(data_root, area_column=str(args.area_column))
    for s in skipped:
        print(f"[WARN] {s}")

    if long_df.empty:
        print("[ERROR] 未读到任何 prediction.csv，退出", file=sys.stderr)
        sys.exit(1)

    min_pts = max(2, int(args.min_points))
    rows = []  # type: List[Dict[str, Any]]

    grouped = long_df.groupby(["mz", "q3"], dropna=False)
    for (mz, q3), g in grouped:
        g2 = g.sort_values("concentration_ppb").drop_duplicates(
            subset=["concentration_ppb", "sample_folder"], keep="first"
        )
        conc = np.asarray(g2["concentration_ppb"], dtype=np.float64)
        area = np.asarray(pd.to_numeric(g2["area"], errors="coerce"), dtype=np.float64)
        name = g2["compound_name"].iloc[0] if "compound_name" in g2.columns else ""
        nid = ""
        if "native_id" in g2.columns and pd.notna(g2["native_id"].iloc[0]):
            nid = str(g2["native_id"].iloc[0]).strip()

        n = int(np.sum(np.isfinite(conc) & np.isfinite(area)))
        if n < min_pts:
            k, b, r2 = np.nan, np.nan, np.nan
            r2_5 = np.nan
        else:
            k, b, r2 = linear_fit_r2(conc, area)
            r2_5 = r2_5pts_after_remove_2_outliers(conc, area) if len(conc) >= 7 else np.nan

        rows.append(
            {
                "compound_name": name,
                "mz": float(mz) if pd.notna(mz) else np.nan,
                "q3": float(q3) if pd.notna(q3) else np.nan,
                "native_id": nid,
                "n_concentrations": n,
                "slope": k,
                "intercept": b,
                "r2": r2,
                "r2_5pts_remove2outliers": r2_5,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["mz", "q3"], na_position="last")

    # 合格判定：≥7 点且五点点拟合 R² 可用 → 用 r2_5pts_remove2outliers；否则用全点 r2
    n_pts = out_df["n_concentrations"].astype(float).values
    r2_full = pd.to_numeric(out_df["r2"], errors="coerce").values
    r2_5pt = pd.to_numeric(out_df["r2_5pts_remove2outliers"], errors="coerce").values
    use_5pt_gate = (n_pts >= 7.0) & np.isfinite(r2_5pt)
    passes = np.where(
        use_5pt_gate,
        r2_5pt > 0.995,
        np.isfinite(r2_full) & (r2_full > 0.995),
    )
    gate_label = np.where(use_5pt_gate, "5pt_after_remove2outliers", "linear_all_points")
    out_df["passes_0p995"] = passes.astype(np.int32)
    out_df["r2_gate_rule"] = gate_label

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[OK] %d 个 transition，已写入: %s" % (len(out_df), out_path))

    total_n = int(len(out_df))
    n_gt = int(np.sum(passes))
    n_used_5pt = int(np.sum(use_5pt_gate))
    n_used_linear = total_n - n_used_5pt
    pct_of_all = (100.0 * n_gt / total_n) if total_n > 0 else 0.0
    n_eligible = int(np.sum(np.isfinite(np.where(use_5pt_gate, r2_5pt, r2_full))))
    pct_of_eligible = (100.0 * n_gt / n_eligible) if n_eligible > 0 else 0.0

    summary_lines = [
        "阈值: 见下方判定规则（五点点拟合优先于全点线性）",
        "判定: 浓度点数>=7 且 r2_5pts_remove2outliers 为有效数值时，用其 >0.995；否则用全点 r2 >0.995",
        "大于 0.995（按上规则合格）的数量: %d" % n_gt,
        "总数量（transition 行数）: %d" % total_n,
        "占比（相对总数量）: %.4f%% (%d/%d)" % (pct_of_all, n_gt, total_n),
        "其中采用「5点拟合 R2」判定的条数: %d；采用「全点线性 R2」判定的条数: %d" % (n_used_5pt, n_used_linear),
        "存在可用门限 R2 的条数（非 nan）: %d" % n_eligible,
        "占比（相对可用门限 R2 条数）: %.4f%% (%d/%d)" % (pct_of_eligible, n_gt, n_eligible),
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = out_path.parent / "r2_gt_0.995_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("")
    print("========== R2 > 0.995 汇总（五点点拟合优先）==========")
    print("规则: 点数>=7 且 5点拟合R2 有效 → 用 r2_5pts_remove2outliers>0.995；否则用 全点 r2>0.995")
    print("合格数量: %d" % n_gt)
    print("总数量: %d" % total_n)
    print("占比（相对总数量）: %.4f%%  (%d/%d)" % (pct_of_all, n_gt, total_n))
    print("（5点门限条数: %d；全点门限条数: %d）" % (n_used_5pt, n_used_linear))
    print("已写入: %s" % summary_path)
    print("====================================================")


if __name__ == "__main__":
    main()
