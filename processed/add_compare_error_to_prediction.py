"""
将 compare_detail.csv 中的相对误差加入 prediction.csv 对应行。

按 image 匹配：compare_detail 每行对应一个 ROI 图像，prediction 中同 image 的行获得 relative_error_percent。

用法:
  python add_compare_error_to_prediction.py --dir "D:\pycharm\QuanFormer-main\results\batch_predictions\20240330_PES_CMA_加标-STD_20ppb_青菜"
  python add_compare_error_to_prediction.py --dir <目录路径> [--prediction_file prediction.csv] [--output prediction_with_error.csv]
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="将 compare_detail.csv 的相对误差加入 prediction.csv"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=r"D:\pycharm\QuanFormer-main\results\batch_predictions\20240330_PES_CMA_加标-STD_20ppb_青菜",
        help="目录路径，内含 compare_detail.csv 与 prediction.csv",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        default="prediction.csv",
        help="prediction 文件名，默认 prediction.csv",
    )
    parser.add_argument(
        "--compare_file",
        type=str,
        default="compare_detail.csv",
        help="compare 明细文件名，默认 compare_detail.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径，默认覆盖原 prediction.csv；可指定新文件名",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir).resolve()
    compare_path = base_dir / args.compare_file
    pred_path = base_dir / args.prediction_file

    if not compare_path.exists():
        print(f"[ERROR] 未找到: {compare_path}")
        return
    if not pred_path.exists():
        print(f"[ERROR] 未找到: {pred_path}")
        return

    compare_df = pd.read_csv(compare_path)
    pred_df = pd.read_csv(pred_path)

    if "relative_error_percent" not in compare_df.columns:
        print(f"[ERROR] compare_detail 中无 relative_error_percent 列")
        return

    # 按 image 提取相对误差（及 manual_area 若存在）
    err_cols = ["image", "relative_error_percent"]
    if "manual_area" in compare_df.columns:
        err_cols.append("manual_area")
    err_df = compare_df[err_cols].drop_duplicates(subset="image", keep="first")
    merged = pred_df.merge(err_df, on="image", how="left")

    out_path = Path(args.output) if args.output else pred_path
    try:
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 已保存: {out_path}")
    except PermissionError:
        fallback = base_dir / "prediction_with_error.csv"
        merged.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[WARN] 无法写入 {out_path}，已另存为: {fallback}")

    n_match = merged["relative_error_percent"].notna().sum()
    print(f"[INFO] 匹配行数: {n_match} / {len(pred_df)}")


if __name__ == "__main__":
    main()
