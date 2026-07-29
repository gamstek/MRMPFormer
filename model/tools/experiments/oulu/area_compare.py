# -*- coding: utf-8 -*-
"""
读取欧路标注结果目录下全部 CSV，按第一列「样品名」映射到欧陆 pipeline，做面积对比。

用法:
  python -m <包名>.experiments.oulu.area_compare
  python -m <包名>.experiments.oulu.area_compare --threshold 0.10
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..common import compute_relative_error, load_os_areas
from ..._shared.table_io import normalize_compound_name, parse_area, read_table

# === 默认路径（示例，实际使用需通过参数覆盖） ===
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STD_DIR = _SCRIPT_DIR.parent.parent.parent / "61公司数据" / "欧路标注结果"
DEFAULT_RESULT_DIR = _SCRIPT_DIR.parent.parent.parent / "61公司数据" / "result"

SAMPLE_TO_PIPELINE = {
    1: "欧陆_27",
    2: "欧陆_29",
    3: "欧陆_31",
    4: "欧陆_33",
    5: "欧陆_35",
    6: "欧陆_37",
}

MZ_TOL = 0.15
Q3_TOL = 0.2


def load_pipeline_predictions(result_dir: Path, pipeline_name: str) -> pd.DataFrame:
    """加载 pipeline 的 prediction_refined.csv。"""
    snr_dir = result_dir / "snr_filtered" / pipeline_name
    candidates = [
        snr_dir / "SNR_box_3" / "prediction_refined.csv",
        snr_dir / "SNR_box_5" / "prediction_refined.csv",
    ]
    for p in candidates:
        if p.is_file():
            return read_table(p)
    # 也尝试 SNR_box_0 等
    for d in sorted(snr_dir.glob("SNR_box_*")):
        p = d / "prediction_refined.csv"
        if p.is_file():
            return read_table(p)
    raise FileNotFoundError("未找到欧陆 pipeline prediction: %s" % pipeline_name)


def match_by_mz_q3(
    std_row: pd.Series,
    pred_df: pd.DataFrame,
    mz_tol: float = MZ_TOL,
    q3_tol: float = Q3_TOL,
) -> Optional[pd.Series]:
    """按 (mz, q3) 在 prediction 中匹配标准品行。"""
    std_mz = parse_area(std_row.get("mz") or std_row.get("Q1"))
    std_q3 = parse_area(std_row.get("q3") or std_row.get("Q3"))
    if std_mz is None or std_q3 is None:
        return None

    for _, r in pred_df.iterrows():
        p_mz = parse_area(r.get("mz"))
        p_q3 = parse_area(r.get("q3"))
        if p_mz is None or p_q3 is None:
            continue
        if abs(p_mz - std_mz) <= mz_tol and abs(p_q3 - std_q3) <= q3_tol:
            return r
    return None


def main():
    ap = argparse.ArgumentParser(description="欧陆面积对比")
    ap.add_argument("--std_dir", type=str, default=None, help="标准品目录")
    ap.add_argument("--result_dir", type=str, default=None, help="pipeline result 目录")
    ap.add_argument("--threshold", type=float, default=0.20)
    args = ap.parse_args()

    std_dir = Path(args.std_dir) if args.std_dir else DEFAULT_STD_DIR
    result_dir = Path(args.result_dir) if args.result_dir else DEFAULT_RESULT_DIR

    if not std_dir.is_dir():
        print("[ERROR] 标准品目录不存在: %s" % std_dir)
        sys.exit(1)

    # 收集所有标准品 CSV
    std_files = list(std_dir.glob("*.csv")) + list(std_dir.glob("*.txt"))
    if not std_files:
        print("[ERROR] 标准品目录下无 CSV/TXT 文件")
        sys.exit(1)

    all_results = []
    stats_by_sample = defaultdict(lambda: {"total": 0, "valid": 0, "over": 0})

    for std_file in std_files:
        try:
            df_std = read_table(std_file)
        except Exception as e:
            print("[SKIP] 无法读取 %s: %s" % (std_file.name, e))
            continue

        # 按样品名列分组
        sample_col = df_std.columns[0]
        for sample_no in range(1, 7):
            pipeline_name = SAMPLE_TO_PIPELINE.get(sample_no)
            if not pipeline_name:
                continue

            try:
                df_pred = load_pipeline_predictions(result_dir, pipeline_name)
            except FileNotFoundError:
                continue

            for _, std_row in df_std.iterrows():
                matched = match_by_mz_q3(std_row, df_pred)
                if matched is None:
                    continue

                std_area = parse_area(std_row.get("Area") or std_row.get("峰面积"))
                ai_area = parse_area(matched.get("main_area"))
                err = compute_relative_error(ai_area, std_area) if std_area else None

                stats_by_sample[sample_no]["total"] += 1
                if err is not None:
                    stats_by_sample[sample_no]["valid"] += 1
                    if err > args.threshold:
                        stats_by_sample[sample_no]["over"] += 1

                all_results.append({
                    "sample": sample_no,
                    "pipeline_name": pipeline_name,
                    "compound": str(std_row.get("Component Name", "")),
                    "std_area": std_area,
                    "ai_area": ai_area,
                    "rel_error": err,
                })

    # 输出
    print("\n=== 欧陆面积对比结果 ===")
    print("阈值: %.0f%%" % (args.threshold * 100))
    for s in sorted(stats_by_sample.keys()):
        st = stats_by_sample[s]
        print("  样品%d (%s): 总计%d, 有效%d, 超阈值%d (%.1f%%)"
              % (s, SAMPLE_TO_PIPELINE[s], st["total"], st["valid"],
                 st["over"], st["over"] / st["valid"] * 100 if st["valid"] > 0 else 0))

    if all_results:
        df = pd.DataFrame(all_results)
        out = Path("oulu_area_compare_details.csv")
        df.to_csv(str(out), index=False, encoding="utf-8-sig")
        print("[OK] 详细结果: %s" % out)


if __name__ == "__main__":
    raise SystemExit(main())
