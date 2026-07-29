# -*- coding: utf-8 -*-
"""
欧路标注标准 vs snr_filtered 主峰(main_area)面积全量对比报告。

对齐规则:
  - 样品: 标准表第一列「样品名」→ snr_filtered/欧陆_XX
  - 物质: 化合物名 → AI (mz, q3) → 取 main_area（每张图仅主峰）
  - 浓度: 记录 分析物浓度 / 计算浓度；统计按样品与全局汇总

用法:
  python -m <包名>.experiments.oulu.full_report
"""
import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..._shared.table_io import normalize_compound_name, parse_area, read_table

# === 默认路径（示例，实际使用需通过参数覆盖） ===
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STD_DIR = _SCRIPT_DIR.parent.parent.parent / "61公司数据" / "欧路标注结果"
DEFAULT_SNR_DIR = _SCRIPT_DIR.parent.parent.parent / "61公司数据" / "result" / "snr_filtered"
DEFAULT_RESULT_DIR = _SCRIPT_DIR.parent.parent.parent / "61公司数据" / "result"
DEFAULT_OUT_DIR = _SCRIPT_DIR.parent.parent.parent / "61公司数据" / "欧陆面积对比报告"

EXCLUSIVE_BAND_LABELS = [
    "(0%,1%]", "(1%,2%]", "(2%,5%]", "(5%,10%]",
    "(10%,50%]", "(50%,100%]", ">100%",
]

SAMPLE_TO_PIPELINE = {
    1: "欧陆_27", 2: "欧陆_29", 3: "欧陆_31",
    4: "欧陆_33", 5: "欧陆_35", 6: "欧陆_37",
}
PIPELINE_TO_SAMPLE = {v: k for k, v in SAMPLE_TO_PIPELINE.items()}

MZ_TOL = 0.15
Q3_TOL = 0.2
ERR_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.50, 1.00)
ERR_LABELS = ("1%", "2%", "5%", "10%", "50%", "100%")


def _load_snr_prediction(snr_dir: Path, pipeline_name: str) -> Optional[pd.DataFrame]:
    """加载 SNR 过滤后的 prediction_refined.csv。"""
    sample_dir = snr_dir / pipeline_name
    if not sample_dir.is_dir():
        return None
    for d in sorted(sample_dir.glob("SNR_box_*")):
        p = d / "prediction_refined.csv"
        if p.is_file():
            return read_table(p)
    return None


def _build_prediction_index(df_pred: pd.DataFrame) -> Dict:
    """构建 (mz, q3) → row 索引。"""
    idx = {}
    for _, r in df_pred.iterrows():
        mz = parse_area(r.get("mz"))
        q3 = parse_area(r.get("q3"))
        if mz is not None and q3 is not None:
            key = (round(mz, 4), round(q3, 2))
            if key not in idx:
                idx[key] = r
    return idx


def _match_prediction(std_row: pd.Series, pred_idx: Dict) -> Optional[pd.Series]:
    """按 (mz, q3) 匹配。"""
    mz = parse_area(std_row.get("mz") or std_row.get("Q1"))
    q3 = parse_area(std_row.get("q3") or std_row.get("Q3"))
    if mz is None or q3 is None:
        return None
    key = (round(mz, 4), round(q3, 2))
    # 精确匹配
    if key in pred_idx:
        return pred_idx[key]
    # 模糊匹配
    for (pmz, pq3), row in pred_idx.items():
        if abs(pmz - mz) <= MZ_TOL and abs(pq3 - q3) <= Q3_TOL:
            return row
    return None


def main():
    ap = argparse.ArgumentParser(description="欧陆全量面积对比报告")
    ap.add_argument("--std_dir", type=str, default=None)
    ap.add_argument("--snr_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    std_dir = Path(args.std_dir) if args.std_dir else DEFAULT_STD_DIR
    snr_dir = Path(args.snr_dir) if args.snr_dir else DEFAULT_SNR_DIR
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not std_dir.is_dir():
        print("[ERROR] 标准品目录不存在: %s" % std_dir)
        sys.exit(1)

    all_rows = []
    stats_global = {"total": 0, "valid": 0}
    band_counts = {b: 0 for b in EXCLUSIVE_BAND_LABELS}

    for std_file in sorted(std_dir.glob("*.csv")) + sorted(std_dir.glob("*.txt")):
        try:
            df_std = read_table(std_file)
        except Exception:
            continue

        sample_col = df_std.columns[0]
        for sample_no, pipeline_name in SAMPLE_TO_PIPELINE.items():
            df_pred = _load_snr_prediction(snr_dir, pipeline_name)
            if df_pred is None:
                continue
            pred_idx = _build_prediction_index(df_pred)

            for _, std_row in df_std.iterrows():
                matched = _match_prediction(std_row, pred_idx)
                if matched is None:
                    continue

                std_area = parse_area(std_row.get("Area") or std_row.get("峰面积"))
                ai_area = parse_area(matched.get("main_area"))

                rel_err = None
                if std_area and ai_area and std_area > 0:
                    rel_err = abs(ai_area - std_area) / std_area

                # 误差分档
                band = ">100%"
                if rel_err is not None:
                    if rel_err <= 0.01:
                        band = "(0%,1%]"
                    elif rel_err <= 0.02:
                        band = "(1%,2%]"
                    elif rel_err <= 0.05:
                        band = "(2%,5%]"
                    elif rel_err <= 0.10:
                        band = "(5%,10%]"
                    elif rel_err <= 0.50:
                        band = "(10%,50%]"
                    elif rel_err <= 1.00:
                        band = "(50%,100%]"

                stats_global["total"] += 1
                if rel_err is not None:
                    stats_global["valid"] += 1
                band_counts[band] = band_counts.get(band, 0) + 1

                all_rows.append({
                    "sample": sample_no,
                    "pipeline_name": pipeline_name,
                    "compound": str(std_row.get("Component Name", "")),
                    "std_area": std_area,
                    "ai_main_area": ai_area,
                    "rel_error": rel_err,
                    "error_band": band,
                })

    # 输出报告
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(str(out_dir / "oulu_full_comparison.csv"), index=False, encoding="utf-8-sig")

    print("=== 欧陆全量面积对比报告 ===")
    print("总对比条目: %d, 有效: %d" % (stats_global["total"], stats_global["valid"]))
    print("误差分布:")
    for band in EXCLUSIVE_BAND_LABELS:
        print("  %s: %d" % (band, band_counts.get(band, 0)))
    print("[OK] 报告目录: %s" % out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
