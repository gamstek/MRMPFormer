# -*- coding: utf-8 -*-
"""
检查 pipeline_batch_mzml 中与 SNR / chrom 对齐相关的风险（不跑模型，只读盘）：

1) 同 (Q1,Q3 四位+二位) 多通道：与 mzml_box_outside_snr_pipeline._chrom_lookup_by_mzq3 相同键会覆盖，仅保留最后一条。
2) Q1 或 Q3 缺失：无法走 dict 匹配，prediction 只能靠 compound_name 对齐 chroms 顺序。
3) roi_windows.csv 的 image 与 batch_predictions/.../prediction.csv 的 image 是否完全一致。

不依赖 testXIC/pyopenms：chrom JSON 解析逻辑与 testXIC.load_raw_chroms_from_json_dir 保持一致。

用法（与 main.py pipeline 目录结构一致）：
  python -m <包名>.diagnostics.check_chrom_snr_alignment ^
    --batch_dir "D:\\...\\json" ^
    --result_dir "D:\\...\\result"

仅查 JSON、尚未跑 result 时可省略 --result_dir。
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .._shared.artifacts import read_csv_safe
from .._shared.chrom_json import load_chrom_json_directory


def _snr_dict_key(q1, q3):
    """与 mzml_box_outside_snr_pipeline._chrom_lookup_by_mzq3 一致。"""
    if q1 is None or q3 is None:
        return None
    return (round(float(q1), 4), round(float(q3), 2))


def resolve_chrom_json_sample_dirs(batch_dir):
    """batch_dir 下每个含 *.json 的子文件夹 → 一个样品；根目录直铺 *.json 则整目录为一样品。"""
    batch_path = Path(batch_dir).resolve()
    if not batch_path.is_dir():
        raise FileNotFoundError("[ERROR] batch_dir 不存在: %s" % batch_path)
    subs = sorted(p for p in batch_path.iterdir() if p.is_dir())
    pairs = [(p.name, p) for p in subs if list(p.glob("*.json"))]
    if pairs:
        return pairs
    if list(batch_path.glob("*.json")):
        return [(batch_path.name, batch_path)]
    return []


def check_sample(
    stem: str, chrom_dir: Path, result_dir: Optional[Path], integration: str
) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("[样品] stem=%s  chrom_dir=%s" % (stem, chrom_dir))
    print("=" * 72)

    issues = {
        "mzq3_collision": False,
        "mz_or_q3_missing": False,
        "image_mismatch": False,
        "compound_name_bad": False,
        "feature_row_mismatch": False,
        "missing_image_column": False,
    }

    records = load_chrom_json_directory(str(chrom_dir))
    n = len(records)
    print("[INFO] load_chroms_from_json_dir: %d 条" % n)
    if n == 0:
        return issues

    by_key = {}
    collisions = []
    key_positions = defaultdict(list)
    for i, rec in enumerate(records):
        k = _snr_dict_key(rec.get("q1"), rec.get("q3"))
        if k is None:
            continue
        key_positions[k].append(i)
        by_key[k] = i
    for k, idxs in key_positions.items():
        if len(idxs) > 1:
            collisions.append((k, idxs, by_key[k]))

    if collisions:
        issues["mzq3_collision"] = True
        print("[WARN] SNR (mz,q3) 查找表键冲突：同键多条 chrom，运行时只会匹配到最后一条 source_index")
        for k, idxs, winner in collisions[:20]:
            print("       键 %s  行号(0-based)=%s  将保留索引=%d" % (k, idxs, winner))
        if len(collisions) > 20:
            print("       ... 另有 %d 组冲突未列出" % (len(collisions) - 20))
    else:
        print("[OK] 无 (Q1,Q3) 全有效且四舍五入后重复的 dict 冲突")

    both_ok = sum(1 for r in records if r.get("q1") is not None and r.get("q3") is not None)
    q1_na = sum(1 for r in records if r.get("q1") is None)
    q3_na = sum(1 for r in records if r.get("q3") is None)
    print(
        "[INFO] Q1+Q3 均有效: %d / %d；Q1 缺失: %d；Q3 缺失: %d"
        % (both_ok, n, q1_na, q3_na)
    )
    if both_ok < n:
        issues["mz_or_q3_missing"] = True
        print(
            "[WARN] 存在 Q1 或 Q3 缺失：SNR 阶段 _match_chrom 不能用 (mz,q3) dict，"
            "需 prediction.csv 的 compound_name 与 chrom 顺序一致（1..N）"
        )

    if result_dir is None:
        return issues

    roi_csv = result_dir / "xic-roi-batch" / stem / "roi_windows.csv"
    pred_name = "prediction.csv" if integration == "linear" else "prediction_%s.csv" % integration
    pred_csv = result_dir / "batch_predictions" / stem / pred_name

    if not roi_csv.is_file():
        print("[SKIP] 无 ROI 表: %s" % roi_csv)
    if not pred_csv.is_file():
        print("[SKIP] 无 prediction: %s" % pred_csv)
    if not roi_csv.is_file() or not pred_csv.is_file():
        return issues

    try:
        df_roi = read_csv_safe(roi_csv)
    except Exception as e:
        print("[ERROR] 无法读取 ROI CSV %s: %s" % (roi_csv, e))
        return issues
    try:
        df_pr = read_csv_safe(pred_csv)
    except Exception as e:
        print("[ERROR] 无法读取 prediction CSV %s: %s" % (pred_csv, e))
        return issues

    if "image" not in df_roi.columns or "image" not in df_pr.columns:
        issues["missing_image_column"] = True
        print("[ERROR] roi 或 prediction 缺少 image 列")
        return issues

    n_pr, n_roi = len(df_pr), len(df_roi)
    if n_pr != n or n_roi != n:
        print(
            "[WARN] 行数: chrom JSON=%d, prediction=%d, roi_windows=%d；"
            "仅以 compound_name / image 对齐时请注意是否漏检或多行"
            % (n, n_pr, n_roi)
        )

    set_roi = set(str(x).strip() for x in df_roi["image"].dropna().unique())
    set_pr = set(str(x).strip() for x in df_pr["image"].dropna().unique())
    only_pred = sorted(set_pr - set_roi)
    only_roi = sorted(set_roi - set_pr)
    if only_pred or only_roi:
        issues["image_mismatch"] = True
        if only_pred:
            print("[WARN] 仅在 prediction 中出现的 image: %s" % only_pred[:30])
        if only_roi:
            print("[WARN] 仅在 roi_windows 中出现的 image: %s" % only_roi[:30])
    else:
        print("[OK] image 列完全一致（prediction ↔ roi_windows）")
    return issues


def main():
    ap = argparse.ArgumentParser(description="检查 chrom JSON ↔ SNR ↔ prediction 对齐风险")
    ap.add_argument("--batch_dir", required=True, help="chrom JSON 目录（含子文件夹则为多样品）")
    ap.add_argument("--result_dir", default=None, help="result 根目录（含 xic-roi-batch、batch_predictions）")
    ap.add_argument("--integration", default="linear", help="prediction 文件后缀：linear → prediction.csv")
    args = ap.parse_args()

    pairs = resolve_chrom_json_sample_dirs(args.batch_dir)
    if not pairs:
        print("[ERROR] 未找到任何含 *.json 的样品目录")
        sys.exit(1)

    result_root = Path(args.result_dir) if args.result_dir else None
    all_issues = []
    for stem, d in pairs:
        issues = check_sample(stem, d, result_root, args.integration)
        all_issues.append((stem, issues))

    print("\n" + "=" * 72)
    print("[SUMMARY]")
    n_any = sum(1 for _, iss in all_issues if any(iss.values()))
    print("样品总数 %d，存在问题的 %d" % (len(all_issues), n_any))
    if n_any > 0:
        print("问题详情见上。")
    return 0 if n_any == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
