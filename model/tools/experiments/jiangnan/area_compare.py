# -*- coding: utf-8 -*-
"""
对比江南大学农残 CSV 与人工加标 OS.txt 的 Area，统计相对误差。

相对误差 = |Area_CSV - Area_OS| / Area_OS

用法:
  python -m <包名>.experiments.jiangnan.area_compare
  python -m <包名>.experiments.jiangnan.area_compare --all_pairs
  python -m <包名>.experiments.jiangnan.area_compare --threshold 0.10 --output_dir "..."
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..common import compute_relative_error, load_csv_areas, load_os_areas, pair_area_comparison
from ..._shared.table_io import normalize_compound_name, parse_area, read_table

# === 默认路径（示例，实际使用需通过参数覆盖） ===
# 原路径: D:\pycharm\QuanFormer-main\江南大学仪器对比原始数据\解析\农残\标注
DEFAULT_BASE = Path(__file__).resolve().parent.parent.parent.parent / "江南大学仪器对比原始数据" / "解析" / "农残" / "标注"
DEFAULT_OS = DEFAULT_BASE / "江南大学农残数据人工加标os.txt"
DEFAULT_CSV_10PPB = DEFAULT_BASE / "江南大学农残10ppb.csv"
DEFAULT_OS_SAMPLE_10PPB = "jizhi_10ppb-1"

ALL_PAIRS = [
    ("江南大学农残10ppb.csv", "jizhi_10ppb-1"),
    ("江南大学农残20ppb.csv", "jizhi_20ppb-1"),
    ("江南大学农残50ppb.csv", "jizhi_50ppb-1"),
]


def compare_one(
    csv_path: Path,
    os_sample: str,
    os_path: Path,
    threshold: float = 0.20,
) -> Tuple[List[Dict], Dict]:
    """对比单个 CSV 与 OS 样本。返回 (details, summary)。"""
    csv_areas = load_csv_areas(csv_path)
    os_areas = load_os_areas(os_path, os_sample)

    results = pair_area_comparison(csv_areas, os_areas)

    n_total = len(results)
    n_valid = sum(1 for r in results if r["rel_error"] is not None)
    n_over = sum(1 for r in results if r["rel_error"] is not None and r["rel_error"] > threshold)

    summary = {
        "csv_file": str(csv_path.name),
        "os_sample": os_sample,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_over_threshold": n_over,
        "pct_over": n_over / n_valid * 100 if n_valid > 0 else 0,
        "threshold": threshold,
    }
    return results, summary


def main():
    ap = argparse.ArgumentParser(description="江南大学 CSV vs OS 面积对比")
    ap.add_argument("--all_pairs", action="store_true", help="对比所有三组浓度对")
    ap.add_argument("--threshold", type=float, default=0.20, help="相对误差阈值")
    ap.add_argument("--output_dir", type=str, default=None, help="输出目录")
    ap.add_argument("--csv", type=str, default=None, help="指定 CSV 文件")
    ap.add_argument("--os", type=str, default=None, help="指定 OS 文件")
    ap.add_argument("--os_sample", type=str, default=None, help="OS 中样品名")
    args = ap.parse_args()

    os_path = Path(args.os) if args.os else DEFAULT_OS
    if not os_path.is_file():
        print("[ERROR] OS 文件不存在: %s（请通过 --os 指定）" % os_path)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path("jiangnan_area_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_pairs:
        pairs = [(DEFAULT_BASE / csv_name, sample) for csv_name, sample in ALL_PAIRS]
    elif args.csv:
        pairs = [(Path(args.csv), args.os_sample or "jizhi_10ppb-1")]
    else:
        pairs = [(DEFAULT_CSV_10PPB, DEFAULT_OS_SAMPLE_10PPB)]

    all_rows = []
    for csv_path, sample in pairs:
        if not csv_path.is_file():
            print("[SKIP] CSV 不存在: %s" % csv_path)
            continue
        print("\n=== %s vs %s ===" % (csv_path.name, sample))
        details, summary = compare_one(csv_path, sample, os_path, args.threshold)

        print("  总化合物: %d, 有效对比: %d, 超阈值(>%.0f%%): %d (%.1f%%)"
              % (summary["n_total"], summary["n_valid"],
                 args.threshold * 100, summary["n_over_threshold"], summary["pct_over"]))

        for r in details:
            r["pair"] = "%s_vs_%s" % (csv_path.stem, sample)
            all_rows.append(r)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_csv = out_dir / "area_comparison_details.csv"
        df.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
        print("\n[OK] 详细结果: %s" % out_csv)

    print("[DONE]")


if __name__ == "__main__":
    raise SystemExit(main())
