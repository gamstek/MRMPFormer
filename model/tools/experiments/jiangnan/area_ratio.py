# -*- coding: utf-8 -*-
"""
江南大学农残 两浓度 CSV 与 OS 面积比对比。

基础量（每个化合物）:
  R_csv = Area_low_CSV / Area_high_CSV
  R_os  = Area_low_OS  / Area_high_OS

指标:
  metric_diff = |R_csv - R_os|

用法:
  python -m <包名>.experiments.jiangnan.area_ratio --pair 10_20 --plot
  python -m <包名>.experiments.jiangnan.area_ratio --pair 20_50
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..common import load_csv_areas, load_os_areas
from ..._shared.table_io import normalize_compound_name, parse_area, read_table

# === 默认路径（示例，实际使用需通过参数覆盖） ===
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = _SCRIPT_DIR.parent.parent.parent / "江南大学仪器对比原始数据" / "解析" / "农残" / "标注"
DEFAULT_OS = DEFAULT_BASE / "江南大学农残数据人工加标os.txt"

PAIR_PRESETS = {
    "10_20": {
        "csv_low": DEFAULT_BASE / "江南大学农残10ppb.csv",
        "csv_high": DEFAULT_BASE / "江南大学农残20ppb.csv",
        "os_low": "jizhi_10ppb-1",
        "os_high": "jizhi_20ppb-1",
        "label_low": "10ppb",
        "label_high": "20ppb",
        "report_dir": "area_ratio_10_20_report",
    },
    "20_50": {
        "csv_low": DEFAULT_BASE / "江南大学农残20ppb.csv",
        "csv_high": DEFAULT_BASE / "江南大学农残50ppb.csv",
        "os_low": "jizhi_20ppb-1",
        "os_high": "jizhi_50ppb-1",
        "label_low": "20ppb",
        "label_high": "50ppb",
        "report_dir": "area_ratio_20_50_report",
    },
}

DIFF_SPEC = {
    "column": "metric_diff",
    "label": "差值 |R_csv - R_os|",
    "bin_edges": (0.0, 0.05, 0.1, 0.15, 0.2),
    "bin_labels": ("[0, 0.05)", "[0.05, 0.1)", "[0.1, 0.15)", "[0.15, 0.2]"),
    "tail_label": "> 0.2",
    "hist_xmax_default": 1.0,
    "hist_xmin_default": 0.0,
}


def compute_ratios(
    csv_low_path: Path,
    csv_high_path: Path,
    os_path: Path,
    os_low_sample: str,
    os_high_sample: str,
) -> pd.DataFrame:
    """计算两浓度面积比及差值。返回 DataFrame。"""
    csv_low = load_csv_areas(csv_low_path)
    csv_high = load_csv_areas(csv_high_path)
    os_low = load_os_areas(os_path, os_low_sample)
    os_high = load_os_areas(os_path, os_high_sample)

    all_compounds = set(csv_low.keys()) | set(csv_high.keys()) | set(os_low.keys()) | set(os_high.keys())
    rows = []
    for name in sorted(all_compounds):
        cl = csv_low.get(name)
        ch = csv_high.get(name)
        ol = os_low.get(name)
        oh = os_high.get(name)

        r_csv = (cl / ch) if (cl and ch and ch > 0) else None
        r_os = (ol / oh) if (ol and oh and oh > 0) else None
        diff = abs(r_csv - r_os) if (r_csv is not None and r_os is not None) else None

        rows.append({
            "compound": name,
            "csv_low": cl,
            "csv_high": ch,
            "os_low": ol,
            "os_high": oh,
            "R_csv": r_csv,
            "R_os": r_os,
            "metric_diff": diff,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="江南大学面积比对比")
    ap.add_argument("--pair", default="10_20", choices=("10_20", "20_50"), help="浓度对")
    ap.add_argument("--plot", action="store_true", help="生成直方图")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--csv_low", type=str, default=None, help="覆盖低浓度 CSV")
    ap.add_argument("--csv_high", type=str, default=None, help="覆盖高浓度 CSV")
    ap.add_argument("--os", type=str, default=None, help="覆盖 OS 文件")
    args = ap.parse_args()

    preset = PAIR_PRESETS[args.pair]

    csv_low = Path(args.csv_low) if args.csv_low else preset["csv_low"]
    csv_high = Path(args.csv_high) if args.csv_high else preset["csv_high"]
    os_path = Path(args.os) if args.os else DEFAULT_OS

    if not csv_low.is_file():
        print("[ERROR] CSV low 不存在: %s" % csv_low)
        sys.exit(1)
    if not csv_high.is_file():
        print("[ERROR] CSV high 不存在: %s" % csv_high)
        sys.exit(1)
    if not os_path.is_file():
        print("[ERROR] OS 文件不存在: %s" % os_path)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path(preset["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = compute_ratios(csv_low, csv_high, os_path, preset["os_low"], preset["os_high"])

    valid = df["metric_diff"].notna()
    print("[INFO] 总化合物: %d, 有效比率对比: %d" % (len(df), valid.sum()))

    if valid.sum() > 0:
        diffs = df.loc[valid, "metric_diff"]
        print("[STATS] mean=%.4f median=%.4f stdev=%.4f min=%.4f max=%.4f"
              % (diffs.mean(), diffs.median(), diffs.std(), diffs.min(), diffs.max()))

        # 分箱统计
        edges = DIFF_SPEC["bin_edges"]
        for i in range(len(edges) - 1):
            cnt = ((diffs >= edges[i]) & (diffs < edges[i+1])).sum()
            print("  %s: %d (%.1f%%)" % (DIFF_SPEC["bin_labels"][i], cnt, cnt / valid.sum() * 100))
        cnt_tail = (diffs >= edges[-1]).sum()
        print("  %s: %d (%.1f%%)" % (DIFF_SPEC["tail_label"], cnt_tail, cnt_tail / valid.sum() * 100))

    out_csv = out_dir / "area_ratio_details.csv"
    df.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
    print("\n[OK] 详细结果: %s" % out_csv)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(diffs.dropna(), bins=30, edgecolor="black", alpha=0.7)
            ax.set_xlabel(DIFF_SPEC["label"])
            ax.set_ylabel("Count")
            ax.set_title("Area Ratio Difference: %s vs %s" % (preset["label_low"], preset["label_high"]))
            fig.tight_layout()
            fig.savefig(str(out_dir / "area_ratio_hist.png"), dpi=150)
            plt.close(fig)
            print("[OK] 直方图: %s" % (out_dir / "area_ratio_hist.png"))
        except ImportError:
            print("[WARN] matplotlib 不可用，跳过绘图")

    print("[DONE]")


if __name__ == "__main__":
    raise SystemExit(main())
