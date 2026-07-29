# -*- coding: utf-8 -*-
"""
对比两浓度 CSV + 两 OS 样本中，哪些化合物名只出现在某一侧（only）。

用法:
  python -m <包名>.experiments.jiangnan.compound_presence --pair 20_50
  python -m <包名>.experiments.jiangnan.compound_presence --pair 10_20
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..common import load_csv_areas, load_os_areas
from .area_ratio import DEFAULT_BASE, DEFAULT_OS, PAIR_PRESETS


def _display_name(maps: List[Dict], key: str) -> str:
    for m in maps:
        if key in m and m[key].get("compound_name"):
            return m[key]["compound_name"]
    return key


def _classify_only(in_cl: bool, in_ch: bool, in_ol: bool, in_oh: bool):
    """返回 only 类型标签。"""
    flags = {
        "csv_low": in_cl,
        "csv_high": in_ch,
        "os_low": in_ol,
        "os_high": in_oh,
    }
    present = [k for k, v in flags.items() if v]
    if len(present) == 0:
        return "none", present
    if len(present) == 1:
        return "only_" + present[0], present
    tags = []
    if in_cl and not in_ch and not in_ol and not in_oh:
        tags.append("only_csv_low")
    if in_ch and not in_cl and not in_ol and not in_oh:
        tags.append("only_csv_high")
    if in_ol and not in_cl and not in_ch and not in_oh:
        tags.append("only_os_low")
    if in_oh and not in_cl and not in_ch and not in_ol:
        tags.append("only_os_high")
    if in_cl and in_ch and not in_ol and not in_oh:
        tags.append("only_csv_both")
    if in_ol and in_oh and not in_cl and not in_ch:
        tags.append("only_os_both")
    if not tags:
        tags.append("present_multiple")
    return ";".join(tags), present


def main():
    ap = argparse.ArgumentParser(description="江南大学化合物存在性对比")
    ap.add_argument("--pair", default="10_20", choices=("10_20", "20_50"))
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    preset = PAIR_PRESETS[args.pair]
    csv_low = preset["csv_low"]
    csv_high = preset["csv_high"]
    os_path = DEFAULT_OS

    if not csv_low.is_file():
        print("[ERROR] CSV low 不存在: %s" % csv_low)
        sys.exit(1)
    if not csv_high.is_file():
        print("[ERROR] CSV high 不存在: %s" % csv_high)
        sys.exit(1)
    if not os_path.is_file():
        print("[ERROR] OS 不存在: %s" % os_path)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path("compound_only_report") / args.pair
    out_dir.mkdir(parents=True, exist_ok=True)

    areas_cl = load_csv_areas(csv_low)
    areas_ch = load_csv_areas(csv_high)
    areas_ol = load_os_areas(os_path, preset["os_low"])
    areas_oh = load_os_areas(os_path, preset["os_high"])

    all_names = sorted(set(areas_cl.keys()) | set(areas_ch.keys()) | set(areas_ol.keys()) | set(areas_oh.keys()))

    rows = []
    only_groups: Dict[str, List[str]] = {}
    for name in all_names:
        in_cl = name in areas_cl
        in_ch = name in areas_ch
        in_ol = name in areas_ol
        in_oh = name in areas_oh
        label, present = _classify_only(in_cl, in_ch, in_ol, in_oh)
        rows.append({
            "compound": name,
            "in_csv_low": in_cl,
            "in_csv_high": in_ch,
            "in_os_low": in_ol,
            "in_os_high": in_oh,
            "only_type": label,
        })
        if label.startswith("only_"):
            only_groups.setdefault(label, []).append(name)

    df = pd.DataFrame(rows)
    df.to_csv(str(out_dir / "compound_presence_all.csv"), index=False, encoding="utf-8-sig")

    # 汇总
    summary_rows = []
    for label in sorted(only_groups.keys()):
        summary_rows.append({"only_type": label, "count": len(only_groups[label])})
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(str(out_dir / "compound_only_summary.csv"), index=False, encoding="utf-8-sig")

    # 各来源独占名单
    for label, names in only_groups.items():
        pd.DataFrame({"compound": names}).to_csv(
            str(out_dir / ("%s.csv" % label)), index=False, encoding="utf-8-sig"
        )

    print("[INFO] 总化合物: %d" % len(all_names))
    for label in sorted(only_groups.keys()):
        print("  %s: %d" % (label, len(only_groups[label])))
    print("[OK] 报告目录: %s" % out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
