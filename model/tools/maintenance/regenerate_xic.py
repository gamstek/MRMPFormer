# -*- coding: utf-8 -*-
"""
从 chrom JSON 目录重新生成与 testXIC.extract_xic_from_chrom_json_dir 相同的输出，
包括 xic_matrix.npy、feature.csv、roi_windows.csv、ROI jpeg、pipeline_qc_excluded.csv（若有剔除）。

用于 batch 目录下缺失 xic_matrix.npy 时补全。

用法：
  python -m <包名>.maintenance.regenerate_xic ^
    --chrom_json_dir "D:\\...\\20251120-01\\json" ^
    --output_dir "D:\\...\\result\\xic-roi-batch\\json" ^
    --smooth_sigma 0.8 ^
    --min_chrom_points 10 ^
    --min_max_intensity 1000
"""
import argparse
import sys
from pathlib import Path

# 副本外依赖：testXIC 模块
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from testXIC import _load_standard_rt_refs, extract_xic_from_chrom_json_dir


def main():
    ap = argparse.ArgumentParser(
        description="从 chrom JSON 补全 xic_matrix.npy（及 feature / roi_windows / ROI 图）"
    )
    ap.add_argument("--chrom_json_dir", required=True, help="含 *.json 的 chrom 目录")
    ap.add_argument("--output_dir", required=True, help="写入目录")
    ap.add_argument("--smooth_sigma", type=float, default=0.0)
    ap.add_argument("--min_chrom_points", type=int, default=10)
    ap.add_argument("--min_max_intensity", type=float, default=1000.0)
    ap.add_argument("--standard_refs_csv", default=None, help="可选标准品 RT 表")
    args = ap.parse_args()

    chrom = Path(args.chrom_json_dir).resolve()
    out = Path(args.output_dir).resolve()
    if not chrom.is_dir():
        print("[ERROR] chrom_json_dir 不存在: %s" % chrom)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)

    std_key, std_mz = ({}, {})
    if args.standard_refs_csv and str(args.standard_refs_csv).strip():
        p = Path(args.standard_refs_csv).resolve()
        if not p.is_file():
            print("[ERROR] standard_refs_csv 不存在: %s" % p)
            sys.exit(1)
        std_key, std_mz = _load_standard_rt_refs(str(p))

    print("[INFO] chrom: %s" % chrom)
    print("[INFO] output: %s" % out)
    print("[INFO] smooth_sigma=%s min_chrom_points=%s min_max_intensity=%s"
          % (args.smooth_sigma, args.min_chrom_points, args.min_max_intensity))

    extract_xic_from_chrom_json_dir(
        str(chrom), str(out),
        smooth_sigma=args.smooth_sigma,
        min_chrom_points=args.min_chrom_points,
        min_max_intensity=args.min_max_intensity,
        standard_key=std_key,
        standard_mz=std_mz,
    )
    print("[DONE]")


if __name__ == "__main__":
    raise SystemExit(main())
