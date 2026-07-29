# -*- coding: utf-8 -*-
"""
将 mzML 中的可读内容导出为 CSV。

说明：
  mzML 本质是 XML（标签嵌套），不是「一行一条字段」的文本表。
  本脚本按色谱 chromatogram / 谱图 spectrum（若有）拆开，导出：
    1) 摘要表：每条色谱一行（sample_name、native id、Q1/Q3、点数、RT 范围、强度范围等）
    2) 可选：每个数据点一行（chrom_index + rt_sec + intensity），数据量大时用 --points

用法：
  python -m <包名>.mzml.inspect --mzml "D:\\...\\file.mzML" --out_dir "D:\\...\\out"
  python -m <包名>.mzml.inspect --mzml "..." --xlsx
  python -m <包名>.mzml.inspect --mzml "..." --points
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import decode_native_id, parse_q1_q3_from_native_id

# PSI mzML 默认命名空间
_MZML_NS = "http://psi.hupo.org/ms/mzml"
_CHROM_TAG = "{%s}chromatogram" % _MZML_NS


def _chromatogram_ids_from_xml(path: Path) -> List[Dict[str, Any]]:
    """
    收集 <chromatogram id="…"/>；若均有 index 属性则按 index 排序对齐 pyopenms，否则按文档顺序。
    """
    rows = []
    try:
        for event, elem in ET.iterparse(str(path), events=("start",)):
            tag = elem.tag
            if tag.endswith("chromatogram") or tag == _CHROM_TAG:
                cid = elem.get("id", "")
                idx = elem.get("index")
                rows.append({"id": cid, "index": int(idx) if idx is not None else None})
            elem.clear()
    except ET.ParseError as e:
        print("[WARN] XML 解析出错: %s" % e)
    if rows and all(r["index"] is not None for r in rows):
        rows.sort(key=lambda r: r["index"])
    return rows


def _load_mzml(mzml_path: Path):
    """延迟加载 pyopenms。"""
    try:
        from pyopenms import MSExperiment, MzMLFile
    except ImportError:
        print("[ERROR] 需要 pyopenms，请在 pipeline 同一 conda/venv 中运行。", file=sys.stderr)
        sys.exit(1)

    exp = MSExperiment()
    MzMLFile().load(str(mzml_path), exp)
    return exp


def inspect_mzml(
    mzml_path: Path,
    out_dir: Optional[Path] = None,
    export_points: bool = False,
    to_xlsx: bool = False,
) -> Dict[str, Any]:
    """检查 mzML 文件并返回摘要。"""
    mzml_path = mzml_path.resolve()
    if not mzml_path.is_file():
        raise FileNotFoundError("mzML 文件不存在: %s" % mzml_path)

    exp = _load_mzml(mzml_path)

    # 样本信息
    info: Dict[str, Any] = {
        "file": str(mzml_path),
        "n_chromatograms": exp.getNrChromatograms(),
        "n_spectra": exp.getNrSpectra(),
    }

    # 色谱摘要
    chrom_rows = []
    for i in range(exp.getNrChromatograms()):
        chrom = exp.getChromatogram(i)
        nid = decode_native_id(chrom.getNativeID())
        times, intensities = chrom.get_peaks()
        q1, q3 = None, None
        try:
            q1 = float(chrom.getPrecursor().getMZ())
        except Exception:
            pass
        try:
            q3 = float(chrom.getProduct().getMZ())
        except Exception:
            pass
        if q1 is None or q3 is None:
            q1p, q3p = parse_q1_q3_from_native_id(nid)
            if q1 is None:
                q1 = q1p
            if q3 is None:
                q3 = q3p

        rt_lo = float(np.min(times)) if len(times) > 0 else float("nan")
        rt_hi = float(np.max(times)) if len(times) > 0 else float("nan")
        int_max = float(np.max(intensities)) if len(intensities) > 0 else float("nan")
        int_avg = float(np.mean(intensities)) if len(intensities) > 0 else float("nan")

        chrom_rows.append({
            "chrom_index": i,
            "native_id": nid,
            "q1": q1,
            "q3": q3,
            "n_points": len(times),
            "rt_min_min": rt_lo / 60.0 if rt_lo > 200 else rt_lo,
            "rt_max_min": rt_hi / 60.0 if rt_hi > 200 else rt_hi,
            "intensity_max": int_max,
            "intensity_mean": int_avg,
        })

    df_chrom = pd.DataFrame(chrom_rows)
    info["chrom_summary"] = df_chrom

    # 谱图摘要
    spec_rows = []
    for i in range(exp.getNrSpectra()):
        spec = exp.getSpectrum(i)
        mz_arr, int_arr = spec.get_peaks()
        spec_rows.append({
            "spec_index": i,
            "ms_level": spec.getMSLevel(),
            "n_peaks": len(mz_arr),
            "rt_sec": spec.getRT(),
        })
    df_spec = pd.DataFrame(spec_rows) if spec_rows else pd.DataFrame()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = mzml_path.stem

        # 色谱摘要
        chrom_csv = out_dir / ("%s_chrom_summary.csv" % stem)
        df_chrom.to_csv(str(chrom_csv), index=False, encoding="utf-8-sig")
        print("[OK] 色谱摘要: %s" % chrom_csv)

        # 谱图摘要
        if not df_spec.empty:
            spec_csv = out_dir / ("%s_spectrum_summary.csv" % stem)
            df_spec.to_csv(str(spec_csv), index=False, encoding="utf-8-sig")
            print("[OK] 谱图摘要: %s" % spec_csv)

        if to_xlsx:
            try:
                xlsx_path = out_dir / ("%s_inspect.xlsx" % stem)
                with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
                    df_chrom.to_excel(writer, sheet_name="chromatograms", index=False)
                    if not df_spec.empty:
                        df_spec.to_excel(writer, sheet_name="spectra", index=False)
                print("[OK] Excel: %s" % xlsx_path)
            except Exception as e:
                print("[WARN] Excel 导出失败: %s" % e)

        # 数据点导出
        if export_points:
            pts_dir = out_dir / ("%s_points" % stem)
            pts_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(exp.getNrChromatograms(), 200)):
                chrom = exp.getChromatogram(i)
                times, intensities = chrom.get_peaks()
                if len(times) == 0:
                    continue
                nid = decode_native_id(chrom.getNativeID())
                safe_name = re.sub(r'[<>:"/\\|?*]', '_', nid)[:100] if nid else "chrom_%d" % i
                df_pts = pd.DataFrame({
                    "rt_sec": times,
                    "intensity": intensities,
                })
                df_pts.to_csv(str(pts_dir / ("%s.csv" % safe_name)), index=False, encoding="utf-8-sig")
            print("[OK] 数据点导出: %s" % pts_dir)

    return info


def main():
    ap = argparse.ArgumentParser(description="mzML 文件检查与 CSV 导出")
    ap.add_argument("--mzml", required=True, help="输入 .mzML 路径")
    ap.add_argument("--out_dir", default=None, help="输出目录")
    ap.add_argument("--points", action="store_true", help="导出每条色谱的数据点 CSV")
    ap.add_argument("--xlsx", action="store_true", help="额外输出 Excel")
    args = ap.parse_args()

    mzml = Path(args.mzml)
    out = Path(args.out_dir) if args.out_dir else mzml.parent / (mzml.stem + "_inspect")

    info = inspect_mzml(mzml, out, args.points, args.xlsx)

    # 终端摘要
    print("\n=== mzML 摘要 ===")
    print("文件: %s" % mzml)
    print("色谱数: %d" % info["n_chromatograms"])
    print("谱图数: %d" % info["n_spectra"])
    if not info["chrom_summary"].empty:
        print("色谱 RT 范围: %.2f ~ %.2f min"
              % (info["chrom_summary"]["rt_min_min"].min(), info["chrom_summary"]["rt_max_min"].max()))


if __name__ == "__main__":
    raise SystemExit(main())
