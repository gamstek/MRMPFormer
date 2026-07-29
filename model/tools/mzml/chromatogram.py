# -*- coding: utf-8 -*-
"""
mzML 色谱查看与导出工具。

子命令:
  list    列出所有色谱的 native ID
  show    显示单条色谱的摘要信息
  export  导出单条色谱的全部 (RT, 强度) 数据点

用法:
  python -m <包名>.mzml.chromatogram list --mzml "D:\\...\\file.mzML"
  python -m <包名>.mzml.chromatogram show --mzml "..." --name "阿维菌素-1"
  python -m <包名>.mzml.chromatogram export --mzml "..." --name "阿维菌素-1"
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import decode_native_id, parse_q1_q3_from_native_id


def _load_mzml_experiment(mzml_path: Path):
    """延迟加载 pyopenms 并返回 MSExperiment。"""
    try:
        from pyopenms import MSExperiment, MzMLFile
    except ImportError:
        print("[ERROR] 需要 pyopenms，请在 pipeline 同一 conda/venv 中运行。", file=sys.stderr)
        sys.exit(1)
    exp = MSExperiment()
    MzMLFile().load(str(mzml_path), exp)
    return exp


def _collect_native_ids(exp) -> List[str]:
    """收集所有色谱的 native ID。"""
    ids = []
    for i in range(exp.getNrChromatograms()):
        chrom = exp.getChromatogram(i)
        nid = decode_native_id(chrom.getNativeID())
        ids.append(nid)
    return ids


def _find_chrom_index(native_ids: List[str], name: Optional[str] = None,
                      chrom_index: Optional[int] = None) -> int:
    """按名称或序号查找色谱索引。"""
    if chrom_index is not None:
        idx = int(chrom_index)
        if idx < 0 or idx >= len(native_ids):
            raise IndexError("chrom_index=%d 超出范围 0..%d" % (idx, len(native_ids) - 1))
        return idx
    if not name or not str(name).strip():
        raise ValueError("请指定 --name 或 --chrom_index")
    key = str(name).strip()
    # 精确匹配
    for i, nid in enumerate(native_ids):
        if nid == key:
            return i
    # 包含匹配
    hits = [i for i, nid in enumerate(native_ids) if key in nid]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        examples = [native_ids[i] for i in hits[:10]]
        raise ValueError("名称 '%s' 匹配到 %d 条，请写更完整，例如: %s" % (key, len(hits), examples[:5]))
    raise ValueError("未找到 chromatogram id 含 '%s' 的色谱" % key)


def _chrom_ids_from_bytes(mzml_path: Path, n_chrom: int) -> List[str]:
    """从 mzML 原始字节中提取 chromatogram id（兼容中文编码问题）。"""
    data = mzml_path.read_bytes()
    ids = []
    pos = 0
    while len(ids) < n_chrom:
        i = data.find(b"<chromatogram", pos)
        if i < 0:
            break
        j = data.find(b">", i)
        if j < 0:
            break
        tag = data[i:j].decode("utf-8", errors="replace")
        m = re.search(r'''id\s*=\s*["']([^"']*)["']''', tag)
        if m:
            ids.append(m.group(1))
        else:
            ids.append("")
        pos = j + 1
    while len(ids) < n_chrom:
        ids.append("")
    return ids


def cmd_list(args):
    """列出所有色谱。"""
    mzml_path = Path(args.mzml).expanduser().resolve()
    if not mzml_path.is_file():
        print("[ERROR] 文件不存在: %s" % mzml_path)
        return 1

    exp = _load_mzml_experiment(mzml_path)
    n = exp.getNrChromatograms()
    print("文件: %s" % mzml_path)
    print("色谱总数: %d\n" % n)

    # 尝试从 XML bytes 读取 id（更可靠的中文支持）
    xml_ids = _chrom_ids_from_bytes(mzml_path, n)

    for i in range(n):
        chrom = exp.getChromatogram(i)
        nid = decode_native_id(chrom.getNativeID())
        xml_id = xml_ids[i] if i < len(xml_ids) else ""
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

        display_id = xml_id or nid
        print("[%d] %s" % (i, display_id))
        if q1 or q3:
            print("     Q1=%.4f  Q3=%.2f" % (q1 or 0, q3 or 0))
        print("     points=%d" % len(times))
        if i >= 29 and n > 30:
            print("... 省略剩余 %d 条" % (n - 30))
            break

    return 0


def cmd_show(args):
    """显示单条色谱摘要。"""
    mzml_path = Path(args.mzml).expanduser().resolve()
    if not mzml_path.is_file():
        print("[ERROR] 文件不存在: %s" % mzml_path)
        return 1

    exp = _load_mzml_experiment(mzml_path)
    native_ids = _collect_native_ids(exp)
    idx = _find_chrom_index(native_ids, args.name, args.chrom_index)

    chrom = exp.getChromatogram(idx)
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

    print("序号: %d" % idx)
    print("native_id: %s" % nid)
    print("Q1: %.4f  Q3: %.2f" % (q1 or 0, q3 or 0))
    print("数据点数: %d" % len(times))
    if len(times) > 0:
        rt = times
        if np.max(rt) > 200:
            rt = rt / 60.0
        print("RT 范围: %.4f ~ %.4f min" % (np.min(rt), np.max(rt)))
        print("强度范围: %.2f ~ %.2f" % (np.min(intensities), np.max(intensities)))
        print("强度均值: %.2f" % np.mean(intensities))
    return 0


def cmd_export(args):
    """导出单条色谱数据点。"""
    mzml_path = Path(args.mzml).expanduser().resolve()
    if not mzml_path.is_file():
        print("[ERROR] 文件不存在: %s" % mzml_path)
        return 1

    exp = _load_mzml_experiment(mzml_path)
    native_ids = _collect_native_ids(exp)
    idx = _find_chrom_index(native_ids, args.name, args.chrom_index)

    chrom = exp.getChromatogram(idx)
    nid = decode_native_id(chrom.getNativeID())
    times, intensities = chrom.get_peaks()

    if len(times) == 0:
        print("[WARN] 该色谱无数据点")
        return 1

    # 输出路径
    if args.out:
        out_path = Path(args.out)
    else:
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', nid)[:80] if nid else "chrom_%d" % idx
        out_path = mzml_path.parent / ("%s_%s_points.csv" % (mzml_path.stem, safe_name))

    df = pd.DataFrame({
        "rt_sec": times,
        "intensity": intensities,
    })
    df.to_csv(str(out_path), index=False, encoding=args.encoding)
    print("[OK] 导出 %d 个数据点 → %s" % (len(times), out_path))
    return 0


def main():
    ap = argparse.ArgumentParser(description="mzML 色谱查看与导出")
    sub = ap.add_subparsers(dest="command", help="子命令")

    # list
    p_list = sub.add_parser("list", help="列出所有色谱")
    p_list.add_argument("--mzml", required=True, help=".mzML 文件路径")

    # show
    p_show = sub.add_parser("show", help="显示单条色谱摘要")
    p_show.add_argument("--mzml", required=True, help=".mzML 文件路径")
    p_show.add_argument("--name", default=None, help="chromatogram id（支持部分匹配，需唯一）")
    p_show.add_argument("--chrom_index", type=int, default=None, help="色谱序号，0=TIC")

    # export
    p_export = sub.add_parser("export", help="导出单条色谱数据点")
    p_export.add_argument("--mzml", required=True, help=".mzML 文件路径")
    p_export.add_argument("--name", default=None, help="chromatogram id")
    p_export.add_argument("--chrom_index", type=int, default=None, help="色谱序号")
    p_export.add_argument("--out", default=None, help="输出 CSV 路径")
    p_export.add_argument("--encoding", default="utf-8-sig",
                          choices=("utf-8-sig", "utf-8", "gbk"))

    args = ap.parse_args()
    if args.command == "list":
        return cmd_list(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "export":
        return cmd_export(args)
    else:
        ap.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
