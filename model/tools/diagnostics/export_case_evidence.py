# -*- coding: utf-8 -*-
"""
给定 refined_plots 下的 *_refined.png，自动定位「原始数据」并出图：

  1) prediction_refined.csv 中对应行（image / compound_name / 主峰 RT 等）
  2) xic_matrix.npy（数值型原始 XIC，与 post_newtest 一致）
  3) roi_windows.csv 中该图的 RT 窗
  4) 若磁盘上仍存在 CNN 输入 ROI 图（同名 .jpeg/.jpg），复制到 refined_plots 并写报告

输出（默认同 refined_plots 目录）：
  - <stem>_source_report.txt   数据源说明
  - <stem>_xic_roi.png         ROI 时间窗内原始 XIC（无平滑）
  - <stem>_xic_full.png        整条公共 RT 轴上的 XIC
  - <stem>_source_roi.jpeg     若找到 CNN 输入图则复制（可选）

示例：
  python -m <包名>.diagnostics.export_case_evidence
  python -m <包名>.diagnostics.export_case_evidence --refined_png "D:\\其它\\xxx_refined.png"
"""
import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from .._shared.artifacts import (
    find_row_for_refined_png,
    image_to_row_index,
    load_roi_map,
    locate_roi_csv,
    locate_xic_npy,
    read_csv_safe,
    refined_core_stem,
    resolve_rt_window,
)

# 延迟导入绘图相关（避免 --help 时加载 matplotlib）
_plot_func = None


def _get_plot_func():
    global _plot_func
    if _plot_func is None:
        from ..visualization.plot_refined_xic import plot_xic_from_refined_png as _f
        _plot_func = _f
    return _plot_func


def find_cnn_roi_jpeg(snr_dir: Path, image_cell: str) -> Optional[Path]:
    """
    在 SNR 样品目录附近查找与 roi_windows / prediction 中 image 列一致的 ROI 图像文件。
    """
    rel = str(image_cell).strip().replace("\\", "/")
    bn = Path(rel).name
    bases = [snr_dir, snr_dir.parent, snr_dir.parent.parent]
    for base in bases:
        if not base.is_dir():
            continue
        cand = (base / rel).resolve()
        try:
            if cand.is_file():
                return cand
        except OSError:
            pass
        cand2 = base / bn
        if cand2.is_file():
            return cand2
    root = snr_dir.parent
    if root.is_dir():
        for p in root.rglob(bn):
            if p.is_file():
                return p
    return None


def run(refined_png: Path, xic_matrix: Optional[Path], out_dir: Optional[Path]) -> int:
    refined_png = refined_png.resolve()
    if not refined_png.is_file():
        print("[ERROR] 找不到 refined 图: %s" % refined_png)
        return 1

    snr_dir = refined_png.parent.parent
    out_dir = (out_dir or refined_png.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = refined_core_stem(refined_png)
    pr = snr_dir / "prediction_refined.csv"
    if not pr.is_file():
        print("[ERROR] 缺少 prediction_refined.csv: %s" % pr)
        return 1

    df = read_csv_safe(pr)
    row = find_row_for_refined_png(df, refined_png)
    if row is None:
        print("[ERROR] 在 CSV 中无法匹配 refined 图: %s" % refined_png.name)
        return 1

    image_name = str(row.get("image", "")).strip()
    row_idx = image_to_row_index(image_name, row.get("compound_name"))
    if row_idx is None:
        print("[ERROR] 无法解析 XIC 行号: image=%r" % image_name)
        return 1

    xic_path = xic_matrix or locate_xic_npy(snr_dir)
    roi_csv = locate_roi_csv(snr_dir, None)
    roi_map = load_roi_map(roi_csv) if roi_csv.is_file() else {}
    rw, _ = resolve_rt_window(roi_map, image_name)

    jpeg_src = find_cnn_roi_jpeg(snr_dir, image_name)
    report_lines = [
        "=== refined 图对应「原始数据」定位报告 ===",
        "refined_png: %s" % refined_png,
        "SNR 根目录: %s" % snr_dir,
        "",
        "[1] prediction_refined.csv: %s" % pr,
        "    image: %s" % image_name,
        "    compound_name: %s" % row.get("compound_name", ""),
        "    xic_matrix 行号(1-based): %d" % (row_idx + 1),
        "    main_rt_peak: %s" % row.get("main_rt_peak", ""),
        "",
        "[2] xic_matrix.npy: %s" % (xic_path if xic_path and xic_path.is_file() else "【未找到】"),
        "",
        "[3] roi_windows.csv: %s" % roi_csv,
        "    本图 RT 窗: %s" % (str(rw) if rw else "【未命中】"),
        "",
        "[4] CNN 输入 ROI 图像（testXIC/newtest 生成的 .jpeg）:",
        "    %s" % (jpeg_src if jpeg_src else "【未在样品目录下搜到】"),
        "",
        "输出目录: %s" % out_dir,
    ]
    report_path = out_dir / ("%s_source_report.txt" % stem)
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    print("\n[OK] 报告已写入: %s" % report_path)

    if jpeg_src is not None:
        dst_j = out_dir / ("%s_source_roi%s" % (stem, jpeg_src.suffix.lower()))
        shutil.copy2(str(jpeg_src), str(dst_j))
        print("[OK] 已复制 CNN 输入 ROI 图: %s" % dst_j)

    if not xic_path or not xic_path.is_file():
        print("[ERROR] 未找到 xic_matrix.npy，无法绘制 XIC。请用 --xic_matrix 指定路径。")
        return 1

    plot_fn = _get_plot_func()
    plot_fn(
        refined_png=refined_png,
        xic_matrix=xic_path,
        roi_windows_csv=roi_csv if roi_csv.is_file() else None,
        pred_refined_csv=pr,
        output_png=out_dir / ("%s_xic_roi.png" % stem),
        window="roi",
        smooth_sigma=0.0,
    )
    plot_fn(
        refined_png=refined_png,
        xic_matrix=xic_path,
        roi_windows_csv=roi_csv if roi_csv.is_file() else None,
        pred_refined_csv=pr,
        output_png=out_dir / ("%s_xic_full.png" % stem),
        window="full",
        smooth_sigma=0.0,
    )
    return 0


def main():
    ap = argparse.ArgumentParser(description="定位 refined 图原始数据并导出 XIC / 报告 / ROI 图")
    ap.add_argument("--refined_png", type=str, required=True, help="refined 图路径")
    ap.add_argument("--xic_matrix", type=str, default=None, help="可选，显式指定 xic_matrix.npy")
    ap.add_argument("--out_dir", type=str, default=None, help="输出目录，默认同 refined_plots")
    args = ap.parse_args()
    refined = Path(args.refined_png)
    xic = Path(args.xic_matrix).resolve() if args.xic_matrix else None
    odir = Path(args.out_dir).resolve() if args.out_dir else None
    return run(refined, xic, odir)


if __name__ == "__main__":
    raise SystemExit(main())
