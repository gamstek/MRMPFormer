# -*- coding: utf-8 -*-
"""评估与可视化工具共用的读取、定位类辅助函数。"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def read_csv_safe(path: Path) -> pd.DataFrame:
    """读取 CSV，依次尝试 utf-8-sig / utf-8 / gbk，规避 Windows 中文路径问题。

    空文件（如仅 BOM/空行的空表）返回空 DataFrame，避免 EmptyDataError
    被误当成编码问题回退到 gbk 而产生误导性报错。
    """
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(str(path), "r", encoding=enc, newline="") as f:
                return pd.read_csv(f)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception as e:
            last_err = e
    raise last_err


def safe_float(v, default: float = np.nan) -> float:
    """转 float，失败或为空时返回 default。"""
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def load_roi_map(path: Path) -> Dict[str, Tuple[float, float]]:
    """读取 roi_windows.csv，返回 image 全路径及文件名到 (rt_lo, rt_hi) 的映射。"""
    if not path.is_file():
        return {}
    df = read_csv_safe(path)
    if not {"image", "rt_lo", "rt_hi"}.issubset(df.columns):
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for _, r in df.iterrows():
        key = str(r["image"]).strip().replace("\\", "/")
        out[key] = (float(r["rt_lo"]), float(r["rt_hi"]))
        bn = os.path.basename(key)
        if bn not in out:
            out[bn] = out[key]
    return out


def resolve_rt_window(
    roi_map: Dict[str, Tuple[float, float]],
    image_cell: str,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """在 roi_map 中匹配图像的 RT 窗口，返回 (窗口, 匹配方式)。"""
    s = str(image_cell).strip().replace("\\", "/")
    if not s:
        return None, "empty_image"
    candidates = [s]
    name = os.path.basename(s)
    if name not in candidates:
        candidates.append(name)
    # roi 表中的键可能只写文件名，补一个去路径的候选
    for c in list(candidates):
        tail = c.replace("\\", "/").split("/")[-1]
        if tail not in candidates:
            candidates.append(tail)
    for c in candidates:
        if c in roi_map:
            return roi_map[c], "key=%r" % c
    return None, "no_match_tried=%s" % candidates[:5]


def image_to_row_index(image_name: str, compound_name) -> Optional[int]:
    """解析 XIC 矩阵的 0 基行索引，两者均按 1 基编号记录。

    优先取 compound_name 的数值；否则看图像名前缀 "N_mz..." 中的 N。
    """
    c = safe_float(compound_name, np.nan)
    if np.isfinite(c) and c > 0:
        return int(c) - 1
    stem = Path(str(image_name)).stem
    low = stem.lower()
    if "_mz" in low:
        prefix = stem.split("_mz", 1)[0]
        if prefix.isdigit():
            n = int(prefix)
            if n > 0:
                return n - 1
    return None


def locate_xic_npy(snr_dir: Path) -> Optional[Path]:
    """定位 SNR 目录对应的 xic_matrix.npy。

    依次尝试 SNR 目录自身、父目录、祖父目录，
    再到祖父（及其上级）下的 xic_roi/<样品名>/ 或 xic-roi-batch/<样品名>/。
    """
    cands = [
        snr_dir / "xic_matrix.npy",
        snr_dir.parent / "xic_matrix.npy",
        snr_dir.parent.parent / "xic_matrix.npy",
    ]
    sample_name = snr_dir.parent.name
    gp = snr_dir.parent.parent
    for roi_dir_name in ("xic_roi", "xic-roi-batch"):
        cands.append(gp / roi_dir_name / sample_name / "xic_matrix.npy")
        cands.append(gp.parent / roi_dir_name / sample_name / "xic_matrix.npy")
    for p in cands:
        if p.is_file():
            return p
    return None


def locate_roi_csv(snr_dir: Path, explicit: Optional[Path] = None) -> Path:
    """定位 roi_windows.csv：explicit 优先，其次 SNR 目录自身，最后到 xic_roi/xic-roi-batch 目录。"""
    if explicit is not None and explicit.is_file():
        return explicit
    rw = snr_dir / "roi_windows.csv"
    if rw.is_file():
        return rw
    sample_name = snr_dir.parent.name
    gp = snr_dir.parent.parent
    for roi_dir_name in ("xic_roi", "xic-roi-batch"):
        alt = gp / roi_dir_name / sample_name / "roi_windows.csv"
        if alt.is_file():
            return alt
    return rw


def resolve_pred_root(result_root: Path) -> Path:
    """预测输出根目录：predictions_model，缺失时退回 batch_predictions。"""
    new_root = result_root / "predictions_model"
    if new_root.is_dir():
        return new_root
    return result_root / "batch_predictions"


def resolve_roi_root(result_root: Path) -> Path:
    """ROI 根目录：xic_roi，缺失时退回 xic-roi-batch。"""
    new_root = result_root / "xic_roi"
    if new_root.is_dir():
        return new_root
    return result_root / "xic-roi-batch"


def resolve_snr_root(result_root: Path) -> Path:
    """SNR/精修结果根目录：prediction_refined，缺失时退回 snr_filtered。"""
    new_root = result_root / "prediction_refined"
    if new_root.is_dir():
        return new_root
    return result_root / "snr_filtered"


def refined_core_stem(png_path: Path) -> str:
    """refined PNG 文件名去掉 _refined 后缀。"""
    s = png_path.stem
    suf = "_refined"
    if s.lower().endswith(suf.lower()):
        return s[: -len(suf)]
    return s


def find_row_for_refined_png(df: pd.DataFrame, png_path: Path) -> Optional[pd.Series]:
    """在 prediction_refined.csv 中找到 refined PNG 对应的行：先精确匹配 stem，再前缀包含匹配。"""
    core = refined_core_stem(png_path)
    exact = []
    prefixed = []
    for _, r in df.iterrows():
        img = str(r.get("image", "")).strip()
        if not img:
            continue
        st = Path(img).stem
        if st == core:
            exact.append(r)
        elif core.endswith("_" + st) or core.endswith("-" + st):
            prefixed.append(r)
    if len(exact) >= 1:
        return exact[0]
    if len(prefixed) >= 1:
        return prefixed[0]
    return None
