# -*- coding: utf-8 -*-
"""
公共 artifact 工具：CSV 安全读取、ROI map 加载、RT 窗口解析、XIC matrix 定位、
图像名到行索引转换等。

所有函数均保持与原脚本相同的行为，不修改业务逻辑。
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def read_csv_safe(path: Path) -> pd.DataFrame:
    """安全读取 CSV，兼容 Windows 中文路径（避免 C 引擎 OSError）。"""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(str(path), "r", encoding=enc, newline="") as f:
                return pd.read_csv(f)
        except Exception as e:
            last_err = e
    raise last_err


def safe_float(v, default: float = np.nan) -> float:
    """安全转换为 float，失败时返回 default。"""
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def load_roi_map(path: Path) -> Dict[str, Tuple[float, float]]:
    """
    从 roi_windows.csv 加载 image → (rt_lo, rt_hi) 映射。
    同时注册 basename 键，方便按文件名匹配。
    """
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
    """
    按多种键匹配 roi_windows（basename、去目录前缀等）。
    返回 (window_or_none, match_note)。

    match_note 说明匹配方式：'key=...' 表示精确命中，'no_match_tried=...' 表示未命中。
    """
    s = str(image_cell).strip().replace("\\", "/")
    if not s:
        return None, "empty_image"
    candidates = [s]
    name = os.path.basename(s)
    if name not in candidates:
        candidates.append(name)
    # SNR 有时写 snr_kept/xxx.jpeg（旧产物为 筛选保留/xxx.jpeg）；roi 表也可能只有 xxx 或带路径
    for c in list(candidates):
        if "/" in c or "\\" in c:
            tail = c.replace("\\", "/").split("/")[-1]
            if tail not in candidates:
                candidates.append(tail)
    for c in candidates:
        if c in roi_map:
            return roi_map[c], "key=%r" % c
    return None, "no_match_tried=%s" % candidates[:5]


def image_to_row_index(image_name: str, compound_name) -> Optional[int]:
    """
    从 image 名称或 compound_name 解析 XIC 矩阵中的 0 基行索引（Zero-based Index）。

    优先级：
    1. compound_name 是纯数字 → 减 1 得行号（1 基编号 → 0 基索引）
    2. image 名前缀 "N_mz..." 中 N 为数字 → 减 1 得行号
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
            if n > 0:  # 1 基编号必须 > 0
                return n - 1
    return None


def locate_xic_npy(snr_dir: Path) -> Optional[Path]:
    """
    在 SNR 目录附近查找 xic_matrix.npy。
    搜索顺序：
    1. SNR 子目录自身
    2. 父目录
    3. 祖父目录
    4. xic-roi-batch/<样品名>/
    """
    cands = [
        snr_dir / "xic_matrix.npy",
        snr_dir.parent / "xic_matrix.npy",
        snr_dir.parent.parent / "xic_matrix.npy",
    ]
    sample_name = snr_dir.parent.name
    gp = snr_dir.parent.parent
    cands.append(gp / "xic-roi-batch" / sample_name / "xic_matrix.npy")
    cands.append(gp.parent / "xic-roi-batch" / sample_name / "xic_matrix.npy")
    for p in cands:
        if p.is_file():
            return p
    return None


def locate_roi_csv(snr_dir: Path, explicit: Optional[Path] = None) -> Path:
    """
    在 SNR 目录附近查找 roi_windows.csv。
    若提供 explicit 且存在则直接返回；否则依次搜索 SNR 目录自身、xic-roi-batch。
    """
    if explicit is not None and explicit.is_file():
        return explicit
    rw = snr_dir / "roi_windows.csv"
    if rw.is_file():
        return rw
    sample_name = snr_dir.parent.name
    gp = snr_dir.parent.parent
    alt = gp / "xic-roi-batch" / sample_name / "roi_windows.csv"
    if alt.is_file():
        return alt
    return rw


def refined_core_stem(png_path: Path) -> str:
    """从 refined PNG 文件名去掉 _refined 后缀，得到核心 stem。"""
    s = png_path.stem
    suf = "_refined"
    if s.lower().endswith(suf.lower()):
        return s[: -len(suf)]
    return s


def find_row_for_refined_png(df: pd.DataFrame, png_path: Path) -> Optional[pd.Series]:
    """
    在 prediction_refined.csv 中匹配 refined PNG 对应的行。
    先精确匹配 stem，再尝试前缀包含匹配。
    """
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
