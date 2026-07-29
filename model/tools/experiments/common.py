# -*- coding: utf-8 -*-
"""
江南、欧陆实验共用的比较和报表辅助函数。

不包含正式峰识别、SNR、积分或定量算法。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .._shared.table_io import (
    find_area_column,
    find_compound_column,
    normalize_compound_name,
    parse_area,
    read_table,
)


def load_csv_areas(
    csv_path: Path,
    area_candidates: Optional[List[str]] = None,
    compound_candidates: Optional[List[str]] = None,
) -> Dict[str, float]:
    """从 CSV 加载化合物→面积映射。返回 {normalized_name: area}。"""
    df = read_table(csv_path)
    area_col = find_area_column(df, area_candidates)
    compound_col = find_compound_column(df, compound_candidates)
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        name = normalize_compound_name(r.get(compound_col, ""))
        if not name:
            continue
        a = parse_area(r.get(area_col))
        if a is not None and a > 0:
            out[name] = a
    return out


def load_os_areas(
    os_path: Path,
    sample_name: str,
    sep: str = "\t",
) -> Dict[str, float]:
    """从 OS txt（人工加标）加载指定样品的化合物→面积映射。"""
    df = read_table(os_path, sep=sep)
    sample_col = None
    for c in df.columns:
        if "sample" in str(c).casefold():
            sample_col = c
            break
    if sample_col is None:
        sample_col = df.columns[0]

    area_col = find_area_column(df)
    compound_col = find_compound_column(df)

    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        if str(r.get(sample_col, "")).strip().casefold() != sample_name.casefold():
            continue
        name = normalize_compound_name(r.get(compound_col, ""))
        if not name:
            continue
        a = parse_area(r.get(area_col))
        if a is not None and a > 0:
            out[name] = a
    return out


def compute_relative_error(csv_area: Optional[float], os_area: float) -> Optional[float]:
    """相对误差 = |csv - os| / os。os 为 0 或 None 时返回 None。"""
    if os_area is None or os_area == 0:
        return None
    if csv_area is None:
        return None
    return abs(csv_area - os_area) / os_area


def pair_area_comparison(
    csv_areas: Dict[str, float],
    os_areas: Dict[str, float],
    compound_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    """对比两组面积映射，返回逐化合物比较结果列表。"""
    results = []
    all_names = set(csv_areas.keys()) | set(os_areas.keys())
    for name in sorted(all_names):
        if compound_filter and name not in compound_filter:
            continue
        ca = csv_areas.get(name)
        oa = os_areas.get(name)
        err = compute_relative_error(ca, oa) if ca is not None and oa is not None else None
        results.append({
            "compound": name,
            "csv_area": ca,
            "os_area": oa,
            "rel_error": err,
        })
    return results
