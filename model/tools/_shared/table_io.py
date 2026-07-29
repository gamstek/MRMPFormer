# -*- coding: utf-8 -*-
"""
表格 I/O 公共函数：CSV/Excel 读取、面积解析、化合物名称标准化。

保持与原脚本完全一致的行为。
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def read_table(path, sep: str = ",") -> pd.DataFrame:
    """
    安全读取表格文件（CSV/TSV），自动尝试多种编码。

    编码尝试顺序：utf-8-sig, utf-8, gb18030, gbk。
    """
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, engine="python")
        except Exception:
            continue
    raise OSError("无法读取文件: %s" % path)


def parse_area(val) -> Optional[float]:
    """
    解析面积值，支持多种格式：
    - 纯数字字符串
    - 科学计数法（如 1.23e5）
    - 千分位逗号
    - N/A、NA、NaN、<2 POINTS 等标记 → None

    返回 float 或 None。
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.upper() in {"N/A", "NA", "NAN", "<2 POINTS"}:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    # 科学计数法（含空格，如 1.23 e 5）
    m = re.match(r"^([\d\.]+)\s*[eE]\s*([+\-]?\d+)$", s)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))
    # 千分位逗号
    s2 = s.replace(",", "")
    try:
        return float(s2)
    except ValueError:
        return None


def normalize_compound_name(name) -> str:
    """
    标准化化合物名称：去空格、转小写（casefold）。

    用于跨表格的化合物名称匹配。
    """
    return str(name).strip().casefold()


def find_area_column(df: pd.DataFrame, candidates=None) -> str:
    """
    在 DataFrame 中自动查找面积列。
    先尝试 candidates 候选名，再按列名关键词匹配（area / 峰面积）。
    """
    if candidates is None:
        candidates = ["Area", "area", "峰面积", "Area_CSV"]
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    for c in df.columns:
        cs = str(c)
        if "area" in cs.casefold() or "峰面积" in cs:
            return c
    raise KeyError("未找到 Area 列，现有列: %s" % list(df.columns))


def find_compound_column(df: pd.DataFrame, candidates=None) -> str:
    """
    在 DataFrame 中自动查找化合物名称列。
    先尝试 candidates 候选名，再按列名关键词匹配。
    """
    if candidates is None:
        candidates = [
            "Component Name", "component_name", "Compound Name",
            "compound_name", "Compound", "化合物名称", "名称",
        ]
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    for c in df.columns:
        cs = str(c)
        if "compound" in cs.casefold() or "名称" in cs or "name" in cs.casefold():
            return c
    raise KeyError("未找到化合物名称列，现有列: %s" % list(df.columns))
