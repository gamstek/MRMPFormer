# -*- coding: utf-8 -*-
"""
chrom JSON 解析公共函数：时间/强度提取、Q1/Q3 解析、目录批量加载。

与 testXIC.load_raw_chroms_from_json_dir 保持语义一致（仅依赖 json/numpy）。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_time_intensity(data: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 chrom JSON 中提取 RT（秒）和强度数组。

    返回 (rt_sec, intensity)，两者均为 np.float64 数组。
    若 time 单位非秒，自动转换为秒（minute → ×60）。
    若数据无效（长度 < 2 或不一致），返回 (None, None)。
    """
    # 时间
    tb = data.get("time")
    if isinstance(tb, dict):
        tvals = tb.get("values") or []
        unit = str(tb.get("unit", "minute")).lower()
    else:
        tvals = data.get("rt") or data.get("time") or []
        unit = "minute"

    # 强度
    ib = data.get("intensity")
    if isinstance(ib, dict):
        ivals = ib.get("values") or []
    else:
        ivals = data.get("intensity") or []

    if len(tvals) < 2 or len(ivals) < 2 or len(tvals) != len(ivals):
        return None, None

    tarr = np.asarray(tvals, dtype=np.float64)
    iarr = np.asarray(ivals, dtype=np.float64)

    # 单位转换
    if unit in ("second", "seconds", "sec", "s"):
        rt_sec = tarr
    else:
        rt_sec = tarr * 60.0

    return rt_sec, iarr


def parse_q1_q3(data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    从 chrom JSON 中提取 Q1（前体离子 m/z）和 Q3（产物离子 m/z）。

    支持两种 JSON 格式：
    1. mz 为 dict：含 precursor_mz / product_mz
    2. mz 为标量：含 mz / q3 字段
    """
    mz_b = data.get("mz")
    if isinstance(mz_b, dict):
        q1 = mz_b.get("precursor_mz")
        q3 = mz_b.get("product_mz")
    else:
        q1 = data.get("mz")
        q3 = data.get("q3")

    if q1 is not None:
        try:
            q1 = float(q1)
        except (TypeError, ValueError):
            q1 = None
    else:
        q1 = None

    if q3 is not None:
        try:
            q3 = float(q3)
        except (TypeError, ValueError):
            q3 = None
    else:
        q3 = None

    return q1, q3


def load_chrom_json_directory(json_dir) -> List[Dict[str, Any]]:
    """
    加载目录下所有 chrom JSON 文件，返回记录列表。

    每条记录包含：native_id, q1, q3, rt_sec, intensity。
    与 testXIC.load_raw_chroms_from_json_dir 同语义。
    自动跳过 _result 后缀的 JSON 和解析失败的文件。
    """
    root = Path(json_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError("[ERROR] chrom JSON 目录不存在: %s" % root)

    seen = set()
    out = []
    for path in sorted(root.glob("*.json")):
        if path.stem.endswith("_result"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print("[WARN] 跳过 JSON %s: %s" % (path.name, e))
            continue

        rt_sec, intensity = parse_time_intensity(data)
        if rt_sec is None:
            print("[WARN] 跳过 %s: 无效 time/rt 或 intensity" % path.name)
            continue

        q1, q3 = parse_q1_q3(data)

        key = (
            round(q1, 4) if q1 is not None else None,
            round(q3, 2) if q3 is not None else None,
            path.name,
        )
        if key in seen:
            continue
        seen.add(key)

        native_id = str(data.get("native_id", ""))
        out.append({
            "native_id": native_id,
            "q1": q1,
            "q3": q3,
            "rt_sec": rt_sec,
            "intensity": np.asarray(intensity, dtype=np.float64).copy(),
        })

    return out
