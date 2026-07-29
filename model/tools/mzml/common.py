# -*- coding: utf-8 -*-
"""mzML 公共工具：文件打开、编码兼容、native_id 修复、Q1/Q3 读取。"""

import html
import re
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


def decode_native_id(native_id: Any) -> str:
    """
    修复 pyopenms 在 Windows 上的中文乱码问题。

    chromatogram / spectrum 的 native id 在 mzML 里多为 UTF-8。
    pyopenms 在 Windows 上偶发：bytes 非 utf-8、或 str 实为 UTF-8 字节被误解释为 Latin-1。
    """
    if native_id is None:
        return ""
    if isinstance(native_id, bytes):
        for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
            try:
                return native_id.decode(enc)
            except UnicodeDecodeError:
                continue
        return native_id.decode("utf-8", errors="replace")
    s = str(native_id)
    if not s:
        return ""
    # 已含 CJK 且无大量替换符时，多半是正确 Unicode
    if any("\u4e00" <= c <= "\u9fff" for c in s) and "\ufffd" not in s:
        return s
    # 误解码常见修复：Latin-1 码位 ↔ 原始 UTF-8 字节
    try:
        decoded = s.encode("latin-1").decode("utf-8")
        if any("\u4e00" <= c <= "\u9fff" for c in decoded):
            return html.unescape(decoded)
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return html.unescape(s)


def parse_q1_q3_from_native_id(native_id_text: str) -> Tuple[Optional[float], Optional[float]]:
    """从 native_id 文本解析 Q1/Q3（兼容无 Q1=/Q3= 的厂商格式）。"""
    text = str(native_id_text)
    q1 = q3 = None
    for pat in (r"Q1=([\d\.]+)", r"q1=([\d\.]+)", r"precursor[=:_ ]([\d\.]+)"):
        m1 = re.search(pat, text)
        if m1:
            q1 = float(m1.group(1))
            break
    for pat in (r"Q3=([\d\.]+)", r"q3=([\d\.]+)", r"product[=:_ ]([\d\.]+)"):
        m3 = re.search(pat, text)
        if m3:
            q3 = float(m3.group(1))
            break
    return q1, q3


def _load_ms_experiment_pyopenms(mzml_path: Path) -> Tuple[Any, Optional[str]]:
    """
    使用 pyopenms 加载 mzML 文件。
    返回 (MSExperiment, tmp_path_or_None)。
    若文件编码有问题，尝试修复后重新加载。
    """
    from pyopenms import MSExperiment, MzMLFile

    exp = MSExperiment()
    MzMLFile().load(str(mzml_path), exp)
    return exp, None


def get_chromatogram_count(exp) -> int:
    """返回 MSExperiment 中 chromatogram 数量。"""
    try:
        return exp.getNrChromatograms()
    except Exception:
        return len(exp.getChromatograms()) if hasattr(exp, 'getChromatograms') else 0


def get_spectrum_count(exp) -> int:
    """返回 MSExperiment 中 spectrum 数量。"""
    try:
        return exp.getNrSpectra()
    except Exception:
        return len(exp.getSpectra()) if hasattr(exp, 'getSpectra') else 0
