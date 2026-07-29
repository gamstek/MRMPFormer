# -*- coding: utf-8 -*-
"""从 mzML 文件可靠读取 chromatogram 的 id（UTF-8），以及生成可放在文件名里的 nid 片段。"""
import hashlib
import html
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Optional, Sequence

_MZML_NS = "http://psi.hupo.org/ms/mzml"
_CHROM_TAG = "{%s}chromatogram" % _MZML_NS


def _decode_attr_value_bytes(raw: bytes) -> str:
    if not raw:
        return ""
    s = None
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            s = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if s is None:
        s = raw.decode("utf-8", errors="replace")
    s = html.unescape(s)
    if any("\u4e00" <= c <= "\u9fff" for c in s) and "\ufffd" not in s:
        return s
    try:
        s2 = s.encode("latin-1").decode("utf-8")
        if any("\u4e00" <= c <= "\u9fff" for c in s2):
            return s2
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return s


def chromatogram_ids_from_mzml_xml(path: Path) -> List[str]:
    rows = []
    try:
        for _event, elem in ET.iterparse(str(path), events=("end",)):
            tag = elem.tag
            if tag != _CHROM_TAG and not str(tag).endswith("}chromatogram"):
                continue
            cid = elem.get("id") or ""
            ix_raw = elem.get("index")
            ix = None
            if ix_raw is not None:
                try:
                    ix = int(ix_raw)
                except ValueError:
                    ix = None
            rows.append((ix, cid))
            elem.clear()
    except (ET.ParseError, OSError):
        return []
    if not rows:
        return []
    if all(r[0] is not None for r in rows):
        rows.sort(key=lambda r: r[0])
        return [r[1] for r in rows]
    return [r[1] for r in rows]


def chromatogram_ids_from_mzml_raw_bytes(path: Path) -> List[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return []
    rows = []
    pos = 0
    while True:
        i = data.find(b"<chromatogram", pos)
        if i < 0:
            break
        j = data.find(b">", i)
        if j < 0:
            break
        tag = data[i:j]
        idm = re.search(br'\bid\s*=\s*"([^"]*)"', tag, re.I)
        if not idm:
            idm = re.search(br"\bid\s*=\s*'([^']*)'", tag, re.I)
        if idm:
            cid = _decode_attr_value_bytes(idm.group(1))
            ixm = re.search(br'\bindex\s*=\s*"(\d+)"', tag, re.I)
            ix = int(ixm.group(1)) if ixm else None
            rows.append((ix, cid))
        pos = j + 1
    if not rows:
        return []
    if all(r[0] is not None for r in rows):
        rows.sort(key=lambda r: r[0])
        return [r[1] for r in rows]
    return [r[1] for r in rows]


def pick_chrom_native_ids_from_mzml_file(path: Path, n_chrom: int) -> Optional[List[str]]:
    raw = chromatogram_ids_from_mzml_raw_bytes(path)
    if len(raw) == n_chrom and raw:
        return raw
    et = chromatogram_ids_from_mzml_xml(path)
    if len(et) == n_chrom and et:
        return et
    return None


def resolve_native_ids_for_chromatograms(
    mzml_path: str,
    chromatograms: Sequence[Any],
    native_id_to_str,
) -> List[str]:
    """与 pyopenms 色谱列表等长：优先 mzML XML/字节 id，否则 chrom.getNativeID()。"""
    p = Path(mzml_path).resolve()
    n = len(chromatograms)
    picked = pick_chrom_native_ids_from_mzml_file(p, n) if p.is_file() else None
    out: List[str] = []
    for i, chrom in enumerate(chromatograms):
        if picked is not None and i < len(picked):
            out.append(picked[i] or "")
        else:
            out.append(native_id_to_str(chrom.getNativeID()))
    return out


def filesystem_slug_for_native_id(s: str, max_len: int = 56) -> str:
    """
    用于文件名：Unicode 规范化、去掉 Windows 非法字符、过长则截断 + 短 hash。
    """
    if s is None:
        return "empty"
    t = unicodedata.normalize("NFKC", str(s)).strip()
    if not t:
        return "empty"
    bad = '<>:"/\\|?*\n\r\t'
    buf = []
    for c in t:
        if c in bad or ord(c) < 32:
            buf.append("_")
        else:
            buf.append(c)
    t = "".join(buf)
    while "__" in t:
        t = t.replace("__", "_")
    t = t.strip("._")
    if not t:
        h = hashlib.sha256(str(s).encode("utf-8", errors="replace")).hexdigest()[:10]
        return "nid_%s" % h
    if len(t) > max_len:
        h = hashlib.sha256(str(s).encode("utf-8", errors="replace")).hexdigest()[:8]
        t = t[: max(8, max_len - 9)].rstrip("._") + "_" + h
    return t


def roi_image_stem(compound_index_1based: int, q1: Any, q3: Any, native_id: str) -> str:
    """与 newtest / SNR 流水线兼容的前缀：{N}_mz... ；后缀含 q3 与 nid。"""
    n = int(compound_index_1based)
    if q1 is not None:
        try:
            q1f = float(q1)
            if q1f == q1f:
                q1s = "%.4f" % q1f
            else:
                q1s = "nan"
        except (TypeError, ValueError):
            q1s = "nan"
    else:
        q1s = "nan"
    if q3 is not None:
        try:
            q3f = float(q3)
            if q3f == q3f:
                q3s = "%.2f" % q3f
            else:
                q3s = "nan"
        except (TypeError, ValueError):
            q3s = "nan"
    else:
        q3s = "nan"
    slug = filesystem_slug_for_native_id(native_id or "")
    return "%d_mz%s_q3%s_nid%s" % (n, q1s, q3s, slug)


def transition_dedup_key(native_id: str, q1: Any, q3: Any):
    nid = (native_id or "").strip()
    k1 = None
    if q1 is not None:
        try:
            f = float(q1)
            if math.isfinite(f):
                k1 = round(f, 4)
        except (TypeError, ValueError):
            pass
    k3 = None
    if q3 is not None:
        try:
            f = float(q3)
            if math.isfinite(f):
                k3 = round(f, 2)
        except (TypeError, ValueError):
            pass
    return (nid, k1, k3)
