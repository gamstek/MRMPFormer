"""临时脚本2.0：标注 xlsx 删除 ert 列。
规则：rt 为空 && peak_label=="0" && ert 有值 → 先用 ert 回填 rt，再整列删除 ert。
输出统计信息，用完即删。
"""
import glob
import os
import re
import shutil
import zipfile
from xml.etree import ElementTree as ET

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ET.register_namespace("", NS)
NSQ = "{%s}" % NS


def col_letter(n):
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def col_num(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def shared_load(contents):
    shared = []
    if "xl/sharedStrings.xml" in contents:
        sroot = ET.fromstring(contents["xl/sharedStrings.xml"])
        for si in sroot.findall(NSQ + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NSQ + "t")))
    return shared


def cell_text(c, shared):
    if c.attrib.get("t") == "inlineStr":
        isp = c.find(NSQ + "is")
        return "".join(t.text or "" for t in isp.iter(NSQ + "t")) if isp is not None else ""
    v = c.find(NSQ + "v")
    if v is None:
        return ""
    txt = (v.text or "").strip()
    return shared[int(txt)] if c.attrib.get("t") == "s" and txt.isdigit() else txt


for f in sorted(glob.glob(r"d:\yinlibo\MRMPFormer\data\label\*.xlsx")):
    name = os.path.basename(f)
    with zipfile.ZipFile(f) as zin:
        names = zin.namelist()
        contents = {n: zin.read(n) for n in names}
    shared = shared_load(contents)
    changed_sheets = []

    for nm in list(contents):
        m = re.match(r"xl/worksheets/sheet\d+\.xml$", nm)
        if not m:
            continue
        root = ET.fromstring(contents[nm])
        rows_el = list(root.iter(NSQ + "row"))
        if not rows_el:
            continue
        hdr_map = {}
        for c in rows_el[0].findall(NSQ + "c"):
            letters = re.match(r"[A-Z]+", c.attrib["r"]).group()
            hdr_map[letters] = cell_text(c, shared).strip()
        rev = {v: k for k, v in hdr_map.items()}
        ert_l = rev.get("ert")
        if not ert_l:
            continue
        rt_l = rev.get("rt")
        pl_l = rev.get("peak_label")
        ert_c, rt_c, pl_c = col_num(ert_l), col_num(rt_l) if rt_l else None, col_num(pl_l) if pl_l else None

        filled = 0
        for row in rows_el[1:]:
            cells = {re.match(r"[A-Z]+", c.attrib["r"]).group(): c for c in row.findall(NSQ + "c")}
            ce, crt, cpl = cells.get(ert_l), cells.get(rt_l) if rt_l else None, cells.get(pl_l) if pl_l else None
            need_fill = (
                crt is not None and (crt.find(NSQ + "v") is None or not (crt.find(NSQ + "v").text or "").strip())
                and cell_text(crt, shared).strip() == ""
            ) if crt is not None else True  # rt 单元格整体缺失视为空
            if crt is not None and cell_text(crt, shared).strip() != "":
                need_fill = False
            ert_txt = cell_text(ce, shared).strip() if ce is not None else ""
            pl_txt = cell_text(cpl, shared).strip() if cpl is not None else ""
            if need_fill and ce is not None and ert_txt and pl_txt == "0":
                import copy
                clone = copy.deepcopy(ce)
                clone.attrib["r"] = f"{rt_l}{row.attrib.get('r', '')}"
                row.insert(list(row).index(ce), clone)  # 保持升序：插到 ert 前
                filled += 1

        # 删除 ert 整列并左移右侧各列
        n_rows_affected = 0
        for row in root.iter(NSQ + "row"):
            for c in row.findall(NSQ + "c"):
                letters = re.match(r"[A-Z]+", c.attrib["r"]).group()
                suffix = c.attrib["r"][len(letters):]
                num = col_num(letters)
                if num == ert_c:
                    row.remove(c)
                elif num > ert_c:
                    c.attrib["r"] = f"{col_letter(num - 1)}{suffix}"
            if row.findall(NSQ + "c"):
                n_rows_affected += 1

        dim = root.find(NSQ + "dimension")
        if dim is not None and dim.get("ref"):
            mm = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", dim.get("ref"))
            if mm:
                hi = max(col_num(mm.group(1)), col_num(mm.group(3)))
                dim.set("ref", f"A{mm.group(2)}:{col_letter(hi - 1)}{mm.group(4)}")
        cols_el = root.find(NSQ + "cols")
        if cols_el is not None:
            for col in list(cols_el):
                lo, hi = int(col.get("min")), int(col.get("max"))
                if lo > hi:
                    cols_el.remove(col)
                    continue
                lo2 = lo - 1 if lo > ert_c else lo
                hi2 = hi - 1 if hi >= ert_c else hi
                if hi2 < lo2:
                    hi2 = lo2
                col.set("min", str(lo2))
                col.set("max", str(hi2))
        contents[nm] = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
        changed_sheets.append((nm, filled, n_rows_affected))

    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, contents[n])
    print(f"{name}: 回填rt={[(s[1]) for s in changed_sheets]}, 删ert列 -> {changed_sheets}")
