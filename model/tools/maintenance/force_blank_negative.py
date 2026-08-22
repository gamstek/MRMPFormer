# -*- coding: utf-8 -*-
"""
强制指定样品（默认 BLANK 空白样）的所有行为负样本：peak_label / peak_count → 0。

背景：traindata3 的 BLANK 空白样 82% 行被标为 peak_label=1（多为进样残留/污染峰）。
若这些行作正样本进入训练，模型会把"空白响应"学成真峰，真实样品定量时假阳性上升。
本脚本将 sample_id 以 --sample_prefix 开头的样品行的 peak_label/peak_count 强制改为 0，
使 BLANK 仅以负样本（有 ROI 图、无 bbox）进入数据集。构建逻辑见
preprocessing.coco_annotation.build_coco_for_mzml（peak_label=0 → 生成 ROI 图但不输出 bbox）。

用法（model/ 目录下执行）：
  # 预览（不写文件）
  python -m tools.maintenance.force_blank_negative \
      --xlsx ../data/label/traindata3.xlsx --dry_run
  # 正式执行（自动备份 <xlsx>.bak_<时间戳>，重复执行幂等）
  python -m tools.maintenance.force_blank_negative --xlsx ../data/label/traindata3.xlsx
"""
import argparse
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _load_xlsx(xlsx):
    """返回 (共享字符串列表, sheet1 根元素)。"""
    z = zipfile.ZipFile(xlsx)
    ss = ["".join(t.text or "" for t in si.iter(_NS + "t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    return ss, sheet


def _col_letter(cell_r):
    return "".join(ch for ch in cell_r if ch.isalpha())


def _cell_value(cell, ss):
    v = cell.find(_NS + "v")
    if v is None:
        return ""
    return ss[int(v.text)] if cell.get("t") == "s" else (v.text or "")


def main():
    ap = argparse.ArgumentParser(description="强制指定样品行为负样本（peak_label/peak_count → 0）")
    ap.add_argument("--xlsx", default="../data/label/traindata3.xlsx", help="标注 xlsx 路径")
    ap.add_argument("--sample_prefix", default="traindata3-BLANK",
                    help="sample_id 以此开头的样品将被强制为负样本")
    ap.add_argument("--dry_run", action="store_true", help="仅预览改动，不写文件")
    args = ap.parse_args()

    xlsx = Path(args.xlsx).resolve()
    ss, sheet = _load_xlsx(str(xlsx))

    # 定位列（表头行 r=1）
    header_row = next(sheet.iter(_NS + "row"))
    col_map = {}
    for c in header_row.iter(_NS + "c"):
        col_map[_col_letter(c.get("r", ""))] = _cell_value(c, ss).strip()
    sid_col = next((c for c, n in col_map.items() if n == "sample_id"), None)
    pl_col = next((c for c, n in col_map.items() if n == "peak_label"), None)
    pc_col = next((c for c, n in col_map.items() if n == "peak_count"), None)
    missing = [n for n, c in [("sample_id", sid_col), ("peak_label", pl_col), ("peak_count", pc_col)] if c is None]
    if missing:
        raise SystemExit(f"[ERROR] 缺少列: {missing}（表头: {col_map}）")

    # peak_label/peak_count 为数值型单元格（<v> 直接存 "0"/"1"），无需处理共享字符串
    n_cells, n_rows = 0, 0
    changed_samples = set()
    for row in sheet.iter(_NS + "row"):
        if row.get("r") == "1":
            continue
        cells = {_col_letter(c.get("r", "")): c for c in row.iter(_NS + "c")}
        sid = _cell_value(cells.get(sid_col), ss).strip()
        if not sid.startswith(args.sample_prefix):
            continue
        n_rows += 1
        changed = False
        for col in (pl_col, pc_col):
            cell = cells.get(col)
            if cell is None:
                continue
            v = cell.find(_NS + "v")
            if v is None:
                continue
            if v.text and v.text.strip() == "1":
                v.text = "0"
                changed = True
                n_cells += 1
        if changed:
            changed_samples.add(sid)

    print(f"[INFO] 命中样品（{len(changed_samples)} 个）: {sorted(changed_samples)}")
    print(f"[INFO] 命中行: {n_rows} | 将被改为负样本的 peak_label/peak_count 单元格: {n_cells}")
    if args.dry_run:
        print("[DRY RUN] 仅预览，未写文件")
        return
    if n_cells == 0:
        print("[INFO] 无需修改（已是负样本或非字符串值）")
        return

    bak = xlsx.with_name(f"{xlsx.stem}.xlsx.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(xlsx, bak)
    print("[备份]", bak)

    import xml.etree.ElementTree as _ET
    _ET.register_namespace("", _NS)
    new_sheet = _ET.tostring(sheet, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(xlsx, "w", zipfile.ZIP_DEFLATED) as zout:
        with zipfile.ZipFile(bak) as zin:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "xl/worksheets/sheet1.xml":
                    data = new_sheet
                zout.writestr(item, data)

    # 验证：前缀样品 peak_label/peak_count 应全为 0
    ss2, sheet2 = _load_xlsx(str(xlsx))
    bad = 0
    for row in sheet2.iter(_NS + "row"):
        if row.get("r") == "1":
            continue
        cells = {_col_letter(c.get("r", "")): c for c in row.iter(_NS + "c")}
        sid = _cell_value(cells.get(sid_col), ss2).strip()
        if not sid.startswith(args.sample_prefix):
            continue
        pl = _cell_value(cells.get(pl_col), ss2).strip()
        pc = _cell_value(cells.get(pc_col), ss2).strip()
        if pl != "0" or pc != "0":
            bad += 1
    print(f"[验证] 前缀命中样品中 peak_label/peak_count 仍非 0 的行数: {bad}")
    if bad:
        print("[WARN] 验证未通过，请人工检查（备份仍保留）")
    else:
        print("[DONE] 全部改为负样本完成")


if __name__ == "__main__":
    main()
