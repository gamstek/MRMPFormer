# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"d:\yinlibo\MRMPFormer\model")
import pandas as pd
from pathlib import Path

for name in ["test1.xlsx", "test1_traditional.xlsx", "traindata3.xlsx"]:
    p = Path(r"d:\yinlibo\MRMPFormer\data\label") / name
    if not p.exists():
        print(name, "不存在"); continue
    xl = pd.ExcelFile(p)
    print("=" * 60)
    print(name, "sheets:", xl.sheet_names)
    for sh in xl.sheet_names[:2]:
        df = xl.parse(sh)
        print(f"-- sheet {sh}: shape={df.shape}")
        print("   columns:", list(df.columns))
        print("   前2行:")
        print(df.head(2).to_string())
