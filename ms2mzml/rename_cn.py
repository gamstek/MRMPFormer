"""
将 data/ 目录下含中文名的 .msdata 文件重命名为英文名。
仅预览模式，确认无误后删掉 --dry-run 执行。
"""

import argparse
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 中文 → 英文 映射表
CN_MAP = {
    "化合物优化参数": "Opt",
    "扫描方法1": "Method1",
    "离子源温度": "TEM",
    "离子源电压": "ISVF",
    "雾化气体": "GS1",
    "辅助加热气": "GS2",
    "碰撞气体": "CAD",
    "碰撞能量": "CE",
    "去簇电压": "DP",
    "化合物": "Cpd",
    "氯霉素": "Chloramphenicol",
    "利血平": "Reserpine",
    "雌二醇": "Estradiol",
    "羟孕酮": "Hydroxyprogesterone",
    "二硝基酚": "Dinitrophenol",
    "欧陆": "Eurofins",
    "线性": "Linearity",
    "重复性": "Repeatability",
    "正负切换": "PolaritySwitch",
}


def to_english(name: str) -> str:
    """将中文文件名逐段替换为英文"""
    result = name
    for cn, en in CN_MAP.items():
        result = result.replace(cn, en)
    # 清理残留的非 ASCII 字符（中文标点等）
    result = re.sub(r'[^\x00-\x7F]+', '', result)
    # 去掉多余的连字符和连续点
    result = re.sub(r'-{2,}', '-', result)
    result = re.sub(r'\.{2,}', '.', result)
    result = result.strip('-')
    return result


def main():
    parser = argparse.ArgumentParser(description="中文 .msdata 文件名 → 英文")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="预览模式（默认开启），加 --no-dry-run 执行重命名")
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run")
    args = parser.parse_args()

    files = sorted(DATA_DIR.glob("*.msdata"))
    cn_files = [(f, to_english(f.name)) for f in files if f.name != to_english(f.name)]

    if not cn_files:
        print("没有需要重命名的中文文件")
        return

    print(f"找到 {len(cn_files)} 个含中文的文件:\n")
    for old, new in cn_files:
        print(f"  {old.name}")
        print(f"  → {new}\n")

    if args.dry_run:
        print(f"以上为预览。确认无误后执行:")
        print(f"  python rename_cn.py --no-dry-run")
        return

    renamed = 0
    for old_path, new_name in cn_files:
        new_path = old_path.with_name(new_name)
        if new_path.exists():
            print(f"跳过（目标已存在）: {old_path.name}")
            continue
        old_path.rename(new_path)
        renamed += 1
        print(f"OK: {old_path.name} → {new_name}")

    print(f"\n完成: {renamed}/{len(cn_files)} 个文件已重命名")


if __name__ == "__main__":
    main()
