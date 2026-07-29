# -*- coding: utf-8 -*-
"""
将 pipeline 输出目录下分散在各级子目录中的文件，按「类别」归并到少数文件夹。

类别规则（默认识别常见结果后缀）：
  - .csv / .npy / .json / .txt：按文件名去掉扩展名分目录
  - .png / .jpg / .jpeg：统一归入 images/
  - 其它扩展名：归入 other/

用法：
  python -m <包名>.maintenance.organize_results --root "D:\\...\\result"
  python -m <包名>.maintenance.organize_results --root "..." --out "D:\\...\\merged"
  python -m <包名>.maintenance.organize_results --root "..." --dry-run
  python -m <包名>.maintenance.organize_results --root "..." --move   # 移动（慎用）

默认复制，不修改原树；输出默认在 <root>/organized_by_type。
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

_DEFAULT_SUFFIXES = {".csv", ".npy", ".png", ".jpg", ".jpeg", ".txt", ".json"}


def _safe_flat_name(rel: Path) -> str:
    parts = [p for p in rel.parts if p != "."]
    base = "__".join(parts)
    for c in '\\/:*?"<>|':
        base = base.replace(c, "_")
    return base or "unnamed"


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def organize(
    root: Path,
    out_root: Path,
    *,
    dry_run: bool = False,
    do_move: bool = False,
    all_files: bool = False,
    skip_output_tree: bool = True,
):
    root = root.resolve()
    out_root = out_root.resolve()

    skip_first_part = None
    if skip_output_tree and _is_under(out_root, root) and out_root != root:
        rel_out = out_root.relative_to(root)
        skip_first_part = rel_out.parts[0] if rel_out.parts else None

    n_done = 0
    n_skip = 0

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dpath = Path(dirpath)

        if skip_first_part is not None:
            try:
                rel_d = dpath.relative_to(root)
                if rel_d.parts and rel_d.parts[0] == skip_first_part:
                    dirnames[:] = []
                    continue
            except ValueError:
                pass

        dirnames[:] = [d for d in dirnames if d != "__pycache__" and not d.startswith(".")]

        for fn in filenames:
            src = dpath / fn
            try:
                rel = src.relative_to(root)
            except ValueError:
                n_skip += 1
                continue

            suffix = src.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg"}:
                cat = "images"
                dst_name = _safe_flat_name(rel)
            elif suffix in _DEFAULT_SUFFIXES or all_files:
                cat = Path(fn).stem
                dst_name = _safe_flat_name(rel)
            else:
                if not all_files:
                    n_skip += 1
                    continue
                cat = "other"
                dst_name = _safe_flat_name(rel)

            dst = out_root / cat / dst_name
            if dry_run:
                print("[DRY-RUN] %s → %s" % (src, dst))
                n_done += 1
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if do_move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
                n_done += 1

    print("[DONE] %d 个文件已%s，跳过 %d" % (n_done, "移动" if do_move else "复制", n_skip))


def main():
    ap = argparse.ArgumentParser(description="归并 pipeline 输出文件")
    ap.add_argument("--root", required=True, help="pipeline result 根目录")
    ap.add_argument("--out", default=None, help="输出根目录，默认 <root>/organized_by_type")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--move", action="store_true", help="移动而非复制（慎用）")
    ap.add_argument("--all-files", action="store_true", help="包含所有扩展名")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "organized_by_type"
    if not root.is_dir():
        print("[ERROR] root 目录不存在: %s" % root)
        sys.exit(1)

    organize(root, out, dry_run=args.dry_run, do_move=args.move, all_files=args.all_files)


if __name__ == "__main__":
    raise SystemExit(main())
