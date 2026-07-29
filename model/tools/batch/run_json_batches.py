# -*- coding: utf-8 -*-
"""
批量对 truedata 下每个子文件夹运行 main.py --mode batch_json_dir。

用法：
  python -m <包名>.batch.run_json_batches
  python -m <包名>.batch.run_json_batches --base truedata/20260204-01_result --model resources/checkpoint0029.pth
"""
import argparse
import subprocess
import sys
from pathlib import Path

# 项目根目录（main.py 所在）
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    root = _REPO_ROOT
    parser = argparse.ArgumentParser(description="批量 batch_json_dir")
    parser.add_argument(
        "--base",
        type=str,
        default=str(root / "truedata" / "20260204-01_result"),
        help="包含多个样本子文件夹的目录",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(root / "resources" / "checkpoint0029.pth"),
        help="QuanFormer .pth 权重路径",
    )
    parser.add_argument("--no-plot", action="store_true", help="不加 --plot")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    model = Path(args.model).resolve()
    if not base.is_dir():
        print("[ERROR] 目录不存在: %s" % base, file=sys.stderr)
        return 1
    if not model.is_file():
        print("[ERROR] 模型文件不存在: %s" % model, file=sys.stderr)
        return 1

    subdirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not subdirs:
        print("[WARN] 未找到子文件夹: %s" % base)
        return 0

    plot_args = [] if args.no_plot else ["--plot"]
    ok, fail = 0, 0

    for d in subdirs:
        print("\n" + "=" * 60)
        print("[RUN] %s" % d)
        print("=" * 60)
        cmd = [
            sys.executable,
            str(root / "main.py"),
            "--mode", "batch_json_dir",
            "--model", str(model),
            "--batch_dir", str(d),
            *plot_args,
        ]
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
            print("[FAIL] %s exit=%s" % (d.name, r.returncode), file=sys.stderr)

    print("\n[DONE] 成功: %d  失败: %d" % (ok, fail))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
