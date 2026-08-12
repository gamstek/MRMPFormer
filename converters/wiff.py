"""
使用 msconvert.exe (ProteoWizard) 批量将 data/ 目录下的 .wiff 文件转换为 .mzML 格式。
注意: 项目路径不能包含中文字符。
"""

import subprocess
import os
import re
import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WIFF_BIN_DIR = BASE_DIR / "wiff_bin"
LEGACY_BIN_DIR = BASE_DIR / "bin"
BIN_DIR = WIFF_BIN_DIR if WIFF_BIN_DIR.exists() else LEGACY_BIN_DIR
MSCONVERT_EXE = BIN_DIR / "msconvert.exe"


def convert_file(input_file: Path, no_peak_picking: bool = False):
    """转换单个 .wiff 为 .mzML，输出到 data/<basename>/ 子目录中。
    返回 (成功标志, 信息字符串)"""
    output_dir = DATA_DIR / input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(MSCONVERT_EXE),
        str(input_file),
        "--mzML",
        "--continueOnError",
        "-o", str(output_dir),
    ]
    if not no_peak_picking:
        cmd += ["--filter", "peakPicking true 1-"]

    print(f"[msconvert] {input_file.name} ... ", end="", flush=True)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except Exception as e:
        print("CRASH")
        return False, f"进程异常退出: {e}"

    mzml_files = list(output_dir.glob("*.mzML"))
    if mzml_files:
        total_bytes = sum(f.stat().st_size for f in mzml_files)
        stderr = result.stderr.strip() if result.stderr else ""
        if "Conversion failed" in stderr:
            match = re.search(r"Conversion failed for (\d+) runs?", stderr)
            failed_runs = match.group(1) if match else "?"
            print(f"PARTIAL ({len(mzml_files)} 个 mzML, {total_bytes} bytes, {failed_runs} runs 失败)")
        else:
            print(f"OK ({len(mzml_files)} 个 mzML, {total_bytes} bytes)")
        return True, ""
    else:
        print("FAILED")
        stderr = result.stderr.strip() if result.stderr else ""
        stdout = result.stdout.strip() if result.stdout else ""
        reason = stderr or stdout or f"退码 {result.returncode}，未生成 mzML 文件"
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        return False, reason


def main():
    parser = argparse.ArgumentParser(description="批量转换 .wiff / .wiff2 为 .mzML")
    parser.add_argument("--input", type=str, default=None,
                        help="指定单个文件路径，不传则处理 data/*.wiff 和 data/*.wiff2")
    parser.add_argument("--no-peak-picking", action="store_true",
                        help="不做峰检测，保留原始 profile 数据")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出待处理文件，不执行转换")
    args = parser.parse_args()

    if not MSCONVERT_EXE.exists():
        print(f"错误: 未找到 {MSCONVERT_EXE}")
        sys.exit(1)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 文件不存在: {args.input}")
            sys.exit(1)
        files = [input_path]
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(DATA_DIR.glob("*.wiff")) + sorted(DATA_DIR.glob("*.wiff2"))

    if not files:
        print("未找到 .wiff / .wiff2 文件，请将文件放入 data/ 目录")
        return

    print(f"找到 {len(files)} 个文件\n")
    if args.dry_run:
        for f in files:
            print(f"  {f}")
        return

    success_list = []
    fail_list = []

    for f in files:
        ok, info = convert_file(f, no_peak_picking=args.no_peak_picking)
        if ok:
            success_list.append(f.name)
        else:
            fail_list.append((f.name, info))

    total = len(files)
    ok_count = len(success_list)
    fail_count = len(fail_list)

    print(f"\n{'='*40}")
    rate = ok_count / total * 100 if total > 0 else 0
    fail_rate = fail_count / total * 100 if total > 0 else 0
    print(f"成功: {ok_count}/{total} ({rate:.1f}%)  |  失败: {fail_count}/{total} ({fail_rate:.1f}%)")
    if fail_list:
        for name, reason in fail_list:
            print(f"  - {name}: {reason}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
