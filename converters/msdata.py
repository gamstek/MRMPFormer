"""
使用 msdata2mzml.exe 批量将 data/msdata/ 目录下的 .msdata 文件转换为 .mzML 格式，
转换结果统一输出到 data/mzml/ 目录（按源文件名分目录存放）。
注意: 项目路径不能包含中文字符，否则 OpenMS C++ 层会报路径不存在。

用法（项目根目录下）：
  # 全部（data/msdata/*.msdata）
  python converters/msdata.py
  # 指定单个文件
  python converters/msdata.py --input data/msdata/20260818_transform.msdata
  # 指定目录（递归扫描其中的 *.msdata，批量转换某个实验）
  python converters/msdata.py --input data/msdata/20260818_transform
  # 仅预览将转换哪些文件
  python converters/msdata.py --dry-run --input data/msdata/20260818_transform
"""

import subprocess
import os
import argparse
import shutil
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "msdata"
OUTPUT_DIR = BASE_DIR.parent / "data" / "mzml"
MSDATA_BIN_DIR = BASE_DIR / "msdata_bin"
LEGACY_BIN_DIR = BASE_DIR / "bin"
BIN_DIR = MSDATA_BIN_DIR if MSDATA_BIN_DIR.exists() else LEGACY_BIN_DIR
MSDATA2MZML_EXE = BIN_DIR / "msdata2mzml.exe"
OPENMS_SHARE = BIN_DIR / "share" / "OpenMS"


def convert_file(input_file: Path):
    """转换单个 .msdata 为 .mzML。exe 会在输入文件同级生成 <stem>/ 子目录，
    脚本随后将其中生成的 .mzML 移动到 data/mzml/<stem>/ 统一输出。
    返回 (成功标志, 信息字符串)"""
    env = os.environ.copy()
    env["OPENMS_DATA_PATH"] = str(OPENMS_SHARE)

    rel_input = os.path.relpath(input_file, BIN_DIR)
    cmd = [str(MSDATA2MZML_EXE), rel_input]

    print(f"[msdata2mzml] {input_file.name} ... ", end="", flush=True)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env=env, cwd=str(BIN_DIR),
        )
    except Exception as e:
        print("CRASH")
        return False, f"进程异常退出: {e}"

    # exe 输出到输入文件同级目录下的 <stem>/ 子目录
    exe_output_dir = input_file.parent / input_file.stem
    mzml_files = list(exe_output_dir.glob("*.mzML")) if exe_output_dir.exists() else []

    if mzml_files:
        # 统一移动到 data/mzml/<stem>/
        target_dir = OUTPUT_DIR / input_file.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        for mzml in mzml_files:
            shutil.move(str(mzml), str(target_dir / mzml.name))
        # msdata2mzml.exe 会附带生成 *.mzML.json（元信息），本脚本只保留 mzML：转换后清理 json 并移除空目录
        json_left = list(exe_output_dir.glob("*.json")) if exe_output_dir.exists() else []
        for j in json_left:
            try:
                j.unlink()
                print(f"[cleanup] 删除附带 json: {j.name}")
            except OSError as e:
                print(f"[WARN] 无法删除附带 json {j.name}: {e}")
        if exe_output_dir.exists():
            try:
                exe_output_dir.rmdir()  # 仅删空目录；非空（未知残留）保留并提示
                if not json_left:
                    print(f"[cleanup] 移除空目录: {exe_output_dir.name}")
            except OSError:
                print(f"[INFO] 目录非空，保留: {exe_output_dir}")
        total_bytes = sum(f.stat().st_size for f in target_dir.glob("*.mzML"))
        print(f"OK ({len(mzml_files)} 个 mzML, {total_bytes} bytes)")
        return True, ""
    else:
        print("FAILED")
        stderr = result.stderr.strip() if result.stderr else ""
        stdout = result.stdout.strip() if result.stdout else ""
        reason = stderr or stdout or f"退码 {result.returncode}，未生成 mzML 文件"
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False, reason


def collect_msdata_files(input_arg):
    """收集待转换的 .msdata 文件：
    --input 缺省 → data/msdata/*.msdata（顶层全部）；
    --input 为文件 → 该文件；
    --input 为目录 → 递归扫描目录内 *.msdata（可指定某个实验的批量文件）。"""
    if input_arg:
        p = Path(input_arg)
        if p.is_file():
            return [p]
        if p.is_dir():
            files = sorted(p.rglob("*.msdata"))
            if not files:
                print(f"错误: 目录内未找到 .msdata 文件: {p}")
                sys.exit(1)
            return files
        print(f"错误: 路径不存在: {input_arg}")
        sys.exit(1)
    return sorted(DATA_DIR.glob("*.msdata"))


def main():
    parser = argparse.ArgumentParser(description="批量转换 .msdata 为 .mzML")
    parser.add_argument("--input", type=str, default=None,
                        help="单个 .msdata 文件路径，或目录（递归扫描 *.msdata 批量转换）；缺省扫描 data/msdata/*.msdata")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not MSDATA2MZML_EXE.exists():
        print("错误: 未找到 bin/msdata2mzml.exe，请将 msdata2mzml_20260608_1/Release/ 下的所有文件复制到 bin/")
        return

    files = collect_msdata_files(args.input)

    if not files:
        print("未找到 .msdata 文件，请将文件放入 data/msdata/ 目录")
        return

    print(f"找到 {len(files)} 个文件\n")
    if args.dry_run:
        for f in files:
            print(f"  {f}")
        return

    success_list = []
    fail_list = []  # (filename, reason)

    for f in files:
        ok, info = convert_file(f)
        if ok:
            success_list.append(f.name)
        else:
            fail_list.append((f.name, info))

    total = len(files)
    ok_count = len(success_list)
    fail_count = len(fail_list)

    print(f"\n{'='*40}")
    print(f"成功: {ok_count}/{total} ({ok_count/total*100:.1f}%)  |  失败: {fail_count}/{total} ({fail_count/total*100:.1f}%)")
    if fail_list:
        for name, reason in fail_list:
            print(f"  - {name}: {reason}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
