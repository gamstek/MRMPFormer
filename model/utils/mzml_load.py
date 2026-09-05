# -*- coding: utf-8 -*-
"""Load mzML via pyopenms with Windows path and invalid UTF-8 repair."""
import os
import sys
import tempfile
from pathlib import Path

from pyopenms import MSExperiment, MzMLFile


def is_valid_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _repair_invalid_utf8_bytes(raw: bytes) -> bytes:
    """局部解码修复：ASCII 段原样保留；非 ASCII 连续段依次试 UTF-8 → GBK → GB18030 解码，
    统一以 UTF-8 重新编码。这样既保留文件中本已合法的 UTF-8 中文（样品名/色谱 id），
    也能正确还原仪器写进 userParam/method_path 的 GBK 中文（不再被替换成 �）。"""
    out = bytearray()
    i, n = 0, len(raw)
    while i < n:
        if raw[i] < 0x80:
            out.append(raw[i])
            i += 1
            continue
        j = i
        while j < n and raw[j] >= 0x80:
            j += 1
        run = raw[i:j]
        decoded = None
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                decoded = run.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            decoded = run.decode("utf-8", errors="replace")
        out.extend(decoded.encode("utf-8"))
        i = j
    return bytes(out)


def repair_mzml_bytes_for_openms(raw: bytes) -> bytes:
    """Replace invalid UTF-8 bytes so OpenMS XML parser can read the file (GBK preserved)."""
    if is_valid_utf8(raw):
        return raw
    print(
        "[WARN] mzML has non-UTF-8 bytes (often GBK in userParam/method_path); "
        "decoding GBK runs and re-encoding to UTF-8 before load"
    )
    return _repair_invalid_utf8_bytes(raw)


def _mzml_load_ok(exp: MSExperiment, path_str: str) -> bool:
    try:
        MzMLFile().load(path_str, exp)
        return True
    except RuntimeError:
        return False


def load_ms_experiment(mzml_path, verbose=True):
    """
    Load mzML into MSExperiment.
    Tries: long path -> Windows short path -> temp copy -> UTF-8 repaired temp copy.
    """
    path = Path(mzml_path).resolve()
    if not path.is_file():
        raise FileNotFoundError("mzML not found: %s" % path)
    resolved = str(path)
    raw = path.read_bytes()

    def _try_load(path_str, label):
        exp = MSExperiment()
        if _mzml_load_ok(exp, path_str):
            if verbose:
                print("[INFO] MzML %s: %s" % (label, path_str))
            return exp
        return None

    # 内容预检：raw 非法 UTF-8 时原路径/临时副本（字节相同）必然被 OpenMS 拒绝，
    # 且 C++ 层会往 stderr 刷告警 —— 直接跳到 UTF-8 修复路径，省 2-3 次注定失败的尝试
    valid = is_valid_utf8(raw)

    if valid:
        exp = _try_load(resolved, "loaded via long path")
        if exp is not None:
            return exp

        if sys.platform == "win32":
            try:
                import ctypes

                buf = ctypes.create_unicode_buffer(4096)
                if ctypes.windll.kernel32.GetShortPathNameW(resolved, buf, 4096):
                    short = buf.value
                    if short and short != resolved and Path(short).is_file():
                        exp = _try_load(short, "loaded via short path")
                        if exp is not None:
                            return exp
                    elif short and verbose:
                        print("[WARN] short path not usable, skipped: %s" % short)
            except Exception as ex:
                if verbose:
                    print("[WARN] GetShortPathNameW failed: %s" % ex)

    fd, tmp_path = tempfile.mkstemp(suffix=".mzML", prefix="mzml_")
    os.close(fd)
    try:
        if valid:
            # 路径问题兜底：ASCII 临时副本装原始字节再试一次
            Path(tmp_path).write_bytes(raw)
            exp = _try_load(tmp_path, "loaded via temp copy")
            if exp is not None:
                return exp

        repaired = repair_mzml_bytes_for_openms(raw)
        if repaired != raw:
            Path(tmp_path).write_bytes(repaired)
            exp = _try_load(tmp_path, "loaded via UTF-8 repaired temp copy")
            if exp is not None:
                return exp

        raise RuntimeError(
            "MzMLFile.load failed (long path, short path, temp copy, UTF-8 repair): %s"
            % resolved
        )
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


def _load_ms_experiment_mzml(mzml_path: Path):
    exp = load_ms_experiment(mzml_path, verbose=True)
    return exp, None


def fix_mzml_encoding(root_dir):
    """一次性批量修复：将 root_dir 递归下所有 *.mzML 的非 UTF-8 字节就地转正为 UTF-8。

    修复后文件满足 is_valid_utf8，后续加载直接走长路径，不再触发 WARN / 临时副本。
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError("not a directory: %s" % root)
    fixed, already_valid = [], 0
    for p in root.rglob("*"):
        if p.suffix.lower() != ".mzml":
            continue
        raw = p.read_bytes()
        if is_valid_utf8(raw):
            already_valid += 1
            continue
        repaired = _repair_invalid_utf8_bytes(raw)
        if repaired == raw:
            already_valid += 1
            continue
        p.write_bytes(repaired)
        fixed.append(str(p))
    print("[FIX] %d file(s) repaired, %d already valid UTF-8" % (len(fixed), already_valid))
    for f in fixed:
        print("  repaired: %s" % f)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="mzML 编码修复工具")
    ap.add_argument(
        "--fix-dir", metavar="DIR",
        help="递归修复该目录下所有 mzML 的非 UTF-8 字节（就地写回 UTF-8）",
    )
    args = ap.parse_args(argv)
    if args.fix_dir:
        fix_mzml_encoding(args.fix_dir)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
