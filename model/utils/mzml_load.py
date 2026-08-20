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


def repair_mzml_bytes_for_openms(raw: bytes) -> bytes:
    """Replace invalid UTF-8 bytes so OpenMS XML parser can read the file."""
    if is_valid_utf8(raw):
        return raw
    print(
        "[WARN] mzML has non-UTF-8 bytes (often GBK in userParam/method_path); "
        "replacing invalid sequences before load"
    )
    return raw.decode("utf-8", errors="replace").encode("utf-8")


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
