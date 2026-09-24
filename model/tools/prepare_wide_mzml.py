"""Convert wide-peak raw files from copies, leaving data/wide_data untouched.

Run from the repository root with the build environment's Python:
    python model/tools/prepare_wide_mzml.py
"""

from __future__ import annotations

import shutil
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from converters.msdata import convert_file  # noqa: E402
from tools.sanitize_msdata_mzml import sanitize  # noqa: E402


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_one(raw_name, copy_name, expected_count, output):
    source = ROOT / "data/wide_data" / raw_name
    if not source.is_file():
        raise FileNotFoundError(source)
    raw_copy = output / "raw_copy" / copy_name
    raw_copy.parent.mkdir(parents=True, exist_ok=True)
    if not raw_copy.exists():
        shutil.copy2(source, raw_copy)
    elif sha256(raw_copy) != sha256(source):
        raise ValueError(f"existing working copy differs from source: {raw_copy}")

    converted_dir = output / "mzml" / raw_copy.stem
    converted = sorted(converted_dir.glob("*.mzML"))
    if len(converted) != expected_count:
        if converted:
            raise ValueError(f"partial conversion in {converted_dir}: {len(converted)} files")
        ok, reason = convert_file(raw_copy.resolve(), (output / "mzml").resolve(), timeout=1200)
        if not ok:
            raise RuntimeError(f"conversion failed for {raw_name}: {reason}")
        converted = sorted(converted_dir.glob("*.mzML"))
    if len(converted) != expected_count:
        raise ValueError(f"expected {expected_count} mzML files for {raw_name}, got {len(converted)}")

    clean_dir = output / "clean_mzml" / raw_name.removesuffix(".msdata")
    clean_dir.mkdir(parents=True, exist_ok=True)
    replacements = 0
    for path in converted:
        target = clean_dir / path.name
        if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
            replacements += sanitize(path, target)
    print(f"{raw_name}: {len(converted)} mzML files ready; replaced {replacements} invalid UTF-8 bytes")


def main():
    output = ROOT / "output/wide_finetune"
    # This converter exited abnormally on the original 20260919 filename,
    # but succeeded on an identical short-named copy (SHA-256 verified).
    prepare_one("20260919-01.msdata", "a.msdata", 62, output)
    prepare_one("20260910-01.msdata", "20260910-01.msdata", 9, output)
    existing = ROOT / "data/mzml/20251111-01/20251111-01_1.mzML"
    if not existing.is_file():
        raise FileNotFoundError(f"existing 20251111 conversion not found: {existing}")
    print(f"20251111-01.msdata: using existing read-only mzML {existing}")


if __name__ == "__main__":
    main()
