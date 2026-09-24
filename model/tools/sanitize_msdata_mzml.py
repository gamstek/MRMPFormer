"""Repair mixed-encoding text emitted by msdata2mzml in derived mzML files.

The converter occasionally writes legacy-encoded metadata bytes into an XML
document declared as UTF-8. Binary chromatogram arrays are base64 ASCII and
are not changed. The original .msdata and converter output remain untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def sanitize(source: Path, destination: Path) -> int:
    raw = source.read_bytes()
    decoded = raw.decode("utf-8", errors="replace")
    replacement_count = decoded.count("\ufffd")
    if replacement_count:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(decoded.encode("utf-8"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return replacement_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Converter mzML file or directory")
    parser.add_argument("destination", type=Path, help="Separate output file or directory")
    args = parser.parse_args()
    paths = sorted(args.source.glob("*.mzML")) if args.source.is_dir() else [args.source]
    if not paths:
        parser.error("no mzML files found")
    for path in paths:
        target = args.destination / path.name if args.source.is_dir() else args.destination
        if path.resolve() == target.resolve():
            parser.error("source and destination must differ")
        count = sanitize(path, target)
        print(f"{path.name}: {count} invalid UTF-8 bytes replaced -> {target}")


if __name__ == "__main__":
    main()
