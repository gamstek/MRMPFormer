# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expected")
    parser.add_argument("actual")
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    expected = load(args.expected)["items"]
    actual = load(args.actual)["items"]
    errors = []
    if len(expected) != len(actual):
        errors.append(f"item count: expected={len(expected)} actual={len(actual)}")
    for index, (left, right) in enumerate(zip(expected, actual)):
        prefix = f"items[{index}]"
        for field in ("uid", "status"):
            if left.get(field) != right.get(field):
                errors.append(f"{prefix}.{field}: expected={left.get(field)!r} actual={right.get(field)!r}")
        lp, rp = left.get("peaks", []), right.get("peaks", [])
        if len(lp) != len(rp):
            errors.append(f"{prefix}.peaks count: expected={len(lp)} actual={len(rp)}")
            continue
        for peak_index, (a, b) in enumerate(zip(lp, rp)):
            for field in ("a", "b", "c"):
                difference = abs(float(a[field]) - float(b[field]))
                if difference > args.atol:
                    errors.append(f"{prefix}.peaks[{peak_index}].{field}: diff={difference:.12g}")
    if errors:
        print("FAIL")
        print("\n".join(errors[:100]))
        raise SystemExit(1)
    print(f"PASS: {len(expected)} items match within atol={args.atol:g}")


if __name__ == "__main__":
    main()
