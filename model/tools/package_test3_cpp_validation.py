# -*- coding: utf-8 -*-
"""Build the self-contained test3 validation pack for the C API release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_ROOT.parent
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from inference.massnova import extract_full_xics
from inference.massnova_runtime import MassNovaArrayRuntime
from utils.mzml_load import load_ms_experiment


GOLDEN_SPECS = {
    "test3_1": (),
    "test3_3": (),
    "test3_7": (219, 388, 393, 397, 635, 641),
    "test3_56": (),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def q3_for_chromatogram(chrom) -> float:
    try:
        value = float(chrom.getProduct().getMZ())
    except Exception:
        value = 0.0
    return value if np.isfinite(value) else 0.0


def extract_items(mzml_path: Path):
    features, excluded = extract_full_xics(
        mzml_path,
        smooth_sigma=0.0,
        min_chrom_points=0,
        min_max_intensity=0.0,
        verbose=False,
    )
    experiment = load_ms_experiment(str(mzml_path), verbose=False)
    chromatograms = experiment.getChromatograms()
    items = []
    indices = []
    for feature in features:
        chrom_index = int(feature["chrom_index"])
        uid = str(feature["uid"])
        items.append({
            "uid": uid,
            "name": str(feature.get("compound_name", uid)),
            "channel": uid,
            "mzq1": float(feature["q1"]),
            "mzq3": q3_for_chromatogram(chromatograms[chrom_index]),
            # Zero means: use QfConfig.smooth_sigma (the release default is 0.8).
            "smooth_sigma": 0.0,
            "x": np.asarray(feature["rt"], dtype=np.float64).tolist(),
            "y": np.asarray(feature["intensity"], dtype=np.float64).tolist(),
        })
        indices.append(chrom_index)
    return items, indices, excluded


def stratified_positions(count: int, total: int = 64, required_indices=(), chrom_indices=()):
    if count <= total:
        return list(range(count))
    selected = set(int(round(value)) for value in np.linspace(0, count - 1, total))
    by_chrom = {int(chrom): position for position, chrom in enumerate(chrom_indices)}
    for chrom in required_indices:
        if int(chrom) in by_chrom:
            selected.add(by_chrom[int(chrom)])
    # Keep required cases, then fill/trim deterministically across the full range.
    ordered = sorted(selected)
    if len(ordered) > total:
        required_positions = {by_chrom[int(chrom)] for chrom in required_indices if int(chrom) in by_chrom}
        optional = [position for position in ordered if position not in required_positions]
        keep_optional = max(0, total - len(required_positions))
        if keep_optional and optional:
            picks = {optional[int(round(value))] for value in np.linspace(0, len(optional) - 1, keep_optional)}
        else:
            picks = set()
        ordered = sorted(required_positions | picks)
    return ordered[:total]


def write_compare_script(path: Path) -> None:
    content = r'''# -*- coding: utf-8 -*-
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
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_readme(path: Path, manifest: dict) -> None:
    text = f"""# test3 C++ 推理验收包

本包对应 Git 提交 `{manifest['git_commit']}`，用于验证新的 `mrmpformer.dll`
是否与 Python MassNova 数组入口返回相同的逐峰 `a/b/c`。

## 固定配置

- ONNX SHA256: `{manifest['model_sha256']}`
- threshold: `0.5`，严格执行 `score > threshold`
- smooth_sigma: `0.8`
- use_gpu: `-1`（金标准固定 CPU）
- batch_size: `128`
- min_chrom_points: `10`
- min_max_intensity: `1000.0`

`inputs/full/` 是 58 个样本的完整 DLL JSON 输入；每个 `item` 是一条独立
MRM transition，`x` 为分钟单位原始 RT，`y` 为未平滑强度。
`smooth_sigma=0` 表示使用全局 0.8，禁止调用方提前再次平滑。

`golden/inputs/` 与 `golden/expected/` 是严格数值验收集。先逐个调用
`qf_process`，将 `qf_get_result_json` 得到的纯 JSON 保存到文件，再执行：

```powershell
python tools/compare_cpp_result.py `
  golden/expected/test3_3_stratified64.expected.json `
  actual/test3_3_stratified64.actual.json `
  --atol 1e-6
```

必须满足：item 顺序、uid、status、峰数量完全一致；每组 a/b/c 的绝对误差
不超过 1e-6。信号兜底峰同样位于 `peaks[]` 且通道状态为 `ok`。

`source_mzml/` 只用于数据溯源；DLL 本身不解析 mzML。旧的
`output/inference/massnova_test3` 使用 PTH 和阈值 0.8，不能作为本包标准答案。
传统算法 Excel、置信度预警图和候选窗口图片不属于 C++ 推理验收范围。
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def zip_tree(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
            relative = file_path.relative_to(source.parent)
            compression = zipfile.ZIP_STORED if file_path.suffix.lower() == ".onnx" else zipfile.ZIP_DEFLATED
            archive.write(file_path, relative.as_posix(), compress_type=compression, compresslevel=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "mzml" / "test3")
    parser.add_argument("--output", type=Path, default=MODEL_ROOT / "project_review" / "test3_cpp_validation_1d4ba0f")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    (output / "inputs" / "full").mkdir(parents=True)
    (output / "source_mzml").mkdir(parents=True)

    model_path = MODEL_ROOT / "checkpoint" / "mrmpformerv2.onnx"
    config_path = MODEL_ROOT / "configs" / "massnova.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime_config = dict(config)
    runtime_config.update({
        "min_chrom_points": int(config["pipeline_min_chrom_points"]),
        "min_max_intensity": float(config["pipeline_min_max_intensity"]),
        "use_gpu": -1,
    })

    all_items = {}
    all_indices = {}
    sample_rows = []
    mzml_files = sorted(args.source.glob("test3_*.mzML"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(mzml_files) != 58:
        raise SystemExit(f"expected 58 mzML files, found {len(mzml_files)}")
    for number, mzml_path in enumerate(mzml_files, start=1):
        key = mzml_path.stem
        print(f"[extract {number:02d}/58] {key}", flush=True)
        items, indices, excluded = extract_items(mzml_path)
        dump_json(output / "inputs" / "full" / f"{key}.json", {"items": items})
        shutil.copy2(mzml_path, output / "source_mzml" / mzml_path.name)
        if key in GOLDEN_SPECS:
            all_items[key] = items
            all_indices[key] = indices
        sample_rows.append({
            "sample": key,
            "items": len(items),
            "front_end_excluded": len(excluded),
            "mzml_sha256": sha256(mzml_path),
        })

    print("[golden] loading ONNX once", flush=True)
    runtime = MassNovaArrayRuntime(model_path, runtime_config)
    golden_rows = []
    for key, required in GOLDEN_SPECS.items():
        items = all_items[key]
        positions = stratified_positions(
            len(items), 64, required_indices=required, chrom_indices=all_indices[key])
        selected = [items[position] for position in positions]
        input_name = f"{key}_stratified64.json"
        expected_name = f"{key}_stratified64.expected.json"
        print(f"[golden] {key}: {len(selected)} channels", flush=True)
        dump_json(output / "golden" / "inputs" / input_name, {"items": selected})
        expected = runtime.process_items(selected)
        dump_json(output / "golden" / "expected" / expected_name, expected)
        golden_rows.append({
            "sample": key,
            "input": f"golden/inputs/{input_name}",
            "expected": f"golden/expected/{expected_name}",
            "positions": positions,
            "chrom_indices": [all_indices[key][position] for position in positions],
            "items": len(selected),
            "peaks": sum(len(item["peaks"]) for item in expected["items"]),
            "ok": sum(item["status"] == "ok" for item in expected["items"]),
            "alert": sum(item["status"] == "alert" for item in expected["items"]),
        })

    (output / "model").mkdir(parents=True)
    (output / "config").mkdir(parents=True)
    (output / "include").mkdir(parents=True)
    (output / "examples").mkdir(parents=True)
    shutil.copy2(model_path, output / "model" / model_path.name)
    shutil.copy2(config_path, output / "config" / config_path.name)
    shutil.copy2(REPO_ROOT / "cpp" / "include" / "mrmpformer.h", output / "include" / "mrmpformer.h")
    shutil.copy2(REPO_ROOT / "cpp" / "examples" / "batch_example.c", output / "examples" / "batch_example.c")
    write_compare_script(output / "tools" / "compare_cpp_result.py")

    manifest = {
        "package": output.name,
        "git_commit": "1d4ba0f966a14ef1b2058ebe4f2284447cb8fba2",
        "model_sha256": sha256(model_path),
        "config_sha256": sha256(config_path),
        "model": "model/mrmpformerv2.onnx",
        "config": {
            "threshold": 0.5,
            "threshold_semantics": "score > threshold",
            "smooth_sigma": 0.8,
            "use_gpu": -1,
            "batch_size": 128,
            "min_chrom_points": 10,
            "min_max_intensity": 1000.0,
        },
        "samples": sample_rows,
        "golden_sets": golden_rows,
    }
    dump_json(output / "manifest.json", manifest)
    write_readme(output / "README.md", manifest)

    files = sorted(path for path in output.rglob("*") if path.is_file())
    with (output / "hashes.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for file_path in files:
            handle.write(f"{sha256(file_path)}  {file_path.relative_to(output).as_posix()}\n")

    if args.zip:
        zip_path = output.with_suffix(".zip")
        print(f"[zip] {zip_path}", flush=True)
        zip_tree(output, zip_path)
        print(f"[done] {zip_path} sha256={sha256(zip_path)}", flush=True)
    else:
        print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
