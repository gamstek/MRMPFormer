"""Build, assemble and validate the Windows x64 SDK in build/windows."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile



ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
CPP = ROOT / "cpp" / "build"
EXTENSIONS = ROOT / "model" / "build" / "python" / "inference"
RUNTIME_REQUIREMENTS = ROOT / "cpp/requirements-runtime.txt"


def run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(list(map(str, args)), check=True, **kwargs)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def read_runtime_requirements(path=RUNTIME_REQUIREMENTS):
    """Read PEP 508 runtime roots from the deployment requirements file."""
    from packaging.requirements import Requirement

    requirements = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
            if requirement.url:
                raise ValueError("URL requirements cannot be checked against installed versions")
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            requirements.append(requirement)
    return requirements


def resolve_runtime_distributions(requirements):
    """Follow installed metadata, validating root/transitive versions and extras."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pending = list(requirements)
    processed = set()
    distributions = {}
    while pending:
        requirement = pending.pop()
        dist = metadata.distribution(requirement.name)
        if requirement.url or (requirement.specifier and dist.version not in requirement.specifier):
            raise RuntimeError(f"Unsatisfied runtime requirement: {requirement}; installed {dist.version}")
        canonical = canonicalize_name(dist.metadata["Name"])
        key = (canonical, frozenset(requirement.extras))
        if key in processed:
            continue
        processed.add(key)
        distributions[canonical] = dist
        for value in dist.requires or ():
            dependency = Requirement(value)
            if dependency.marker is None or any(dependency.marker.evaluate({"extra": extra})
                                               for extra in {"", *requirement.extras}):
                pending.append(dependency)
    return distributions


def copy_dependencies(destination: Path) -> dict[str, str]:
    """Copy the runtime dependency closure, excluding unrelated build packages."""
    distributions = resolve_runtime_distributions(read_runtime_requirements())
    versions = {}
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    for canonical, dist in distributions.items():
        versions[canonical] = dist.version
        for relative in dist.files or ():
            source = Path(dist.locate_file(relative)).resolve()
            if not source.is_relative_to(site) or not source.is_file():
                continue
            if any(part in {"__pycache__", "tests", "test", "benchmarks"} for part in source.parts) or source.suffix in {
                    ".pyc", ".whl", ".pdb", ".lib", ".a", ".h", ".hpp", ".c", ".cpp", ".pyx", ".pxd", ".pxi"}:
                continue
            copy_file(source, destination / source.relative_to(site))
    return versions


def assemble(stage: Path) -> dict[str, str]:
    try:
        metadata.distribution("onnxruntime")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("Remove CPU-only onnxruntime, then reinstall onnxruntime-gpu; they share import files")
    release = CPP / "Release"
    for name in ("mrmpformer.dll", "mrmpformer.lib", "python311.dll"):
        copy_file(release / name, stage / name)
    copy_file(ROOT / "cpp/include/mrmpformer.h", stage / "mrmpformer.h")
    copy_file(ROOT / "model/checkpoint/mrmpformerv2.onnx", stage / "mrmpformerv2.onnx")

    base = Path(sys.base_prefix)
    copy_file(base / "LICENSE.txt", stage / "licenses/Python-LICENSE.txt")
    for source in base.glob("*.dll"):
        copy_file(source, stage / source.name)
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installation = run(vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                       "-property", "installationPath", capture_output=True, text=True).stdout.strip()
    crt_dirs = sorted((Path(installation) / "VC/Redist/MSVC").glob("*/x64/Microsoft.VC*.CRT")) if installation else []
    if not crt_dirs:
        raise RuntimeError("Visual Studio x64 CRT redistributable directory is required for packaging")
    for source in crt_dirs[-1].glob("*.dll"):
        copy_file(source, stage / source.name)
    shutil.copytree(base / "DLLs", stage / "python/DLLs", ignore=shutil.ignore_patterns("*.pdb", "*.lib"))
    with zipfile.ZipFile(stage / "python311.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in (base / "Lib").rglob("*"):
            relative = source.relative_to(base / "Lib")
            if not source.is_file() or source.suffix == ".pyc":
                continue
            if any(part in {"site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter", "ensurepip"}
                   for part in relative.parts):
                continue
            archive.write(source, relative.as_posix())
    # CPython resolves this file relative to python311.dll, independently of the host EXE.
    (stage / "python311._pth").write_text(
        "python311.zip\npython/DLLs\npython\npython/Lib/site-packages\nimport site\n",
        encoding="utf-8",
    )
    for source in EXTENSIONS.glob("*.pyd"):
        copy_file(source, stage / "python/inference" / source.name)
    for name in ("__init__.py", "massnova_bridge.py", "two_round_detection.py"):
        copy_file(ROOT / "model/inference" / name, stage / "python/inference" / name)
    for folder in ("utils", "preprocessing"):
        # Helpers are shared source modules; exclude datasets, caches and generated artifacts.
        for source in (ROOT / "model" / folder).glob("*.py"):
            copy_file(source, stage / "python" / folder / source.name)
    versions = copy_dependencies(stage / "python/Lib/site-packages")
    cuda_provider = stage / "python/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll"
    if not cuda_provider.is_file():
        raise RuntimeError("Unified package must include onnxruntime_providers_cuda.dll")
    copy_file(ROOT / "docs/WINDOWS_INTEGRATION.md", stage / "README_INTEGRATION.md")
    copy_file(ROOT / "docs/WINDOWS_PACKAGE_CHANGES.md", stage / "CHANGELOG.md")
    return versions


def synthetic_items() -> list[dict]:
    x = [1.0 + i * 0.005 for i in range(401)]
    y = [10 + 10000 * math.exp(-((v - 1.5) / .06) ** 2)
         + 8000 * math.exp(-((v - 2.5) / .06) ** 2) for v in x]
    return [dict(uid="double", channel="100>50", mzq1=100, mzq3=50, x=x, y=y),
            dict(uid="low", channel="100>51", mzq1=100, mzq3=51, x=x, y=[1.0] * len(x))]


def validate(stage: Path) -> dict:
    items = synthetic_items()
    # The reference uses the normal .py implementation, not the bundled Cython modules.
    sys.path.insert(0, str(ROOT / "model"))
    from inference.massnova_runtime import MassNovaArrayRuntime
    import numpy as np
    reference = MassNovaArrayRuntime(ROOT / "model/checkpoint/mrmpformerv2.onnx", {
        "threshold": .5, "smooth_sigma": float(np.float32(.8)), "use_gpu": -1,
        "batch_size": 128, "min_chrom_points": 10, "min_max_intensity": 1000.0,
    }).process_items(items)
    # Relocate to a path containing spaces/non-ASCII characters and strip all developer paths.
    with tempfile.TemporaryDirectory(prefix="mrmpformer SDK 中文 ") as work:
        relocated = Path(work) / "runtime"
        shutil.copytree(stage, relocated)
        copy_file(CPP / "Release/batch_example.exe", relocated / "verify_runtime.exe")
        (Path(work) / "input.json").write_text(json.dumps({"items": items}), encoding="utf-8")
        env = os.environ.copy()
        for key in list(env):
            if key.upper().startswith(("PYTHON", "CONDA", "VIRTUAL_ENV", "MRMPFORMER")):
                env.pop(key)
        env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
        env["PYTHONHOME"] = str(Path(work) / "nonexistent-python")
        env["PYTHONPATH"] = str(Path(work) / "nonexistent-modules")
        env["MPLCONFIGDIR"] = str(Path(work) / "matplotlib")
        env["OPENBLAS_NUM_THREADS"] = "1"
        result = run(relocated / "verify_runtime.exe", "runtime/mrmpformerv2.onnx",
                     "input.json", "-1", cwd=work, env=env,
                     capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        actual = json.loads(result.stdout[result.stdout.index('{'):])
        if len(actual["items"][0]["peaks"]) != 2:
            raise RuntimeError(f"Expected two peaks: {actual}")
        maximum = 0.0
        for got, expected in zip(actual["items"], reference["items"], strict=True):
            if (got["uid"], got["status"], got["alerts"], len(got["peaks"])) != (
                    expected["uid"], expected["status"], expected["alerts"], len(expected["peaks"])):
                raise RuntimeError(f"C/Python result mismatch: {got}, {expected}")
            for a, b in zip(got["peaks"], expected["peaks"], strict=True):
                for key in ("a", "b", "c"):
                    maximum = max(maximum, abs(a[key] - b[key]))
        if maximum > 1e-12:
            raise RuntimeError(f"C/Python peak delta {maximum} exceeds 1e-12")
    return {"relocated_clean_environment": True, "channels": len(items),
            "peaks": 2, "max_absolute_delta": maximum, "provider": "CPUExecutionProvider"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="Package existing cpp/build and model/build outputs")
    parser.add_argument("--install-deps", action="store_true", help="Install build/runtime dependencies into the current interpreter")
    args = parser.parse_args()
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        raise SystemExit("Use Windows x64 CPython 3.11 with cpp/requirements-build.txt installed")
    if args.install_deps:
        try:
            metadata.distribution("onnxruntime")
            had_cpu = True
        except metadata.PackageNotFoundError:
            had_cpu = False
        if had_cpu:
            run(sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime")
        run(sys.executable, "-m", "pip", "install", "-r", ROOT / "cpp/requirements-build.txt")
        if had_cpu:
            # CPU/GPU distributions share files; repair GPU files after CPU removal.
            gpu_requirement = next(r for r in read_runtime_requirements() if r.name == "onnxruntime-gpu")
            run(sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(gpu_requirement))
    if not args.skip_build:
        run(sys.executable, ROOT / "model/tools/build_massnova_bridge.py", cwd=ROOT)
        run("cmake", "-S", ROOT / "cpp", "-B", CPP, "-DBUILD_TESTS=ON",
            f"-DPython3_EXECUTABLE={sys.executable}", f"-DPython3_ROOT_DIR={sys.prefix}", cwd=ROOT)
        run("cmake", "--build", CPP, "--config", "Release", cwd=ROOT)
        run("ctest", "--test-dir", CPP, "-C", "Release", "--output-on-failure", cwd=ROOT)
    if len(list(EXTENSIONS.glob("*.pyd"))) != 4:
        raise RuntimeError("Expected four Cython extensions in model/build/python/inference")
    BUILD.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".windows-stage-", dir=BUILD) as staging:
        stage = Path(staging) / "windows"
        stage.mkdir()
        print("Assembling private runtime and dependencies...", flush=True)
        versions = assemble(stage)
        print("Validating relocated package through the C API...", flush=True)
        report = validate(stage)
        # Fail packaging if the compiled C APIs diverge from the source Python pipeline.
        parity_output = ROOT / "tests/results"
        run(sys.executable, ROOT / "tests/run_parity.py",
            "--package", stage, "--output", parity_output, cwd=ROOT)
        run(sys.executable, ROOT / "tests/check_devices.py", "--package", stage,
            "--output", parity_output / "device_verification.json", cwd=ROOT)
        (parity_output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        files = {str(p.relative_to(stage)).replace('\\', '/'):
                 hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file()}
        (parity_output / "package_manifest.json").write_text(json.dumps({"python": sys.version.split()[0],
            "packages": versions, "files_sha256": files}, indent=2), encoding="utf-8")
        target = BUILD / "windows"
        if target.exists():
            backup = BUILD / ("windows.previous-" + time.strftime("%Y%m%d-%H%M%S"))
            target.rename(backup)
            print(f"Previous package preserved: {backup}")
        stage.rename(target)
    print(f"Validated integration package: {target}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
