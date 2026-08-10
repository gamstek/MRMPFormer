#!/usr/bin/env python3
"""
MRMPFormer 环境依赖检测脚本
=============================
用法: python check_env.py [--json] [--quiet] [--target-env NAME]

输出结构化检测报告，覆盖:
  - Conda 环境（当前环境名称、是否为 quanformer）
  - Python 版本
  - pip 包依赖 (基于 model/requirements.txt)
  - PyTorch / CUDA / MPS 可用性 + GPU 架构匹配
  - R 运行时 + Bioconductor 包 (MSnbase, xcms)
  - 模型权重文件
  - 磁盘空间

选项:
  --json        以 JSON 格式输出结果
  --quiet       仅输出失败项（静默模式）
  --target-env   指定要检测的 conda 环境名称（在该环境中运行检测）
  --help        显示帮助
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

# 项目根目录（脚本位于 .github/skills/check-dependencies/ → 向上 4 级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# requirements.txt 路径
REQUIREMENTS_PATH = PROJECT_ROOT / "model" / "requirements.txt"

# 模型权重路径
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "checkpoint" / "checkpoint0029.pth"

# 期望的 Python 版本范围
PYTHON_MIN = (3, 10)
PYTHON_MAX = (3, 11)

# 磁盘空间阈值 (GB)
DISK_MIN_GB = 2

# 期望的模型文件最小大小 (MB)
CHECKPOINT_MIN_MB = 300

# GPU 架构 → 期望 PyTorch 版本映射
GPU_ARCH_RULES = {
    (12,): {"min_torch": "2.7.0", "cuda_tag": "cu128", "label": "RTX 50 系 (Blackwell)"},
    (8, 9): {"min_torch": "2.6.0", "cuda_tag": "cu124", "label": "RTX 40 系 (Ada Lovelace)"},
    (8, 6): {"min_torch": "2.6.0", "cuda_tag": "cu124", "label": "RTX 30 系 (Ampere)"},
    (7, 5): {"min_torch": "2.6.0", "cuda_tag": "cu124", "label": "RTX 20 / GTX 16 系 (Turing)"},
}

# R 需要检查的包
R_PACKAGES = ["MSnbase", "xcms"]

# 默认 conda 环境名
DEFAULT_CONDA_ENV = "mrmpformer"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def run_cmd(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """运行命令，返回 (returncode, stdout, stderr).

    强制使用 UTF-8 编码，避免 Windows 下 GBK/管道编码错配导致中文乱码。
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=(sys.platform == "win32"),
            encoding="utf-8",
            errors="replace",
        )
        stdout = proc.stdout.strip() if proc.stdout else ""
        stderr = proc.stderr.strip() if proc.stderr else ""
        return proc.returncode, stdout, stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"
    except subprocess.TimeoutExpired:
        return -2, "", f"Command timed out: {cmd[0] if isinstance(cmd, list) else cmd}"


def parse_requirements(path: Path) -> Dict[str, str]:
    """
    解析 requirements.txt，返回 {package_name: version_spec}.
    处理注释、--index-url、空行。
    """
    packages: Dict[str, str] = {}
    if not path.exists():
        return packages

    content = path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        # 跳过注释、空行、--index-url
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        # 解析包名和版本
        # 支持格式: pkg==1.0, pkg>=1.0, pkg>=1.0,<2.0, pkg>=1.0 ; python_version >= "3.10"
        match = re.match(r"^([a-zA-Z0-9_\-\.]+)\s*((?:[><=!~]+\s*[0-9\.\*]+(?:\s*,\s*[><=!~]+\s*[0-9\.\*]+)*)?)", line)
        if match:
            name = match.group(1).lower().replace("_", "-")
            spec = match.group(2).strip().replace(" ", "") if match.group(2) else "*"
            packages[name] = spec
    return packages


def parse_pip_list() -> Dict[str, str]:
    """通过 pip list --format=json 获取已安装包 {name: version}."""
    ret, out, err = run_cmd([sys.executable, "-m", "pip", "list", "--format=json"])
    if ret != 0:
        # 尝试用 --format=columns 回退
        ret2, out2, _ = run_cmd([sys.executable, "-m", "pip", "list", "--format=columns"])
        if ret2 != 0:
            return {}
        packages: Dict[str, str] = {}
        for line in out2.splitlines()[2:]:  # 跳过表头
            parts = line.split()
            if len(parts) >= 2:
                packages[parts[0].lower().replace("_", "-")] = parts[1]
        return packages

    try:
        data = json.loads(out)
        return {item["name"].lower().replace("_", "-"): item["version"] for item in data}
    except json.JSONDecodeError:
        return {}


def version_tuple(v: str) -> Tuple[int, ...]:
    """将版本字符串转为可比较的元组，如 '2.7.0+cu128' → (2, 7, 0)."""
    v = v.split("+")[0].split("-")[0]  # 去掉 +cu128, -rc1 等后缀
    parts = v.split(".")
    return tuple(int(p) for p in parts if p.isdigit())


def check_version_satisfied(installed: str, spec: str) -> bool:
    """检查已安装版本是否满足 requirements.txt 中的约束."""
    if spec == "*":
        return True
    if not installed:
        return False

    iv = version_tuple(installed)

    # 处理逗号分隔的多重约束，如 ">=6.7.0,<6.8.0"
    constraints = [c.strip() for c in spec.split(",") if c.strip()]
    for constraint in constraints:
        m = re.match(r"^([><=!~]+)\s*([0-9\.]+)$", constraint)
        if not m:
            continue
        op, target = m.group(1), m.group(2)
        tv = version_tuple(target)
        if op == "==" and iv != tv:
            return False
        elif op == ">=" and iv < tv:
            return False
        elif op == ">" and iv <= tv:
            return False
        elif op == "<=" and iv > tv:
            return False
        elif op == "<" and iv >= tv:
            return False
        elif op == "!=" and iv == tv:
            return False
        elif op == "~=":
            # ~=X.Y 表示 >=X.Y, ==X.*
            if iv < tv:
                return False
            if len(tv) >= 2 and iv[:2] != tv[:2]:
                return False
    return True


def detect_conda() -> Dict[str, Any]:
    """检测 conda 环境信息."""
    result: Dict[str, Any] = {
        "conda_available": False,
        "in_conda_env": False,
        "env_name": None,
        "env_prefix": None,
        "is_mrmpformer_env": False,
        "all_envs": [],
    }

    # 检查 conda 是否可用
    ret, out, _ = run_cmd(["conda", "--version"])
    if ret != 0:
        return result
    result["conda_available"] = True

    # 获取当前环境名 (通过 CONDA_DEFAULT_ENV 或 conda info)
    env_name = os.environ.get("CONDA_DEFAULT_ENV", None)
    env_prefix = os.environ.get("CONDA_PREFIX", None)

    if env_name:
        result["in_conda_env"] = True
        result["env_name"] = env_name
        result["env_prefix"] = env_prefix
        result["is_mrmpformer_env"] = (env_name == DEFAULT_CONDA_ENV)
    else:
        # 尝试通过 conda info 获取
        ret2, out2, _ = run_cmd(["conda", "info", "--json"])
        if ret2 == 0:
            try:
                info = json.loads(out2)
                active_prefix = info.get("active_prefix_name") or info.get("active_prefix")
                if active_prefix:
                    result["in_conda_env"] = True
                    if isinstance(active_prefix, str) and not active_prefix.startswith("/"):
                        result["env_name"] = active_prefix
                    else:
                        result["env_name"] = os.path.basename(str(active_prefix))
                    result["is_mrmpformer_env"] = (result["env_name"] == DEFAULT_CONDA_ENV)
            except json.JSONDecodeError:
                pass

    # 获取所有 conda 环境列表
    ret3, out3, _ = run_cmd(["conda", "env", "list"])
    if ret3 == 0:
        for line in out3.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 格式: env_name  /path/to/env  或  env_name  *  /path/to/env
            parts = line.split()
            if parts:
                name = parts[0]
                if name not in ("base",) and name not in result["all_envs"]:
                    # 跳过路径行（以 / 或 \ 或盘符开头的）
                    if not re.match(r"^[A-Za-z]:|^/|^\\", name):
                        result["all_envs"].append(name)

    return result


def detect_gpu() -> Optional[Dict[str, Any]]:
    """检测 NVIDIA GPU 信息."""
    ret, out, _ = run_cmd(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    if ret != 0 or not out:
        return None
    parts = out.split(",")
    if len(parts) < 2:
        return None
    name = parts[0].strip()
    cc_str = parts[1].strip()
    try:
        cc = tuple(int(x) for x in cc_str.split("."))
    except ValueError:
        cc = ()
    return {"name": name, "compute_capability": cc, "cc_str": cc_str}


def get_gpu_arch_rule(cc: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
    """根据计算能力返回对应的架构规则."""
    if not cc:
        return None
    major = cc[0]
    minor = cc[1] if len(cc) > 1 else 0
    # 检查精确匹配
    for cc_key, rule in GPU_ARCH_RULES.items():
        if major == cc_key[0]:
            if len(cc_key) == 1:
                return rule  # 仅按主版本号匹配 (如 12.x → RTX 50)
            if len(cc_key) >= 2 and minor == cc_key[1]:
                return rule
    # 回退：按主版本号大致匹配
    if major >= 12:
        return GPU_ARCH_RULES[(12,)]
    elif major >= 8:
        return GPU_ARCH_RULES[(8, 9)]
    elif major >= 7:
        return GPU_ARCH_RULES[(7, 5)]
    return None


def detect_torch() -> Dict[str, Any]:
    """检测 PyTorch 信息."""
    code = """
import sys
try:
    import torch
    ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else 'N/A'
    print(f"VERSION:{ver}")
    print(f"CUDA:{cuda_avail}")
    print(f"MPS:{mps_avail}")
    print(f"GPU_NAME:{gpu_name}")
except ImportError:
    print("VERSION:NOT_INSTALLED")
    print("CUDA:False")
    print("MPS:False")
    print("GPU_NAME:N/A")
"""
    ret, out, _ = run_cmd([sys.executable, "-c", code])
    result: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "mps_available": False,
        "gpu_name": "N/A",
    }
    if ret != 0:
        return result
    for line in out.splitlines():
        if line.startswith("VERSION:"):
            ver = line.split(":", 1)[1].strip()
            if ver != "NOT_INSTALLED":
                result["installed"] = True
                result["version"] = ver
        elif line.startswith("CUDA:"):
            result["cuda_available"] = line.split(":", 1)[1].strip() == "True"
        elif line.startswith("MPS:"):
            result["mps_available"] = line.split(":", 1)[1].strip() == "True"
        elif line.startswith("GPU_NAME:"):
            result["gpu_name"] = line.split(":", 1)[1].strip()
    return result


def check_r() -> Dict[str, Any]:
    """检测 R 运行时和包."""
    result: Dict[str, Any] = {"installed": False, "version": None, "packages": {}}

    # 检测 R 是否安装
    ret, out, _ = run_cmd(["R", "--version"])
    if ret != 0:
        return result

    result["installed"] = True
    for line in out.splitlines():
        if "R version" in line:
            m = re.search(r"(\d+\.\d+\.\d+)", line)
            if m:
                result["version"] = m.group(1)
            break

    # 检测 R 包
    for pkg in R_PACKAGES:
        ret, out, _ = run_cmd(
            ["R", "-e", f'cat(if("{pkg}" %in% rownames(installed.packages())) "YES" else "NO")']
        )
        result["packages"][pkg] = (ret == 0 and "YES" in out)

    return result


def check_checkpoint() -> Dict[str, Any]:
    """检测模型权重文件."""
    result: Dict[str, Any] = {"exists": False, "size_mb": 0, "path": str(CHECKPOINT_PATH)}
    if CHECKPOINT_PATH.exists():
        result["exists"] = True
        result["size_mb"] = round(CHECKPOINT_PATH.stat().st_size / (1024 * 1024), 1)
    return result


def check_disk() -> Dict[str, Any]:
    """检测磁盘空间."""
    target = CHECKPOINT_PATH if CHECKPOINT_PATH.parent.exists() else PROJECT_ROOT
    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "total_gb": round(usage.total / (1024**3), 1),
        "used_gb": round(usage.used / (1024**3), 1),
        "free_gb": round(usage.free / (1024**3), 1),
    }


# ---------------------------------------------------------------------------
# 主检测逻辑
# ---------------------------------------------------------------------------

class CheckResult:
    """单项检测结果."""
    def __init__(self, category: str, item: str, expected: str, actual: str,
                 passed: bool, required: bool = True, fix: str = ""):
        self.category = category
        self.item = item
        self.expected = expected
        self.actual = actual
        self.passed = passed
        self.required = required  # True=强制, False=条件必需
        self.fix = fix


def run_all_checks() -> List[CheckResult]:
    """运行全部检测，返回结果列表."""
    results: List[CheckResult] = []

    # ---- Conda 环境 ----
    conda_info = detect_conda()
    if conda_info["conda_available"]:
        if conda_info["in_conda_env"]:
            env_label = conda_info["env_name"] or "(未知)"
            is_quanformer = conda_info["is_mrmpformer_env"]
            results.append(CheckResult(
                "Conda", "当前环境",
                DEFAULT_CONDA_ENV,
                env_label,
                is_quanformer,
                required=False,
                fix=f"建议创建名为 '{DEFAULT_CONDA_ENV}' 的 conda 环境: conda create -n {DEFAULT_CONDA_ENV} python=3.11"
                    if not is_quanformer else ""
            ))
            # 列出所有相关环境
            other_envs = [e for e in conda_info["all_envs"] if e != env_label]
            if other_envs:
                results.append(CheckResult(
                    "Conda", "其他环境",
                    "—",
                    ", ".join(other_envs[:5]) + ("..." if len(other_envs) > 5 else ""),
                    True,
                    required=False
                ))
        else:
            results.append(CheckResult(
                "Conda", "Conda 环境",
                f"已激活 conda 环境",
                "未激活任何 conda 环境（在 base 中）",
                True,
                required=False,
                fix=f"建议创建并激活 '{DEFAULT_CONDA_ENV}': conda create -n {DEFAULT_CONDA_ENV} python=3.11 && conda activate {DEFAULT_CONDA_ENV}"
            ))
    else:
        results.append(CheckResult(
            "Conda", "Conda",
            "已安装",
            "未安装或不在 PATH",
            True,
            required=False,
            fix="建议安装 Miniconda/Anaconda 以管理 Python 环境"
        ))

    # ---- Python 版本 ----
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = PYTHON_MIN <= sys.version_info[:2] <= PYTHON_MAX
    results.append(CheckResult(
        "Python", "版本",
        f"{PYTHON_MIN[0]}.{PYTHON_MIN[1]} ~ {PYTHON_MAX[0]}.{PYTHON_MAX[1]}",
        py_ver,
        py_ok,
        fix="请安装 Python 3.10 或 3.11，不要使用 3.8 或 3.12+"
    ))

    # ---- pip 包依赖 ----
    required_pkgs = parse_requirements(REQUIREMENTS_PATH)
    installed_pkgs = parse_pip_list()

    if not required_pkgs:
        results.append(CheckResult(
            "配置文件", "requirements.txt",
            f"存在于 {REQUIREMENTS_PATH}",
            "文件不存在或无法解析",
            False,
            fix=f"确认 {REQUIREMENTS_PATH} 存在且格式正确"
        ))
    else:
        # torch/torchvision 由 PyTorch 专项检测处理，此处跳过
        _PYTORCH_SKIP = {"torch", "torchvision"}
        for pkg_name, spec in required_pkgs.items():
            pkg_key = pkg_name.lower().replace("_", "-")
            if pkg_key in _PYTORCH_SKIP:
                continue

            installed_ver = installed_pkgs.get(pkg_key, None)

            # 生成 fix 命令
            def _make_fix(name: str, spec_str: str) -> str:
                """生成可执行的 pip install 修复命令."""
                if spec_str == "*" or not spec_str:
                    return f"pip install {name}"
                if spec_str.startswith("=="):
                    return f"pip install {name}{spec_str}"
                if any(op in spec_str for op in (">=", "<=", "!=", "~=", ",")):
                    return f"pip install '{name}{spec_str}'"
                if spec_str[0].isdigit():
                    return f"pip install {name}=={spec_str}"
                return f"pip install '{name}{spec_str}'"

            if installed_ver is None:
                results.append(CheckResult(
                    "pip", pkg_name,
                    spec if spec != "*" else "任意版本",
                    "未安装",
                    False,
                    fix=_make_fix(pkg_name, spec)
                ))
            elif not check_version_satisfied(installed_ver, spec):
                results.append(CheckResult(
                    "pip", pkg_name,
                    spec,
                    installed_ver,
                    False,
                    fix=_make_fix(pkg_name, spec)
                ))
            else:
                results.append(CheckResult(
                    "pip", pkg_name, spec, installed_ver, True
                ))

    # ---- PyTorch + GPU 匹配 ----
    # 优先用 pip list 判断安装状态，再用 detect_torch() 获取运行时信息
    torch_pip_ver = installed_pkgs.get("torch", None)
    torchvision_pip_ver = installed_pkgs.get("torchvision", None)
    torch_info = detect_torch()
    gpu_info = detect_gpu()
    is_mps = torch_info.get("mps_available", False)
    has_cuda = torch_info.get("cuda_available", False)
    device_type = "CUDA" if has_cuda else ("MPS" if is_mps else "CPU")

    # torch 安装状态（以 pip list 为准）
    if not torch_pip_ver:
        results.append(CheckResult(
            "PyTorch", "torch",
            "已安装",
            "未安装",
            False,
            fix="请根据 GPU 型号安装对应版本 PyTorch，详见 model/requirements.txt"
        ))
    else:
        tv_clean = torch_pip_ver.split("+")[0]

        # GPU 架构匹配检查
        if gpu_info and gpu_info.get("compute_capability"):
            cc = gpu_info["compute_capability"]
            rule = get_gpu_arch_rule(cc)
            if rule:
                expected_min = rule["min_torch"]
                expected_tag = rule["cuda_tag"]
                label = rule["label"]

                torch_version_ok = version_tuple(tv_clean) >= version_tuple(expected_min)
                cuda_tag_ok = expected_tag in torch_pip_ver

                if not torch_version_ok:
                    results.append(CheckResult(
                        "PyTorch", "torch",
                        f"≥ {expected_min}+{expected_tag} ({label})",
                        torch_pip_ver,
                        False,
                        fix=f"GPU 为 {label}，需要 PyTorch ≥ {expected_min}+{expected_tag}。请按 model/requirements.txt 中对应段安装。"
                    ))
                elif not cuda_tag_ok:
                    results.append(CheckResult(
                        "PyTorch", "torch",
                        f"{expected_tag} 标签 ({label})",
                        torch_pip_ver,
                        True,  # 仅警告，不阻塞
                        fix=f"当前 PyTorch CUDA 标签与 GPU ({label}) 不完全匹配，建议使用 +{expected_tag} 版本"
                    ))
                else:
                    results.append(CheckResult(
                        "PyTorch", "torch",
                        f"≥ {expected_min}+{expected_tag} ({label})",
                        torch_pip_ver,
                        True
                    ))
            else:
                results.append(CheckResult(
                    "PyTorch", "torch",
                    "已安装",
                    torch_pip_ver,
                    True
                ))
        else:
            # 无 NVIDIA GPU — 期望 CPU 或 MPS 版本
            results.append(CheckResult(
                "PyTorch", "torch",
                "≥ 2.6.0 (CPU/MPS)",
                torch_pip_ver,
                version_tuple(tv_clean) >= (2, 6, 0),
                fix="请安装 PyTorch ≥ 2.6.0 (CPU 版)"
            ))

        # CUDA / MPS 运行时可用性
        results.append(CheckResult(
            "PyTorch", "设备加速",
            "CUDA / MPS / CPU 任一可用",
            device_type,
            True  # CPU 也通过
        ))

    # torchvision
    if not torchvision_pip_ver:
        results.append(CheckResult(
            "PyTorch", "torchvision",
            "已安装",
            "未安装",
            False,
            fix="pip install torchvision"
        ))
    else:
        results.append(CheckResult(
            "PyTorch", "torchvision",
            "已安装",
            torchvision_pip_ver,
            True
        ))

    # GPU 名称
    if gpu_info:
        results.append(CheckResult(
            "硬件", "GPU",
            "NVIDIA GPU",
            f"{gpu_info['name']} (CC {gpu_info['cc_str']})",
            True
        ))
    else:
        results.append(CheckResult(
            "硬件", "GPU",
            "NVIDIA GPU (可选)",
            "无 NVIDIA GPU — 将使用 CPU/MPS",
            True,
            required=False
        ))

    # ---- R 运行时 + 包 ----
    r_info = check_r()
    if r_info["installed"]:
        results.append(CheckResult(
            "R", "R 运行时",
            "≥ 4.0",
            r_info.get("version", "未知"),
            r_info["version"] is not None and version_tuple(r_info["version"]) >= (4, 0, 0),
            fix="请安装 R 4.0+",
            required=False
        ))
        for pkg, ok in r_info["packages"].items():
            results.append(CheckResult(
                "R", f"R 包: {pkg}",
                "已安装",
                "已安装" if ok else "未安装",
                ok,
                fix=f'R -e \'if(!"{pkg}" %in% rownames(installed.packages())) {{ install.packages("BiocManager"); BiocManager::install("{pkg}") }}\'',
                required=False
            ))
    else:
        results.append(CheckResult(
            "R", "R 运行时",
            "≥ 4.0 (仅 Untargeted 模式需要)",
            "未安装或不在 PATH",
            True,  # R 不是所有模式都需要
            fix="Untargeted 模式需安装 R 4.0+。下载: https://cran.r-project.org/",
            required=False
        ))
        for pkg in R_PACKAGES:
            results.append(CheckResult(
                "R", f"R 包: {pkg}",
                "已安装 (仅 Untargeted 模式)",
                "R 未安装，无法检测",
                True,
                required=False
            ))

    # ---- 模型权重文件 ----
    cp = check_checkpoint()
    results.append(CheckResult(
        "文件", "checkpoint0029.pth",
        f"存在且 >{CHECKPOINT_MIN_MB}MB",
        f"{'存在' if cp['exists'] else '缺失'} ({cp['size_mb']}MB)",
        cp["exists"] and cp["size_mb"] >= CHECKPOINT_MIN_MB,
        fix="请将模型权重文件放置到 model/checkpoint/checkpoint0029.pth（需 >300MB）"
    ))

    # ---- 磁盘空间 ----
    disk = check_disk()
    disk_ok = disk["free_gb"] >= DISK_MIN_GB
    results.append(CheckResult(
        "磁盘", "可用空间",
        f"≥ {DISK_MIN_GB}GB",
        f"{disk['free_gb']}GB (总计 {disk['total_gb']}GB)",
        disk_ok,
        fix=f"磁盘空间不足，请清理至少 {DISK_MIN_GB}GB 可用空间",
        required=False
    ))

    return results


# ---------------------------------------------------------------------------
# 输出格式化
# ---------------------------------------------------------------------------

def render_markdown(results: List[CheckResult]) -> str:
    """渲染 Markdown 报告."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed_count = sum(1 for r in results if r.passed)
    failed_required = sum(1 for r in results if not r.passed and r.required)
    failed_optional = sum(1 for r in results if not r.passed and not r.required)
    total = len(results)

    if failed_required == 0 and failed_optional == 0:
        overall = "✅ 全部通过"
    elif failed_required == 0:
        overall = f"⚠️ {failed_optional} 项条件依赖未满足（不影响核心功能）"
    else:
        overall = f"❌ {failed_required} 项强制依赖失败"

    lines = [
        "## 环境依赖检测报告",
        "",
        f"**检测时间**: {now}",
        f"**操作系统**: {sys.platform}",
        f"**Python 路径**: {sys.executable}",
        f"**Conda 环境**: {os.environ.get('CONDA_DEFAULT_ENV', '无')}",
        f"**项目根目录**: {PROJECT_ROOT}",
        "",
        f"### 总体结果: {overall}",
        f"",
        f"| 类别 | 检测项 | 期望 | 实际 | 状态 |",
        f"|------|--------|------|------|:--:|",
    ]

    for r in results:
        status = "✅" if r.passed else ("⚠️" if not r.required else "❌")
        req_mark = "" if r.required else " (可选)"
        lines.append(
            f"| {r.category} | {r.item}{req_mark} | {r.expected} | {r.actual} | {status} |"
        )

    # 修复建议部分
    failed_items = [r for r in results if not r.passed]
    if failed_items:
        lines.append("")
        lines.append("### 修复建议")
        lines.append("")
        for r in failed_items:
            req_label = "**[强制]**" if r.required else "[可选]"
            lines.append(f"- {req_label} **{r.item}**: {r.fix}")

    # 下一步建议
    lines.append("")
    lines.append("### 下一步")
    lines.append("")
    if failed_required == 0 and failed_optional == 0:
        lines.append("> ✅ 环境完全就绪，可直接运行：")
        lines.append("> ```bash")
        lines.append("> python model/main.py    # CLI 推理")
        lines.append("> python model/GUI/ms-main.py  # 启动 GUI")
        lines.append("> ```")
    elif failed_required == 0:
        lines.append("> ⚠️ 核心依赖全部通过。条件依赖缺失仅影响特定功能：")
        lines.append("> - Untargeted 模式需要 R 及 Bioconductor 包")
        lines.append("> - 磁盘空间不足可能影响大文件处理")
    else:
        lines.append("> ❌ 请先按上述修复建议解决强制依赖失败项后再尝试运行。")

    return "\n".join(lines)


def render_json(results: List[CheckResult]) -> str:
    """渲染 JSON 报告."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed_required = all(r.passed for r in results if r.required)
    passed_all = all(r.passed for r in results)

    items = []
    for r in results:
        items.append({
            "category": r.category,
            "item": r.item,
            "expected": r.expected,
            "actual": r.actual,
            "passed": r.passed,
            "required": r.required,
            "fix": r.fix if not r.passed else "",
        })

    return json.dumps({
        "timestamp": now,
        "platform": sys.platform,
        "python": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed_required": sum(1 for r in results if not r.passed and r.required),
            "failed_optional": sum(1 for r in results if not r.passed and not r.required),
            "overall": "PASS" if passed_all else ("WARN" if passed_required else "FAIL"),
        },
        "results": items,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MRMPFormer 环境依赖检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    parser.add_argument("--quiet", action="store_true", help="仅输出失败项")
    parser.add_argument("--target-env", type=str, default=None, metavar="NAME",
                        help=f"指定要检测的 conda 环境名称（默认: 当前环境）。"
                             f"若指定且不同于当前环境，将通过 'conda run -n NAME' 重新执行本脚本。")
    parser.add_argument("--outfile", type=str, default=None, metavar="PATH",
                        help="将 JSON 结果写入指定文件（而非 stdout）。供 fix_env.py 调用使用。")
    args = parser.parse_args()

    # 如果指定了 --target-env 且不同于当前环境，通过 conda run 重新执行
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if args.target_env and args.target_env != current_env:
        # 重建命令行参数
        passthru = []
        if args.json:
            passthru.append("--json")
        if args.quiet:
            passthru.append("--quiet")
        script_path = Path(__file__).resolve()
        cmd = ["conda", "run", "-n", args.target_env, sys.executable, str(script_path)] + passthru
        ret, out, err = run_cmd(cmd, timeout=120)
        if ret != 0:
            print(f"[ERROR] 无法在 conda 环境 '{args.target_env}' 中运行检测:")
            print(err or out)
            sys.exit(2)
        print(out)
        sys.exit(ret if args.json else 0)

    results = run_all_checks()

    # 如果指定了 --outfile，写入文件
    if args.outfile:
        json_str = render_json(results)
        Path(args.outfile).write_text(json_str, encoding="utf-8")
        # 仍然输出简短摘要到 stdout
        failed_required = sum(1 for r in results if not r.passed and r.required)
        print(f"检测完成: {sum(1 for r in results if r.passed)}/{len(results)} 通过, "
              f"{failed_required} 项强制失败 → {args.outfile}")
        sys.exit(0 if failed_required == 0 else 1)

    if args.json:
        print(render_json(results))
    elif args.quiet:
        failed = [r for r in results if not r.passed]
        if failed:
            for r in failed:
                print(f"[{'FAIL' if r.required else 'WARN'}] {r.category}/{r.item}: {r.actual} (期望: {r.expected})")
                if r.fix:
                    print(f"  → {r.fix}")
        else:
            print("✅ All checks passed.")
    else:
        print(render_markdown(results))

    # 返回码
    failed_required = sum(1 for r in results if not r.passed and r.required)
    sys.exit(0 if failed_required == 0 else 1)


if __name__ == "__main__":
    main()
