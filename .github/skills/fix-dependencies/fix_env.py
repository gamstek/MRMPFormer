#!/usr/bin/env python3
"""
QuanFormer 环境依赖修复脚本
=============================
由 fix-dependencies skill 调用，处理机械性修复操作。

用法:
  python fix_env.py --find-env              # 查找 quanformer conda 环境
  python fix_env.py --create-env [NAME]     # 创建 conda 环境 (默认 quanformer)
  python fix_env.py --check ENV             # 在指定环境中运行依赖检测
  python fix_env.py --fix ENV [--dry-run]   # 自动修复可修复的依赖
  python fix_env.py --verify ENV            # 修复后验证

选项:
  --dry-run   仅打印修复命令，不实际执行
  --yes       跳过确认，直接执行
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECK_SCRIPT = Path(__file__).resolve().parent.parent / "check-dependencies" / "check_env.py"
DEFAULT_ENV_NAME = "quanformer"
DEFAULT_PYTHON = "3.11"

# 不可自动修复的类别
MANUAL_ONLY_CATEGORIES = {"Python", "R", "文件", "磁盘", "硬件"}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def run_cmd(cmd: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    """运行命令."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            shell=(sys.platform == "win32"),
            encoding="utf-8", errors="replace",  # 避免 Windows GBK 编码错误
        )
        stdout = proc.stdout.strip() if proc.stdout else ""
        stderr = proc.stderr.strip() if proc.stderr else ""
        return proc.returncode, stdout, stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"
    except subprocess.TimeoutExpired:
        return -2, "", "Command timed out"


def list_conda_envs() -> Dict[str, Any]:
    """列出所有 conda 环境，检查 quanformer 是否存在."""
    result: Dict[str, Any] = {
        "conda_available": False,
        "quanformer_exists": False,
        "quanformer_path": None,
        "all_envs": [],
    }

    ret, out, _ = run_cmd(["conda", "--version"])
    if ret != 0:
        return result
    result["conda_available"] = True

    ret2, out2, _ = run_cmd(["conda", "env", "list"])
    if ret2 != 0:
        return result

    for line in out2.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        # 跳过路径格式的行
        if re.match(r"^[A-Za-z]:|^/|^\\\\", name):
            continue
        if name == "base":
            continue
        result["all_envs"].append(name)
        if name == DEFAULT_ENV_NAME:
            result["quanformer_exists"] = True
            # 提取路径
            for p in parts[1:]:
                if p not in ("*", ""):
                    result["quanformer_path"] = p
                    break

    return result


def create_conda_env(name: str = DEFAULT_ENV_NAME, python_ver: str = DEFAULT_PYTHON) -> bool:
    """创建 conda 环境."""
    print(f"[创建] 正在创建 conda 环境 '{name}' (Python {python_ver})...")
    ret, out, err = run_cmd(
        ["conda", "create", "-n", name, f"python={python_ver}", "-y"],
        timeout=300
    )
    if ret == 0:
        print(f"[成功] 环境 '{name}' 创建完成。")
        return True
    else:
        print(f"[失败] 创建环境失败:\n{err or out}")
        return False


def run_check_in_env(env_name: str) -> Optional[Dict[str, Any]]:
    """在指定 conda 环境中运行 check_env.py，返回解析后的 JSON 结果.

    策略：使用临时文件传递 JSON 输出，避免 conda run 对 stdout 的 GBK 编码问题。
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name

    try:
        # 方案 A: 用 --outfile 写到临时文件
        cmd = f'conda run -n {env_name} python "{CHECK_SCRIPT}" --json --outfile "{tmp_path}"'
        ret, out, err = run_cmd([cmd] if sys.platform == "win32" else cmd.split(), timeout=120)

        # 如果 conda run 执行成功，读取临时文件
        if Path(tmp_path).exists():
            result_text = Path(tmp_path).read_text(encoding="utf-8")
            if result_text.strip():
                return json.loads(result_text)

        # 方案 B: 回退 — 直接通过 stdout 解析
        cmd2 = f'conda run -n {env_name} python "{CHECK_SCRIPT}" --json'
        ret2, out2, err2 = run_cmd([cmd2] if sys.platform == "win32" else cmd2.split(), timeout=120)
        if ret2 not in (0, 1):
            print(f"[错误] 无法在环境 '{env_name}' 中运行检测")
            if err2:
                print(f"  stderr: {err2[:300]}")
            return None

        json_start = out2.find("{")
        if json_start < 0:
            print(f"[错误] 检测输出中未找到 JSON")
            return None

        return json.loads(out2[json_start:])
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_fixable_items(check_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从检测结果中提取可自动修复的失败项."""
    fixable: List[Dict[str, Any]] = []
    for item in check_result.get("results", []):
        if item.get("passed", True):
            continue
        if item.get("category") in MANUAL_ONLY_CATEGORIES:
            continue
        if not item.get("fix"):
            continue
        fixable.append(item)
    return fixable


def parse_fix_command(fix_str: str, env_name: str) -> Optional[str]:
    """将 fix 字符串转换为 conda run 命令."""
    if not fix_str:
        return None
    fix_str = fix_str.strip()
    # 如果已经是完整的 conda run 命令，直接返回
    if fix_str.startswith("conda run"):
        return fix_str
    # pip install 命令
    if fix_str.startswith("pip install"):
        return f"conda run -n {env_name} {fix_str}"
    # pip install with index-url (PyTorch)
    if fix_str.startswith("pip "):
        return f"conda run -n {env_name} {fix_str}"
    # 如果是更复杂的命令（含换行或 &&），逐行包装
    lines = fix_str.split("\n")
    wrapped = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("conda run") or line.startswith("#"):
            wrapped.append(line)
        elif line.startswith("pip"):
            wrapped.append(f"conda run -n {env_name} {line}")
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


def apply_fixes(env_name: str, fixable_items: List[Dict[str, Any]], dry_run: bool = False) -> Tuple[int, int]:
    """执行修复命令. 返回 (success_count, fail_count)."""
    success = 0
    fail = 0

    # 先去重：同一个 package 只修一次
    seen: set = set()
    unique_fixes: List[Tuple[str, str]] = []  # (item_name, command)

    for item in fixable_items:
        name = item["item"]
        if name in seen:
            continue
        seen.add(name)
        cmd = parse_fix_command(item["fix"], env_name)
        if cmd:
            unique_fixes.append((name, cmd))

    # 优先处理 PyTorch
    torch_fixes = [(n, c) for n, c in unique_fixes if n == "torch"]
    other_fixes = [(n, c) for n, c in unique_fixes if n != "torch"]
    ordered_fixes = torch_fixes + other_fixes

    for name, cmd in ordered_fixes:
        print(f"\n[修复] {name}...")
        print(f"  → {cmd}")
        if dry_run:
            print("  (dry-run, 未实际执行)")
            success += 1
            continue

        ret, out, err = run_cmd(cmd.split() if not sys.platform == "win32" else [cmd], timeout=300)
        if ret == 0:
            print(f"  ✅ 成功")
            success += 1
        else:
            print(f"  ❌ 失败: {err or out[:200]}")
            fail += 1

    return success, fail


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="QuanFormer 环境依赖修复",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="操作")

    # --find-env
    sub.add_parser("find-env", help="查找 quanformer conda 环境")

    # --create-env [NAME]
    p_create = sub.add_parser("create-env", help="创建 conda 环境")
    p_create.add_argument("name", nargs="?", default=DEFAULT_ENV_NAME, help="环境名 (默认: quanformer)")
    p_create.add_argument("--python", default=DEFAULT_PYTHON, help="Python 版本 (默认: 3.11)")

    # --check ENV
    p_check = sub.add_parser("check", help="在指定环境中运行依赖检测")
    p_check.add_argument("env", help="conda 环境名")

    # --fix ENV
    p_fix = sub.add_parser("fix", help="自动修复可修复的依赖")
    p_fix.add_argument("env", help="conda 环境名")
    p_fix.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    p_fix.add_argument("--yes", action="store_true", help="跳过确认")

    # --verify ENV
    p_verify = sub.add_parser("verify", help="修复后验证")
    p_verify.add_argument("env", help="conda 环境名")

    args = parser.parse_args()

    if args.command == "find-env":
        envs = list_conda_envs()
        print(json.dumps(envs, ensure_ascii=False, indent=2))
        sys.exit(0 if envs["quanformer_exists"] else 1)

    elif args.command == "create-env":
        ok = create_conda_env(args.name, args.python)
        sys.exit(0 if ok else 1)

    elif args.command == "check":
        result = run_check_in_env(args.env)
        if result is None:
            sys.exit(2)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "fix":
        # 1. 先运行检测
        print(f"[检测] 正在环境 '{args.env}' 中运行依赖检测...")
        result = run_check_in_env(args.env)
        if result is None:
            sys.exit(2)

        # 2. 提取可修复项
        fixable = extract_fixable_items(result)
        if not fixable:
            print("[结果] 没有可自动修复的失败项。")
            sys.exit(0)

        print(f"\n[待修复] 共 {len(fixable)} 项可自动修复:")
        for item in fixable:
            print(f"  - {item['category']}/{item['item']}: {item['actual']} → {item['fix']}")

        # 3. 列出不可自动修复的项
        manual = [i for i in result.get("results", [])
                  if not i.get("passed") and i.get("category") in MANUAL_ONLY_CATEGORIES]
        if manual:
            print(f"\n[需手动] 共 {len(manual)} 项需手动处理:")
            for item in manual:
                print(f"  - {item['category']}/{item['item']}: {item['actual']}")

        # 4. 执行修复
        if not args.yes and not args.dry_run:
            resp = input("\n是否继续执行修复? [Y/n]: ").strip().lower()
            if resp and resp != "y":
                print("已取消。")
                sys.exit(0)

        success, fail = apply_fixes(args.env, fixable, dry_run=args.dry_run)
        print(f"\n[完成] 成功 {success}, 失败 {fail}")

    elif args.command == "verify":
        ret, out, _ = run_cmd(
            ["conda", "run", "-n", args.env, sys.executable, str(CHECK_SCRIPT)],
            timeout=120
        )
        print(out)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
