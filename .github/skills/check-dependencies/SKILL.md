---
name: check-dependencies
description: "Use when: 检查环境依赖/check dependencies/环境检测/依赖检查/验证环境/本机是否满足依赖/环境就绪/依赖诊断. Diagnoses whether the local machine meets ALL MRMPFormer project requirements — Python version, pip packages, PyTorch/CUDA compatibility, R/Bioconductor, model weights, and system prerequisites. Produces a structured pass/fail report with fix suggestions."
argument-hint: "检查本机是否满足 MRMPFormer 全部依赖"
user-invocable: true
---

# 环境依赖检测

检测 Conda 环境、Python 版本、pip 包、PyTorch/CUDA、GPU、模型权重、R、磁盘空间。

## 使用方式

```bash
# GUI 弹窗（推荐，交互式，支持一键修复）
python .github/skills/check-dependencies/check_gui.py

# 终端文本报告
python .github/skills/check-dependencies/check_env.py

# 仅失败项 / JSON / 指定 conda 环境
python .github/skills/check-dependencies/check_env.py --quiet
python .github/skills/check-dependencies/check_env.py --json
python .github/skills/check-dependencies/check_env.py --target-env gamstekpeaking
```

检测基于根目录 `requirements.txt`，覆盖 `==`、`>=`、`>=X,<Y` 等版本约束。对 PyTorch 会按 GPU 计算能力自动匹配正确版本。
