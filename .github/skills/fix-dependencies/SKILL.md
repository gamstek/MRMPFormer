---
name: fix-dependencies
description: "Use when: 修复依赖/fix dependencies/安装依赖/自动修复环境/补齐依赖/环境修复. After check-dependencies reports failures, automatically repairs the environment — finds or creates the 'quanformer' conda env, installs missing pip packages with correct versions, and verifies the fix."
argument-hint: "根据检测报告自动修复 QuanFormer 环境依赖"
user-invocable: true
---

# 环境依赖修复

在 `check-dependencies` 检测出失败项后，自动修复 pip 包依赖。

> **推荐用户直接使用 GUI**（`check_gui.py`），弹窗内有一键修复按钮，无需手动调用本 skill。
> 本 skill 主要用于 AI 辅助修复场景。

## 使用方式

```bash
# 查找 conda 环境
python .github/skills/fix-dependencies/fix_env.py find-env

# 在指定环境中检测 + 修复
python .github/skills/fix-dependencies/fix_env.py fix quanformer

# 预览修复方案（不执行）
python .github/skills/fix-dependencies/fix_env.py fix quanformer --dry-run

# 修复后验证
python .github/skills/fix-dependencies/fix_env.py verify quanformer
```

> ⚠️ 修复前会列出待安装的包清单，**必须让用户确认后再执行**。不可自动修复的项（Python 版本、R、模型文件、磁盘空间）应告知用户手动处理。
