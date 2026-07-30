# 🏔 GAMSTEKPEAKing

QuanFormer 项目的下一代统一桌面应用 — 质谱代谢组学全流程工作台。

## 快速开始

```bash
# 1. 创建并激活 conda 环境
conda create -n gamstekpeaking python=3.11
conda activate gamstekpeaking

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python main.py
```

## 目录导航

| 目录 | 说明 | 详情 |
|------|------|------|
| `pages/` | 功能页面模块 | [README_pages.md](pages/README_pages.md) |
| `workers/` | 后台工作线程 | [README_workers.md](workers/README_workers.md) |
| `engine/` | 模型推理引擎（预留） | [README_engine.md](engine/README_engine.md) |
| `bin/` | msdata2mzml 运行时 | [README_bin.md](bin/README_bin.md) |
| `assets/` | 静态资源 | [README_assets.md](assets/README_assets.md) |

## 技术栈

前端基于 **PySide6**（Qt for Python）构建，采用深色科技风主题。

## 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| PySide6 | 6.9.3 | GUI 框架（Qt for Python） |
| pymzml | 2.5.11 | mzML 文件解析 |
| numpy | 1.26.4 | 数值计算 |
| tqdm | 4.69.1 | 进度显示 |
| pandas | 2.2.2 | 数据处理 |
| matplotlib | 3.9.2 | 绘图 |
| scipy | 1.13.1 | 科学计算 |
| PyOpenMS | 3.3.0 | 质谱数据处理 |
| torch | 2.6.0+cu124 | 深度学习（RTX 4090D） |

## 架构概览

```
┌──────────────┬────────────────────────────────────┐
│  侧边栏导航   │  QStackedWidget（页面路由）          │
│  ─────────── │  ┌────────────────────────────────┐ │
│  🏔 前处理    │  │ 📦 格式转换  ⚡ 离子天顶       │ │
│  📊 寻峰      │  │     (后续上线)                  │ │
│  📈 定量      │  └────────────────────────────────┘ │
│  🔬 模型      │                                     │
│  ⚙ 设置       │                                     │
└──────────────┴────────────────────────────────────┘
```

- 每个功能板块 = 一个 `pages/*.py` 文件
- 耗时操作 → `workers/*.py` 的 QThread，Signal 驱动 UI 更新
- 视觉风格 → `theme.py` 统一管理（深色科技风）

## 当前状态

| 板块 | 状态 |
|------|------|
| 前处理（格式转换 + 离子天顶） | ✅ 已上线 |
| 寻峰 | 🔒 规划中 |
| 定量 | 🔒 规划中 |
| 模型管理 | 🔒 规划中 |
| 设置 | 🔒 规划中 |
