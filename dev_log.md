# 项目开发日志

## 项目概述

### 目标
QuanFormer 是一个基于深度学习的 LC-MS 代谢组学峰检测与定量工具。结合 CNN（ResNet-50）与 Transformer（DETR 架构），在提取离子色谱图（EIC/ROI）中识别真峰/假峰并定位峰边界以积分定量。

### 输入
- **原始数据**: `.mzML` 格式的高分辨率 LC-MS 数据（Centroided 或 Profile）
- **特征表**（Targeted 模式）: CSV 文件，包含 `Compound Name` / `mz` / `RT` 三列
- **模型权重**: `checkpoint0029.pth`（>300MB）

### 输出
- **定量结果**: `area.csv`（峰面积表）
- **后处理结果**: `post-area.csv`（去重转置后的面积表）
- **EIC 图像**: 每个化合物生成一张 ROI 区域的 JPEG 图像
- **预测图像**: 带检测框标注的 EIC 图像

### 方法介绍
1. 从 mzML 中按 m/z 提取 EIC → 生成 ROI 图像
2. 使用 QuanFormer（ResNet-50 + 1 层 Transformer 编解码器）检测峰
3. 根据预测框边界对 EIC 积分 → 得到峰面积
4. 后处理去重 → 输出定量结果表
支持 Targeted/Untargeted × Centroided/Profile 四种组合模式。

---

## 开发时间线

### 2026-07-07

- 需求分析(全局): 跨平台兼容性审阅 — 识别出 GPU 硬编码、路径分隔符、Python 版本矛盾等 15 个问题
- 代码生成(utils/): 新增 `_get_best_device()` (predict_utils) + `safe_torch_load()` (io_utils) + `_get_dist_device()` (misc)；替换 5 处 `device='cuda'` 硬编码为 auto
- 调试(utils/): 修复 Unix 路径分隔符硬编码 (postprocess)；`--device` 默认值改为 auto (main)；空 MS1 谱图 IndexError 守卫 (extract_eic)
- 重构(GUI/ms-main.py): 全界面中文化 + QSS 简约主题 + 信号阻塞修复 + 图片重复累积 bug 修复
- 重构(requirements.txt): 整合 GPU/CPU 双文件为统一配置，按 GPU 架构分段 (RTX 50/40/30/20 + CPU)
- 文档生成: README 全文汉化 + 跨平台兼容矩阵 + 目录结构；DEPLOY 部署说明同步；PROJECT_PANORAMA 全景文档；PROBLEM 已知问题汇总（已修复 8 项 + 待修复 9 项）
- 其他: RTX 5060 (sm_120) 与 PyTorch 2.6.0 不兼容分析；预测阶段 5 个性能瓶颈识别

### 2026-07-09

- 文档生成(CLAUDE.md): 重写项目级 AI 指令文件，新增项目身份表、技术约束和代码修改记录规则
- 代码生成(.github/skills/): 创建环境依赖检测与修复系统 —— check_env.py（全量检测/JSON/Markdown）、check_gui.py（tkinter 弹窗/一键修复）、fix_env.py（find-env/check/fix/verify）；配套 check-dependencies + fix-dependencies 两个 skill（精简后各 ~30 行）
- 调试(check_env.py + check_gui.py): 修复 Windows GBK/UTF-8 编码错配导致中文乱码 — run_cmd() 统一 utf-8、GUI 改用 --outfile 绕过管道编码
- 文档生成(README.md): 新增「环境检测」小节，一行命令启动 GUI 弹窗
