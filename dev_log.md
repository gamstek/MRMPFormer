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
- 代码生成(utils/predict_utils.py): 新增 `_get_best_device()` 支持 CUDA/MPS/CPU 自动检测
- 代码生成(utils/io_utils.py): 新增 `safe_torch_load()` 跨 PyTorch 版本兼容封装
- 代码生成(quanformer/util/misc.py): 新增 `_get_dist_device()` + `safe_torch_load()`；替换 5 处 `device='cuda'` 硬编码
- 调试(utils/postprocess.py): 修复 Unix 路径分隔符硬编码 → `os.path`
- 调试(quanformer/main.py): `--device` 默认值 `cuda` → `auto` + 自动检测回退逻辑
- 重构(GUI/ms-main.py): 全界面中文化 + QSS 简约主题 + 信号阻塞修复
- 调试(GUI/ms-main.py): 修复 `listWidget_2` 切换样本时图片重复累积 bug
- 文档生成(README.md): 全文汉化 + 跨平台兼容性矩阵 + 版本要求修正
- 文档生成(DEPLOY.md): 同步更新部署说明
- 文档生成(PROJECT_PANORAMA.md): 创建项目全景文档（目录树 + 数据流 + 依赖关系）
- 文档生成(PROBLEM.md): 创建已知问题汇总（已修复 8 项 + 待修复 9 项）
- 其他(PyTorch): 识别 RTX 5060 (sm_120) 与 PyTorch 2.6.0 不兼容 → 推荐升级到 2.11.0+cu128
- 其他(性能诊断): 识别预测阶段 5 个瓶颈（batch_size=1 为首要）
- 重构(requirements.txt): 整合 GPU/CPU 双文件为统一配置，按 GPU 架构分段 (RTX 50/40/30/20 系列 + CPU)，默认激活 RTX 50 系 cu128 配置
- 文档生成(README.md): 新增项目目录结构章节；更新环境要求/安装/依赖表/FAQ 以反映统一 requirements
- 调试(utils/extract_eic.py): 修复空 MS1 谱图 IndexError — 新增 `len(_mzs)==0` 守卫跳过空谱图
