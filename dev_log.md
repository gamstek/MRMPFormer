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

### 2026-07-15

- 需求分析(MRMPFormer/): 确定采用 SimCLR 对比学习方案（方案 B），ResNet50 骨干 + 128-d 投影头，自监督训练无需标注
- 代码生成(MRMPFormer/): 创建完整 SimCLR 训练框架 —— simclr.py（ResNet50 + ProjectionHead）、losses.py（NT-Xent）、augmentations.py（5 种 SimCLR 增强）、dataset.py（无标签图像数据集）、train.py（CosineAnnealing + 梯度累积）、extract_features.py（推理输出 2048-d 特征）
- 文档生成(docs/superpowers/specs/): 创建 2026-07-15-mrmpformer-simclr-design.md 设计文档

### 2026-07-20

- 需求分析(MRMPFormer/): 环境诊断 — 确认 MRMPFormer 复用 quanformer conda 环境 (PyTorch 2.11.0+cu128, RTX 5060)；从 8 个源文件追踪全部依赖链 (torch/torchvision/numpy/Pillow/tqdm/tensorboard)
- 代码生成(MRMPFormer/): 创建 requirements.txt — 按 RTX 50/40/30/20 系 GPU + CPU 分段，50 系默认启用 cu128 索引（当前本机配置），其余注释备用；版本号与 quanformer 环境已验证版本对齐
- 重构(MRMPFormer/train.py + utils/config.py): 实现早停策略 (方案 A: 相对改善率监控) — config 新增 4 个早停字段 (enabled/patience/min_delta/min_epochs)，train.py 训练循环内嵌早停检查逻辑；同时用 tqdm 替换原有 batch 级 print 进度显示，进度条实时展示 running loss
- 重构(MRMPFormer/models/simclr.py + config.py + train.py): 实现 backbone 分阶段冻结 — simclr.py 拆分 Sequential 为 stem/layer1~4/avgpool 独立组件，新增 freeze_stages 参数 (0=全训练, 4=仅训练 layer4+投影头)；config.py 增加 freeze_stages 配置项；train.py 输出可训练/冻结参数量统计
- 代码生成(MRMPFormer/): 实现评估体系 — losses.py 新增 alignment_loss() / uniformity_loss()（Wang & Isola 2020）；evaluate.py（Retrieval P@K + Uniformity 对比脚本，支持单文件/双文件模式，标签自动从中文文件名推断）；train.py 新增加 compute_alignment_uniformity()，每 N epoch 自动计算并写入 TensorBoard；config.py 增加 eval_metrics_every 配置

### 2026-07-23

- 调试(predict_utils.py): 修复 P3 — plot_single_result 重复 Image.open() 读盘，改为 plot_results 主进程预加载图片传入（≤500 张阈值保护，超限回退路径模式）
- 调试(predict_utils.py): 修复 P2 — plot_results 默认 n_jobs=-1 全核并行导致 I/O 争抢，改为 n_jobs=2
- 调试(predict_utils.py): 修复 P4 — plot_results 覆盖原 ROI 图，输出改为原文件名_detected 后缀
- 调试(detection_helper.py): 修复 C1 — Rscript 命令路径参数未加引号，Windows 路径含空格时解析失败
- 调试(plot_utils.py): 修复 Q4 — import bisect 易与标准库混淆，改为 from bisect import bisect_left, bisect_right
- 调试(workbooks/): 修复 Q1 — calcQuantificationResults、peakdetective-application、calcTrueOrFalsemarker、peakAlignment 四个脚本硬编码原作者绝对路径，统一改为相对路径或空占位符