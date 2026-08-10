# 项目开发日志

## 项目概述

### 目标
MRMPFormer 是一个基于深度学习的 LC-MS 代谢组学峰检测与定量工具。结合 CNN（ResNet-50）与 Transformer（DETR 架构），在提取离子色谱图（EIC/ROI）中识别真峰/假峰并定位峰边界以积分定量。

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
2. 使用 MRMPFormer（ResNet-50 + 1 层 Transformer 编解码器）检测峰
3. 根据预测框边界对 EIC 积分 → 得到峰面积
4. 后处理去重 → 输出定量结果表
支持 Targeted/Untargeted × Centroided/Profile 四种组合模式。

---

## 开发时间线

### 2026-08-10

- 文档生成(README.md): 基于当前模型架构与 conda 环境校正 README —— 修正编解码器层数（3层Decoder→1层Decoder，与 checkpoint0029.pth 一致）、补充 num_queries=3 / hidden_dim=256 / nheads=8 等架构细节、明确 conda 环境名 `mrmpformer`（Python 3.11.15 + PyTorch 2.6.0+cu124）、标注本机 GPU 为 8× RTX 4090 D、训练参数表注明 checkpoint 实际 num_queries 与默认值差异、新增 checkpoint 完整训练参数说明

### 2026-07-29

- 重构(tools/batch/): 合并 batch_post_newtest_under_snr_filtered.py 与 rerun_snr_under_snr_filtered.py → 新建 reprocess.py，统一 `--stage snr/post/snr-post` 三种运行模式，消除 post_newtest 参数分叉
- 代码生成(tools/batch/reprocess.py): 合并脚本支持分阶段批量处理 —— SNR 阶段调用 mzml_box_outside_snr_pipeline.run()，post 阶段通过 subprocess 调用 run_unified_peak_workflow.py；统一暴露所有 post_newtest 参数至 CLI
- 文档生成(tools/batch/): 旧脚本加弃用标记，指向 reprocess.py
- 重构(tools/mzml/): 合并 mzml_export_one_chrom.py 与 read_mzml_one_group.py → 新建 chromatogram.py，统一 `list/show/export` 三个子命令
- 代码生成(tools/mzml/chromatogram.py): 共享 inspect.py 依赖，修复旧脚本 import 路径 bug（mzml_inspect_to_csv→inspect）；统一色谱定位、编码处理、数据提取逻辑
- 文档生成(tools/mzml/): 旧脚本加弃用标记，指向 chromatogram.py
- 文档生成(README.md): 全面更新 —— 修正核心目录名 main_model→model、重写项目结构树（新增 gamstekpeaking/ms2mzml/tools/管线脚本）、版本号 v0.3.0→v2.0.0、新增「辅助工具」章节

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

### 2026-08-03

- 文档生成(README.md): 修复完整管线部分 —— 数据路径改为 `../data/...`（`cd model` 后根目录数据）、输出结构图修正（`prediction_refined.csv` 实际位于 `snr_filtered/<样品>/SNR_box_<阈值>/` 下，补充 `predicted_plots/`、`refined_plots/`、计时日志）、轻量模式路径统一
- 文档生成(README.md): 安装部分拆分为「第一步 安装 Python 环境」+「第二步 安装项目依赖」两节，明确 Python 3.10~3.11 与 pip/conda 两种路径
- 文档生成(README.md): 推理参数速查新增「完整参数模板」—— 覆盖 `model/main.py` argparse 全部 42 个参数（基础/QC/SNR/Post 四组），每个参数配注释填写说明，可直接复制运行（bash 语法已验证）
- 其他(model/environment.yml): 补充缺失的 `pyopenms==3.3.0`，与 `requirements.txt` 对齐（testXIC.py 顶层硬依赖，缺失会导致管线无法启动）
