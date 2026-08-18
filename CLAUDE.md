# CLAUDE.md — MRMPFormer 项目级全局指令

> 本文件为 MRMPFormer 项目的 AI 辅助指令文件，会被自动加载进每次模型调用的上下文。
> 任何模型（主对话、子 Agent、外部模型）在本项目中被调用时，**必须**遵守以下规则。

---

## 项目身份

MRMPFormer 是一个基于 **DETR（ResNet-50 + 1 层 Transformer 编解码器）** 的 LC-MS 代谢组学峰检测与定量工具。

| 属性 | 说明 |
|------|------|
| **核心代码** | `model/` — 扁平结构，包含 inference / models / preprocessing / postprocessing / framework / utils / tools 七个子包 |
| **桌面 GUI** | `desktop/` — PySide6 图形界面 |
| **格式转换** | `converters/` — msdata/wiff → mzML |
| **模型权重** | `model/checkpoint/quanformer.pth`（>300MB） |
| **推理入口** | `model/inference/cli.py`（须在 `model/` 目录下用 `python -m inference.cli --mode ...` 调用） |
| **训练入口** | `model/train.py`（须在 `model/` 目录下用 `python -m train ...` 调用） |
| **四种分析模式** | Targeted / Untargeted × Centroided / Profile（⚠️ **当前仅开发 Targeted × Centroided（MRM）**，其余三组合暂不开发；原 `getFeature.py`/R、`testXIC.py` 等非 MRM 代码已从仓库删除，如需恢复从 git history 找回） |
| **输入** | `.mzML` 原始质谱数据（chromatogram 模式） |
| **输出** | `prediction_refined.csv`（峰面积 + 置信度）、`box_outside_snr_report.csv`、EIC 预测图像 |
| **上游来源** | Facebook DETR（`model/framework/` fork 自 DETR） |
| **版权所有** | LinShuhaiLAB, Xiamen University |

---

## 项目技术约束

在做出任何代码修改前，必须遵守以下硬性约束：

### 开发范围（强制）
- **当前唯一开发模式**：Targeted × Centroided（MRM 靶向定量）
- **暂不开发**：Targeted × Profile、Untargeted × Centroided、Untargeted × Profile —— 原 `getFeature.py`/R、`testXIC.py` 等非 MRM 代码已从仓库删除（如需恢复从 git history 找回）；`extract_xic_from_arrays`（外部数组模式）保留现状，❌ 禁止新增功能、重构或大规模改动
- 任何修改不得破坏 MRM 模式现有可用状态；涉及公共代码（`preprocessing/xic_extraction.py`、`inference/predictor.py`、pipeline）时以 MRM 模式为准验证回归

### Python & PyTorch
- **Python 版本**：3.11
- **Conda 环境名**：`gamstekpeaking` ，禁止改名或另建环境，激活命令固定为 `conda activate gamstekpeaking`
- **PyTorch 版本**：按 GPU 架构选择对应版本
  - RTX 50 系（Blackwell, sm_120）：≥ 2.7.0+cu128
  - RTX 40/30/20 系：2.6.0+cu124
  - CPU / Apple Silicon (MPS)：2.6.0
- **依赖文件**：以根目录 `requirements.txt` 为准（model + desktop 合并，含 GPU/CPU 分段配置，默认启用 RTX 40 系/4090 的 cu124 段）

### 设备与路径
- **设备选择**：❌ 严禁硬编码 `device='cuda'`，必须使用 `resolve_torch_device()`（`utils/torch_device.py`，CUDA > MPS > CPU 自动检测）
- **模型加载**：❌ 严禁直接用 `torch.load()`，必须使用 `safe_torch_load()`（`framework/util/misc.py`）以兼容跨 PyTorch 版本
- **路径分隔符**：❌ 严禁硬编码 `/` 或 `\\`，必须使用 `os.path.join()` / `pathlib.Path`

### 命令执行（强制）
- **外部命令优先**：涉及训练（`python -m train ...`）、推理（`python -m inference.cli ...` / `predictor`）、XIC 提取（`extract_xic_with_pyopenms` / `coco_annotation`）等会实际运行模型、写文件或触发 matplotlib/pyopenms 渲染的命令，**优先整理成完整命令交给用户在系统 PowerShell 中执行**，不要反复在沙箱内自跑
- **已知限制**：TRAE 沙箱会拦截 matplotlib 渲染 DLL 延迟加载（崩溃码 `0xc06d007f`）与用户目录 site-packages 写入，导致 XIC 图像生成、推理画图、pip 安装等在其内部不可靠
- **沙箱内仅跑安全操作**：纯数据验证、纯 Python 逻辑单测、JSON/CSV 解析、编译检查（`py_compile`）等不触发渲染的轻量命令可在沙箱内执行
- **给用户的外部命令必须完整可复制**：含 `cd` 到 `model/`、完整参数、以及必要的产物拷贝（如微调后 `Copy-Item output_v2/checkpoint.pth checkpoint/quanformerv2.pth`）；说明预期输出与关注点
<!-- [DISABLED] R 语言相关说明已注释
- **路径空格**：注意 Windows 路径含空格时 R 调用（`find_peaks.R`）可能失败，需加引号
-->

### 架构边界
- `model/models/`：纯 DETR 模型定义（quanformer / mrmpformer / shared，不依赖业务逻辑）
- `model/framework/`：DETR 训练框架（datasets / util / engine）
- `model/inference/`：模型加载 + 预测 + 可视化 + 统一 CLI 入口
- `model/preprocessing/`：mzML 加载 → EIC → ROI 图像（xic_extraction）；MS1 谱图聚合（ion_zenith）
- `model/postprocessing/`：积分定量 + 峰质量 + box↔RT 映射 + SNR 筛选 + 峰区间精修（peak_refinement）
- `model/utils/`：公共工具（io_utils / quantify / mzml_load / roi_rt_mapping 等）
- `model/tools/`：批处理 / 诊断 / 可视化 / 基准测试
- `model/train.py`：训练入口
- `desktop/`：PySide6 图形界面（独立于 CLI 管线；`workers/ion_zenith.py` 为 `preprocessing/ion_zenith.py` 的 Qt 薄包装）
- `converters/`：格式转换工具（msdata/wiff → mzML）
- 各层单向依赖：inference → preprocessing/postprocessing → models/framework → utils

### 已知陷阱（详见 `docs/Bugs.md`）
- `batch_size=1` 逐张推理导致 GPU 利用率仅 ~15%
- `plot_results` 用 `joblib(n_jobs=-1)` 可能导致 I/O 争抢
<!-- [DISABLED] R 语言相关说明已注释
- Untargeted 模式需要安装 R + Bioconductor（`find_peaks.R` 调用 CentWave）
-->
- 模型预测不应每次重新 `torch.load`（GUI 场景）

---

## 强制规则：项目开发日志

### 1. 确保项目文件中有项目开发日志文件
- 文件路径：`dev_log.md`（项目根目录）
- 若不存在，则自动创建

### 2. 开发日志构成
- 开发日志由两个部分组成：项目概述和开发时间线

### 3. 项目概述部分
- 项目概述部分分为若干子部分，内容仅保留说明性文字
- 子部分：`目标` / `输入` / `输出` / `方法介绍`
- 仅在项目定位、架构发生实质变化时更新概述，不要每次任务都改

### 4. 开发时间线格式
- 按日期分组：`### YYYY-MM-DD`
- 条目格式：`- <类型>(<作用域>): <简要描述>`
- 类型：`需求分析` / `数据建模` / `代码生成` / `调试` / `文档生成` / `重构` / `测试` / `其他`
- 作用域即范围（模块/文件/功能），建议使用相对于 `model/` 的路径
- 内容只记录阶段性说明，不展开具体业务细节

### 5. 自动更新 `dev_log.md`
- 触发时机：每次模型完成一次生成、修改代码、测试任务后，以及用户提出"同步更新日志"后
- 不得遗漏，不得延迟到下次会话补记
- 详细规范见 `.github/skills/dev-log-writer/SKILL.md`

---

## 强制规则：分析/调研/回答类任务（只读模式）

### 适用范围
当用户请求以下类型的行为时，该规则**强制生效**：
- 分析（分析原因、分析问题、分析方案）
- 调研（调研方案、调研可行性、调研兼容性）
- 给出解决方案（但未明确要求执行）
- 回复问题（解释、说明、回答）
- 审阅代码（review）

### 规则
1. **禁止修改任何文件**——不调用 `replace_string_in_file`、`create_file`、`multi_replace_string_in_file` 等写入工具
2. **禁止执行任何可能改变环境的命令**——不安装/卸载/升级 pip 包、不修改 conda 环境、不创建/删除 Docker 容器
3. **仅基于当前环境**——只读取已有文件、查询当前安装的包版本、检查运行状态
4. **三轮自审**——任何分析结论必须经过三次自我审查：
   - 第一轮：表面原因分析
   - 第二轮：反驳/质疑第一轮，寻找更深层原因
   - 第三轮：综合前两轮，排除不成立假设，给出最终结论
5. **最终输出**——根据三轮审查结果，总结一版最合适的答案

### 例外
用户明确说"执行"、"改"、"修"、"帮我做"等指令时，本规则不生效，转为正常执行模式。