# CLAUDE.md — QuanFormer 项目级全局指令

> 本文件为 QuanFormer 项目的 AI 辅助指令文件，会被自动加载进每次模型调用的上下文。
> 任何模型（主对话、子 Agent、外部模型）在本项目中被调用时，**必须**遵守以下规则。

---

## 项目身份

QuanFormer 是一个基于 **DETR（ResNet-50 + 1 层 Transformer 编解码器）** 的 LC-MS 代谢组学峰检测与定量工具。

| 属性 | 说明 |
|------|------|
| **核心代码** | `model/` 目录（非项目根目录） |
| **模型权重** | `model/resources/checkpoint0029.pth`（>300MB） |
| **四种分析模式** | Targeted / Untargeted × Centroided / Profile |
| **输入** | `.mzML` 原始质谱数据 + 可选 `feature.csv` |
| **输出** | `area.csv`、`post-area.csv`、EIC 预测图像 |
| **上游来源** | Facebook DETR（`quanformer/` 包 fork 自 DETR） |
| **版权所有** | LinShuhaiLAB, Xiamen University |

---

## 项目技术约束

在做出任何代码修改前，必须遵守以下硬性约束：

### Python & PyTorch
- **Python 版本**：3.10 ~ 3.11（❌ 禁止使用 3.8 或 3.12+）
- **PyTorch 版本**：按 GPU 架构选择对应版本
  - RTX 50 系（Blackwell, sm_120）：≥ 2.7.0+cu128
  - RTX 40/30/20 系：2.6.0+cu124
  - CPU / Apple Silicon (MPS)：2.6.0
- **依赖文件**：以 `model/requirements.txt` 为准（已整合 GPU/CPU 分段配置）

### 设备与路径
- **设备选择**：❌ 严禁硬编码 `device='cuda'`，必须使用 `_get_best_device()`（`utils/predict_utils.py`，CUDA > MPS > CPU 自动检测）
- **模型加载**：❌ 严禁直接用 `torch.load()`，必须使用 `safe_torch_load()`（`utils/io_utils.py`）以兼容跨 PyTorch 版本
- **路径分隔符**：❌ 严禁硬编码 `/` 或 `\\`，必须使用 `os.path.join()` / `pathlib.Path`
- **路径空格**：注意 Windows 路径含空格时 R 调用（`find_peaks.R`）可能失败，需加引号

### 架构边界
- `quanformer/` 包：模型定义 + 训练（纯 DETR，不依赖业务逻辑）
- `utils/` 包：推理辅助（EIC 提取、定量、后处理、绑图）
- `GUI/`：PySide6 图形界面（独立于 CLI 管线）
- 两个包职责不同，不要交叉引用

### 已知陷阱（详见 `docs/Bugs.md`）
- `batch_size=1` 逐张推理导致 GPU 利用率仅 ~15%
- `plot_results` 用 `joblib(n_jobs=-1)` 可能导致 I/O 争抢
- Untargeted 模式需要安装 R + Bioconductor（`find_peaks.R` 调用 CentWave）
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

## 强制规则：代码修改记录

每次通过编辑工具（`replace_string_in_file`、`create_file` 等）完成文件修改后，在内存中暂存改动信息。任务结束时**一次性**批量写入 `docs/Modified.md`。
- 每次编辑工具调用对应一条记录，不合并
- 改动人仅在任务结束时询问一次（不每次打断）
- 详细规范见 `.github/skills/code-change-logger/SKILL.md`

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