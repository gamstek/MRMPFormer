# QuanFormer

## 简介

QuanFormer 是一个基于 Python 的峰（特征）检测与定量方法，用于原始 profile 模式 LC-MS 数据。
其核心思路是：结合 CNN 与 Transformer 训练目标检测网络，在 ROI 中识别真实峰（判断真峰/假峰）并定位峰边界以进行积分面积定量。
本方法目前面向高分辨率 LC-MS 代谢组学数据开发，但也可应用于其他以峰为检测目标的场景。

**支持的格式**: `.mzML`

当前开发版本：v0.2.1

---

## 项目结构

```
Quanformer/
├── README.md                         # 项目说明
├── dev_log.md                        # 开发日志
├── CLAUDE.md                         # 项目级 AI 辅助指令
├── ion_zenith.py                     # 离子天顶角计算脚本
├── data/                             # 测试数据
│   ├── test1/                        # 测试集 1（含 mzML / label / train）
│   └── test2/                        # 测试集 2
├── docs/                             # 文档
│   ├── PROJECT_PANORAMA.md           # 项目全景
│   ├── Bugs.md                       # 已知问题
│   ├── QuanFormer 跨平台部署指南.md   # 部署说明
│   └── 项目方案设计模板.md            # 方案模板
├── paper/                            # 论文 & 补充材料
└── main_model/                       # ⭐ 核心代码
    ├── requirements.txt              # 统一依赖（GPU/CPU 分段配置）
    ├── environment.yml               # Conda 环境定义
    ├── main.py                       # 命令行入口
    ├── getFeature.py                 # Untargeted 特征提取
    ├── GUI/                          # 图形界面
    │   ├── ms-main.py                # GUI 入口
    │   ├── ms.py                     # PySide6 UI 逻辑
    │   └── ms.ui                     # Qt Designer 布局文件
    ├── utils/                        # 工具模块
    │   ├── predict_utils.py          # 推理核心（自动设备检测）
    │   ├── extract_eic.py            # EIC 提取
    │   ├── quantify.py               # 定量积分
    │   ├── postprocess.py            # 后处理去重
    │   ├── io_utils.py               # 跨版本模型加载
    │   ├── detection_helper.py       # 检测辅助
    │   ├── plot_utils.py             # 绑图工具
    │   └── find_peaks.R              # R 脚本（untargeted 峰查找）
    ├── quanformer/                   # DETR 模型包
    │   ├── main.py                   # 训练入口
    │   ├── engine.py                 # 训练/评估引擎
    │   ├── hubconf.py                # Torch Hub 配置
    │   ├── datasets/                 # COCO 数据集 & 数据增强
    │   ├── models/                   # 模型定义
    │   │   ├── backbone.py           # ResNet-50 backbone
    │   │   ├── detr.py               # DETR 整体架构
    │   │   ├── transformer.py        # Transformer 编解码器
    │   │   ├── position_encoding.py  # 位置编码
    │   │   ├── matcher.py            # 匈牙利匹配器
    │   │   └── segmentation.py       # 分割头
    │   └── util/                     # 模型工具（bbox / 可视化）
    ├── resources/                    # 示例数据 & 模型权重
    │   ├── checkpoint0029.pth        # 预训练权重 (>300MB)
    │   └── example/                  # 示例 mzML / feature / 输出
    └── workbooks/                    # 数据分析脚本
        ├── boxplot&CV.ipynb          # 箱线图 & CV 分析
        ├── calcQuantificationResults.py
        ├── peakAlignment.py          # 峰对齐
        └── ...
```

---

## 操作系统兼容性

| 操作系统 | CPU 模式 | GPU 模式 | 备注 |
|----------|:--------:|:--------:|------|
| Windows 10/11 | ✅ | ✅ NVIDIA CUDA | 推荐 |
| Linux (Ubuntu 20.04+) | ✅ | ✅ NVIDIA CUDA | 推荐 |
| macOS (Apple Silicon M1-M4) | ✅ | ⚠️ MPS 加速（实验性） | 见下方说明 |
| macOS (Intel) | ✅ | ❌ | CPU only |

> **GPU 说明**：
> - **NVIDIA GPU**：使用 CUDA 加速，推理速度最快。
> - **Apple Silicon (M1-M4)**：代码已支持 MPS 加速（`torch.backends.mps`），在 `predict_utils.py` 中会自动检测并使用。训练脚本也支持 `--device auto` 自动选择最佳设备。
> - **AMD GPU (ROCm)**：PyTorch 官方未提供 Windows 上的 ROCm 支持；Linux 下可尝试 ROCm 版 PyTorch。
> - **无 GPU**：自动回退 CPU 模式。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | **3.10 ~ 3.11**（⚠️ 不要用 3.8 或 3.12+） |
| PyTorch | 按 GPU 型号选择对应版本（见下表） |
| Conda | 推荐 Miniconda / Anaconda |
| R (可选) | 4.0+（仅 untargeted 模式需要） |

| GPU 系列 | CUDA | torch | torchvision |
|----------|------|-------|-------------|
| RTX 50 (5060–5090) | 12.8 (cu128) | ≥2.7.0 | ≥0.22.0 |
| RTX 40 (4060–4090) | 12.4 (cu124) | 2.6.0 | 0.21.0 |
| RTX 30 / 20 / GTX 16 | 12.4 (cu124) | 2.6.0 | 0.21.0 |
| CPU / Apple Silicon | — | 2.6.0 | 0.21.0 |

> **Python 版本说明**：PyTorch 不支持 Python 3.8 及 3.12+。本项目的 `requirements.txt` 和 `environment.yml` 均以 Python 3.10/3.11 为准。
> `requirements.txt` 已内置上述所有配置段，按需取消注释即可。

---

## 环境检测（推荐先运行 ✅）

在安装之前，运行以下命令即可**自动检测**当前机器是否满足所有依赖：

```bash
python .github/skills/check-dependencies/check_gui.py
```

弹窗会展示：

| 检测内容 | 说明 |
|----------|------|
| Conda 环境 | 是否存在 `quanformer` 环境，支持一键创建 |
| Python 版本 | 是否在 3.10 ~ 3.11 范围内 |
| pip 包 | 逐一比对 `requirements.txt` 版本约束（`==` / `>=` / 范围） |
| PyTorch / CUDA | 是否按 GPU 架构（RTX 50/40/30/20）正确安装 |
| GPU | 型号 + 计算能力 |
| 模型权重 | `checkpoint0029.pth` 是否存在且 >300MB |
| R (可选) | Untargeted 模式所需运行时和包 |
| 磁盘空间 | 是否 ≥ 2GB 可用 |

如有缺失，弹窗支持**一键修复**（自动安装缺失的 pip 包到正确版本）。

> 纯终端环境可用 `python .github/skills/check-dependencies/check_env.py` 输出文本报告。

---

## 安装

### 方式一：Conda（推荐 ✅）

```bash
# 1. 进入项目目录
cd Quanformer/main_model

# 2. 创建 conda 环境（Python 3.11）
conda env create -f environment.yml
conda activate quanformer

# 3. 无 GPU 时替换为 CPU 版 PyTorch:
# pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0

# 4. 验证安装
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
python -c "import pymzml; print('pymzml OK')"
```

### 方式二：pip + venv

```bash
cd Quanformer/main_model

# 创建虚拟环境
python -3.11 -m venv venv

# 激活
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装（RTX 50 系默认激活；其他 GPU 编辑 requirements.txt 切换对应段落后再执行）
pip install -r requirements.txt

# 验证
python -c "import torch; print('OK')"
```

> **注意**：确保 `resources/checkpoint0029.pth` 模型文件存在且大于 300MB。
> 若缺失可从 [模型权重](resources/checkpoint0029.pth) 下载。

---

## 不同平台注意事项

### Windows
- 路径中如有空格（如 `C:\Program Files\...`），untargeted 模式下调用 R 可能会失败。建议使用不含空格的路径。
- `pycocotools` 在 Windows 上安装可能需要 Visual C++ Build Tools。如遇安装问题可尝试 `pip install pycocotools-windows`。

### Linux (Ubuntu)
- 无桌面环境的服务器运行 GUI 模式需要 X11 转发（`ssh -X`）或虚拟显示器（`xvfb`）。
- Untargeted 模式需要安装 R 和 Bioconductor 包（见下方说明）。

### macOS
- **Apple Silicon (M1-M4)**：PyTorch 会自动使用 MPS 加速，推理性能优于纯 CPU。
- **Intel Mac**：仅支持 CPU 模式。
- 首次运行 GUI 可能需要允许未签名应用（「系统偏好设置 → 安全性与隐私」）。

---

## 使用

### 1. 数据准备

1. 在 `mzML` 文件夹中放入 `.mzML` 文件，并准备 `feature.csv`：
   ```
   ├── mzML
      ├── BC1.mzML
      ├── BC2.mzML
      └── BC3.mzML
   ```
2. 若运行 targeted 定量，需准备以下格式的 `feature.csv`；否则跳过：
   ```
   feature.csv 包含以下列：
   1. Compound Name（唯一编号）
   2. mz
   3. RT
   ```

### 2. 参数说明

#### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--type` | `mzML` | 原始数据类型，目前仅支持 mzML |
| `--ppm` | `10` | ROI 提取的 PPM 容差 |
| `--source` | `resources/example/centroided` | 原始数据目录 |
| `--feature` | `resources/example/centroided_feature.csv` | 特征文件路径。非空则使用 targeted 模式；留空则使用 untargeted 模式 |
| `--images_path` | `resources/example/centroided_output` | ROI 输出路径 |
| `--output` | `.../area.csv` | 结果输出路径 |
| `--threshold` | `0.99` | 仅保留置信度 > 0.99 的预测 |
| `--model` | `resources/checkpoint0029.pth` | 峰检测模型路径 |
| `--roi_plot` | `True` | 是否绘制 ROI（首次运行须为 True） |
| `--plot` | `True` | 是否绘制预测结果 |
| `--num_classes` | `1` | 类别数 |
| `--smooth_sigma` | `0` | 平滑 sigma 值 |
| `--processes_number` | `1` | 并行进程数 |

#### Untargeted 模式参数 (centWave 算法)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--polarity` | `positive` | 极性 (positive/negative) |
| `--minWidth` | `5` | 最小峰宽 |
| `--maxWidth` | `50` | 最大峰宽 |
| `--s2n` | `5` | 信噪比阈值 |
| `--noise` | `100` | 噪声水平 |
| `--mzDiff` | `0.005` | m/z 差异 |
| `--prefilter` | `3` | 预过滤参数 |

### 3. 命令行运行

#### 3.1 Targeted 模式（Centroided 数据）

```shell
python main.py --ppm 10 \
  --source resources/example/centroided \
  --feature resources/example/centroided_feature.csv \
  --images_path resources/example/centroided_output \
  --output resources/example/centroided_output/area.csv \
  --model resources/checkpoint0029.pth
```

#### 3.2 Targeted 模式（Profile 数据）

示例数据下载：[Google Drive](https://drive.google.com/drive/folders/1JopRY0mgMxRGg45iXiBgbT-i7uG3M3tS?usp=drive_link)

```shell
python main.py --ppm 10 \
  --source resources/example/profile \
  --feature resources/example/profile_feature.csv \
  --images_path resources/example/profile_output \
  --output resources/example/profile_output/area.csv \
  --model resources/checkpoint0029.pth
```

#### 3.3 安装 R（Untargeted 模式前置条件）

- **R 版本**：4.4.2，xcms 版本：4.4.0
- R 依赖包打包下载：[Google Drive](https://drive.google.com/file/d/1oEIANtyXztyRkKUcWpUh3jznCG2trHwv/view?usp=drive_link)

首先检查 R 是否已安装：

```shell
R --version
```

**Ubuntu/Debian 安装 R**（详见 [CRAN](https://cran.r-project.org/bin/linux/ubuntu/)）：

```shell
sudo apt-get update
sudo apt update -qq
sudo apt install --no-install-recommends software-properties-common dirmngr
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
sudo apt install --no-install-recommends r-base
sudo apt-get install libxml2-dev
```

安装 R 包（详见 [Bioconductor](https://www.bioconductor.org/packages/release/bioc/html/xcms.html)）：

```shell
sudo R
```

在 R 控制台中执行：

```r
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("xcms")
BiocManager::install("MSnbase")
install.packages("dplyr")
```

#### 3.4 Untargeted 模式

`--feature` 参数留空或不设置即可进入 untargeted 模式：

```shell
python main.py --ppm 10 \
  --source resources/example/centroided \
  --polarity positive --minWidth 5 --maxWidth 50 \
  --s2n 5 --noise 100 --mzDiff 0.005 --prefilter 3 \
  --images_path resources/example/untargeted_centroided_output \
  --output resources/example/untargeted_centroided_output/area.csv \
  --model resources/checkpoint0029.pth \
  --processes_number 2
```

> **注意**：若出现 `FileNotFoundError: ... xcms_peak_list.csv`，说明 R 环境或依赖包未正确安装，请回到步骤 3.3。

### 4. GUI 模式

#### 4.1 Targeted 模式

```shell
python GUI/ms-main.py
```

#### 4.2 Untargeted 模式

由于 centWave 模块配置复杂且运行耗时，建议先在命令行中运行 ROI 搜索，再用 GUI 读取生成的 feature 表：

```shell
python getFeature.py --source resources/example/centroided \
  --polarity positive --ppm 10 --minWidth 5 --maxWidth 50 \
  --s2n 5 --noise 100 --mzDiff 0.015 --prefilter 3
```

![GUI](resources/GUI.png)

---

## 依赖版本说明

核心依赖及其版本锁定策略：

| 包 | 版本 | 说明 |
|----|------|------|
| `torch` | 2.6.0 / ≥2.7.0 | 随 GPU 架构不同，见环境要求表 |
| `torchvision` | 0.21.0 / ≥0.22.0 | 与 torch 版本配套 |
| `numpy` | 1.26.4 | 精确锁定，避免兼容问题 |
| `pandas` | 2.2.2 | 精确锁定 |
| `scipy` | 1.13.1 | 精确锁定 |
| `pymzml` | 2.5.10 | 质谱数据解析 |
| `pyside6` | ≥6.7.0, <6.8.0 | GUI 框架 |
| `pycocotools` | ≥2.0.0 | COCO 评估工具 |
| `matplotlib` | 3.9.2 | 绑图 |
| `joblib` | 1.4.2 | 并行处理 |
| `pillow` | ≥10.0.0, <11.0.0 | 图像处理 |

> 若遇到依赖冲突，可将 `requirements.txt` 中 `==` 改为 `>=` 尝试。

---

## 训练模型（高级）

```shell
cd main_model
python quanformer/main.py \
  --coco_path data/peak-all \
  --output_dir output \
  --device auto
```

`--device auto` 会自动选择最佳可用设备（CUDA > MPS > CPU）。

---

## 常见问题 (FAQ)

<details>
<summary><b>Q: 运行时提示 CUDA 不可用？</b></summary>

确认 NVIDIA 驱动已安装且 `nvidia-smi` 正常。运行：
```shell
python -c "import torch; print(torch.cuda.is_available())"
```
若返回 `False`，请重装对应 CUDA 版本的 PyTorch。
</details>

<details>
<summary><b>Q: macOS M1/M2/M3/M4 上如何加速？</b></summary>

代码会自动检测 MPS 并使用。验证：
```shell
python -c "import torch; print(torch.backends.mps.is_available())"
```
若返回 `True`，推理将自动使用 GPU 加速。
</details>

<details>
<summary><b>Q: 无 GPU 可以运行吗？</b></summary>

可以。编辑 `requirements.txt`，取消 `CPU Only` 段注释（含 `--index-url .../cpu`），注释掉其他 GPU 段，再 `pip install -r requirements.txt` 即可。代码会自动使用 CPU。
</details>

<details>
<summary><b>Q: Windows 上出现路径相关错误？</b></summary>

尽量避免路径中包含**空格**和**中文字符**。推荐使用类似 `D:\data\quanformer\` 的简洁路径。
</details>

---

更详细的模型训练说明参见：[User Guide.pdf](User%20Guide.pdf)。

> 跨平台部署详情参见：[docs/QuanFormer 跨平台部署指南.md](docs/QuanFormer%20跨平台部署指南.md)。
