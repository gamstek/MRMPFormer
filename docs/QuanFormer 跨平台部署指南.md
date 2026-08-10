# MRMPFormer 跨平台部署指南

## 支持的平台

| 操作系统 | CPU 模式 | GPU 模式 (CUDA) | 备注 |
|----------|----------|-----------------|------|
| **Windows 10/11** | ✅ | ✅ (NVIDIA) | 推荐 |
| **Linux (Ubuntu 20.04+)** | ✅ | ✅ (NVIDIA) | 推荐 |
| **macOS (Apple Silicon)** | ✅ | ⚠️ MPS 加速 | 实验性 |

## 前置条件

### 所有平台
- **Python 3.10 或 3.11**（不要用 3.12+，PyTorch 2.6 不兼容）
- **Git**（可选，用于克隆仓库）

### 可选：Untargeted 模式需要
- **R 4.0+** + 包 `MSnbase`、`xcms`
  ```bash
  # 安装 R 后运行:
  R -e "install.packages('BiocManager')"
  R -e "BiocManager::install(c('MSnbase','xcms'))"
  ```

---

## 快速开始

### 方式一：Conda（推荐 ✅）

```bash
# 1. 克隆或解压项目到本地
cd MRMPFormer/main_model

# 2. 创建环境 (有 GPU)
conda env create -f environment.yml
conda activate mrmpformer

# 如果没有 GPU，改为 CPU 模式:
# conda env create -f environment.yml
# conda activate mrmpformer
# pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0

# 3. 验证安装
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import matplotlib; print('matplotlib OK')"
python -c "import pymzml; print('pymzml OK')"
```

### 方式二：pip + venv

```bash
cd MRMPFormer/main_model

# 1. 创建虚拟环境
python -3.11 -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. 安装依赖
# 有 GPU:
pip install -r requirements.txt

# 无 GPU:
pip install -r requirements-cpu.txt

# 3. 验证
python -c "import torch; print('OK')"
```

---

## 运行项目

### 启动 GUI
```bash
cd main_model/GUI
python ms-main.py
```
> 首次启动可能较慢（PySide6 初始化），请耐心等待。

### 命令行模式
```bash
cd main_model

# 预测 (targeted)
python main.py \
  --source resources/example/centroided \
  --feature resources/example/centroided_feature.csv \
  --images_path resources/example/centroided_output

# 训练模型
python mrmpformer/main.py \
  --coco_path data/peak-all \
  --output_dir output
```

---

## 硬件兼容性说明

### GPU（NVIDIA CUDA）
- 需要 NVIDIA 显卡 + CUDA 12.4 驱动
- 训练推荐 VRAM ≥ 8GB
- 推理/预测 VRAM ≥ 4GB 即可

### CPU Only
- 预测速度约为 GPU 的 5-10 倍慢，但完全可用
- 训练不建议仅用 CPU（极慢）

### Apple Silicon (M1/M2/M3)
- PyTorch 2.6 支持 MPS 后端加速
- 安装 CPU 版依赖后，PyTorch 会自动使用 MPS
- GUI (PySide6) 在 macOS 上原生支持

---

## 常见问题排查

| 症状 | 原因 | 解决方案 |
|------|------|----------|
| `KeyboardInterrupt` 在 matplotlib 导入时 | pyparsing 版本过高 | `pip install pyparsing==3.1.4` |
| `Weights only load failed` | PyTorch 2.6 安全策略变更 | 已修复，更新代码即可 |
| `Rscript not found` | 未安装 R | 仅 untargeted 模式需要；targeted 模式不受影响 |
| `No module named 'pycocotools'` | 依赖缺失 | `pip install pycocotools` |
| GUI 闪退 | 显卡驱动/PySide6 | 更新显卡驱动；尝试 `pip install pyside6==6.7.2` |
| `CUDA out of memory` | 显存不足 | 减小 batch size 或用 CPU 模式 |
| `FileNotFoundError: *.mzML` | 路径配置错误 | 检查 `--source` 参数指向正确的数据目录 |

---

## 项目文件结构（部署相关）

```
main_model/
├── requirements.txt          # GPU 依赖
├── requirements-cpu.txt      # CPU 依赖
├── environment.yml           # Conda 环境定义
├── main.py                   # 命令行入口
├── GUI/
│   ├── ms-main.py            # GUI 入口 ⭐
│   ├── ms.py                 # PySide6 UI 代码
│   └── ms.ui                 # Qt Designer 文件
├── utils/                    # 工具模块
│   ├── find_peaks.R          # R 脚本 (untargeted 需要)
│   ├── predict_utils.py      # 预测核心
│   └── ...
├── mrmpformer/               # DETR 模型包
└── resources/                # 示例数据 & 模型权重
    ├── checkpoint0029.pth    # 预训练权重
    └── example/              # 示例数据
```

---

## 版本兼容性矩阵（已验证）

| 组件 | 版本 | 备注 |
|------|------|------|
| Python | 3.10 ~ 3.11 | ❌ 3.12+ 不可用 |
| PyTorch | 2.6.0 | 最低 2.0，推荐 2.6 |
| torchvision | 0.21.0 | 与 PyTorch 版本绑定 |
| CUDA | 12.4 (GPU) | 或 CPU 模式 |
| matplotlib | 3.9.x | pyparsing 需 ≤3.1.4 |
| pymzml | 2.5.x | mzML 文件解析 |
| pandas | 2.2.x | |
| R | ≥4.0 | 仅 untargeted 模式 |

---

> 最后更新: 2026-07-07
