# MRMPFormer 色谱图专用增强 — 设计文档

> 日期：2026-07-20 | 状态：待审阅（v2 — 新增配置文件设计）

---

## 1. 背景与动机

### 1.1 问题

MRMPFormer 当前使用标准 SimCLR 增强策略（`RandomResizedCrop(0.08~1.0)` + `ColorJitter` + `Grayscale` + `GaussianBlur(kernel=23)`），该策略为 ImageNet 自然图像设计。但训练数据是 **XIC 时间窗口 ROI 截取图像**（400×300, 4:3），两者存在本质差异：

| 维度 | ImageNet 自然图像 | XIC ROI 图像 |
|------|:---:|:---:|
| 颜色 | 真实物体颜色，色彩是强语义信号 | 伪彩色/灰度，颜色无物理意义 |
| 结构 | 物体可出现在任意位置和大小 | 色谱峰有固定 x 轴（RT）语义 |
| 尺度 | 极端缩放仍有语义 | 缩小到 0.08 倍 = 噪声块 |
| 变异源 | 视角/光照/遮挡 | 保留时间漂移/仪器分辨率/离子化效率 |

### 1.2 目标

构建一套**色谱图专用增强策略（方案 B）**，与标准 SimCLR（方案 A）进行对比实验，通过表征质量指标判定最优方案。

---

## 2. 数据概况

| 属性 | 值 |
|------|-----|
| 图片数量 | 699 |
| 分辨率 | 400×300 (W:H = 4:3) |
| 颜色模式 | RGB |
| 图像类型 | XIC 提取离子流图（ROI 时间窗口） |
| 化合物种类 | 378 种 |
| 每类样本数 | 1~2 张（min=1, max=2, avg=1.8） |

---

## 3. 方案 B 设计

### 3.1 增强 Pipeline

```
方案 A (SimCLR)                         方案 B (Chromatogram)
────────────────────────                ────────────────────────────
                                        RandomRTShift(±8%)          ← 新增
RandomResizedCrop(0.08~1.0)       →    Resize(168,224) + Pad(28,28) ← 替换: 零损失
HorizontalFlip                    →    HorizontalFlip               ← 保留
ColorJitter                       →    (删除)                       ← 不适用
RandomGrayscale                   →    (删除)                       ← 不适用
GaussianBlur(kernel=23)           →    GaussianBlur(kernel=5)        ← 替换: 弱化
ToTensor                          →    ToTensor                     ← 保留
Normalize(ImageNet)               →    Normalize(ImageNet)           ← 保留
```

### 3.2 各步骤详解

| 序号 | 操作 | 参数 | 设计理由 |
|:---:|------|------|------|
| ① | **RandomRTShift** | `max_shift=0.08` (±32px) | 模拟 LC 柱保留时间漂移，覆盖 ±0.1~0.3 min 典型范围 |
| ② | **Resize + Pad** | `resize=(168,224)` → `pad(top=28,bottom=28)` | 4:3→1:1 零信息损失；pad_mode='edge' 基线自然延续 |
| ③ | **HorizontalFlip** | `p=0.5` | 色谱峰天然对称，翻转不变性合理 |
| ④ | **GaussianBlur** | `kernel=5, sigma=(0.1,1.0)` | 模拟不同分辨率质谱仪采集效果，远弱于原方案 kernel=23 |
| ⑤ | **ToTensor** | — | 标准转换 |
| ⑥ | **Normalize** | ImageNet μ/σ | 与方案 A 一致，保证对比公平 |

### 3.3 宽高比处理：Resize + Pad 策略

```
原始: 400×300 (4:3)
  → Resize((168, 224)): 宽边=224, 高=224×3/4=168
  → Pad(top=28, bottom=28): 补成 224×224

信息损失: 0%
```

Padding 模式：
- **默认**: `mode='edge'` — 边缘像素延续，基线自然延伸，无人工痕迹
- **注释备用**: `mode='constant', fill=255` — 纯白边填充

### 3.4 刻意不做的增强 & 理由

| 不做的操作 | 理由 |
|------|------|
| ColorJitter (hue/saturation) | XIC 伪彩色无物理意义，引入无意义变异 |
| RandomGrayscale | 图已是灰度/伪彩色，灰度化无区分度 |
| RandomVerticalFlip | 强度轴有方向性（峰向上） |
| RandomRotation | RT 轴是水平的，旋转破坏时间/强度关系 |
| IntensityJitter (brightness/contrast) | 可能产生"假信号"（放大基线噪声）、改变峰高比（分析关键信息） |
| 极端裁剪 (scale<0.3) | 可能切碎色谱峰 |

### 3.5 代码结构

```
MRMPFormer/
├── augmentations.py    ← 新增: RandomRTShift 类
│                         ← 新增: get_chromatogram_augmentations(config)
│                         ← 不变: get_simclr_augmentations()
│                         ← 不变: get_test_augmentations()
├── utils/
│   ├── __init__.py      ← 新增导出: ExperimentConfig, PRESETS
│   ├── config.py        ← 新建: 所有训练/模型/增强参数集中定义
│   └── losses.py        ← 不变
├── train.py            ← 重写: get_args() 精简为仅 --config 参数
│                         ← 修改: main() 从 config 读取所有参数
├── dataset.py          ← 不变
├── models/simclr.py    ← 不变
└── extract_features.py ← 不变（后续可复用 ExperimentConfig）
```

**新增文件**：
- `utils/config.py`：`ExperimentConfig` dataclass + 2 组预设实验配置

**新增类**：
- `RandomRTShift(nn.Module)`: 水平随机平移，`padding_mode='edge'`

**新增函数**：
- `get_chromatogram_augmentations(config: ExperimentConfig)`: 从配置构建色谱专用增强 pipeline

### 3.6 配置文件设计

所有参数从 `train.py` 的 `argparse` 迁移到 `utils/config.py` 的 `ExperimentConfig` dataclass，每个参数附带中文注释说明用途。

**参数分组**：

```python
@dataclass
class ExperimentConfig:
    name: str                      # 实验标识
    data_dir: str                  # 数据
    proj_hidden_dim: int           # 模型架构
    proj_output_dim: int
    pretrained: bool
    batch_size: int                # 训练超参
    epochs: int
    lr: float
    weight_decay: float
    temperature: float
    gradient_accumulation: int
    augmentation: str              # 数据增强
    rt_shift: float                # (chromatogram 专用)
    blur_kernel: int               # (chromatogram 专用)
    pad_mode: str                  # (chromatogram 专用)
    output_dir: str                # 保存与日志
    log_dir: str
    save_every: int
    num_workers: int               # 运行环境
    seed: int
```

**预设配置**：

```python
SIMCLR_BASELINE = ExperimentConfig(
    name="simclr_baseline",
    augmentation="simclr",
)

CHROMATOGRAM_V1 = ExperimentConfig(
    name="chromatogram_v1",
    augmentation="chromatogram",
    rt_shift=0.08,
    blur_kernel=5,
    pad_mode="edge",
)

PRESETS = {"simclr_baseline": SIMCLR_BASELINE,
           "chromatogram_v1": CHROMATOGRAM_V1}
```

**train.py 接入**：

```python
# get_args() 精简为：
def get_args():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--config', type=str, default='simclr_baseline',
                        choices=['simclr_baseline', 'chromatogram_v1'])
    return parser.parse_args()

# main() 中一行获取所有参数：
config = PRESETS[args.config]
```

---

## 4. 对比实验设计

### 4.1 实验命令

```bash
# 实验 1：标准 SimCLR（使用预设配置 simclr_baseline）
python train.py --config simclr_baseline

# 实验 2：色谱图专用增强（使用预设配置 chromatogram_v1）
python train.py --config chromatogram_v1
```

权重自动保存到按实验名称隔离的目录：
```
checkpoints/simclr_baseline/best_model.pth
checkpoints/chromatogram_v1/best_model.pth
```

### 4.2 控制变量

两组实验使用**完全相同**的：
- 训练数据（699 张 XIC ROI 图像）
- 模型架构（ResNet50 + 投影头 MLP）
- 训练超参（batch_size=256, lr=3e-4, τ=0.5, epochs=300）
- 损失函数（NT-Xent）
- 优化器与调度器（AdamW + CosineAnnealing）

### 4.3 评估指标体系

#### L1：训练过程指标

| 指标 | 方法 | 判断标准 |
|------|------|----------|
| Loss 收敛曲线 | TensorBoard 叠加两组 | 稳定下降至 < 2.0 |
| 收敛速度 | 达到 loss=3.0 的 epoch 数 | 差异 >50 epoch 需关注 |
| 防坍塌 | 监控 loss 不快速降至 < 0.5 | loss>0.5 正常 |

#### L2：表征质量指标

| 指标 | 方法 | 判断标准 |
|------|------|----------|
| 正样本相似度 | `cos(view_a, view_b)` 均值 | > 0.6 |
| 负样本分离度 | 不同化合物 pair 余弦相似度分布 | 均值≈0, std 大 |
| t-SNE 可视化 | 2048-d → 2D，按化合物着色 | 同类聚集 |
| 维度利用率 | 协方差矩阵有效秩 | > 500/2048 |
| 方差分布 | 各维度方差排序图 | 长尾不陡降 |

#### L3：判定规则

```
≥3 项指标胜出 → 推荐该方案
否则 → 差异不显著，优先选参数更少的 Chromatogram
```

---

## 5. 参数完整对照

| 参数 | 方案 A (SimCLR) | 方案 B (Chromatogram) |
|------|:---:|:---:|
| 裁剪/缩放 | RandomResizedCrop 224, scale=(0.08,1.0) | Resize(168,224) + Pad(28,28) |
| RT 漂移 | ❌ | RandomRTShift(±8%) |
| 翻转 | HorizontalFlip p=0.5 | HorizontalFlip p=0.5 |
| 颜色抖动 | ColorJitter(0.4)×4, p=0.8 | ❌ |
| 灰度化 | RandomGrayscale p=0.2 | ❌ |
| 模糊 | GaussianBlur kernel=23, sigma=(0.1,2.0) | GaussianBlur kernel=5, sigma=(0.1,1.0) |
| 归一化 | ImageNet | ImageNet |
| 信息保留 | 不定 (0.08~100%) | 100% (零损失) |
| 变异源数量 | 5 | 3 |

---

## 6. 执行清单

- [ ] 1. 新建 `utils/config.py`：`ExperimentConfig` dataclass + `SIMCLR_BASELINE` + `CHROMATOGRAM_V1` 预设
- [ ] 2. 修改 `utils/__init__.py`：导出配置相关符号
- [ ] 3. 实现 `augmentations.py`：`RandomRTShift` 类 + `get_chromatogram_augmentations(config)` 函数
- [ ] 4. 修改 `train.py`：`get_args()` 精简 + `main()` 从 config 读取 + 输出目录按 `config.name` 隔离
- [ ] 5. 运行方案 A：`python train.py --config simclr_baseline`
- [ ] 6. 运行方案 B：`python train.py --config chromatogram_v1`
- [ ] 7. 提取两组特征
- [ ] 8. 运行评估脚本，生成对比报告
- [ ] 9. 根据判定规则选择最优方案

---

## 7. 技术约束

- Python 3.10~3.11（符合 CLAUDE.md）
- 路径使用 `pathlib.Path`
- 设备自动检测 `get_best_device()`（CUDA > MPS > CPU）
- 不硬编码路径分隔符
- 遵循 `CLAUDE.md` 所有规范
