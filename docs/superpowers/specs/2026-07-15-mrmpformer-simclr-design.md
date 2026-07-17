# MRMPFormer SimCLR 对比学习特征提取器 — 设计文档

> 日期：2026-07-15 | 状态：待审阅

---

## 1. 目标

在 `MRMPFormer/` 内搭建基于 SimCLR 对比学习的 ResNet50 特征提取器：
- **输入**：任意二维图像（无标签）
- **输出**：2048-d 语义特征向量（ResNet50 backbone 输出）
- **训练方式**：自监督对比学习（NT-Xent loss），无需标注

---

## 2. 文件结构

```
MRMPFormer/
├── data/
│   └── images/              ← 训练图像平铺存放（.png/.jpg，无需子文件夹）
├── models/
│   ├── __init__.py
│   └── simclr.py            ← SimCLR 模型（ResNet50 + 投影头 MLP）
├── utils/
│   ├── __init__.py
│   └── losses.py            ← NT-Xent (InfoNCE) 对比损失
├── augmentations.py         ← SimCLR 数据增强策略
├── dataset.py               ← 图像数据集加载器
├── train.py                 ← 训练主脚本
└── extract_features.py      ← 训练后特征提取（去掉投影头，输出 2048-d）
```

---

## 3. 模型架构

```
Input: (3, 224, 224)
    │
    ▼
ResNet50 (ImageNet pretrained, 去掉 FC)
    │  output: (2048,)
    ▼
Projection Head (训练时):
    Linear(2048 → 512) → ReLU → Linear(512 → 128)
    │  output: (128,), L2-normalized
    ▼
NT-Xent Loss (对比学习)

推理时: 去掉投影头, 直接用 2048-d 输出
```

---

## 4. 数据增强（SimCLR 标准策略）

| 序号 | 增强 | 参数 |
|------|------|------|
| 1 | RandomResizedCrop | 224×224, scale=(0.08, 1.0) |
| 2 | RandomHorizontalFlip | p=0.5 |
| 3 | ColorJitter | brightness/contrast/saturation=0.4, hue=0.1 |
| 4 | RandomGrayscale | p=0.2 |
| 5 | GaussianBlur | kernel=23, sigma=(0.1, 2.0) |

---

## 5. 损失函数

- **NT-Xent (InfoNCE)**：余弦相似度 + 温度系数 τ=0.5
- 每个 batch: N 张原图 → 2N 个 view，正样本对对称计算

---

## 6. 训练超参数

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW, lr=3e-4, weight_decay=1e-4 |
| 学习率调度 | CosineAnnealing → 1e-6 |
| Batch size | 256 |
| 温度 τ | 0.5 |
| 投影头维度 | 2048 → 512 → 128 |
| Epochs | 300（默认） |
| 设备 | 自动检测（CUDA > MPS > CPU） |

---

## 7. 技术约束（遵循 CLAUDE.md）

- Python 3.10~3.11
- 路径使用 `pathlib.Path`
- 设备自动检测，不硬编码 `cuda`
- 模型加载使用安全加载方式

---

## 8. 自审清单

- [x] 无标签依赖 → 自监督，符合需求
- [x] 训练/推理分离 → 投影头仅训练时使用
- [x] 设备选择不硬编码
- [x] 路径使用 pathlib
- [x] 无冗余依赖
