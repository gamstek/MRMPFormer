# MRMPFormer 项目全景文档

> 自动生成于 2026-07-07 · 基于全项目遍历

---

## 📁 项目目录树

```
MRMPFormer/
├── CLAUDE.md                          ← 项目级 AI 指令（开发日志规则）
├── dev_log.md                         ← 项目开发日志
├── PROJECT_PANORAMA.md                ← 本文件
├── PROBLEM.md                         ← 已知问题汇总
├── 项目方案设计模板.md                   ← 通用方案设计模板
│
├── paper/                             ← 📄 论文
│   ├── MRMPFormer-A Transformer Based Precise Peak.pdf
│   ├── MRMPFormer-translated.pdf
│   └── MRMPFormer_Supporting Information.pdf
│
├── data/                              ← 🏷️ 训练/测试数据
│   └── test/
│       ├── mzML/test.mzML             ← 原始质谱
│       ├── train/B1~B3/ (*.jpeg)      ← 训练 EIC 图像
│       └── label/B1~B3/ (*.png)       ← 标注真值
│
├── .cursor/skills/dev-log-writer/     ← 🛠️ IDE Skill 定义
├── .vscode/settings.json
├── .idea/                             ← JetBrains IDE 配置
│
└── main_model/                        ← 🧠 核心代码
    ├── main.py                        ← 【推理入口】命令行脚本
    ├── getFeature.py                  ← 【Untargeted】独立 CLI
    ├── environment.yml                ← Conda 环境 (Python 3.11)
    ├── requirements.txt               ← pip GPU 依赖 (CUDA 12.4)
    ├── requirements-cpu.txt           ← pip CPU 依赖
    ├── DEPLOY.md                      ← 跨平台部署指南
    ├── README.md                      ← 项目说明（中文）
    ├── User Guide.pdf                 ← 用户手册
    ├── LICENSE
    │
    ├── GUI/                           ← 🖥️ 图形界面
    │   ├── ms-main.py  (~800行)       ← PySide6 主程序
    │   ├── ms.py       (~800行)       ← 自动生成的 UI 代码
    │   └── ms.ui       (~300行)       ← Qt Designer XML
    │
    ├── mrmpformer/                    ← 🧬 深度学习核心库
    │   ├── main.py     (~150行)       ← 【训练入口】
    │   ├── engine.py   (~120行)       ← 训练/评估引擎
    │   ├── hubconf.py  (~30行)        ← PyTorch Hub 入口
    │   ├── models/
    │   │   ├── detr.py                ← DETR 模型 + 损失
    │   │   ├── backbone.py            ← ResNet-50 骨干
    │   │   ├── transformer.py         ← Transformer 编解码
    │   │   ├── position_encoding.py   ← 正弦位置编码
    │   │   ├── matcher.py             ← 匈牙利匹配
    │   │   └── segmentation.py        ← 分割头（扩展）
    │   ├── datasets/
    │   │   ├── coco.py                ← COCO 数据加载
    │   │   ├── coco_eval.py           ← COCO 评估 (AP/mAP)
    │   │   └── transforms.py          ← 数据增强
    │   └── util/
    │       ├── misc.py                ← 通用工具 + 分布式
    │       ├── box_ops.py             ← 边界框操作 (GIoU)
    │       └── plot_utils.py          ← 训练可视化
    │
    ├── utils/                         ← 🔧 推理辅助
    │   ├── __init__.py                ← 导出 build_roi()
    │   ├── detect_helper.py  (~200行) ← PeakList + CentWave
    │   ├── extract_eic.py    (~200行) ← EIC 提取核心
    │   ├── io_utils.py       (~150行) ← I/O + safe_torch_load
    │   ├── plot_utils.py     (~80行)  ← EIC 绘图
    │   ├── predict_utils.py  (~200行) ← 预测管道
    │   ├── quantify.py       (~100行) ← 峰面积定量
    │   ├── postprocess.py    (~50行)  ← 去重 + 转置
    │   └── find_peaks.R      (~30行)  ← R/CentWave 脚本（已禁用，整体注释）
    │
    ├── workbooks/                     ← 📊 分析工具
    │   ├── boxplot&CV.ipynb
    │   ├── interpretability.ipynb
    │   ├── mergePeakonlyWIthFeature.ipynb
    │   ├── calcQuantificationResults.py
    │   ├── calcTrueOrFalsemarker.py
    │   ├── compare_fc.py
    │   ├── peakAlignment.py
    │   └── peakdetective-application.py
    │
    └── resources/                     ← 📦 资源文件
        ├── quanformer.pth         ← 模型权重 (>300MB)
        ├── GUI.png
        └── example/
            ├── centroided/            ← Centroided 输入 (3×mzML)
            ├── profile/               ← Profile 输入 (需下载)
            ├── centroided_feature.csv ← Targeted 特征 (6 化合物)
            ├── profile_feature.csv    ← Profile 特征 (14 化合物)
            ├── peak_list.csv          ← Untargeted 特征
            ├── centroided_output/     ← ✅ 前人结果 (18 JPEG)
            ├── output/                ← ✅ 另一批 (18 JPEG)
            ├── profile_output/        ← ✅ Profile (472 JPEG)
            └── untargeted_centroided_output/ ← ✅ Untargeted (~7746 JPEG)
```

---

## 🧠 三种分析模式

### 维度一：Targeted vs Untargeted

| | Targeted（靶向） | Untargeted（非靶向） |
|--|--|--|
| **思路** | 已知化合物 → 按清单找 | 未知 → 算法自动扫 |
| **输入** | feature CSV（m/z + RT） | 无（自动检测） |
| **算法** | 直接按 m/z 提取 EIC | R + XCMS CentWave |
| **特征数** | ~6（手动指定） | ~数百~数千 |
| **依赖** | 纯 Python | 需安装 R + Bioconductor |

### 维度二：Centroided vs Profile

| | Centroided | Profile |
|--|--|--|
| **数据** | 峰提取后压缩数据 | 原始连续谱图 |
| **大小** | 小（几百 KB/文件） | 大 |
| **峰形** | 离散点 | 完整高斯形状 |

### 四种组合

| | Targeted | Untargeted |
|--|:--:|:--:|
| **Centroided** | ✅ 最快 demo | ✅ 需装 R |
| **Profile** | ✅ 需下载数据 | ⚠️ 理论可行 |

---

## 🔄 数据流

```
mzML 文件                     feature.csv
    │                              │
    ▼                              ▼
extract_eic()  ←──────────  read_targeted_features()
    │                              │
    ▼                              │
ROI 图像 (.jpeg)                   │
    │                              │
    ▼                              ▼
build_predictor()  ───→  quan_former (ResNet50 + Transformer)
    │
    ▼
预测结果 (prob + bbox)
    │
    ▼
quantify()  ───→  area.csv
    │
    ▼
post_process()  ───→  post-area.csv
```

---

## 🛠️ 依赖关系

### 核心推理链
```
torch ≥ 2.11 (CUDA) → torchvision → PyTorch Hub → ResNet-50
pymzml → 解析 .mzML
numpy / scipy → 数值计算 / 积分
Pillow → 图像读写
matplotlib → EIC / 预测框绘图
joblib → 并行处理
```

### 训练链
```
+ pycocotools → COCO 评估
+ seaborn → 训练曲线
```

### GUI 链
```
+ PySide6 → Qt 界面
```

### Untargeted 链
```
+ R + Bioconductor (MSnbase, xcms) → CentWave
```

---

## ⚠️ 关键路径约束

- **Python**: 3.10 ~ 3.11（不要 3.12+, 不要 3.8）
- **PyTorch**: ≥ 2.11.0+cu128（支持 Blackwell sm_120 / RTX 5060）
- **模型权重**: `checkpoint/quanformer.pth` 必须存在且 >300MB
- **R**: 仅 Untargeted 模式需要
- **路径**: 避免空格和中文
