"""
MRMPFormer 实验配置。

所有训练超参数、模型参数、增强参数均在此集中定义。
通过预设配置切换实验，train.py 通过 --config 参数选择。
"""
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """
    一次完整训练实验的参数定义。

    使用方式:
        from utils import SIMCLR_BASELINE, CHROMATOGRAM_V1
        config = SIMCLR_BASELINE   # 或 CHROMATOGRAM_V1
        print(config.name)          # 'simclr_baseline'
    """

    # ===================================================================
    # 实验标识
    # ===================================================================

    # 实验名称（用于输出目录和日志区分，不同实验的权重不会相互覆盖）
    name: str = "simclr_baseline"

    # ===================================================================
    # 数据
    # ===================================================================

    # 训练图像存放目录，支持 .png / .jpg / .jpeg / .bmp / .tiff / .webp 格式
    data_dir: str = "MRMPFormer/data/images"

    # ===================================================================
    # 模型架构
    # ===================================================================

    # 投影头隐藏层维度：MLP 的中间层大小（backbone_dim → proj_hidden_dim → proj_output_dim）
    proj_hidden_dim: int = 512

    # 投影头输出维度：最终投影向量的维数，用于对比学习的余弦相似度计算
    proj_output_dim: int = 128

    # 是否加载 ImageNet 预训练权重作为 ResNet50 backbone 初始化
    pretrained: bool = True

    # 冻结 backbone 前 N 个 stage (0=全部训练, 4=仅训练 layer4+投影头, 5=线性探针)
    # stage 顺序: stem → layer1 → layer2 → layer3 → layer4
    freeze_stages: int = 4

    # ===================================================================
    # 训练超参数
    # ===================================================================

    # 每批处理的图像数量，受 GPU 显存限制（显存不足时可减小此值或增大 gradient_accumulation）
    batch_size: int = 40

    # 训练总轮数，每轮遍历全部数据一次
    epochs: int = 300

    # 优化器初始学习率（AdamW），训练过程中随 CosineAnnealing 调度器降至 1e-6
    lr: float = 3e-4

    # 权重衰减系数（L2 正则化），防止过拟合
    weight_decay: float = 1e-4

    # NT-Xent 损失函数的温度系数 τ，越小对难负样本越敏感（典型范围 0.1 ~ 1.0）
    temperature: float = 0.5

    # 梯度累积步数：当 batch_size 太大无法放入 GPU 时，增大此值可模拟更大的有效 batch
    gradient_accumulation: int = 2

    # ===================================================================
    # 早停 (Early Stopping)
    # ===================================================================

    # 是否启用早停（设为 False 则训练满所有 epoch）
    early_stopping_enabled: bool = True

    # 容忍轮数：连续 N 个 epoch 无显著改善后停止训练
    early_stopping_patience: int = 50

    # 最小改善阈值（相对值）：|best - curr| / best < 此值视为无改善
    # 1e-4 = 0.01%，通常设 1e-4 ~ 1e-3
    early_stopping_min_delta: float = 1e-4

    # 早停保护期：前 N 个 epoch 不触发早停（给 CosineAnnealing 充分的快速下降时间）
    early_stopping_min_epochs: int = 100

    # ===================================================================
    # 评估指标
    # ===================================================================

    # 每隔多少 epoch 计算一次 Alignment + Uniformity（0 = 不计算）
    # 计算使用一个 batch 的数据，无梯度，开销极小
    eval_metrics_every: int = 10

    # ===================================================================
    # 数据增强
    # ===================================================================

    # 增强策略名称
    #   "simclr"        — 标准 SimCLR 增强（RandomResizedCrop + ColorJitter + Grayscale + GaussianBlur）
    #   "chromatogram"  — 色谱图专用增强（RandomRTShift + ResizePad + HorizontalFlip + MildGaussianBlur）
    augmentation: str = "simclr"

    # ----- 以下参数仅在 augmentation="chromatogram" 时生效 -----

    # 保留时间漂移最大比例：模拟 LC 色谱柱老化导致的 RT 整体偏移
    # 值的含义 = 最大平移像素 / 图像宽度（0.08 = ±32px on 400px 宽图像）
    rt_shift: float = 0.08

    # 高斯模糊核大小：模拟不同分辨率质谱仪的采集效果
    # 必须是奇数，越小保留的峰细节越多（5 = 轻度模糊，23 = 重度模糊）
    blur_kernel: int = 5

    # 补边模式（将 4:3 图像补成 1:1 方图时的填充策略）
    #   "edge"      — 边缘像素延续（基线自然延伸，无人工痕迹）【默认推荐】
    #   "constant"  — 纯白边填充（fill=255，人为引入边界信号）
    pad_mode: str = "edge"

    # ===================================================================
    # 保存与日志
    # ===================================================================

    # 模型权重保存目录，会在其下按 config.name 创建子目录
    # 例如: checkpoints/simclr_baseline/best_model.pth
    output_dir: str = "MRMPFormer/checkpoints"

    # TensorBoard 日志保存目录，会在其下按 config.name 创建子目录
    log_dir: str = "MRMPFormer/logs"

    # 每隔多少 epoch 保存一次中间 checkpoint（不影响 best_model 的自动保存逻辑）
    save_every: int = 50

    # ===================================================================
    # 运行环境
    # ===================================================================

    # DataLoader 并行加载数据的工作进程数
    # Windows 下建议 0 ~ 4，Linux/macOS 下可设更高（受 CPU 核数限制）
    num_workers: int = 4

    # 随机种子，固定后可复现训练结果，对比实验必须保持一致
    seed: int = 42


# ===================================================================
# 预设实验配置
# ===================================================================

# 实验 A：标准 SimCLR 增强（适用于自然图像的通用对比学习增强策略）
SIMCLR_BASELINE = ExperimentConfig(
    name="simclr_baseline",
    augmentation="simclr",
)

# 实验 B：色谱图专用增强（为 XIC ROI 图像定制的物理合理增强策略）
CHROMATOGRAM_V1 = ExperimentConfig(
    name="chromatogram_v1",
    augmentation="chromatogram",
    rt_shift=0.08,          # ±8% 水平平移，模拟保留时间漂移
    blur_kernel=5,          # 轻度模糊，模拟仪器分辨率差异
    pad_mode="edge",        # 边缘像素延续，基线自然延伸
    # pad_mode="constant",  # 注释备用：纯白边填充（如需对比可取消注释）
)


# 预设配置字典，train.py 通过 --config 参数的名称在此查找
PRESETS = {
    "simclr_baseline": SIMCLR_BASELINE,
    "chromatogram_v1": CHROMATOGRAM_V1,
}
