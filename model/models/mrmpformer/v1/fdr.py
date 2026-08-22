# -*- coding: utf-8 -*-
"""
MRMPFormer v1 — FDR（Fine-grained Distribution Refinement）核心组件。

包含：
  1. make_bin_values / FDRBinBuffer       : 非均匀 Bin 候选偏移 W(n)
  2. FDRHead                              : 每层独立的边界分布 FFN 头
  3. decode_expected_offsets              : 概率分布 → 期望偏移（全程可导）
  4. BoundaryPositionMLP                  : (x_L, x_R) → 边界位置编码
  5. DistributionBoundaryLoss             : FDR 软标签分布监督（工程回退实现，非论文 FGL 原式）
  6. softmax_focal_loss                   : 含显式背景类的 Softmax Focal Loss
  7. dynamic_l1_loss / 权重统计            : 峰中心动态加权 L1
  8. peak_width_weighted_ciou             : PW-CIoU（bar_w 来自训练集统计）

坐标约定（全项目统一）：
  - 归一化坐标 [0,1]，box 格式 cxcywh；
  - FDR 内部精化时用 xyxy 的左右边 x_L / x_R；
  - Δx > 0 表示向右移动，Δx < 0 表示向左移动。
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# 1. Bin 候选偏移 W(n)
# ---------------------------------------------------------------------------
def make_bin_values(num_bins: int, power: float = 2.0) -> Tensor:
    """生成非均匀 Bin 候选偏移 W(n)，n = 0..N-1。

    工程默认方案（非论文原式，项目/论文中未给出精确定义时使用）：
        u_n   = 2n/(N-1) - 1                          ∈ [-1, 1]
        W(n)  = sign(u_n) * |u_n|^p                   默认 p=2.0
    性质：严格单调递增、对称覆盖 0、长度恰为 N。
    允许配置直接传入显式 bin_values 覆盖该生成方式（见 MRMPFormer.__init__）。
    """
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")
    u = torch.linspace(-1.0, 1.0, num_bins, dtype=torch.float32)
    w = torch.sign(u) * torch.abs(u) ** float(power)
    # 防御：浮点误差可能导致 sign(0)*0^p = 0，此处恒为 0，无需特殊处理
    return w


class FDRHead(nn.Module):
    """每层独立的 FDR 分布头。

    结构（提示词 §4.2 / §5）：
        Linear(D, D) → ReLU → Linear(D, D) → ReLU → Linear(D, 2N)
    输出 reshape 为 [B, Q, 2, N]；维度 2 依次为 [左边界, 右边界]。
    Layer1 输出完整分布 z1；Layer>=2 输出上一层累计 Logits 的残差 Δz_k
    （残差累加由模型 forward 完成，保证 Δz=0 时 z 不变）。

    注意：末层 Linear 采用零初始化，使训练初期 Δx≈0（均匀分布期望偏移为
    mean(W)=0），即"精化从恒等映射起步"，数值稳定且不破坏初始框头的学习。
    """

    def __init__(self, d_model: int, num_bins: int):
        super().__init__()
        self.num_bins = num_bins
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 2 * num_bins)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, Q, D] → z or Δz: [B, Q, 2, N]"""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        z = self.out(h)
        return z.view(*x.shape[:-1], 2, self.num_bins)


# ---------------------------------------------------------------------------
# 2. 期望偏移解码（可导）
# ---------------------------------------------------------------------------
def decode_expected_offsets(fdr_logits: Tensor, bin_values: Tensor,
                           scale: Tensor) -> Tensor:
    """累计分布 → 期望偏移（归一化坐标修正量）。

    Δx = s0 * Σ_n P(n) * W(n),  P = Softmax(z)（沿 bin 维）

    Args:
        fdr_logits: [B, Q, 2, N] 累计 Logits z^(k)（不是概率，也不是单层残差）
        bin_values: [N] buffer W
        scale:      [B, Q, 1] 或可广播的尺度因子 s0（默认初始框宽 w0）
    Returns:
        [B, Q, 2] 期望偏移 (Δx_L, Δx_R)
    """
    probs = F.softmax(fdr_logits.float(), dim=-1)          # [B,Q,2,N]
    delta = (probs * bin_values).sum(dim=-1)               # [B,Q,2]
    return delta * scale


# ---------------------------------------------------------------------------
# 3. Boundary Position MLP（逐层边界反馈）
# ---------------------------------------------------------------------------
class BoundaryPositionMLP(nn.Module):
    """(x_L, x_R) → 边界位置编码 [B, Q, D]。

    用于反馈通路：q_pos^(k+1) = q_pos^base + MLP_pos([x_L^k, x_R^k])
    默认门控 g=1（固定），门控作为配置化扩展，本版本不引入。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(2, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, lr: Tensor) -> Tensor:
        """lr: [B, Q, 2] 左右边界 → [B, Q, D]"""
        h = F.relu(self.fc1(lr))
        return self.fc2(h)


# ---------------------------------------------------------------------------
# 4. FDR 分布监督（DistributionBoundaryLoss，工程回退实现）
# ---------------------------------------------------------------------------
# 说明：提示词 §9.2 明确指出论文补充材料未给出 L_FGL 的完整数学定义，
# 本实现是独立、可替换的工程回退方案（两点线性插值软标签 + 软标签交叉熵），
# 命名 DistributionBoundaryLoss，不得在论文/报告中冒充 FGL 原式。
class DistributionBoundaryLoss(nn.Module):
    """左右边界离散分布监督。

    步骤（提示词 §9.2 工程回退）：
      1) 匹配正样本上计算目标偏移 d = (x_gt - x_init) / s0；
      2) 在有序非均匀 Bin W 上找包围 d 的相邻两个 Bin；
      3) 距离线性插值 → 和为 1 的两点软标签；越界裁到端点并统计比例；
      4) 软标签交叉熵监督累计 Logits z（z1/z2/z3）；
      5) 仅正样本参与；s0 加 eps 防除零。
    """

    def __init__(self, num_bins: int, eps: float = 1e-6):
        super().__init__()
        self.num_bins = num_bins
        self.eps = eps

    @staticmethod
    def soft_labels(target: Tensor, bin_values: Tensor) -> tuple[Tensor, Tensor]:
        """target: [*] 目标偏移 → 两点软标签 (weights: [* , N], overflow: [*] bool)。

        weights 在两个相邻 Bin 上按距离线性插值，和为 1；超出 [W0, W_{N-1}]
        裁到端点（标签 1 落在最近端点），overflow 记录越界。
        """
        n = bin_values.shape[0]
        # 搜索 j 满足 W[j] <= target <= W[j+1] 的左邻索引（W 严格递增）
        # searchsorted(right=True)：返回 W 中 <= target 的元素个数-1 后再夹紧
        idx = torch.searchsorted(bin_values, target.contiguous(), right=True) - 1
        idx = idx.clamp(min=0, max=n - 2)                     # 左 Bin 索引 ∈ [0, N-2]
        wl = bin_values[idx]
        wr = bin_values[idx + 1]
        # 插值权重：target == wl → 1 落左；target == wr → 1 落右
        frac = ((target - wl) / (wr - wl).clamp(min=1e-12)).clamp(0.0, 1.0)

        weights = torch.zeros(*target.shape, n, device=target.device, dtype=target.dtype)
        # 纯函数式散点赋值（对 autograd 安全：weights 不需要梯度，梯度经 log_softmax 流向 z）
        weights.scatter_(-1, (idx + 1).unsqueeze(-1), frac.unsqueeze(-1))
        weights.scatter_(-1, idx.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))

        overflow = (target < bin_values[0]) | (target > bin_values[-1])
        return weights, overflow

    def forward(self, fdr_logits: Tensor, target_offsets: Tensor,
                bin_values: Tensor) -> dict:
        """软标签交叉熵。

        Args:
            fdr_logits:    [M, 2, N] 匹配正样本的累计 Logits z^(k)
            target_offsets:[M, 2]    目标偏移 d_L, d_R（已除 s0）
        Returns:
            dict(loss: 标量 tensor, left/right: 标量, overflow_ratio: 标量 tensor,
                 exp_offset_err: 标量 tensor 期望偏移误差 |E[P]-d|)
        """
        if fdr_logits.numel() == 0:
            zero = fdr_logits.sum()  # 保持计算图连接的 0（空目标 batch）
            return {'loss': zero, 'left': zero, 'right': zero,
                    'overflow_ratio': zero.detach(), 'exp_offset_err': zero.detach()}

        tgt = target_offsets.float()
        weights, overflow = self.soft_labels(tgt, bin_values)      # [M,2,N]
        log_p = F.log_softmax(fdr_logits.float(), dim=-1)          # [M,2,N]
        ce = -(weights * log_p).sum(dim=-1)                        # [M,2]

        # 期望偏移误差（诊断用：|E[P] - d|，期望偏移单位为 bin 值域）
        exp_off = (log_p.exp() * bin_values).sum(dim=-1)           # [M,2]
        exp_err = (exp_off - tgt).abs().mean()

        return {
            'loss': ce.mean(),
            'left': ce[:, 0].mean(),
            'right': ce[:, 1].mean(),
            'overflow_ratio': overflow.float().mean().detach(),
            'exp_offset_err': exp_err.detach(),
        }


# ---------------------------------------------------------------------------
# 5. Softmax Focal Loss（含显式背景类，与 QuanFormer 分类头语义一致）
# ---------------------------------------------------------------------------
def softmax_focal_loss(src_logits: Tensor, target_classes: Tensor,
                       num_classes: int, alpha: float = 0.25,
                       gamma: float = 2.0, eos_coef: float = 1.0) -> Tensor:
    """Softmax 形式 Focal Loss（替换 DETR 的 CE + eos_coef 机制）。

    L_FL = -w_t * alpha_t * (1 - p_t)^gamma * log(p_t)
      p_t     = Softmax(logits)[target]
      alpha_t = alpha（前景/峰类）、1 - alpha（背景/no-object 类）
      w_t     = eos_coef（背景，承接 DETR no-object 权重）、1（前景）

    eos_coef 说明（提示词 §8.2 要求正确处理 no-object 权重）：
    QuanFormer 的 CE 用 empty_weight 把 no-object 类降权（默认 0.1），
    换 Focal 后必须保留这层背景压制，否则负样本质量（~34/batch×0.75）
    会压倒正样本（~14/batch×0.25），分类头卡死在 p_peak≈0.3 的梯度
    平衡点（argmax 恒背景、class_error=100）。

    本项目分类头为 [峰, no-object] 两类 Softmax（背景在最后一列），
    因此不能直接使用 Sigmoid Focal Loss；本实现目标编码与 CE 版一致：
    背景查询 target = num_classes（no-object 索引）。
    输入必须是原始 Logits（禁止先 Softmax 再传入）。
    """
    probs = F.softmax(src_logits.float(), dim=-1)                       # [B,Q,C+1]
    target_p = probs.gather(-1, target_classes.unsqueeze(-1)).squeeze(-1).clamp_min(1e-7)
    log_p = target_p.log()

    # alpha_t：前景类（0..C-1）用 alpha，背景 no-object（C）用 1-alpha
    is_bg = (target_classes == num_classes)
    alpha_t = torch.where(is_bg, 1.0 - alpha, alpha).to(src_logits.dtype)
    # w_t：背景额外乘 eos_coef（DETR no-object 权重），前景恒 1
    w_t = torch.where(is_bg, eos_coef, 1.0).to(src_logits.dtype)

    fl = -w_t * alpha_t * torch.pow(1.0 - target_p, gamma) * log_p
    return fl.mean()


# ---------------------------------------------------------------------------
# 6. 动态加权 L1
# ---------------------------------------------------------------------------
def dynamic_l1_weights(target_boxes_cxcywh: Tensor, eps: float,
                       center_weight_clip: Optional[float],
                       normalize: bool) -> tuple[Tensor, dict]:
    """λ_c = 1 / (w_gt + eps)，λ_w = λ_h = 1（可配置）。

    Args:
        target_boxes_cxcywh: [M, 4] (cx, cy, w, h) 归一化 GT
    Returns:
        (weights: [M, 4] 逐坐标权重, stats: P50/P90/P99/Max 中心权重诊断)
    """
    w_gt = target_boxes_cxcywh[:, 2]
    lam_c = 1.0 / (w_gt + eps)
    if center_weight_clip is not None:
        lam_c = lam_c.clamp(max=float(center_weight_clip))
    if normalize:
        lam_c = lam_c / lam_c.mean().clamp_min(eps)
    weights = torch.stack([lam_c, lam_c,
                           torch.ones_like(lam_c), torch.ones_like(lam_c)], dim=-1)
    with torch.no_grad():
        # q 张量必须与输入同设备（CUDA 训练时 CPU 张量会报错）
        q = torch.quantile(lam_c.detach().float(),
                           torch.tensor([0.5, 0.9, 0.99], device=lam_c.device))
        stats = {'p50': q[0].item(), 'p90': q[1].item(),
                 'p99': q[2].item(), 'max': lam_c.max().item()}
    return weights, stats


def dynamic_l1_loss(src_boxes: Tensor, target_boxes: Tensor, eps: float,
                    lambda_w: float, lambda_h: float,
                    center_weight_clip: Optional[float], normalize: bool):
    """动态加权 L1（提示词 §10.1）。

    L = λ_c(|cx-ĉx|+|cy-ĉy|) + λ_w|w-ŵ| + λ_h|h-ĥ|
    坐标顺序为 cxcywh（cx,cy 为中心坐标，勿与左上角混淆）。
    """
    weights, stats = dynamic_l1_weights(target_boxes, eps,
                                        center_weight_clip, normalize)
    weights = weights * torch.tensor([1.0, 1.0, lambda_w, lambda_h],
                                     device=weights.device, dtype=weights.dtype)
    loss = ((src_boxes - target_boxes).abs() * weights).sum(dim=-1)
    return loss, stats


# ---------------------------------------------------------------------------
# 7. PW-CIoU（峰宽加权 CIoU）
# ---------------------------------------------------------------------------
def peak_width_weighted_ciou(boxes1: Tensor, boxes2: Tensor,
                             mean_width: Tensor,
                             weight_mode: str = 'ratio',
                             eps: float = 1e-6,
                             weight_clip: Optional[float] = None):
    """PW-CIoU = IoU - (bar_w/(w_gt+eps)) * ρ²/(c²+eps) - αv

    Args:
        boxes1: [N, 4] 预测框 xyxy
        boxes2: [M, 4] GT 框 xyxy
        mean_width: 标量 tensor，bar_w——训练集 GT 平均峰宽（归一化），
                    必须来自训练集统计，禁止用当前 mini-batch 均值。
        weight_mode:
            'ratio'         → bar_w/(w_gt+eps)，w_gt=bar_w 时权重=1（与原 CIoU 一致）
            'one_plus_ratio'→ 1 + bar_w/(w_gt+eps)，w_gt=bar_w 时权重=2，
                              ≠ 原 CIoU（仅探索实验用）
    Returns:
        (pw_ciou: [N, M], weight_stats: dict)
    """
    if weight_mode == 'plain':
        # 原 CIoU 基线：中心项权重恒 1（供 pw_ciou_enabled=false 消融）
        center_weight = torch.ones_like(boxes2[:, 2 - 2]).expand(boxes2.shape[0])
    else:
        w_gt0 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0.0)
        if weight_mode == 'ratio':
            center_weight = mean_width / (w_gt0 + eps)
        elif weight_mode == 'one_plus_ratio':
            center_weight = 1.0 + mean_width / (w_gt0 + eps)
        else:
            raise ValueError(
                f"pw_ciou_weight_mode must be ratio|one_plus_ratio|plain, got {weight_mode}")
        if weight_clip is not None:
            center_weight = center_weight.clamp(max=float(weight_clip))
    return _ciou_with_center_weight(boxes1, boxes2, center_weight, eps,
                                    weight_mode, center_weight)


def _ciou_with_center_weight(boxes1: Tensor, boxes2: Tensor,
                             center_weight: Tensor, eps: float,
                             stat_mode: str, stat_weight: Tensor):
    """CIoU 核心计算，中心距离项 ρ²/c² 乘以逐 GT 权重 [M]。"""
    # —— 复用 box_ops 的面积/IoU/闭合框逻辑（逐元素成对计算） ——
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    N, M = boxes1.shape[0], boxes2.shape[0]

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])   # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])   # [N,M,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp_min(eps)

    # 闭合框对角线平方 c² 与中心距平方 ρ²
    lt_e = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_e = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_e = (rb_e - lt_e).clamp(min=0)
    c2 = wh_e[..., 0] ** 2 + wh_e[..., 1] ** 2 + eps

    x1, y1 = (boxes1[:, 0] + boxes1[:, 2]) / 2, (boxes1[:, 1] + boxes1[:, 3]) / 2
    x2, y2 = (boxes2[:, 0] + boxes2[:, 2]) / 2, (boxes2[:, 1] + boxes2[:, 3]) / 2
    rho2 = (x1[:, None] - x2[None, :]) ** 2 + (y1[:, None] - y2[None, :]) ** 2

    # 宽高比一致性 v（标准 CIoU 定义）
    w_p = (boxes1[:, None, 2] - boxes1[:, None, 0]).clamp_min(eps)
    h_p = (boxes1[:, None, 3] - boxes1[:, None, 1]).clamp_min(eps)
    w_g = (boxes2[None, :, 2] - boxes2[None, :, 0]).clamp_min(eps)
    h_g = (boxes2[None, :, 3] - boxes2[None, :, 1]).clamp_min(eps)
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w_p / h_p) - torch.atan(w_g / h_g), 2)
    alpha = v / (1 - iou + v + eps)

    # —— 峰宽权重作用于中心距离项 ——
    pw_ciou = iou - center_weight[None, :] * rho2 / c2 - alpha * v

    with torch.no_grad():
        if M > 0:
            q = torch.quantile(stat_weight.detach().float(),
                               torch.tensor([0.5, 0.9, 0.99], device=stat_weight.device))
            stats = {'p50': q[0].item(), 'p90': q[1].item(),
                     'p99': q[2].item(), 'max': stat_weight.max().item()}
        else:
            stats = {'p50': 0.0, 'p90': 0.0, 'p99': 0.0, 'max': 0.0}
    return pw_ciou, stats


# ---------------------------------------------------------------------------
# 辅助：量化分位数日志工具（供 Criterion 汇总多个 batch 的权重统计）
# ---------------------------------------------------------------------------
def merge_weight_stats(prefix: str, stats: dict) -> dict:
    """把 dynamic_l1 / pw_ciou 的权重分位数展平成日志键。"""
    return {f'{prefix}_w_{k}': v for k, v in stats.items()}
