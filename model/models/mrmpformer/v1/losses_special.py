# -*- coding: utf-8 -*-
"""
[隔离实验] special_peak_v1 —— 特殊峰专项损失组件。

与原版 mrmpformer/v1/fdr.py 完全隔离：原版损失行为不变。
本模块提供：
  1. quality_focal_loss   : QFL（质量感知分类，GFL 式双类 Softmax 适配版）
                            正样本目标 = IoU(pred, gt)（detach），score 与定位质量耦合；
  2. log_width_l1_loss    : 尺度等变宽度回归
                            宽度项 |log ŵ - log w|（宽窄峰等权），
                            中心项权重 1/sqrt(w_gt)（较原版 1/w 软化）；
坐标约定与全项目一致：归一化 [0,1]，cxcywh。
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def quality_focal_loss(src_logits: Tensor, target_classes: Tensor,
                       quality: Tensor, num_classes: int,
                       eos_coef: float = 0.1,
                       gamma: float = 2.0) -> Tensor:
    """QFL（Softmax 双类适配版）。输入必须是原始 Logits。

    target_classes: [B,Q] 背景 = num_classes；
    quality:        [B,Q] 正样本 IoU 目标（detach），背景位忽略。

    正样本（前景）：L = -[q·log(p) + (1-q)·log(1-p)]，p = P(峰)（softmax 第 0 列）
      —— score 回归到定位质量 q，而非恒 1；
    负样本（背景）：L = -eos · p^gamma · log(1-p)（难负样本加权，压制重复框）。
    """
    eps = 1e-7
    probs = F.softmax(src_logits.float(), dim=-1)            # [B,Q,C+1]
    p_peak = probs[..., 0].clamp(eps, 1 - eps)               # [B,Q] 峰概率
    is_bg = (target_classes == num_classes)
    q = quality.float().clamp(min=0.05, max=1.0)             # 防 IoU≈0 时正样本梯度归零

    pos_loss = -(q * p_peak.clamp(min=eps).log()
                 + (1 - q) * (1 - p_peak).clamp(min=eps).log())
    neg_loss = -eos_coef * p_peak.pow(gamma) * (1 - p_peak).clamp(min=eps).log()

    loss = torch.where(is_bg, neg_loss, pos_loss)
    return loss.mean()


def log_width_l1_loss(src_boxes: Tensor, target_boxes: Tensor,
                      center_exp: float = 0.5,
                      width_scale: Optional[float] = None,
                      eps: float = 1e-6):
    """尺度等变 L1（cxcywh）。

    L = λ_c·(|Δcx|+|Δcy|) + w_ref·|log ŵ - log w| + |Δh|
      λ_c   = 1/(w_gt^center_exp)（center_exp=0.5 较原版 1/w 软化；=0 退化为常数）
      w_ref = width_scale（训练集 GT 平均峰宽，量纲桥接；None 用 w_gt 均值）

    返回 (loss_per_box [M], stats dict)。
    """
    w_gt = target_boxes[:, 2].clamp_min(eps)
    w_pred = src_boxes[:, 2].clamp_min(eps)

    lam_c = 1.0 / w_gt.pow(center_exp) if center_exp > 0 else torch.ones_like(w_gt)
    if width_scale is None:
        width_scale = float(w_gt.mean())
    w_ref = torch.as_tensor(float(width_scale), device=w_gt.device, dtype=w_gt.dtype)

    center = (src_boxes[:, 0] - target_boxes[:, 0]).abs() + \
             (src_boxes[:, 1] - target_boxes[:, 1]).abs()
    width = (w_pred.log() - w_gt.log()).abs() * w_ref
    height = (src_boxes[:, 3] - target_boxes[:, 3]).abs()

    loss = lam_c * center + width + height
    with torch.no_grad():
        stats = {
            'p50': float(torch.quantile(lam_c.detach().float(), 0.5)),
            'max': float(lam_c.max()),
        }
    return loss, stats
