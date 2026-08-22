# -*- coding: utf-8 -*-
"""
MRMPFormer v1 — 三层 Decoder + FDR 边界逐层精化（提示词文档实现）。

架构（docs/MRMPFormer_FDR_Architecture_Agent_Prompt.md）：
  Decoder Layer 1 → 完整二维初始框 b0=(cx,cy,w,h) + FDR 分布 z1
  Decoder Layer 2 → 残差 Δz2, z2 = z1 + Δz2
  Decoder Layer 3 → 残差 Δz3, z3 = z2 + Δz3
  每层左右边界解码后经 BoundaryPositionMLP 反馈到下一层 query_pos；
  最终二维框 = (xL3, yT0, xR3, yB0)，即左右边界取第 3 层、上下边界取第 1 层；
  最终分类 = 第 3 层 class_logits（唯一推理口径，禁止层间平均）。

输出接口（提示词 §13）：
  pred_logits  : [B,Q,C+1] 第 3 层分类 Logits（Softmax 含背景类，与 QuanFormer 一致）
  pred_boxes   : [B,Q,4]   最终框 cxcywh（归一化）——与 Matcher/Loss/PostProcess 一致
  initial_boxes: [B,Q,4]   第 1 层初始框 cxcywh
  fdr_logits   : [z1,z2,z3] each [B,Q,2,N]  累计 Logits
  fdr_deltas   : [Δz2,Δz3] each [B,Q,2,N]   各层实际预测的残差
  refined_lr   : [lr1,lr2,lr3] each [B,Q,2] 每层解码的左右边界
  aux_outputs  : 中间层 logits/boxes（仅辅助监督，配置控制）
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from framework.util import box_ops
from framework.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                                 accuracy, get_world_size,
                                 is_dist_avail_and_initialized)

from ...shared.backbone import build_backbone
from ...shared.matcher import build_matcher
from ...quanformer.detr import MLP, PostProcess  # 复用现有 MLP 与后处理（cxcywh→xyxy 缩放语义一致）
from .fdr import (FDRHead, BoundaryPositionMLP, DistributionBoundaryLoss,
                  make_bin_values, decode_expected_offsets,
                  softmax_focal_loss, dynamic_l1_loss,
                  peak_width_weighted_ciou, merge_weight_stats)
from .transformer import build_fdr_transformer

MODEL_VERSION = 'mrmpformer_v1'


class MRMPFormer(nn.Module):
    """三层 Decoder + FDR 边界逐层精化模型。"""

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 num_decoder_layers=3,
                 num_fdr_bins=33, fdr_bin_power=2.0,
                 fdr_bin_values=None,
                 fdr_scale_mode='initial_box_width',
                 detach_boundary_feedback=False,
                 fdr_min_width=1e-4,
                 aux_loss=True):
        super().__init__()
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.transformer = transformer
        self.num_classes = num_classes
        hidden_dim = transformer.d_model

        # —— 共享分类头（三层共用，推理只用第 3 层输出）——
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # —— 第 1 层初始二维框头（结构同 QuanFormer：MLP(D,D,D,4)+Sigmoid）——
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        # —— FDR：每层独立分布头 + 边界位置反馈 MLP ——
        self.fdr_heads = nn.ModuleList(
            [FDRHead(hidden_dim, num_fdr_bins) for _ in range(num_decoder_layers)])
        self.boundary_pos_mlp = BoundaryPositionMLP(hidden_dim)

        # —— Bin 候选偏移 W(n)：buffer 随设备移动、随 checkpoint 保存、不参与训练 ——
        if fdr_bin_values is not None:
            bins = torch.as_tensor(fdr_bin_values, dtype=torch.float32).flatten()
            if bins.numel() != num_fdr_bins:
                raise ValueError(
                    f"显式 bin_values 长度 {bins.numel()} != num_fdr_bins {num_fdr_bins}")
            if not torch.all(bots_diff_pos(bins)):
                raise ValueError("bin_values 必须严格单调递增")
            bins = bins - bins.mean()  # 对称覆盖 0 的工程保证（提示词 §6.1）
        else:
            bins = make_bin_values(num_fdr_bins, fdr_bin_power)
        self.register_buffer('fdr_bin_values', bins)

        if fdr_scale_mode not in ('initial_box_width', 'roi_width'):
            raise ValueError(f"未知 fdr_scale_mode: {fdr_scale_mode}")
        self.fdr_scale_mode = fdr_scale_mode
        self.detach_boundary_feedback = detach_boundary_feedback
        self.fdr_min_width = float(fdr_min_width)
        self.aux_loss = aux_loss
        self.version = MODEL_VERSION  # 模型结构版本号，防新旧 checkpoint 混淆

    # ------------------------------------------------------------------
    # 边界位置反馈回调：第 k 层解码完成后，把该层精化边界编码进下一层 query_pos
    # ------------------------------------------------------------------
    def _make_boundary_pos_fn(self):
        """闭包内独立维护 z 的累计链（与 forward 末尾的输出重算完全同构：
        FDRHead/BoundaryPositionMLP/BBox 头均不含 Dropout，两次前向确定性一致，
        梯度经两条通路汇入同一组参数，autograd 自动求和）。"""
        state = {'z': None, 'init_xyxy': None, 's0': None}

        def boundary_pos_fn(k: int, h: torch.Tensor) -> torch.Tensor:
            z_k = self.fdr_heads[k](h)                       # k=0: z1；k>=1: Δz
            state['z'] = z_k if state['z'] is None else state['z'] + z_k
            if state['init_xyxy'] is None:
                # 初始框只能由第 1 层 Query 特征预测
                init_cxcywh = self.bbox_embed(h).sigmoid()
                state['init_xyxy'] = box_ops.box_cxcywh_to_xyxy(init_cxcywh)
                state['s0'] = self._scale_factor(init_cxcywh)
            dx = decode_expected_offsets(state['z'], self.fdr_bin_values, state['s0'])
            x_l = state['init_xyxy'][..., 0] + dx[..., 0]
            x_r = state['init_xyxy'][..., 2] + dx[..., 1]
            lr = torch.stack([x_l, x_r], dim=-1)             # [B,Q,2]
            bpos = self.boundary_pos_mlp(lr)
            if self.detach_boundary_feedback:
                bpos = bpos.detach()  # 仅消融实验使用，默认不 detach
            return bpos

        return boundary_pos_fn

    def _scale_factor(self, initial_boxes_cxcywh: torch.Tensor) -> torch.Tensor:
        """s0：默认初始框宽 w0；roi_width 模式取 1.0（全图归一化尺度）。
        必须与坐标系统一（归一化 [0,1]），禁止混用像素坐标。"""
        if self.fdr_scale_mode == 'initial_box_width':
            return initial_boxes_cxcywh[..., 2:3]            # [B,Q,1] w0
        return torch.ones_like(initial_boxes_cxcywh[..., 2:3])

    # ------------------------------------------------------------------
    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        hs, _memory = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1],
            boundary_pos_fn=self._make_boundary_pos_fn())    # hs: [K,B,Q,D]

        B, Q, _ = hs.shape[1], hs.shape[2], hs.shape[3]
        K = hs.shape[0]

        # —— 分类：三层共享头；pred_logits 只取第 3 层（K-1）——
        class_logits_all = self.class_embed(hs)              # [K,B,Q,C+1]

        # —— 初始二维框（第 1 层）——
        initial_box_cxcywh = self.bbox_embed(hs[0]).sigmoid()          # [B,Q,4]
        initial_edges_ltrb = box_ops.box_cxcywh_to_xyxy(initial_box_cxcywh)
        s0 = self._scale_factor(initial_box_cxcywh)

        # —— FDR：z1 = FFN1(h1)；z_k = z_{k-1} + Δz_k（Logits 残差累加）——
        fdr_logits = []
        fdr_deltas = []
        z = None
        for k in range(K):
            out_k = self.fdr_heads[k](hs[k])                 # [B,Q,2,N]
            if k == 0:
                z = out_k
            else:
                fdr_deltas.append(out_k)                     # Δz_k（k 从 2 层起记录）
                z = z + out_k                                # z_k = z_{k-1} + Δz_k
            fdr_logits.append(z)

        # —— 每层左右边界：累计分布相对【初始边界】解码，禁止坐标残差双重累计 ——
        refined_lr = []
        for z_k in fdr_logits:
            dx = decode_expected_offsets(z_k, self.fdr_bin_values, s0)  # [B,Q,2]
            x_l = initial_edges_ltrb[..., 0] + dx[..., 0]
            x_r = initial_edges_ltrb[..., 2] + dx[..., 1]
            refined_lr.append(torch.stack([x_l, x_r], dim=-1))

        # —— 最终二维框：左右=第 3 层，上下=第 1 层初始框 ——
        initial_top = initial_edges_ltrb[..., 1]
        initial_bottom = initial_edges_ltrb[..., 3]
        lr_final = refined_lr[-1]
        # 训练用连续边界：宽度下限 clamp 保证 cxcywh 有效性（可导，不 detach/argmax）；
        # 非法框比例另行统计（invalid_box_ratio），推理侧可再做安全裁剪。
        width_safe = (lr_final[..., 1] - lr_final[..., 0]).clamp(min=self.fdr_min_width)
        final_boxes_xyxy = torch.stack(
            [lr_final[..., 0], initial_top,
             lr_final[..., 0] + width_safe, initial_bottom], dim=-1)
        final_boxes = box_ops.box_xyxy_to_cxcywh(final_boxes_xyxy)     # [B,Q,4] cxcywh

        out = {
            'pred_logits': class_logits_all[-1],             # 最终峰概率唯一来源：第 3 层
            'pred_boxes': final_boxes,
            'initial_boxes': initial_box_cxcywh,
            'initial_edges_ltrb': initial_edges_ltrb,
            'fdr_logits': fdr_logits,                         # [z1,z2,z3]
            'fdr_deltas': fdr_deltas,                         # [Δz2,Δz3]
            'refined_lr': refined_lr,                         # [lr1,lr2,lr3]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(class_logits_all, refined_lr,
                                                    initial_top, initial_bottom)
        return out

    @torch.jit.unused
    def _set_aux_self_boxes(self, lr, top, bottom):
        """中间层精化框组装（左右=该层，上下=初始框）。"""
        w = (lr[..., 1] - lr[..., 0]).clamp(min=self.fdr_min_width)
        xyxy = torch.stack([lr[..., 0], top, lr[..., 0] + w, bottom], dim=-1)
        return box_ops.box_xyxy_to_cxcywh(xyxy)

    @torch.jit.unused
    def _set_aux_loss(self, class_logits_all, refined_lr, top, bottom):
        # DETR aux_outputs 约定：不含最后一层；每层 {'pred_logits','pred_boxes'}
        return [{'pred_logits': a, 'pred_boxes': self._set_aux_self_boxes(b, top, bottom)}
                for a, b in zip(class_logits_all[:-1], refined_lr[:-1])]


def bots_diff_pos(bins: torch.Tensor) -> torch.Tensor:
    """严格单调递增校验。"""
    return (bins[1:] - bins[:-1]) > 0


# ===========================================================================
# Criterion
# ===========================================================================
class MRMPSetCriterion(nn.Module):
    """MRMPFormer v1 损失（提示词 §8-§11）。

    L_total = λ_cls·L_FL^(3) + λ_box·L_dynL1 + λ_iou·L_PW-CIoU
              + λ_fdr·Σ_k α_k·L_FGL^(k) + L_aux

    说明：
      - L_FGL^(k) 使用工程回退实现 DistributionBoundaryLoss（两点插值软标签
        CE，见 fdr.py），论文未给出 FGL 精确定义，不得冒充原式；
      - 分类主损失只用第 3 层；1/2 层辅助分类损失由 aux_class_loss 配置；
      - 动态 L1 / PW-CIoU 均可开关，保留原始 L1 / CIoU 基线做消融。
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 bin_values: torch.Tensor,
                 classification_loss='focal', focal_alpha=0.25, focal_gamma=2.0,
                 fdr_layer_weights=(0.5, 0.7, 1.0),
                 fdr_scale_mode='initial_box_width',
                 dynamic_l1_enabled=True, dynamic_l1_eps=1e-6,
                 dynamic_l1_lambda_w=1.0, dynamic_l1_lambda_h=1.0,
                 center_weight_clip=None, normalize_dynamic_weights=False,
                 pw_ciou_enabled=True, pw_ciou_weight_mode='ratio',
                 pw_ciou_eps=1e-6, pw_ciou_weight_clip=None,
                 pw_ciou_mean_width=None,
                 aux_class_loss=True,
                 fdr_min_width=1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.classification_loss = classification_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_class_loss = aux_class_loss
        self.fdr_layer_weights = list(fdr_layer_weights)
        self.fdr_scale_mode = fdr_scale_mode
        self.fdr_min_width = float(fdr_min_width)
        self.fdr_loss = DistributionBoundaryLoss(num_bins=bin_values.shape[0])

        self.dynamic_l1_enabled = dynamic_l1_enabled
        self.dynamic_l1_eps = dynamic_l1_eps
        self.dynamic_l1_lambda_w = dynamic_l1_lambda_w
        self.dynamic_l1_lambda_h = dynamic_l1_lambda_h
        self.center_weight_clip = center_weight_clip
        self.normalize_dynamic_weights = normalize_dynamic_weights

        self.pw_ciou_enabled = pw_ciou_enabled
        self.pw_ciou_weight_mode = pw_ciou_weight_mode
        self.pw_ciou_eps = pw_ciou_eps
        self.pw_ciou_weight_clip = pw_ciou_weight_clip
        # bar_w 必须来自训练集统计（配置/数据集元数据），禁止 mini-batch 均值；
        # 为 None 时回退为权重=1（等价原 CIoU），并打 WARN 提示补配。
        if pw_ciou_enabled and pw_ciou_mean_width is None:
            print("[WARN] pw_ciou_mean_width 未配置，PW-CIoU 中心项权重回退为 1 "
                  "（等价原 CIoU）。请用训练集 GT 平均峰宽配置该项。")
            pw_ciou_mean_width = 1.0
        self.register_buffer('pw_ciou_mean_width',
                             torch.as_tensor(float(pw_ciou_mean_width or 1.0)))
        self.register_buffer('fdr_bin_values', bin_values.clone())

        if classification_loss not in ('focal', 'ce'):
            raise ValueError(f"classification_loss 仅支持 focal|ce， got {classification_loss}")
        if classification_loss == 'focal':
            self.cls_main_key = 'loss_cls_focal_main'
        else:
            self.cls_main_key = 'loss_ce'
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    # ------------------------------------------------------------------
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """分类损失：Focal（默认）或 CE 基线。目标编码与 QuanFormer 一致：
        背景查询 target=num_classes（no-object 恒在最后一列）。"""
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.classification_loss == 'focal':
            # Softmax Focal Loss：输入必须为原始 Logits（fdr.softmax_focal_loss 内部做 Softmax）；
            # eos_coef 透传为背景项权重（承接 DETR no-object 权重，防负样本压倒正样本）
            loss = softmax_focal_loss(src_logits, target_classes, self.num_classes,
                                      alpha=self.focal_alpha, gamma=self.focal_gamma,
                                      eos_coef=self.eos_coef)
            losses = {self.cls_main_key: loss}
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                      self.empty_weight)
            losses = {self.cls_main_key: loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        losses = {'cardinality_error': F.l1_loss(card_pred.float(), tgt_lengths.float())}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """定位损失：动态加权 L1（§10.1）+ PW-CIoU（§10.2），均可开关。"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]                       # cxcywh
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        if src_boxes.numel() == 0:
            zero = outputs['pred_boxes'].sum() * 0.0
            losses['loss_bbox'] = zero
            losses['loss_dynamic_l1'] = zero
            losses['loss_ciou'] = zero
            losses['loss_pw_ciou'] = zero
            return losses

        # —— L1 / 动态加权 L1 ——
        if self.dynamic_l1_enabled:
            loss_dyn, dyn_stats = dynamic_l1_loss(
                src_boxes, target_boxes, self.dynamic_l1_eps,
                self.dynamic_l1_lambda_w, self.dynamic_l1_lambda_h,
                self.center_weight_clip, self.normalize_dynamic_weights)
            losses['loss_dynamic_l1'] = loss_dyn.sum() / num_boxes
            for k, v in merge_weight_stats('dyn_l1_center', dyn_stats).items():
                losses[k] = torch.tensor(v, device=src_boxes.device)
        else:
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # —— IoU 损失：PW-CIoU（默认）或原 CIoU 基线 ——
        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
        if self.pw_ciou_enabled:
            pw, pw_stats = peak_width_weighted_ciou(
                src_xyxy, tgt_xyxy, self.pw_ciou_mean_width,
                weight_mode=self.pw_ciou_weight_mode, eps=self.pw_ciou_eps,
                weight_clip=self.pw_ciou_weight_clip)
            losses['loss_pw_ciou'] = (1 - torch.diagonal(pw)).sum() / num_boxes
            for k, v in merge_weight_stats('pw_ciou', pw_stats).items():
                losses[k] = torch.tensor(v, device=src_boxes.device)
        else:
            ciou = peak_width_weighted_ciou(src_xyxy, tgt_xyxy,
                                            self.pw_ciou_mean_width,
                                            weight_mode='plain',
                                            eps=self.pw_ciou_eps)[0]
            losses['loss_ciou'] = (1 - torch.diagonal(ciou)).sum() / num_boxes
        return losses

    def loss_fdr(self, outputs, targets, indices, num_boxes):
        """FDR 分布监督（§9）+ 每层边界 MAE / 每层框 IoU / 越界率诊断。"""
        idx = self._get_src_permutation_idx(indices)
        losses = {}
        if outputs['fdr_logits'][0][idx].numel() == 0:
            zero = outputs['fdr_logits'][0].sum() * 0.0
            for k in range(len(self.fdr_layer_weights)):
                losses[f'loss_fdr_layer_{k + 1}_left'] = zero
                losses[f'loss_fdr_layer_{k + 1}_right'] = zero
            losses['fdr_target_overflow_ratio'] = zero.detach()
            return losses

        init_edges = outputs['initial_edges_ltrb']                  # [B,Q,4] xyxy
        if self.fdr_scale_mode == 'initial_box_width':
            s0 = (outputs['initial_boxes'][..., 2:3]).clamp_min(1e-6)  # w0 + eps 防除零
        else:
            s0 = torch.ones_like(outputs['initial_boxes'][..., 2:3])

        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(
            torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0))

        # 目标偏移 d = (x_gt - x_init) / s0（提示词 §9.2）；s0 加 eps 防除零
        s0_matched = s0[idx][:, 0]                                  # [M]
        d_l = (tgt_xyxy[:, 0] - init_edges[idx][:, 0]) / s0_matched
        d_r = (tgt_xyxy[:, 2] - init_edges[idx][:, 2]) / s0_matched
        d_offsets = torch.stack([d_l, d_r], dim=-1)                 # [M,2]

        overflow_acc = []
        for k, z_k in enumerate(outputs['fdr_logits']):
            res = self.fdr_loss(z_k[idx], d_offsets, self.fdr_bin_values)
            losses[f'loss_fdr_layer_{k + 1}_left'] = res['left']
            losses[f'loss_fdr_layer_{k + 1}_right'] = res['right']
            losses[f'fdr_exp_offset_err_layer_{k + 1}'] = res['exp_offset_err']
            overflow_acc.append(res['overflow_ratio'])

            # 每层左/右边界 MAE 与每层框 IoU（诊断：验证逐层精化趋势）
            lr_k = outputs['refined_lr'][k][idx]
            losses[f'fdr_lr_mae_layer_{k + 1}_left'] = (
                lr_k[:, 0] - tgt_xyxy[:, 0]).abs().mean().detach()
            losses[f'fdr_lr_mae_layer_{k + 1}_right'] = (
                lr_k[:, 1] - tgt_xyxy[:, 2]).abs().mean().detach()
            box_k_xyxy = torch.stack(
                [lr_k[:, 0], init_edges[idx][:, 1],
                 lr_k[:, 1], init_edges[idx][:, 3]], dim=-1)
            iou_k = torch.diagonal(box_ops.box_iou(box_k_xyxy, tgt_xyxy)[0]).mean()
            losses[f'fdr_iou_layer_{k + 1}'] = iou_k.detach()

        losses['fdr_target_overflow_ratio'] = torch.stack(overflow_acc).mean()
        return losses

    # ------------------------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'fdr': self.loss_fdr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # —— 召回相关诊断（§12.1）：GT 数超过 Query 数的 ROI 比例 ——
        n_q = outputs['pred_logits'].shape[1]
        exceed = torch.tensor(
            sum(1 for t in targets if len(t['labels']) > n_q) / max(len(targets), 1),
            device=outputs['pred_logits'].device)
        losses['gt_exceeds_queries_roi_ratio'] = exceed
        losses['matched_positives'] = torch.tensor(
            float(sum(len(i) for i, _ in indices)), device=exceed.device)
        losses['empty_target_rois'] = torch.tensor(
            float(sum(1 for t in targets if len(t['labels']) == 0)), device=exceed.device)
        # 非法框比例：最终层 xL >= xR（宽度低于下限被 clamp 的框）
        lr_last = outputs['refined_lr'][-1].detach()
        invalid = ((lr_last[..., 1] - lr_last[..., 0]) < self.fdr_min_width)
        losses['invalid_box_ratio'] = invalid.float().mean()

        # —— 中间层辅助分类损失（可配置；推理口径不受影响）——
        if self.aux_class_loss and 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets)
                l_dict = self.loss_labels(aux_outputs, targets, aux_indices,
                                          num_boxes, log=False)
                losses.update({f'loss_cls_aux_{i + 1}': v for v in l_dict.values()
                               if v.dim() == 0 or True})
                # 仅保留损失标量键（去掉 log=False 下本就不存在的 class_error）
                losses = {k: v for k, v in losses.items()}
        return losses


# ===========================================================================
# 旧 QuanFormer checkpoint 迁移（提示词 §14.1）
# ===========================================================================
def load_legacy_quanformer_state(model: MRMPFormer, legacy_state_dict: dict,
                                 verbose: bool = True) -> dict:
    """把单层 Decoder 的 QuanFormer checkpoint 迁移到 MRMPFormer v1。

    策略：
      - encoder / decoder.layers.0 / decoder.norm / class_embed / bbox_embed /
        query_embed / input_proj / backbone 直接复用；
      - decoder.layers.0 的参数复制初始化到 layers.1..K-1（新层从旧层热启动）；
      - FDR Heads / BoundaryPositionMLP 保持新建初始化（FDR 末层零初始化，
        起步为恒等精化）；fdr_bin_values 为 buffer，模型自带。
    迁移结果打印分类报告（expected missing / unexpected / shape mismatch），
    禁止静默 strict=False。
    """
    num_dec = model.num_decoder_layers
    migrated = {}
    for k, v in legacy_state_dict.items():
        migrated[k] = v
        if k.startswith('transformer.decoder.layers.0.'):
            suffix = k[len('transformer.decoder.layers.0.'):]
            for i in range(1, num_dec):
                migrated[f'transformer.decoder.layers.{i}.{suffix}'] = v.clone()

    model_keys = set(model.state_dict().keys())
    legacy_keys = set(migrated.keys())
    shape_mismatch = [k for k in migrated
                      if k in model_keys and migrated[k].shape != model.state_dict()[k].shape]
    for k in shape_mismatch:
        migrated.pop(k)
    unexpected = sorted(legacy_keys - model_keys - set(shape_mismatch))
    expected_new = sorted(model_keys - legacy_keys)  # fdr_heads.* / boundary_pos_mlp.* / 版本号 buffer

    missing, unexpected_loaded = model.load_state_dict(migrated, strict=False)
    report = {
        'loaded': sorted(model_keys & legacy_keys - set(shape_mismatch)),
        'expected_missing': expected_new,
        'unexpected': sorted(set(unexpected) | set(unexpected_loaded)),
        'shape_mismatch': sorted(shape_mismatch),
    }
    if verbose:
        print(f"[迁移] QuanFormer→MRMPFormer v1：加载 {len(report['loaded'])} 项；"
              f"decoder L1 参数已复制到 L2..L{num_dec}")
        print(f"[迁移] expected missing（新增模块，保持新初始化）: {report['expected_missing']}")
        print(f"[迁移] unexpected keys（忽略）: {report['unexpected'][:20]}")
        print(f"[迁移] shape mismatch（跳过，需检查 num_queries 等配置）: {report['shape_mismatch']}")
    return report


# ===========================================================================
# build
# ===========================================================================
def build(args):
    num_classes = 1
    if getattr(args, "device", None) == "auto":
        from utils.torch_device import resolve_torch_device
        device = resolve_torch_device(verbose=False)
    else:
        device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_fdr_transformer(args)

    num_dec = getattr(args, 'dec_layers', 3)
    num_bins = getattr(args, 'num_fdr_bins', 33)
    bin_power = getattr(args, 'fdr_bin_power', 2.0)
    bin_values = getattr(args, 'fdr_bin_values', None)

    model = MRMPFormer(
        backbone, transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_decoder_layers=num_dec,
        num_fdr_bins=num_bins,
        fdr_bin_power=bin_power,
        fdr_bin_values=bin_values,
        fdr_scale_mode=getattr(args, 'fdr_scale_mode', 'initial_box_width'),
        detach_boundary_feedback=getattr(args, 'detach_boundary_feedback', False),
        fdr_min_width=getattr(args, 'fdr_min_width', 1e-4),
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)

    # —— 损失权重：全部配置化（提示词 §11）——
    cls_key = 'loss_cls_focal_main' if getattr(args, 'classification_loss', 'focal') == 'focal' else 'loss_ce'
    weight_dict = {cls_key: getattr(args, 'cls_loss_coef', 1.0)}

    if getattr(args, 'dynamic_l1_enabled', True):
        weight_dict['loss_dynamic_l1'] = args.bbox_loss_coef
    else:
        weight_dict['loss_bbox'] = args.bbox_loss_coef
    if getattr(args, 'pw_ciou_enabled', True):
        weight_dict['loss_pw_ciou'] = args.iou_loss_coef
    else:
        weight_dict['loss_ciou'] = args.iou_loss_coef

    fdr_weights = getattr(args, 'fdr_layer_weights', [0.5, 0.7, 1.0])
    if len(fdr_weights) < num_dec:
        fdr_weights = list(fdr_weights) + [1.0] * (num_dec - len(fdr_weights))
    fdr_coef = getattr(args, 'fdr_loss_coef', 2.0)
    for k in range(num_dec):
        for side in ('left', 'right'):
            weight_dict[f'loss_fdr_layer_{k + 1}_{side}'] = fdr_coef * fdr_weights[k]

    if getattr(args, 'aux_class_loss', True):
        for i in range(num_dec - 1):
            weight_dict[f'loss_cls_aux_{i + 1}'] = getattr(args, 'aux_class_loss_coef', 1.0)

    if getattr(args, 'recall_loss_enabled', False):
        raise NotImplementedError(
            "recall_loss_enabled=true：Recall Loss 的论文精确定义未确认（提示词 §12.4），"
            "禁止自行编造公式；请先验证 Focal Loss 基线。")

    losses = ['labels', 'boxes', 'cardinality', 'fdr']
    criterion = MRMPSetCriterion(
        num_classes, matcher=matcher, weight_dict=weight_dict,
        eos_coef=args.eos_coef, losses=losses,
        bin_values=model.fdr_bin_values,
        classification_loss=getattr(args, 'classification_loss', 'focal'),
        focal_alpha=getattr(args, 'focal_alpha', 0.25),
        focal_gamma=getattr(args, 'focal_gamma', 2.0),
        fdr_layer_weights=fdr_weights,
        fdr_scale_mode=getattr(args, 'fdr_scale_mode', 'initial_box_width'),
        dynamic_l1_enabled=getattr(args, 'dynamic_l1_enabled', True),
        dynamic_l1_eps=getattr(args, 'dynamic_l1_eps', 1e-6),
        dynamic_l1_lambda_w=getattr(args, 'dynamic_l1_lambda_w', 1.0),
        dynamic_l1_lambda_h=getattr(args, 'dynamic_l1_lambda_h', 1.0),
        center_weight_clip=getattr(args, 'center_weight_clip', None),
        normalize_dynamic_weights=getattr(args, 'normalize_dynamic_weights', False),
        pw_ciou_enabled=getattr(args, 'pw_ciou_enabled', True),
        pw_ciou_weight_mode=getattr(args, 'pw_ciou_weight_mode', 'ratio'),
        pw_ciou_eps=getattr(args, 'pw_ciou_eps', 1e-6),
        pw_ciou_weight_clip=getattr(args, 'pw_ciou_weight_clip', None),
        pw_ciou_mean_width=getattr(args, 'pw_ciou_mean_width', None),
        aux_class_loss=getattr(args, 'aux_class_loss', True),
    )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    return model, criterion, postprocessors
