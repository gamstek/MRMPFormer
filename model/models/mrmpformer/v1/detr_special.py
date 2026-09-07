# -*- coding: utf-8 -*-
"""
[隔离实验] special_peak_v1 —— 特殊峰专项模型构建。

模型结构 = MRMPFormer v1（完全复用，不改任何参数/forward）；
损失 = SpecialSetCriterion（继承原版 MRMPSetCriterion，仅覆盖分类/定位两项）：
  - 分类：QFL（score 回归 IoU 质量），可配置回退 focal/ce；
  - 定位：log 宽度 L1（尺度等变，替代动态 L1）+ 原 PW-CIoU；
  - FDR/辅助损失等其余项全部继承原版。
隔离保证：原版 detr.build / MRMPSetCriterion 行为零改动。
"""
import torch

from framework.util import box_ops
from models.mrmpformer.v1.detr import MRMPSetCriterion, build as _build_v1
from models.mrmpformer.v1.fdr import peak_width_weighted_ciou, merge_weight_stats
from models.mrmpformer.v1.losses_special import quality_focal_loss, log_width_l1_loss


class SpecialSetCriterion(MRMPSetCriterion):
    """特殊峰专项损失（仅覆盖 loss_labels / loss_boxes，其余继承）。"""

    def __init__(self, *args,
                 special_cls='qfl', qfl_gamma=2.0,
                 log_width_enabled=True, center_exp=0.5,
                 log_width_ref=0.1273,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.special_cls = special_cls
        self.qfl_gamma = qfl_gamma
        self.log_width_enabled = log_width_enabled
        self.center_exp = center_exp
        self.log_width_ref = log_width_ref

    # ---- 分类：QFL（quality = 匹配对 IoU，detach）----
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        if self.special_cls != 'qfl':
            return super().loss_labels(outputs, targets, indices, num_boxes, log=log)

        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # 逐匹配对 IoU 作为质量目标（detach；无匹配处 0，QFL 按背景处理）
        quality = torch.zeros(src_logits.shape[:2], device=src_logits.device)
        if idx[0].numel():
            src_b = outputs['pred_boxes'][idx]
            tgt_b = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou = torch.diagonal(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_b),
                box_ops.box_cxcywh_to_xyxy(tgt_b))[0])
            quality[idx] = iou.detach().clamp(min=0.0, max=1.0)

        loss = quality_focal_loss(src_logits, target_classes, quality,
                                  self.num_classes, eos_coef=self.eos_coef,
                                  gamma=self.qfl_gamma)
        losses = {'loss_cls_qfl': loss}
        if log and idx[0].numel():
            from framework.util.misc import accuracy
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # ---- 定位：log 宽度 L1（替代动态 L1）+ PW-CIoU（继承开关语义）----
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        if src_boxes.numel() == 0:
            zero = outputs['pred_boxes'].sum() * 0.0
            losses['loss_log_width_l1'] = zero
            losses['loss_pw_ciou'] = zero
            losses['loss_ciou'] = zero
            return losses

        if self.log_width_enabled:
            loss_lw, lw_stats = log_width_l1_loss(
                src_boxes, target_boxes,
                center_exp=self.center_exp,
                width_scale=self.log_width_ref)
            losses['loss_log_width_l1'] = loss_lw.sum() / num_boxes
            for k, v in merge_weight_stats('log_width', lw_stats).items():
                losses[k] = torch.tensor(v, device=src_boxes.device)

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


def build(args):
    """构建隔离版：MRMPFormer v1 结构 + SpecialSetCriterion。"""
    model, crit, postprocessors = _build_v1(args)

    criterion = SpecialSetCriterion(
        crit.num_classes, matcher=crit.matcher, weight_dict=crit.weight_dict,
        eos_coef=crit.eos_coef, losses=crit.losses,
        bin_values=crit.fdr_bin_values,
        classification_loss=getattr(args, 'classification_loss', 'focal'),
        focal_alpha=getattr(args, 'focal_alpha', 0.25),
        focal_gamma=getattr(args, 'focal_gamma', 2.0),
        fdr_layer_weights=crit.fdr_layer_weights,
        fdr_scale_mode=crit.fdr_scale_mode,
        dynamic_l1_enabled=False,
        dynamic_l1_eps=crit.dynamic_l1_eps,
        dynamic_l1_lambda_w=crit.dynamic_l1_lambda_w,
        dynamic_l1_lambda_h=crit.dynamic_l1_lambda_h,
        center_weight_clip=crit.center_weight_clip,
        normalize_dynamic_weights=crit.normalize_dynamic_weights,
        pw_ciou_enabled=getattr(args, 'pw_ciou_enabled', True),
        pw_ciou_weight_mode=getattr(args, 'pw_ciou_weight_mode', 'ratio'),
        pw_ciou_eps=getattr(args, 'pw_ciou_eps', 1e-6),
        pw_ciou_weight_clip=getattr(args, 'pw_ciou_weight_clip', None),
        pw_ciou_mean_width=float(crit.pw_ciou_mean_width),
        aux_class_loss=crit.aux_class_loss,
        fdr_layer_sigmas=getattr(args, 'fdr_layer_sigmas', None),
        fdr_layer_progress=getattr(args, 'fdr_layer_progress', None),
        fdr_cascade=crit.fdr_cascade,
        # 隔离版专属
        special_cls=getattr(args, 'special_cls', 'qfl'),
        qfl_gamma=getattr(args, 'qfl_gamma', 2.0),
        log_width_enabled=getattr(args, 'log_width_enabled', True),
        center_exp=getattr(args, 'log_width_center_exp', 0.5),
        log_width_ref=getattr(args, 'log_width_ref', 0.1273),
    )

    # 权重字典对齐：动态 L1 → log 宽度 L1；focal 主键 → QFL 主键
    wd = dict(crit.weight_dict)
    if getattr(args, 'dynamic_l1_enabled', True):
        dyn_w = wd.pop('loss_dynamic_l1', args.bbox_loss_coef)
    else:
        dyn_w = wd.pop('loss_bbox', args.bbox_loss_coef)
    if getattr(args, 'log_width_enabled', True):
        wd['loss_log_width_l1'] = dyn_w
    cls_old = crit.cls_main_key
    if getattr(args, 'special_cls', 'qfl') == 'qfl':
        if cls_old in wd:
            wd['loss_cls_qfl'] = wd.pop(cls_old)
    criterion.weight_dict = wd
    criterion.cls_main_key = 'loss_cls_qfl' if getattr(args, 'special_cls', 'qfl') == 'qfl' else cls_old

    if getattr(args, "device", None) == "auto":
        from utils.torch_device import resolve_torch_device
        device = resolve_torch_device(verbose=False)
    else:
        device = torch.device(args.device)
    criterion.to(device)
    return model, criterion, postprocessors
