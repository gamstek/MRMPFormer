"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

from .util import misc as utils
from .datasets.coco_eval import CocoEvaluator
# from framework.datasets.panoptic_eval import PanopticEvaluator


# ---------------------------------------------------------------------------
# 终端展示设计（仅控制终端；log.txt 与 train_stats 始终保存全量英文 key）
#   逐步行：紧凑英文短码（loss/cls），可 grep、单行不折行
#   汇总块：纯中文短标签、一组一行、精化趋势用箭头 L1→L2→L3 串接
# ---------------------------------------------------------------------------
_LEGEND_PRINTED = False


def _print_legend_once():
    """训练开始时打印一次图例，代替在每个指标名上重复中英对照。"""
    global _LEGEND_PRINTED
    if _LEGEND_PRINTED:
        return
    _LEGEND_PRINTED = True
    print('[图例] loss=加权总损失  cls=分类误差(仅匹配query)  (x)=epoch均值  '
          '→=逐层精化L1→L2→L3  | 全部指标以英文key保存于 log.txt')


def _print_avg_stats(tag, metric_logger):
    """epoch 平均摘要：64 字符宽框、一组一行、趋势箭头压缩多层数据。

    展示层级（epoch 级决策只需这四件事）：
      分类：学没学起来 | 损失：各分量降没降 | 精化：逐层是否变好 | 匹配：有无告警
    其余诊断指标（p99 分位数、期望偏差等）只进 log.txt，不上终端。
    """
    m = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if not m:
        return
    W = 64
    print('─' * W)
    head = f'◆ {tag}'
    if 'loss' in m:
        head += f'  loss {m["loss"]:.4f}'
    if 'lr' in m:
        head += f'  lr {m["lr"]:.0e}'
    print(head)

    def line(name, items, per_line=3):
        items = [x for x in items if x]
        if not items:
            return
        for s in range(0, len(items), per_line):
            chunk = items[s:s + per_line]
            prefix = f'  {name}  ' if s == 0 else '        '
            print(prefix + ' | '.join(chunk))

    # —— 分类 ——
    line('分类', [f'误差 {m["class_error"]:.1f}' if 'class_error' in m else None,
                f'数量 {m["cardinality_error"]:.2f}' if 'cardinality_error' in m else None])

    # —— 损失分量（基线与 v1 的键都覆盖；未知 loss_ 键兜底展示）——
    known_loss = set()
    items = []
    for k, lab in (('loss_cls_focal_main', 'Focal'), ('loss_ce', 'CE'),
                   ('loss_dynamic_l1', '动态L1'), ('loss_bbox', 'L1'),
                   ('loss_pw_ciou', 'PW-CIoU'), ('loss_ciou', 'CIoU'),
                   ('loss_diou', 'DIoU'), ('loss_giou', 'GIoU')):
        known_loss.add(k)
        if k in m:
            items.append(f'{lab} {m[k]:.3f}')
    for i in (1, 2, 3):
        k = f'loss_cls_aux_{i}'
        known_loss.add(k)
        if k in m:
            items.append(f'辅分{i} {m[k]:.3f}')
    fl = [m.get(f'loss_fdr_layer_{i}_left') for i in (1, 2, 3)]
    fr = [m.get(f'loss_fdr_layer_{i}_right') for i in (1, 2, 3)]
    for i in (1, 2, 3):
        known_loss.update((f'loss_fdr_layer_{i}_left', f'loss_fdr_layer_{i}_right'))
    if all(v is not None for v in fl):
        items.append('FDR左 ' + '→'.join(f'{v:.2f}' for v in fl))
        items.append('FDR右 ' + '→'.join(f'{v:.2f}' for v in fr))
    items += [f'{k} {m[k]:.3f}' for k in m
              if k.startswith('loss_') and k not in known_loss]
    line('损失', items)

    # —— 精化趋势（FDR 的故事线：MAE/IoU 应逐层向好）——
    ml_ = [m.get(f'fdr_lr_mae_layer_{i}_left') for i in (1, 2, 3)]
    mr_ = [m.get(f'fdr_lr_mae_layer_{i}_right') for i in (1, 2, 3)]
    io_ = [m.get(f'fdr_iou_layer_{i}') for i in (1, 2, 3)]
    if all(v is not None for v in ml_):
        line('精化', ['左MAE ' + '→'.join(f'{v:.3f}' for v in ml_),
                    '右MAE ' + '→'.join(f'{v:.3f}' for v in mr_),
                    'IoU ' + '→'.join(f'{v:.2f}' for v in io_)])

    # —— 匹配告警（5 项均短，单行展示）——
    line('匹配', [f'{lab} {m[k]:.2f}' for k, lab in
                (('matched_positives', '正样本'), ('empty_target_rois', '空ROI'),
                 ('gt_exceeds_queries_roi_ratio', 'GT超限'),
                 ('invalid_box_ratio', '无效框'),
                 ('fdr_target_overflow_ratio', 'FDR越界')) if k in m], per_line=5)

    # —— 权重分位数（P50/P90/max 三值压缩为一项）——
    w_items = []
    for base, lab in (('dyn_l1_center_w', '动态L1'), ('pw_ciou_w', 'PW')):
        qs = [m.get(f'{base}_{q}') for q in ('p50', 'p90', 'max')]
        if all(v is not None for v in qs):
            w_items.append(f'{lab} {qs[0]:.1f}/{qs[1]:.1f}/{qs[2]:.1f}')
    line('权重', w_items)
    print('─' * W)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # 逐步行：紧凑英文短码（loss/cls）；全部指标进 epoch 汇总与 log.txt
    _print_legend_once()
    metric_logger.display_keys = ['loss', 'class_error']
    metric_logger.label_map = {'loss': 'loss', 'class_error': 'cls'}
    header = 'Epoch {}'.format(epoch)
    print_freq = 20

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # unscaled 指标可由 scaled/损失权重换算，不进 logger（缩短每行输出；log.txt 保留全部 scaled）
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        if 'cardinality_error' in loss_dict_reduced:
            metric_logger.update(cardinality_error=loss_dict_reduced['cardinality_error'])
        # 通用扩展：不在 weight_dict 中的诊断指标（MRMPFormer v1 的 FDR 逐层 MAE/IoU、
        # 越界率、无效框比例、匹配统计、动态权重分位数等）也进入日志
        _extra = {k: v for k, v in loss_dict_reduced.items()
                  if k not in weight_dict and k not in ('class_error', 'cardinality_error')}
        if _extra:
            metric_logger.update(**_extra)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    _print_avg_stats(f"Epoch {epoch} 训练平均", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # 验证逐步行：同训练侧短码（loss/cls）；全部指标进汇总与 log.txt
    metric_logger.display_keys = ['loss', 'class_error']
    metric_logger.label_map = {'loss': 'loss', 'class_error': 'cls'}
    header = 'Val'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 25, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        if 'cardinality_error' in loss_dict_reduced:
            metric_logger.update(cardinality_error=loss_dict_reduced['cardinality_error'])
        # 与 train_one_epoch 一致的通用扩展：未加权诊断指标进日志
        _extra = {k: v for k, v in loss_dict_reduced.items()
                  if k not in weight_dict and k not in ('class_error', 'cardinality_error')}
        if _extra:
            metric_logger.update(**_extra)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    _print_avg_stats("验证集平均", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
