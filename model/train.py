import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from framework import datasets
from framework.util import misc as utils
from framework.datasets import build_dataset, get_coco_api_from_dataset
from framework.engine import evaluate, train_one_epoch
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr_drop', default=35, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--model', default='quanformer', type=str,
                        choices=('quanformer', 'mrmpformer_v1'),
                        help="Model variant: quanformer (baseline) | mrmpformer_v1")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--iou_type', default="ciou", type=str,
                        help="giou, diou, ciou")
    parser.add_argument('--set_cost_iou', default=2, type=float,
                        help="iou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--iou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help='Relative classification weight of the no-object class')

    # * MRMPFormer v1 分类/定位损失参数
    parser.add_argument('--classification_loss', default='focal', type=str,
                        choices=('focal', 'ce'),
                        help='MRMPFormer v1: 分类损失（focal=Softmax Focal / ce=基线交叉熵）')
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='MRMPFormer v1: Focal Loss alpha（前景=alpha，背景=1-alpha）')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='MRMPFormer v1: Focal Loss gamma（聚焦参数，论文 β=2 按 γ 统一）')
    parser.add_argument('--cls_loss_coef', default=1.0, type=float,
                        help='MRMPFormer v1: 主分类损失权重 λ_cls')
    parser.add_argument('--aux_class_loss', default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
                        help='MRMPFormer v1: 中间层（1/2 层）辅助分类损失开关')
    parser.add_argument('--aux_class_loss_coef', default=1.0, type=float,
                        help='MRMPFormer v1: 中间层辅助分类损失权重')
    parser.add_argument('--dynamic_l1_enabled', default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
                        help='MRMPFormer v1: 动态加权 L1（λ_c=1/(w_gt+eps)）开关')
    parser.add_argument('--dynamic_l1_eps', default=1e-6, type=float,
                        help='MRMPFormer v1: 动态 L1 eps（防除零）')
    parser.add_argument('--dynamic_l1_lambda_w', default=1.0, type=float,
                        help='MRMPFormer v1: 动态 L1 宽度项权重 λ_w')
    parser.add_argument('--dynamic_l1_lambda_h', default=1.0, type=float,
                        help='MRMPFormer v1: 动态 L1 高度项权重 λ_h')
    parser.add_argument('--center_weight_clip', default=None, type=float,
                        help='MRMPFormer v1: 动态中心权重上限（null=不裁剪，忠实复现公式）')
    parser.add_argument('--normalize_dynamic_weights', default=False, type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
                        help='MRMPFormer v1: 动态权重按均值归一（默认 false 忠实复现公式）')
    parser.add_argument('--pw_ciou_enabled', default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
                        help='MRMPFormer v1: PW-CIoU 开关（false 回退原 CIoU 基线）')
    parser.add_argument('--pw_ciou_weight_mode', default='ratio', type=str,
                        choices=('ratio', 'one_plus_ratio'),
                        help='MRMPFormer v1: PW-CIoU 中心项权重模式 ratio=bar_w/(w_gt+eps)')
    parser.add_argument('--pw_ciou_eps', default=1e-6, type=float,
                        help='MRMPFormer v1: PW-CIoU eps')
    parser.add_argument('--pw_ciou_weight_clip', default=None, type=float,
                        help='MRMPFormer v1: PW-CIoU 权重上限（null=不裁剪）')
    parser.add_argument('--pw_ciou_mean_width', default=None, type=float,
                        help='MRMPFormer v1: bar_w 训练集 GT 平均峰宽（归一化）。'
                             'merged/train 实测=0.2297；null 回退权重 1（等价原 CIoU）')
    parser.add_argument('--recall_loss_enabled', default=False, type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
                        help='MRMPFormer v1: Recall Loss 实验开关（默认关闭；论文定义未确认，'
                             '启用将报错提示，防止编造公式）')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='../output/train/run',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='auto',
                        help='device to use for training / testing ("auto", "cuda", "cpu", "mps")')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--resume', default='checkpoint.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='微调模式：只加载模型权重，不加载 optimizer/lr_scheduler，start_epoch 归零，'
                             '以当前 lr 从头训练（配合 --resume 做 fine-tune）')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # ---- 简明启动信息（完整参数存档至 output_dir/config_used.txt）----
    _dev_name = device.type.upper()
    if device.type == 'cuda':
        _dev_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    _mode = "微调" if (args.resume and args.reset_optimizer) else ("续训" if args.resume else "从零训练")
    print("=" * 64)
    print(f"{args.model} 训练 | {_mode} | {utils.get_sha()}")
    print("-" * 64)
    print(f"模型   : {args.model} ({args.backbone}, queries={args.num_queries}, "
          f"enc/dec={args.enc_layers}/{args.dec_layers})")
    print(f"数据   : {args.coco_path} | epochs={args.epochs} | batch={args.batch_size}")
    print(f"学习率 : lr={args.lr:g} (backbone {args.lr_backbone:g}) | lr_drop={args.lr_drop}")
    if args.resume:
        print(f"权重   : {args.resume}" +
              (" [重置优化器]" if args.reset_optimizer else f" [从 epoch {args.start_epoch} 续训]"))
    else:
        print("权重   : 随机初始化")
    print(f"设备   : {_dev_name} | 输出: {args.output_dir}/")
    print("=" * 64)
    if args.output_dir:
        _out_dir = Path(args.output_dir)
        _out_dir.mkdir(parents=True, exist_ok=True)
        with (_out_dir / 'config_used.txt').open('w', encoding='utf-8') as _f:
            _f.write(utils.get_sha() + "\n" + str(args) + "\n")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 可训练参数量: {n_parameters / 1e6:.1f}M")

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = utils.safe_torch_load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = utils.safe_torch_load(args.resume, map_location='cpu')
        # 仅当 checkpoint 与当前模型同名字段维度不一致时才跳过（COCO 预训练迁移场景）；
        # 自训 checkpoint 续训/微调时维度一致，分类头与 query_embed 应完整加载，避免静默随机初始化
        _model_state = model_without_ddp.state_dict()
        _skip_keys = []
        for _k, _v in checkpoint['model'].items():
            if _k in _model_state and _v.shape != _model_state[_k].shape:
                _skip_keys.append(_k)
        if _skip_keys:
            print(f"[WARN] resume: 维度不匹配，跳过 {len(_skip_keys)} 个权重: {_skip_keys}")
            for _k in _skip_keys:
                checkpoint['model'].pop(_k, None)
        if args.model == 'mrmpformer_v1':
            # 旧 QuanFormer 单层 checkpoint → v1 迁移：L1 参数复制初始化到 L2/L3，
            # FDR/边界反馈模块保持新初始化；迁移报告分类打印，禁止静默 strict=False
            _is_legacy = any(k.startswith("transformer.decoder.layers.") for k in checkpoint['model']) \
                and not any(k.startswith("fdr_heads.") for k in checkpoint['model'])
            if _is_legacy:
                from models.mrmpformer.v1.detr import load_legacy_quanformer_state
                load_legacy_quanformer_state(model_without_ddp, checkpoint['model'], verbose=True)
            else:
                _report = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                if _report.missing_keys or _report.unexpected_keys:
                    print(f"[WARN] resume: missing={list(_report.missing_keys)[:10]} "
                          f"unexpected={list(_report.unexpected_keys)[:10]}")
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            if args.reset_optimizer:
                # 微调：不延续旧 lr/动量，start_epoch 归零，用当前 config/CLI 的 lr 从头训练
                print(f"[INFO] 微调模式(--reset_optimizer)：跳过 optimizer/lr_scheduler 加载，start_epoch=0")
                args.start_epoch = 0
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("[开始] 训练")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        _t_epoch = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 10 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

        # ---- 每 epoch 简明摘要（详细 loss 见上方 Averaged stats 与 log.txt）----
        _bbox_stats = test_stats.get('coco_eval_bbox')
        _ap = (f" | AP50 {_bbox_stats[1]:.3f} | AP75 {_bbox_stats[2]:.3f}"
               if _bbox_stats else "")
        print("-" * 64)
        print(f"[Epoch {epoch + 1}/{args.epochs}] "
              f"train_loss {train_stats['loss']:.4f} | val_loss {test_stats['loss']:.4f}"
              f"{_ap} | lr {train_stats['lr']:.2e} | {time.time() - _t_epoch:.0f}s")
        print("-" * 64)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"[完成] 训练总耗时 {total_time_str} | 最终权重 {output_dir / 'checkpoint.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MRMPFormer training and evaluation script', parents=[get_args_parser()])
    # 参数配置外置：--config 指定 JSON 配置文件作为默认值，CLI 参数仍可覆盖（improve.md 第 2 项）
    parser.add_argument('--config', type=str, default=None, help='JSON 配置文件路径（作为默认参数，CLI 可覆盖）')
    _known, _ = parser.parse_known_args()
    if _known.config:
        import json as _json
        with open(_known.config, encoding='utf-8') as _f:
            _cfg = _json.load(_f)
        _cfg.pop('config', None)
        _cfg = {_k: _v for _k, _v in _cfg.items() if not _k.startswith('_')}  # 过滤 _comment_* 注释键
        parser.set_defaults(**_cfg)
        print(f"[INFO] 已加载配置: {_known.config}")
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
