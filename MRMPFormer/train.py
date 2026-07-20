"""
MRMPFormer SimCLR 对比学习训练脚本。

所有超参数从 utils/config.py 的预设配置中读取，通过 --config 选择。

用法:
    python train.py --config simclr_baseline      # 标准 SimCLR 增强
    python train.py --config chromatogram_v1      # 色谱图专用增强
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import SimCLR
from utils import NT_XentLoss, alignment_loss, uniformity_loss, PRESETS
from augmentations import get_simclr_augmentations, get_chromatogram_augmentations
from dataset import SimCLRDataset


def get_best_device():
    """自动选择最佳可用设备: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser(
        description='MRMPFormer — SimCLR 对比学习训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 实验选择：其余所有超参数从 utils/config.py 的预设配置中读取
    parser.add_argument('--config', type=str, default='simclr_baseline',
                        choices=['simclr_baseline', 'chromatogram_v1'],
                        help='预设实验配置 (定义在 utils/config.py 中)')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """保存训练 checkpoint (包含完整模型 + 优化器状态)。"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """训练一个 epoch, 返回平均 loss。使用 tqdm 进度条实时显示。"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch:3d}/{config.epochs}',
                unit='batch', leave=False, ncols=100, ascii=True)

    for batch_idx, (view_a, view_b) in enumerate(pbar):
        view_a = view_a.to(device)
        view_b = view_b.to(device)

        # 前向传播
        z_a = model(view_a)  # (N, proj_output_dim)
        z_b = model(view_b)  # (N, proj_output_dim)

        loss = criterion(z_a, z_b)

        # 梯度累积
        loss = loss / config.gradient_accumulation
        loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        current_loss = loss.item() * config.gradient_accumulation
        total_loss += current_loss
        num_batches += 1

        # 实时更新进度条
        running_avg = total_loss / num_batches
        pbar.set_postfix(loss=f'{running_avg:.4f}')

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def compute_alignment_uniformity(model, dataloader, device):
    """
    在一个 batch 上计算 Alignment 和 Uniformity 指标。
    使用 eval 模式，不更新模型参数。

    Returns:
        (align, unif): 两个标量 float
    """
    model.eval()
    view_a, view_b = next(iter(dataloader))
    view_a, view_b = view_a.to(device), view_b.to(device)

    z_a = model(view_a)
    z_b = model(view_b)

    align = alignment_loss(z_a, z_b).item()
    z_all = torch.cat([z_a, z_b], dim=0)
    unif = uniformity_loss(z_all).item()

    model.train()
    return align, unif


def main():
    args = get_args()
    config = PRESETS[args.config]
    set_seed(config.seed)

    # ---- 设备 ----
    device = get_best_device()
    print(f"[设备] 使用: {device}")
    print(f"[实验] {config.name} | 增强: {config.augmentation}")

    # ---- 输出目录（按实验名称隔离）----
    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.log_dir) / config.name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S')))

    # ---- 数据集 ----
    print(f"[数据] 加载目录: {config.data_dir}")
    if config.augmentation == 'simclr':
        transform = get_simclr_augmentations()
    elif config.augmentation == 'chromatogram':
        transform = get_chromatogram_augmentations(config)
    else:
        raise ValueError(f"未知的增强策略: {config.augmentation}")
    dataset = SimCLRDataset(config.data_dir, transform=transform)
    print(f"[数据] 图像数量: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"[数据] Batch 数/epoch: {len(dataloader)} (batch_size={config.batch_size})")

    # ---- 模型 ----
    freeze_stages = getattr(config, 'freeze_stages', 0)
    model = SimCLR(
        proj_hidden_dim=config.proj_hidden_dim,
        proj_output_dim=config.proj_output_dim,
        pretrained=config.pretrained,
        freeze_stages=freeze_stages,
    ).to(device)
    trainable_params, total_params = model.trainable_param_counts()
    frozen_str = f" | 冻结: {total_params - trainable_params:,}" if freeze_stages > 0 else ""
    print(f"[模型] 总参数量: {total_params:,} | 可训练: {trainable_params:,}{frozen_str}")

    # ---- 损失 & 优化器 & 调度器 ----
    criterion = NT_XentLoss(temperature=config.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6,
    )

    es_enabled = getattr(config, 'early_stopping_enabled', False)
    if es_enabled:
        print(f"[早停] 启用 | 容忍: {config.early_stopping_patience} epoch | "
              f"最小改善: {config.early_stopping_min_delta} | "
              f"保护期: {config.early_stopping_min_epochs} epoch")
    print(f"[训练] Epochs: {config.epochs} | LR: {config.lr:.0e} → 1e-6 | τ: {config.temperature}")
    print("=" * 60)

    # ---- 训练循环 ----
    best_loss = float('inf')
    es_counter = 0
    es_stopped_epoch = config.epochs

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch, config
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 日志
        print(
            f"Epoch [{epoch:3d}/{config.epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"LR: {current_lr:.2e} "
            f"Time: {epoch_time:.1f}s"
        )
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # 定期计算 Alignment + Uniformity
        metrics_every = getattr(config, 'eval_metrics_every', 0)
        if metrics_every > 0 and epoch % metrics_every == 0:
            align, unif = compute_alignment_uniformity(model, dataloader, device)
            writer.add_scalar('Metrics/alignment', align, epoch)
            writer.add_scalar('Metrics/uniformity', unif, epoch)
            print(f"  ↳ Alignment: {align:.4f} | Uniformity: {unif:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(output_dir, 'best_model.pth'),
            )
            print(f"  ↳ 最佳模型已保存 (loss={best_loss:.4f})")

        # 定期保存
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch:04d}.pth'),
            )

        # ---- 早停检查 ----
        if es_enabled and epoch >= config.early_stopping_min_epochs:
            if avg_loss < best_loss - config.early_stopping_min_delta * best_loss:
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= config.early_stopping_patience:
                    es_stopped_epoch = epoch
                    print(f"\n⏹ 早停触发 @ epoch {epoch} — "
                          f"连续 {config.early_stopping_patience} epoch "
                          f"loss 改善 < {config.early_stopping_min_delta:.0e}")
                    break

    # 保存最终模型
    save_checkpoint(
        model, optimizer, es_stopped_epoch, avg_loss,
        os.path.join(output_dir, 'last_model.pth'),
    )
    print(f"\n训练完成! 共 {es_stopped_epoch}/{config.epochs} epoch | 最低 loss: {best_loss:.4f}")
    print(f"模型保存在: {output_dir.resolve()}")

    writer.close()


if __name__ == '__main__':
    main()
