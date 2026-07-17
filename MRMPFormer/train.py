"""
MRMPFormer SimCLR 对比学习训练脚本。

用法:
    python train.py --data_dir data/images --batch_size 256 --epochs 300
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

from models import SimCLR
from utils import NT_XentLoss
from augmentations import get_simclr_augmentations
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

    # 数据
    parser.add_argument('--data_dir', type=str, default='data/images',
                        help='训练图像目录')

    # 模型
    parser.add_argument('--proj_hidden_dim', type=int, default=512,
                        help='投影头隐藏层维度')
    parser.add_argument('--proj_output_dim', type=int, default=128,
                        help='投影头输出维度')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='加载 ImageNet 预训练权重')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                        help='不使用预训练权重')

    # 训练
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减 (L2 正则)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='NT-Xent 温度系数 τ')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='梯度累积步数 (batch_size 过大无法放入 GPU 时增大此值)')

    # 保存
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='TensorBoard 日志目录')
    parser.add_argument('--save_every', type=int, default=50,
                        help='每隔多少 epoch 保存一次 checkpoint')

    # 其他
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """训练一个 epoch, 返回平均 loss。"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    optimizer.zero_grad()

    for batch_idx, (view_a, view_b) in enumerate(dataloader):
        view_a = view_a.to(device)
        view_b = view_b.to(device)

        # 前向传播
        z_a = model(view_a)  # (N, proj_output_dim)
        z_b = model(view_b)  # (N, proj_output_dim)

        loss = criterion(z_a, z_b)

        # 梯度累积
        loss = loss / args.gradient_accumulation
        loss.backward()

        if (batch_idx + 1) % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_accumulation
        num_batches += 1

        # 每 20 个 batch 打印进度
        if (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:3d}/{args.epochs}] "
                f"Batch [{batch_idx + 1:4d}/{len(dataloader)}] "
                f"Loss: {loss.item() * args.gradient_accumulation:.4f} "
                f"Time: {elapsed:.1f}s"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    args = get_args()
    set_seed(args.seed)

    # ---- 设备 ----
    device = get_best_device()
    print(f"[设备] 使用: {device}")

    # ---- 输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S')))

    # ---- 数据集 ----
    print(f"[数据] 加载目录: {args.data_dir}")
    transform = get_simclr_augmentations()
    dataset = SimCLRDataset(args.data_dir, transform=transform)
    print(f"[数据] 图像数量: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"[数据] Batch 数/epoch: {len(dataloader)} (batch_size={args.batch_size})")

    # ---- 模型 ----
    model = SimCLR(
        proj_hidden_dim=args.proj_hidden_dim,
        proj_output_dim=args.proj_output_dim,
        pretrained=args.pretrained,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数量: {total_params:,} | 可训练: {trainable_params:,}")

    # ---- 损失 & 优化器 & 调度器 ----
    criterion = NT_XentLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    print(f"[训练] Epochs: {args.epochs} | LR: {args.lr} → 1e-6 | τ: {args.temperature}")
    print("=" * 60)

    # ---- 训练循环 ----
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch, args
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 日志
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"LR: {current_lr:.2e} "
            f"Time: {epoch_time:.1f}s"
        )
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(output_dir, 'best_model.pth'),
            )
            print(f"  ↳ 最佳模型已保存 (loss={best_loss:.4f})")

        # 定期保存
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch:04d}.pth'),
            )

    # 保存最终模型
    save_checkpoint(
        model, optimizer, args.epochs, avg_loss,
        os.path.join(output_dir, 'last_model.pth'),
    )
    print(f"\n训练完成! 最低 loss: {best_loss:.4f}")
    print(f"模型保存在: {output_dir.resolve()}")

    writer.close()


if __name__ == '__main__':
    main()
