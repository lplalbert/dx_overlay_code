"""
水印定位器 — 训练脚本

用法:
    python train.py --data_dir training_data --epochs 100 --batch_size 4
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from network import WatermarkLocatorNet, WatermarkLocatorLoss
from dataset import WatermarkLocatorDataset


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_det = 0
    total_offset = 0
    total_decode = 0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        predictions = model(images)
        losses = criterion(predictions, targets)
        losses['total'].backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses['total'].item()
        total_det += losses['det'].item()
        total_offset += losses['offset'].item() if isinstance(losses['offset'], torch.Tensor) else losses['offset']
        total_decode += losses['decode'].item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}: total={losses['total'].item():.4f} "
                  f"det={losses['det'].item():.4f} "
                  f"decode={losses['decode'].item():.4f}")

    return {
        'loss': total_loss / max(num_batches, 1),
        'det': total_det / max(num_batches, 1),
        'offset': total_offset / max(num_batches, 1),
        'decode': total_decode / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_det = 0
    total_det = 0
    correct_decode = 0
    total_decode = 0
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        predictions = model(images)
        losses = criterion(predictions, targets)

        total_loss += losses['total'].item()
        num_batches += 1

        # 检测精度
        pred_conf = predictions['detections']['s1'][:, 0]  # (B, 4, 6)
        gt_conf = targets['locator_map']  # (B, 4, 6)
        pred_det = (pred_conf > 0.5).float()
        correct_det += (pred_det == gt_conf).sum().item()
        total_det += gt_conf.numel()

        # 解码精度
        pred_cw = predictions['codeword_logits'].argmax(dim=2)  # (B, 24)
        gt_cw = targets['codeword_labels']  # (B, 24)
        correct_decode += (pred_cw == gt_cw).sum().item()
        total_decode += gt_cw.numel()

    return {
        'loss': total_loss / max(num_batches, 1),
        'det_acc': correct_det / max(total_det, 1),
        'decode_acc': correct_decode / max(total_decode, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Watermark Locator")
    parser.add_argument("--data_dir", type=str, default="training_data",
                        help="Training data directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        help="Resize input to (W, H), e.g. --resize 960 540")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--block_rows", type=int, default=4)
    parser.add_argument("--block_cols", type=int, default=6)
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 数据集
    full_dataset = WatermarkLocatorDataset(
        args.data_dir, augment=True,
        resize=tuple(args.resize) if args.resize else None,
    )

    # 划分训练/验证
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # 验证集不用增强
    val_dataset_clean = WatermarkLocatorDataset(
        args.data_dir, augment=False,
        resize=tuple(args.resize) if args.resize else None,
    )
    # 用相同索引
    val_indices = val_dataset.indices
    val_subset = torch.utils.data.Subset(val_dataset_clean, val_indices)

    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # 模型
    model = WatermarkLocatorNet(
        in_ch=3,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        num_classes=17,  # 16个码字 + 1个定位图案
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 损失函数
    criterion = WatermarkLocatorLoss(
        lambda_det=1.0,
        lambda_offset=0.5,
        lambda_decode=1.0,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # 保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    best_val_acc = 0
    print(f"\n=== Training for {args.epochs} epochs ===")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 验证
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: loss={train_metrics['loss']:.4f} "
              f"det={train_metrics['det']:.4f} "
              f"decode={train_metrics['decode']:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f} "
              f"det_acc={val_metrics['det_acc']:.4f} "
              f"decode_acc={val_metrics['decode_acc']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        val_acc = val_metrics['det_acc'] * 0.5 + val_metrics['decode_acc'] * 0.5
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'best_val_acc': best_val_acc,
            }, save_path)
            print(f"  → Saved best model (val_acc={best_val_acc:.4f})")

        # 定期保存checkpoint
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)

    print(f"\n=== Training complete. Best val_acc: {best_val_acc:.4f} ===")


if __name__ == "__main__":
    main()
