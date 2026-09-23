"""
v1-vv1: 独立识别 + U-Net 语义分割

多数据集训练，参考 fftmask/train_cb_v18_pair.py 风格。
支持 /data1/lpl/datasets 下所有子目录的 train/val 自动发现。

用法:
    python train.py --config config.yaml
    python train.py --config config.yaml --device 0
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet'))

from multi_dataset import build_multi_dataset
from unet.unet_model import UNet

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_v1_vv1_unet.log'),
    ]
)
logger = logging.getLogger(__name__)


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = 1 - (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return dice.mean()


def bce_dice_loss(pred, target):
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)
    return bce + dsc


def validate(model, val_loader, device, threshold=0.5):
    """验证: 返回 IoU, Dice, Precision, Recall。"""
    model.eval()
    total_tp = total_fp = total_fn = total_tn = 0
    total_dice = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > threshold).float()

            tp = (preds * masks).sum().item()
            fp = (preds * (1 - masks)).sum().item()
            fn = ((1 - preds) * masks).sum().item()
            tn = ((1 - preds) * (1 - masks)).sum().item()
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()
            dice = (2. * intersection + 1.0) / (union + 1.0)
            total_dice += dice.item()
            n_batches += 1

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    dice = total_dice / max(n_batches, 1)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    return {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall}


def main():
    parser = argparse.ArgumentParser(description='v1-vv1 U-Net 训练 (独立识别)')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default=None, help='GPU编号(覆盖config)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device or cfg.get('device', '0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = cfg.get('output_dir', f'output/v1_vv1_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)

    logger.info("=" * 60)
    logger.info("v1-vv1: Independent U-Net Segmentation")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")

    # ── 数据集 ──
    data_root = cfg.get('data_root', '/data1/lpl/datasets')
    image_size = (cfg.get('image_height', 1080), cfg.get('image_width', 1920))

    logger.info(f"Discovering datasets in: {data_root}")
    train_dataset = build_multi_dataset(data_root, split='train', task='segmentation', image_size=image_size)
    val_dataset = build_multi_dataset(data_root, split='val', task='segmentation', image_size=image_size)

    train_length = cfg.get('train_length', 0)
    val_length = cfg.get('val_length', 0)
    if train_length > 0 and len(train_dataset) > train_length:
        indices = np.random.choice(len(train_dataset), train_length, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    if val_length > 0 and len(val_dataset) > val_length:
        indices = np.random.choice(len(val_dataset), val_length, replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    batch_size = cfg.get('batch_size', 4)
    num_workers = cfg.get('num_workers', 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── 模型 ──
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)

    finetune_cfg = cfg.get('finetune', {})
    weight_path = finetune_cfg.get('weight_path')
    if weight_path and os.path.exists(weight_path):
        logger.info(f"Loading pretrained: {weight_path}")
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        logger.info("Weights loaded")

    epochs = finetune_cfg.get('epochs', cfg.get('epochs', 100))
    lr = finetune_cfg.get('lr', cfg.get('lr', 0.001))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=cfg.get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    save_every = cfg.get('save_every', 10)

    # ── 训练 ──
    best_dice = 0.0
    best_epoch = 0

    logger.info("\nStarting training...")
    logger.info("-" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = bce_dice_loss(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(len(train_loader), 1)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        epoch_time = time.time() - start_time

        logger.info(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s]")
        logger.info(f"  Train Loss: {avg_loss:.4f}")
        logger.info(f"  Val Dice: {val_metrics['dice']:.4f}  IoU: {val_metrics['iou']:.4f}  "
                     f"P: {val_metrics['precision']:.4f}  R: {val_metrics['recall']:.4f}")

        # 保存最佳
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_dice': best_dice,
                'val_metrics': val_metrics,
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"  ✓ Best model saved (dice={best_dice:.4f})")

        # 定期保存
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'best_dice': best_dice,
                'best_epoch': best_epoch,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"  ✓ Checkpoint saved: epoch_{epoch+1}")

    # ── 保存结果 ──
    final_metrics = validate(model, val_loader, device)
    torch.save({'epoch': epochs, 'model': model.state_dict()},
               os.path.join(output_dir, 'final_model.pth'))

    results = {
        'best_epoch': best_epoch,
        'best_dice': float(best_dice),
        'final_metrics': {k: float(v) for k, v in final_metrics.items()},
        'total_epochs': epochs,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'config': cfg,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best epoch: {best_epoch}, Best dice: {best_dice:.4f}")
    logger.info(f"Final: {final_metrics}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
