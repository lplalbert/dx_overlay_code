"""
v2-vv1: 联合识别 + U-Net 语义分割

利用6个定位块的棋盘格固定间隔几何先验，联合拟合网格。
后处理: 全局网格拟合 → 所有框联合定位，而非独立连通域。

棋盘格位置 (block网格 4x6):
    (0,3), (1,1), (1,5), (2,3), (3,1), (3,5)
归一化中心坐标（无噪声时）:
    (0.625, 0.1875), (0.292, 0.438), (0.958, 0.438),
    (0.625, 0.688), (0.292, 0.938), (0.958, 0.938)

用法:
    python train.py --data_dir ../../dataset/data --epochs 50
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet'))
from unet.unet_model import UNet

# ── 棋盘格几何先验 ──
BLOCK_ROWS, BLOCK_COLS = 4, 6
LOCATOR_GRID = [(0, 3), (1, 1), (1, 5), (2, 3), (3, 1), (3, 5)]  # (row, col)
NUM_LOCATORS = 6


def get_expected_centers(screen_w, screen_h):
    """返回6个定位块的期望归一化中心坐标。"""
    block_h = screen_h / BLOCK_ROWS
    block_w = screen_w / BLOCK_COLS
    msg_h = block_h / 2
    msg_w = block_w / 2
    centers = []
    for (bi, bj) in LOCATOR_GRID:
        cx = (bj * block_w + msg_w + msg_w / 2) / screen_w
        cy = (bi * block_h + msg_h + msg_h / 2) / screen_h
        centers.append((cx, cy))
    return np.array(centers)


class WatermarkSegmentationDataset(Dataset):
    """Cb通道 → 定位块mask的分割数据集。"""

    def __init__(self, image_dir, mask_dir, image_size=(1080, 1920)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.image_dir, name))
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cb = ycrcb[:, :, 2].astype(np.float32) / 255.0

        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        image_t = torch.from_numpy(cb).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return image_t, mask_t


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def bce_dice_loss(pred, target):
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)
    return bce + dsc


def postprocess_joint(prob_map, screen_w=1920, screen_h=1080, threshold=0.5):
    """
    v2联合识别: 利用棋盘格几何先验联合拟合。

    1. 在每个期望位置附近搜索连通域
    2. 用找到的质心拟合全局仿射变换（平移+缩放）
    3. 用变换后的网格统一输出所有6个框
    """
    binary = (prob_map > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    expected = get_expected_centers(screen_w, screen_h)  # (6, 2) 归一化
    msg_w = screen_w / BLOCK_COLS / 2
    msg_h = screen_h / BLOCK_ROWS / 2

    # Step 1: 在每个期望位置附近找最近的连通域
    matched_centroids = []
    matched_indices = []
    search_radius = 0.15  # 归一化搜索半径

    for i, (ex, ey) in enumerate(expected):
        best_dist = search_radius
        best_centroid = None
        best_label = 0
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < 500:
                continue
            cx = centroids[j][0] / screen_w
            cy = centroids[j][1] / screen_h
            dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if dist < best_dist:
                best_dist = dist
                best_centroid = (cx, cy)
                best_label = j
        if best_centroid is not None:
            matched_centroids.append(best_centroid)
            matched_indices.append(i)

    # Step 2: 拟合全局仿射变换 (平移+缩放，无旋转)
    if len(matched_centroids) >= 2:
        src = expected[matched_indices]
        dst = np.array(matched_centroids)
        # 简单的平移+各向同性缩放拟合
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        scale = np.sum(src_centered * dst_centered) / (np.sum(src_centered**2) + 1e-8)
        offset = dst_mean - scale * src_mean

        # Step 3: 用拟合的变换输出所有6个框
        boxes = []
        for i in range(NUM_LOCATORS):
            cx, cy = expected[i]
            tcx = scale * cx + offset[0]
            tcy = scale * cy + offset[1]
            x = int((tcx * screen_w) - msg_w / 2)
            y = int((tcy * screen_h) - msg_h / 2)
            boxes.append((x, y, int(msg_w), int(msg_h)))
        return boxes, len(matched_centroids)
    else:
        # 匹配不足，退回独立识别
        boxes = []
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < 500:
                continue
            x = stats[j, cv2.CC_STAT_LEFT]
            y = stats[j, cv2.CC_STAT_TOP]
            w = stats[j, cv2.CC_STAT_WIDTH]
            h = stats[j, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, w, h))
        return boxes, 0


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[v2-vv1] Joint U-Net Segmentation (checkerboard prior)")
    print(f"Device: {device}")
    print(f"Locator grid positions: {LOCATOR_GRID}")

    train_dataset = WatermarkSegmentationDataset(
        os.path.join(args.data_dir, 'vv1', 'images'),
        os.path.join(args.data_dir, 'vv1', 'masks'),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    print(f"训练集: {len(train_dataset)} 张")

    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = bce_dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"v2_vv1_unet_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    print("[v2-vv1] 训练完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../dataset/data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
