#!/usr/bin/env python
"""画 vv1 / vv2 训练关键曲线, 每条曲线单独一张图。

数据源:
  vv1  train_vv1_v2.log                      (Epoch N / Train Loss / Val Dice IoU P R)
  vv2  runs/detect/.../results.csv           (ultralytics 标准 15 列)

输出: docs/figures/curves/*.png
"""
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/data1/lpl/dx_overlay_code'
LOG_VV1 = os.path.join(REPO, 'train_vv1_v2.log')
CSV_VV2 = os.path.join(REPO, 'runs/detect/output/v1_vv2_yolo_v2',
                       'yolo_noisy_3noise_finetune/results.csv')
OUT = os.path.join(REPO, 'docs/figures/curves')


def parse_vv1(path):
    """从训练日志抠出每轮的 train loss 与 val 指标。"""
    epochs, loss, dice, iou, prec, rec = [], [], [], [], [], []
    cur_ep = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'INFO -' not in line:
                continue  # 丢掉 tqdm 进度条
            body = line.split('INFO -', 1)[1]
            m = re.search(r'Epoch (\d+) \[', body)
            if m:
                cur_ep = int(m.group(1))
                continue
            m = re.search(r'Train Loss:\s*([\d.]+)', body)
            if m and cur_ep is not None:
                epochs.append(cur_ep)
                loss.append(float(m.group(1)))
                continue
            m = re.search(r'Val Dice:\s*([\d.]+)\s+IoU:\s*([\d.]+)\s+'
                          r'P:\s*([\d.]+)\s+R:\s*([\d.]+)', body)
            if m and cur_ep is not None:
                dice.append(float(m.group(1)))
                iou.append(float(m.group(2)))
                prec.append(float(m.group(3)))
                rec.append(float(m.group(4)))
    n = min(len(epochs), len(dice), len(loss))
    return {
        'epoch': epochs[:n], 'train_loss': loss[:n],
        'dice': dice[:n], 'iou': iou[:n], 'precision': prec[:n], 'recall': rec[:n],
    }


def parse_vv2(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    keys = {
        'epoch': 'epoch',
        'train_box_loss': 'train/box_loss', 'train_cls_loss': 'train/cls_loss',
        'train_dfl_loss': 'train/dfl_loss',
        'val_box_loss': 'val/box_loss', 'val_cls_loss': 'val/cls_loss',
        'val_dfl_loss': 'val/dfl_loss',
        'precision': 'metrics/precision(B)', 'recall': 'metrics/recall(B)',
        'mAP50': 'metrics/mAP50(B)', 'mAP50-95': 'metrics/mAP50-95(B)',
    }
    out = {}
    for name, col in keys.items():
        out[name] = [float(r[col]) for r in rows]
    out['epoch'] = [int(e) for e in out['epoch']]
    return out


def save_curve(epochs, series, title, ylabel, fname, ylim=None):
    """一条曲线一张图。series = [(label, values, color), ...]"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, vals, color in series:
        ax.plot(epochs, vals, marker='o', ms=3.5, lw=1.8, color=color, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  {fname}')


def main():
    os.makedirs(OUT, exist_ok=True)
    v1 = parse_vv1(LOG_VV1)
    v2 = parse_vv2(CSV_VV2)
    print(f'vv1: {len(v1["epoch"])} epochs   vv2: {len(v2["epoch"])} epochs')

    print('\n[vv1 U-Net]')
    e1 = v1['epoch']
    save_curve(e1, [('train loss', v1['train_loss'], '#1f77b4')],
               'vv1 U-Net — Train Loss', 'Loss', 'vv1_train_loss.png')
    save_curve(e1, [('val Dice', v1['dice'], '#2ca02c')],
               'vv1 U-Net — Val Dice', 'Dice', 'vv1_val_dice.png')
    save_curve(e1, [('val IoU', v1['iou'], '#9467bd')],
               'vv1 U-Net — Val IoU', 'IoU', 'vv1_val_iou.png')
    save_curve(e1, [('val Precision', v1['precision'], '#ff7f0e')],
               'vv1 U-Net — Val Precision', 'Precision', 'vv1_val_precision.png')
    save_curve(e1, [('val Recall', v1['recall'], '#d62728')],
               'vv1 U-Net — Val Recall', 'Recall', 'vv1_val_recall.png')

    print('\n[vv2 YOLOv8]')
    e2 = v2['epoch']
    save_curve(e2, [('train box_loss', v2['train_box_loss'], '#1f77b4'),
                    ('val box_loss', v2['val_box_loss'], '#ff7f0e')],
               'vv2 YOLOv8 — Box Loss', 'Loss', 'vv2_box_loss.png')
    save_curve(e2, [('train cls_loss', v2['train_cls_loss'], '#1f77b4'),
                    ('val cls_loss', v2['val_cls_loss'], '#ff7f0e')],
               'vv2 YOLOv8 — Cls Loss', 'Loss', 'vv2_cls_loss.png')
    save_curve(e2, [('train dfl_loss', v2['train_dfl_loss'], '#1f77b4'),
                    ('val dfl_loss', v2['val_dfl_loss'], '#ff7f0e')],
               'vv2 YOLOv8 — DFL Loss', 'Loss', 'vv2_dfl_loss.png')
    save_curve(e2, [('precision', v2['precision'], '#ff7f0e')],
               'vv2 YOLOv8 — Val Precision', 'Precision', 'vv2_val_precision.png')
    save_curve(e2, [('recall', v2['recall'], '#d62728')],
               'vv2 YOLOv8 — Val Recall', 'Recall', 'vv2_val_recall.png')
    save_curve(e2, [('mAP50', v2['mAP50'], '#2ca02c')],
               'vv2 YOLOv8 — mAP@50', 'mAP@50', 'vv2_val_map50.png')
    save_curve(e2, [('mAP50-95', v2['mAP50-95'], '#9467bd')],
               'vv2 YOLOv8 — mAP@50-95', 'mAP@50-95', 'vv2_val_map5095.png')

    print(f'\noutputs: {OUT}')


if __name__ == '__main__':
    main()
