"""
v1-vv2: 独立识别 + YOLOv8 目标检测

多数据集训练: 自动发现 /data1/lpl/datasets 下所有子目录的 train/val，
合并为统一的 YOLO 格式数据集后训练。

用法:
    python train.py --config config.yaml
    python train.py --config config.yaml --device 0
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset'))
from multi_dataset import discover_datasets, _list_images

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_v1_vv2_yolo.log'),
    ]
)
logger = logging.getLogger(__name__)


def build_yolo_dataset(data_root, output_dir, image_size=(1080, 1920)):
    """
    将多数据集合并为统一的 YOLO 格式目录:
        output_dir/
            images/train/  images/val/
            labels/train/  labels/val/
            watermark.yaml
    """
    datasets = discover_datasets(data_root)
    if not datasets:
        raise RuntimeError(f"No datasets found in {data_root}")

    for split in ('train', 'val'):
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    counter = {'train': 0, 'val': 0}

    for info in datasets:
        ds_name = info['name']
        for split in ('train', 'val'):
            images_dir = info.get(f'{split}_images')
            labels_dir = info.get(f'{split}_labels')
            if images_dir is None:
                continue

            files = _list_images(images_dir)
            for fname in files:
                stem = os.path.splitext(fname)[0]
                out_stem = f"{ds_name}_{stem}"

                # 转换并保存 Cb 通道图像
                img = cv2.imread(os.path.join(images_dir, fname))
                if img is None:
                    continue
                h, w = image_size
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                cb = ycrcb[:, :, 2]
                cb_3ch = cv2.merge([cb, cb, cb])
                cv2.imwrite(os.path.join(output_dir, 'images', split, out_stem + '.png'), cb_3ch)

                # 复制/转换标签
                lbl_src = None
                if labels_dir:
                    for ext in ('.txt',):
                        path = os.path.join(labels_dir, stem + ext)
                        if os.path.exists(path):
                            lbl_src = path
                            break

                lbl_dst = os.path.join(output_dir, 'labels', split, out_stem + '.txt')
                if lbl_src:
                    shutil.copy2(lbl_src, lbl_dst)
                else:
                    # 无标签时写空文件 (负样本)
                    with open(lbl_dst, 'w') as f:
                        pass

                counter[split] += 1

            logger.info(f"  [{ds_name}/{split}] {len(files)} images")

    logger.info(f"Merged dataset: train={counter['train']}, val={counter['val']}")

    # 写 YOLO YAML
    yaml_path = os.path.join(output_dir, 'watermark.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""# Auto-generated YOLO dataset config
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: locator
""")
    logger.info(f"YOLO config: {yaml_path}")
    return yaml_path, counter


def main():
    parser = argparse.ArgumentParser(description='v1-vv2 YOLOv8 训练 (独立识别)')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default=None, help='GPU编号(覆盖config)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg.get('device', '0')
    output_dir = cfg.get('output_dir', f'output/v1_vv2_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)

    logger.info("=" * 60)
    logger.info("v1-vv2: Independent YOLOv8 Detection")
    logger.info("=" * 60)

    # ── 构建合并数据集 ──
    data_root = cfg.get('data_root', '/data1/lpl/datasets')
    image_size = (cfg.get('image_height', 1080), cfg.get('image_width', 1920))

    logger.info(f"Building merged YOLO dataset from: {data_root}")
    merged_dir = os.path.join(output_dir, 'merged_data')
    yaml_path, counts = build_yolo_dataset(data_root, merged_dir, image_size)

    # ── 训练 ──
    model_name = cfg.get('model', 'yolov8n.pt')
    logger.info(f"Loading model: {model_name}")
    model = YOLO(model_name)

    finetune_cfg = cfg.get('finetune', {})
    weight_path = finetune_cfg.get('weight_path')
    if weight_path and os.path.exists(weight_path):
        logger.info(f"Loading pretrained: {weight_path}")
        model = YOLO(weight_path)

    results = model.train(
        data=yaml_path,
        epochs=cfg.get('epochs', 100),
        imgsz=cfg.get('imgsz', 1080),
        batch=cfg.get('batch_size', 8),
        lr0=cfg.get('lr', 0.01),
        device=device,
        project=output_dir,
        name='yolo_train',
        exist_ok=True,
        patience=cfg.get('patience', 30),
        save=True,
        save_period=cfg.get('save_every', 10),
        verbose=True,
    )

    # ── 保存结果 ──
    best_pt = os.path.join(output_dir, 'yolo_train', 'weights', 'best.pt')
    final_pt = os.path.join(output_dir, 'yolo_train', 'weights', 'last.pt')

    summary = {
        'best_weights': best_pt if os.path.exists(best_pt) else None,
        'last_weights': final_pt if os.path.exists(final_pt) else None,
        'train_samples': counts.get('train', 0),
        'val_samples': counts.get('val', 0),
        'config': cfg,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best model: {best_pt}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
