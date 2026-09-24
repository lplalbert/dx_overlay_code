"""
v1-vv2: 独立识别 + YOLOv8 目标检测

多数据集训练: 自动发现 data_root 下所有子目录的 train/val，
合并为统一的 YOLO 格式数据集后训练。

两阶段训练 (推荐): config 里给 `stages`, 先 clean 预训练再 noisy 微调:
    stages:
      - {name: clean_pretrain,  data_root: .../clean, epochs: 10, lr: 0.01}
      - {name: noisy_finetune,  data_root: .../noisy, epochs: 50, lr: 0.005}
后一阶段自动加载前一阶段的 best.pt。不给 `stages` 则退回单阶段。

用法:
    python train.py --config config.yaml
    python train.py --config config.yaml --device 0
"""

import argparse
import json
import logging
import os
import random
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _merge_stamp(data_root, image_size, train_length=0, val_length=0):
    """数据源指纹: 文件数或抽样上限变了才触发重建。"""
    datasets = discover_datasets(data_root)
    n = 0
    for info in datasets:
        for split in ('train', 'val'):
            d = info.get(f'{split}_images')
            if d and os.path.isdir(d):
                n += len(_list_images(d))
    return (f"{os.path.abspath(data_root)}|{image_size[0]}x{image_size[1]}"
            f"|{len(datasets)}ds|{n}img"
            f"|tr{int(train_length or 0)}|va{int(val_length or 0)}")


def build_yolo_dataset(data_root, output_dir, image_size=(1080, 1920), force=False,
                       train_length=0, val_length=0, seed=42):
    """
    将多数据集合并为统一的 YOLO 格式目录:
        output_dir/
            images/train/  images/val/
            labels/train/  labels/val/
            watermark.yaml
    源文件数没变则跳过重建 (.build_stamp 指纹)。
    train_length / val_length = 跨数据集总量上限 (0=全量), 冒烟用。
    """
    stamp = _merge_stamp(data_root, image_size, train_length, val_length)
    stamp_path = os.path.join(output_dir, '.build_stamp')
    yaml_path = os.path.join(output_dir, 'watermark.yaml')
    if not force and os.path.exists(stamp_path) and os.path.exists(yaml_path):
        with open(stamp_path) as f:
            if f.read().strip() == stamp:
                counts = {s: len(_list_images(os.path.join(output_dir, 'images', s)))
                          for s in ('train', 'val')}
                logger.info(f"Reuse merged dataset (stamp match): {output_dir} {counts}")
                return yaml_path, counts

    datasets = discover_datasets(data_root)
    if not datasets:
        raise RuntimeError(f"No datasets found in {data_root}")

    for split in ('train', 'val'):
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    counter = {'train': 0, 'val': 0}
    rng = np.random.RandomState(seed)
    limits = {'train': int(train_length or 0), 'val': int(val_length or 0)}

    for split in ('train', 'val'):
        # 先收集后抽样: train_length 是**跨数据集总量**, 不是每集配额
        items = []  # (ds_name, fname, images_dir, labels_dir)
        for info in datasets:
            images_dir = info.get(f'{split}_images')
            labels_dir = info.get(f'{split}_labels') or info.get(f'{split}_masks')
            if images_dir is None:
                continue
            for fname in _list_images(images_dir):
                items.append((info['name'], fname, images_dir, labels_dir))

        n_all = len(items)
        limit = limits[split]
        if limit > 0 and n_all > limit:
            pick = sorted(rng.choice(n_all, limit, replace=False).tolist())
            items = [items[i] for i in pick]
            logger.info(f"  [{split}] subsample {len(items)}/{n_all}")

        per_ds = {}  # ds -> [n, n_empty]
        for ds_name, fname, images_dir, labels_dir in items:
            stem = os.path.splitext(fname)[0]
            out_stem = f"{ds_name}_{stem}"

            # 保存完整 BGR 三通道 (channel 0 = B = 水印信号所在通道), 不转 Cb
            img = cv2.imread(os.path.join(images_dir, fname))
            if img is None:
                continue
            h, w = image_size
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            cv2.imwrite(os.path.join(output_dir, 'images', split, out_stem + '.png'), img)

            # 复制/转换标签
            lbl_src = None
            if labels_dir:
                for ext in ('.txt',):
                    path = os.path.join(labels_dir, stem + ext)
                    if os.path.exists(path):
                        lbl_src = path
                        break

            lbl_dst = os.path.join(output_dir, 'labels', split, out_stem + '.txt')
            rec = per_ds.setdefault(ds_name, [0, 0])
            rec[0] += 1
            if lbl_src and os.path.getsize(lbl_src) > 0:
                shutil.copy2(lbl_src, lbl_dst)
            else:
                # 无标签时写空文件 (负样本)
                with open(lbl_dst, 'w') as f:
                    pass
                rec[1] += 1

            counter[split] += 1

        for ds_name, (n, n_empty) in sorted(per_ds.items()):
            logger.info(f"  [{ds_name}/{split}] {n} images "
                        f"({n_empty} empty labels)")

    logger.info(f"Merged dataset: train={counter['train']}, val={counter['val']}")

    # 写 YOLO YAML
    with open(yaml_path, 'w') as f:
        f.write(f"""# Auto-generated YOLO dataset config
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: locator
""")
    with open(stamp_path, 'w') as f:
        f.write(stamp)
    logger.info(f"YOLO config: {yaml_path}")
    return yaml_path, counter


def main():
    parser = argparse.ArgumentParser(description='v1-vv2 YOLOv8 训练 (独立识别)')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default=None, help='GPU编号(覆盖config)')
    parser.add_argument('--force_rebuild', action='store_true', help='强制重建合并数据集')
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

    seed = int(cfg.get('seed', 42))
    set_seed(seed)

    stages = cfg.get('stages')
    if not stages:
        stages = [{
            'name': 'single',
            'data_root': cfg.get('data_root', '/data1/lpl/datasets_labeled/noisy'),
            'epochs': cfg.get('epochs', 100),
            'lr': cfg.get('lr', 0.01),
        }]
        logger.info(f"Single-stage mode (no `stages` in config): {stages[0]['name']}")

    image_size = (cfg.get('image_height', 1080), cfg.get('image_width', 1920))
    model_name = cfg.get('model', 'yolov8n.pt')
    # 裸文件名先按本目录解析: 从别的 cwd 跑时 YOLO('yolov8n.pt') 会静默去 GitHub 下载
    if model_name and not os.path.exists(model_name):
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
        if os.path.exists(local):
            model_name = local
    finetune_cfg = cfg.get('finetune', {})
    force = bool(args.force_rebuild or cfg.get('force_rebuild', False))

    logger.info(f"Stages: {[s['name'] for s in stages]}")
    stage_summary = []
    prev_best = None

    for si, stage in enumerate(stages):
        tag = stage.get('tag', stage['name'])
        epochs = int(stage.get('epochs', cfg.get('epochs', 100)))
        lr = float(stage.get('lr', cfg.get('lr', 0.01)))
        # 短阶段不要早停 (patience=0 在 ultralytics 里表示关闭早停)
        patience = int(stage.get('patience', cfg.get('patience', 30)))

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"STAGE [{stage['name']}]  epochs={epochs}  lr={lr}  patience={patience}")
        logger.info("=" * 60)

        # ── 构建/复用合并数据集 ──
        merged_dir = os.path.join(output_dir, f'merged_data_{tag}')
        yaml_path, counts = build_yolo_dataset(
            stage['data_root'], merged_dir, image_size,
            force=force and si == 0,  # 第二阶段的 force 由自己的 stamp 决定
            train_length=cfg.get('train_length', 0),
            val_length=cfg.get('val_length', 0),
            seed=seed + si * 10)

        # ── 初始化: 阶段1用 base/finetune 权重, 之后接上一阶段 best.pt ──
        init = prev_best or finetune_cfg.get('weight_path') or model_name
        if not (init and os.path.exists(init)):
            init = model_name
        logger.info(f"Loading model: {init}")
        model = YOLO(init)

        run_name = f'yolo_{tag}'
        # optimizer=auto 会自行挑 lr 并**忽略 lr0** —— 配置里的 lr 就白写了。
        # 显式指定优化器 (默认 SGD, 即 lr0=0.01 的 YOLO 惯例) 才吃 stages 里的 lr。
        optimizer = stage.get('optimizer', cfg.get('optimizer', 'SGD'))
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=cfg.get('imgsz', 640),
            batch=cfg.get('batch_size', 8),
            lr0=lr,
            optimizer=optimizer,
            device=device,
            project=output_dir,
            name=run_name,
            exist_ok=True,
            patience=patience,
            save=True,
            save_period=stage.get('save_every', cfg.get('save_every', 10)),
            seed=seed,
            verbose=True,
        )

        # ultralytics 会把相对 project 挂到 runs/detect/ 下, 自行拼 output_dir/run_name
        # 找不到 best.pt, 阶段间权重链就断了 —— 以 trainer 的实际 save_dir 为准
        save_dir = str(getattr(model.trainer, 'save_dir', None)
                       or os.path.join(output_dir, run_name))
        best_pt = os.path.join(save_dir, 'weights', 'best.pt')
        last_pt = os.path.join(save_dir, 'weights', 'last.pt')
        if os.path.exists(best_pt):
            prev_best = best_pt
        elif os.path.exists(last_pt):
            prev_best = last_pt

        stage_summary.append({
            'name': stage['name'], 'data_root': stage['data_root'],
            'epochs': epochs, 'lr': lr,
            'best_weights': best_pt if os.path.exists(best_pt) else None,
            'last_weights': last_pt if os.path.exists(last_pt) else None,
            'train_samples': counts.get('train', 0),
            'val_samples': counts.get('val', 0),
        })

    # ── 保存结果 ──
    summary = {
        'stages': stage_summary,
        'best_weights': prev_best,
        'total_epochs': sum(s['epochs'] for s in stage_summary),
        'config': cfg,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Training complete!")
    for s in stage_summary:
        logger.info(f"  [{s['name']}] {s['epochs']} ep  "
                    f"train={s['train_samples']} val={s['val_samples']}  "
                    f"best={s['best_weights']}")
    logger.info(f"Best model: {prev_best}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
