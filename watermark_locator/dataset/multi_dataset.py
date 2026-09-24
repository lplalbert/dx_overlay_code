"""
多数据集加载器 — 支持 /data1/lpl/datasets 下所有子目录的 train/val

目录结构约定:
    /data1/lpl/datasets/
        ├── dataset_a/
        │   ├── train/
        │   │   ├── images/   (或直接放图片)
        │   │   └── masks/    (vv1) 或 labels/ (vv2)
        │   └── val/
        │       ├── images/
        │       └── masks/ 或 labels/
        ├── dataset_b/
        │   ├── train/
        │   └── val/
        └── ...

也支持 images/ 和 masks|labels/ 在 train/ 下，或 train/ 下直接是图片+标注。
"""

import os
import glob
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

logger = logging.getLogger(__name__)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def discover_datasets(root_dir):
    """
    扫描 root_dir 下所有子目录，返回:
        [{name,
          train_images, train_masks, train_labels,
          val_images,   val_masks,   val_labels}, ...]

    masks/ 供 vv1 分割, labels/ 供 vv2 检测; 两者可同时存在, 也可都缺(纯载体)。

    自动适配以下布局:
        root/ds/train/images/ + root/ds/train/masks/ + root/ds/train/labels/
        root/ds/train/ (直接放图片) + root/ds/train_masks|labels/
        root/ds/train_images/ + root/ds/train_masks| + root/ds/train_labels/
        root/ds/images/ + root/ds/masks/ + root/ds/labels/ (单 split)
    """
    datasets = []
    if not os.path.isdir(root_dir):
        logger.warning(f"Dataset root not found: {root_dir}")
        return datasets

    for entry in sorted(os.listdir(root_dir)):
        ds_path = os.path.join(root_dir, entry)
        if not os.path.isdir(ds_path):
            continue

        info = {'name': entry}

        for split in ('train', 'val'):
            images_dir, masks_dir, labels_dir = _find_split_dirs(ds_path, split)
            if images_dir is None:
                logger.warning(f"  [{entry}/{split}] no images dir found, skipping")
                continue
            info[f'{split}_images'] = images_dir
            info[f'{split}_masks'] = masks_dir
            info[f'{split}_labels'] = labels_dir

        if 'train_images' in info:
            datasets.append(info)
            n_train = len(_list_images(info['train_images']))
            n_val = len(_list_images(info.get('val_images', '')))
            logger.info(f"  [{entry}] train={n_train}, val={n_val}, "
                       f"masks={info.get('train_masks') or 'N/A'}, "
                       f"labels={info.get('train_labels') or 'N/A'}")
        else:
            logger.warning(f"  [{entry}] no train split found, skipping")

    return datasets


def _find_split_dirs(ds_path, split):
    """查找 split (train/val) 的 images / masks / labels 目录。

    Returns:
        (images_dir, masks_dir, labels_dir) — 缺失项为 None。
    """
    candidates = [
        # ds/train/images + ds/train/masks + ds/train/labels
        (os.path.join(ds_path, split, 'images'),
         os.path.join(ds_path, split, 'masks'),
         os.path.join(ds_path, split, 'labels')),
        # ds/train (直接放图片) + ds/train_masks + ds/train_labels
        (os.path.join(ds_path, split),
         os.path.join(ds_path, f'{split}_masks'),
         os.path.join(ds_path, f'{split}_labels')),
        # ds/train_images + ds/train_masks + ds/train_labels
        (os.path.join(ds_path, f'{split}_images'),
         os.path.join(ds_path, f'{split}_masks'),
         os.path.join(ds_path, f'{split}_labels')),
        # ds/images + ds/masks + ds/labels (单 split)
        (os.path.join(ds_path, 'images'),
         os.path.join(ds_path, 'masks'),
         os.path.join(ds_path, 'labels')),
    ]

    for img_dir, msk_dir, lbl_dir in candidates:
        if os.path.isdir(img_dir) and _list_images(img_dir):
            return (img_dir,
                    msk_dir if os.path.isdir(msk_dir) else None,
                    lbl_dir if os.path.isdir(lbl_dir) else None)

    return None, None, None


def _list_images(directory):
    if not directory or not os.path.isdir(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTS)
    ])


class LocatorSegmentationDataset(Dataset):
    """v1-vv1: 3 通道 BGR → mask 分割数据集 (单个数据源)。

    输入是完整 BGR 三通道 — 水印模板的信号只写在 B 通道 (黄 B=0 / 白 B=255,
    G/R 恒 255), 所以 channel 0 就是水印所在通道。**不要**再抽 Cb。
    """

    def __init__(self, images_dir, masks_dir, image_size=(1080, 1920)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.files = _list_images(images_dir)
        if masks_dir is None and self.files:
            logger.warning(f"  SegmentationDataset: no masks dir for {images_dir} "
                           f"-> all targets will be EMPTY (pure negatives)")
        logger.info(f"  SegmentationDataset: {len(self.files)} images from {images_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.images_dir, name))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {os.path.join(self.images_dir, name)}")

        # 尝试多种 mask 文件名匹配
        mask = self._load_mask(name)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 统一尺寸
        h, w = self.image_size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3 通道 BGR (channel 0 = B = 水印信号所在通道)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.astype(np.float32) / 255.0

        mask = (mask > 127).astype(np.float32)

        image_t = torch.from_numpy(img.transpose(2, 0, 1))   # (3, H, W) BGR
        mask_t = torch.from_numpy(mask).unsqueeze(0)         # (1, H, W)
        return image_t, mask_t

    def _load_mask(self, image_name):
        if self.masks_dir is None:
            return None
        stem = os.path.splitext(image_name)[0]
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            for suffix in ('', '_mask', '_label'):
                path = os.path.join(self.masks_dir, stem + suffix + ext)
                if os.path.exists(path):
                    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None


class LocatorDetectionDataset(Dataset):
    """v1-vv2: 3 通道 BGR → YOLO bbox 检测数据集 (单个数据源)。

    输入是完整 BGR 三通道 (channel 0 = B = 水印信号所在通道), **不要**抽 Cb。
    返回 (image_tensor, label_tensor) 其中 label_tensor 是 (N, 5):
        [class_id, cx, cy, w, h] 归一化坐标
    """

    def __init__(self, images_dir, labels_dir, image_size=(1080, 1920)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.files = _list_images(images_dir)
        if labels_dir is None and self.files:
            logger.warning(f"  DetectionDataset: no labels dir for {images_dir} "
                           f"-> all targets will be placeholders (pure negatives)")
        logger.info(f"  DetectionDataset: {len(self.files)} images from {images_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.images_dir, name))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {os.path.join(self.images_dir, name)}")

        # 加载 YOLO labels
        labels = self._load_labels(name)

        h, w = self.image_size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        # 3 通道 BGR (channel 0 = B = 水印信号所在通道); 不要抽 Cb
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        image_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0  # (3, H, W) BGR
        label_t = torch.tensor(labels, dtype=torch.float32)  # (N, 5)
        return image_t, label_t

    def _load_labels(self, image_name):
        if self.labels_dir is None:
            return [[0, 0.5, 0.5, 0.0, 0.0]]  # placeholder (无标注)
        stem = os.path.splitext(image_name)[0]
        path = os.path.join(self.labels_dir, stem + '.txt')
        labels = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(float(parts[0]))
                        cx, cy, bw, bh = map(float, parts[1:5])
                        labels.append([cls, cx, cy, bw, bh])
        return labels if labels else [[0, 0.5, 0.5, 0.0, 0.0]]  # placeholder


def build_multi_dataset(root_dir, split='train', task='segmentation',
                        image_size=(1080, 1920)):
    """
    从 root_dir 下所有数据集构建 ConcatDataset。

    Args:
        root_dir: 数据集根目录 (e.g., /data1/lpl/datasets)
        split: 'train' 或 'val'
        task: 'segmentation' (vv1) 或 'detection' (vv2)
        image_size: (H, W)

    Returns:
        ConcatDataset 或单个 Dataset
    """
    datasets_info = discover_datasets(root_dir)
    if not datasets_info:
        raise RuntimeError(f"No valid datasets found in {root_dir}")

    sub_datasets = []
    for info in datasets_info:
        images_dir = info.get(f'{split}_images')
        if images_dir is None:
            continue

        if task == 'segmentation':
            # vv1: masks/ (灰度 PNG) 优先; 兼容旧字段 labels/ 指向 mask 的情况
            ann_dir = info.get(f'{split}_masks') or info.get(f'{split}_labels')
            ds = LocatorSegmentationDataset(images_dir, ann_dir, image_size)
        elif task == 'detection':
            # vv2: labels/ (YOLO txt) 优先
            ann_dir = info.get(f'{split}_labels')
            ds = LocatorDetectionDataset(images_dir, ann_dir, image_size)
        else:
            raise ValueError(f"Unknown task: {task}")

        if len(ds) > 0:
            sub_datasets.append(ds)

    if not sub_datasets:
        raise RuntimeError(f"No valid {split} data found in {root_dir}")

    if len(sub_datasets) == 1:
        return sub_datasets[0]
    return ConcatDataset(sub_datasets)
