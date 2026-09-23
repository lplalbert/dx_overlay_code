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
    扫描 root_dir 下所有子目录，返回 [{name, train_images, train_labels, val_images, val_labels}, ...]

    自动适配以下布局:
        root/ds/train/images/ + root/ds/train/masks|labels/
        root/ds/train/ (直接放图片和标注)
        root/ds/train_images/ + root/ds/train_masks|labels/
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
            images_dir, labels_dir = _find_split_dirs(ds_path, split)
            if images_dir is None:
                logger.warning(f"  [{entry}/{split}] no images dir found, skipping")
                continue
            info[f'{split}_images'] = images_dir
            info[f'{split}_labels'] = labels_dir

        if 'train_images' in info:
            datasets.append(info)
            n_train = len(_list_images(info['train_images']))
            n_val = len(_list_images(info.get('val_images', '')))
            logger.info(f"  [{entry}] train={n_train}, val={n_val}, "
                       f"labels={info.get('train_labels', 'N/A')}")
        else:
            logger.warning(f"  [{entry}] no train split found, skipping")

    return datasets


def _find_split_dirs(ds_path, split):
    """查找 split (train/val) 的 images 和 labels 目录。"""
    candidates = [
        # ds/train/images + ds/train/masks|labels
        (os.path.join(ds_path, split, 'images'),
         [os.path.join(ds_path, split, 'masks'),
          os.path.join(ds_path, split, 'labels')]),
        # ds/train (直接放图片) + ds/train_masks|labels
        (os.path.join(ds_path, split),
         [os.path.join(ds_path, f'{split}_masks'),
          os.path.join(ds_path, f'{split}_labels')]),
        # ds/train_images + ds/train_masks|labels
        (os.path.join(ds_path, f'{split}_images'),
         [os.path.join(ds_path, f'{split}_masks'),
          os.path.join(ds_path, f'{split}_labels')]),
        # ds/images + ds/masks (单 split)
        (os.path.join(ds_path, 'images'),
         [os.path.join(ds_path, 'masks'),
          os.path.join(ds_path, 'labels')]),
    ]

    for img_dir, lbl_dirs in candidates:
        if os.path.isdir(img_dir) and _list_images(img_dir):
            for lbl_dir in lbl_dirs:
                if os.path.isdir(lbl_dir):
                    return img_dir, lbl_dir
            return img_dir, None  # 有图片无标注 (推理用)

    return None, None


def _list_images(directory):
    if not directory or not os.path.isdir(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTS)
    ])


class LocatorSegmentationDataset(Dataset):
    """v1-vv1: Cb通道 → mask 分割数据集 (单个数据源)。"""

    def __init__(self, images_dir, masks_dir, image_size=(1080, 1920)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.files = _list_images(images_dir)
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

        # Cb通道
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cb = ycrcb[:, :, 2].astype(np.float32) / 255.0

        mask = (mask > 127).astype(np.float32)

        image_t = torch.from_numpy(cb).unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W)
        return image_t, mask_t

    def _load_mask(self, image_name):
        stem = os.path.splitext(image_name)[0]
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            for suffix in ('', '_mask', '_label'):
                path = os.path.join(self.masks_dir, stem + suffix + ext)
                if os.path.exists(path):
                    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None


class LocatorDetectionDataset(Dataset):
    """v1-vv2: Cb通道 → YOLO bbox 检测数据集 (单个数据源)。

    返回 (image_tensor, label_tensor) 其中 label_tensor 是 (N, 5):
        [class_id, cx, cy, w, h] 归一化坐标
    """

    def __init__(self, images_dir, labels_dir, image_size=(1080, 1920)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.files = _list_images(images_dir)
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

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cb = ycrcb[:, :, 2]
        cb_3ch = cv2.merge([cb, cb, cb])  # YOLO 需要 3 通道

        image_t = torch.from_numpy(cb_3ch.transpose(2, 0, 1)).float() / 255.0  # (3, H, W)
        label_t = torch.tensor(labels, dtype=torch.float32)  # (N, 5)
        return image_t, label_t

    def _load_labels(self, image_name):
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
        labels_dir = info.get(f'{split}_labels')
        if images_dir is None:
            continue

        if task == 'segmentation':
            ds = LocatorSegmentationDataset(images_dir, labels_dir, image_size)
        elif task == 'detection':
            ds = LocatorDetectionDataset(images_dir, labels_dir, image_size)
        else:
            raise ValueError(f"Unknown task: {task}")

        if len(ds) > 0:
            sub_datasets.append(ds)

    if not sub_datasets:
        raise RuntimeError(f"No valid {split} data found in {root_dir}")

    if len(sub_datasets) == 1:
        return sub_datasets[0]
    return ConcatDataset(sub_datasets)
