"""
复用已生成的 clean 树 → 只加 3 种噪声 (wechat / pimog / pimog_wechat)。

与 prepare_multids_3noise.py 的区别:
  - 不重新生成 clean 水印图 (直接从 --clean_root 读取)
  - 支持 --shard i/n 多进程分片, 可跨 GPU 并行

噪声来源 (与之前一致):
  wechat        wechat_worst_case_compressor.py:preset=mainstream_worst
  pimog         physical_moire.py (EfficientScreenMoireNoise, EXTREME profile)
  pimog_wechat  pimog -> wechat

用法 (单卡):
  python prepare_noise_from_clean.py

用法 (多卡并行, 例如 6 个分片, GPU 0 跑 4 个 + GPU 2 跑 2 个):
  CUDA_VISIBLE_DEVICES=0 python prepare_noise_from_clean.py --shard 0/6 &
  CUDA_VISIBLE_DEVICES=0 python prepare_noise_from_clean.py --shard 1/6 &
  CUDA_VISIBLE_DEVICES=0 python prepare_noise_from_clean.py --shard 2/6 &
  CUDA_VISIBLE_DEVICES=0 python prepare_noise_from_clean.py --shard 3/6 &
  CUDA_VISIBLE_DEVICES=2 python prepare_noise_from_clean.py --shard 4/6 &
  CUDA_VISIBLE_DEVICES=2 python prepare_noise_from_clean.py --shard 5/6 &
  wait
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_dataset import (
    add_wechat_noise,
    add_pimog_noise,
)

NOISE_VARIANTS = [
    ('wechat',       '单独微信压缩',   False),
    ('pimog',        '单独模拟拍照',   True),
    ('pimog_wechat', '拍照后微信压缩', True),
]

# 固定种子盐, 不用 hash() (跨进程不确定)
NOISE_SALT = {'wechat': 11, 'pimog': 23, 'pimog_wechat': 37}


def apply_noise_variant(image, mask, bboxes, kind, rng):
    if kind == 'wechat':
        return add_wechat_noise(image), mask, bboxes
    elif kind == 'pimog':
        return add_pimog_noise(image, mask, bboxes, rng=rng)
    elif kind == 'pimog_wechat':
        img, mask, bboxes = add_pimog_noise(image, mask, bboxes, rng=rng)
        return add_wechat_noise(img), mask, bboxes
    else:
        raise ValueError(kind)


def load_bboxes(label_path):
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                bboxes.append([int(parts[0])] + [float(x) for x in parts[1:5]])
    return bboxes


def save_sample(out_dirs, stem, image, mask, bboxes):
    cv2.imwrite(os.path.join(out_dirs['images'], stem + '.png'), image)
    if mask is not None:
        cv2.imwrite(os.path.join(out_dirs['masks'], stem + '.png'), mask)
    with open(os.path.join(out_dirs['labels'], stem + '.txt'), 'w') as f:
        for bbox in bboxes:
            f.write(f'{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} '
                    f'{bbox[3]:.6f} {bbox[4]:.6f}\n')


def make_dirs(out_root, ds_name, split):
    base = os.path.join(out_root, ds_name, split)
    dirs = {
        'images': os.path.join(base, 'images'),
        'masks': os.path.join(base, 'masks'),
        'labels': os.path.join(base, 'labels'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def collect_clean_stems(clean_root):
    """遍历 clean 树, 返回 [(ds_name, split, stem), ...]"""
    items = []
    if not os.path.isdir(clean_root):
        return items
    for ds_name in sorted(os.listdir(clean_root)):
        ds_dir = os.path.join(clean_root, ds_name)
        if not os.path.isdir(ds_dir):
            continue
        for split in ('train', 'val'):
            img_dir = os.path.join(ds_dir, split, 'images')
            if not os.path.isdir(img_dir):
                continue
            stems = sorted(
                os.path.splitext(f)[0] for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            )
            for stem in stems:
                items.append((ds_name, split, stem))
    return items


def main():
    parser = argparse.ArgumentParser(
        description='复用 clean 树, 只加 3 种噪声 (支持多卡分片)')
    parser.add_argument('--clean_root', default='/data1/lpl/datasets_labeled_3noise/clean')
    parser.add_argument('--out_root', default='/data1/lpl/datasets_labeled_3noise/noisy')
    parser.add_argument('--shard', default='0/1',
                        help='i/n 形式, 第 i 个分片共 n 个')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true',
                        help='已存在的 noisy 文件也重算')
    args = parser.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split('/'))
    assert 0 <= shard_i < shard_n

    device = os.environ.get('CUDA_VISIBLE_DEVICES', 'default')
    log = print
    log('=' * 60)
    log(f'prepare_noise_from_clean  shard {shard_i}/{shard_n}  '
        f'CUDA_VISIBLE_DEVICES={device}')
    log('=' * 60)
    log(f'clean_root : {args.clean_root}')
    log(f'out_root   : {args.out_root}')
    for tag, cn, _ in NOISE_VARIANTS:
        log(f'  {tag:14s} {cn}')

    items = collect_clean_stems(args.clean_root)
    if not items:
        log(f'No clean samples found under {args.clean_root}')
        return

    # 分片: sample i, i+n, i+2n, ...
    items = items[shard_i::shard_n]
    log(f'total clean = ?  this shard = {len(items)}')

    # 预热压缩器 / moire
    import generate_dataset as gd
    gd._get_wechat_compressor('mainstream_worst')
    gd._get_moire_sim()

    t0 = time.time()
    n_done = 0
    n_skip = 0
    noise_hist = {}

    for idx, (ds_name, split, stem) in enumerate(items):
        clean_img_dir = os.path.join(args.clean_root, ds_name, split, 'images')
        clean_msk_dir = os.path.join(args.clean_root, ds_name, split, 'masks')
        clean_lbl_dir = os.path.join(args.clean_root, ds_name, split, 'labels')

        img_path = os.path.join(clean_img_dir, stem + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(clean_img_dir, stem + '.jpg')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        mask_path = os.path.join(clean_msk_dir, stem + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], np.uint8)

        bboxes = load_bboxes(os.path.join(clean_lbl_dir, stem + '.txt'))

        out_dirs = make_dirs(args.out_root, ds_name, split)

        for noise_tag, noise_cn, need_rng in NOISE_VARIANTS:
            noisy_stem = f'{stem}_{noise_tag}'
            out_path = os.path.join(out_dirs['images'], noisy_stem + '.png')
            if os.path.exists(out_path) and not args.overwrite:
                n_skip += 1
                continue

            noise_rng = np.random.RandomState(
                (args.seed * 1_000_003 + idx * 17 + NOISE_SALT[noise_tag] * 1_003) & 0x7FFFFFFF)
            noisy_img, mask_n, bboxes_n = apply_noise_variant(
                image.copy(), mask.copy(), [list(b) for b in bboxes],
                noise_tag, noise_rng)
            save_sample(out_dirs, noisy_stem, noisy_img, mask_n, bboxes_n)
            noise_hist[noise_tag] = noise_hist.get(noise_tag, 0) + 1

        n_done += 1
        if (idx + 1) % 20 == 0 or (idx + 1) == len(items):
            dt = time.time() - t0
            rate = dt / (idx + 1)
            log(f'  [shard {shard_i}/{shard_n}] {idx + 1}/{len(items)}  '
                f'{rate:.2f}s/clean  (3 noisy each)  '
                f'eta {rate * (len(items) - idx - 1) / 60:.1f}min')

    dt = time.time() - t0
    log(f'\nshard {shard_i}/{shard_n} done in {dt / 60:.1f}min')
    log(f'  generated: {n_done} clean -> {sum(noise_hist.values())} noisy')
    log(f'  skipped (existed): {n_skip}')
    log(f'  noise_hist: {noise_hist}')


if __name__ == '__main__':
    main()
