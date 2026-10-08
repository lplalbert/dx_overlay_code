"""
多数据集 → 拼贴画布 + 三种指定噪声各加一遍

与 prepare_multids.py 的区别:
  噪声不再是 pair(identity,wechat,tile_crop,pimog) 随机 2 选,
  而是**固定 3 种噪声各加一遍**, 每个 clean 样本出 3 个 noisy 变体:

    wechat        add_wechat_noise(image)                         单独微信压缩
    pimog         add_pimog_noise(image, mask, bboxes, rng)       单独模拟拍照
    pimog_wechat  add_pimog_noise(...) -> add_wechat_noise(...)   拍照后微信压缩

噪声来源:
    wechat  = wechat_worst_case_compressor.py  (真 JPEG q60 4:2:0)
    pimog   = physical_moire.py                (屏-摄摩尔纹/曝光/PSF/CFA/ISP)

输出布局:
    <output_root>/clean/<ds>/<split>/{images,masks,labels}/   1 份 (stem)
    <output_root>/noisy/<ds>/<split>/{images,masks,labels}/   3 份
        stem_wechat / stem_pimog / stem_pimog_wechat

用法:
    python prepare_multids_3noise.py \
        --dataset_counts coco_minator_dataset=2000,document_ds=2000,bcgd=0
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_dataset import (
    generate_one_sample,
    add_wechat_noise,
    add_pimog_noise,
    apply_noise_tier,
    NOISE_TIERS,
    build_canvas,
    pick_canvas_grid,
    get_locator_positions,
    SCREEN_W, SCREEN_H,
)
from generate_locator_pattern import FIX_FG_MATRIX
from multi_dataset import discover_datasets, _list_images

LOCATOR_PATTERN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'locator_pattern.npy')

# 三种噪声: (文件名后缀, 中文说明, 需要 rng)
NOISE_VARIANTS = [
    ('wechat',       '单独微信压缩',   False),
    ('pimog',        '单独模拟拍照',   True),
    ('pimog_wechat', '拍照后微信压缩', True),
    ('n1',           '档位1 轻',       True),
    ('n2',           '档位2 中',       True),
    ('n3',           '档位3 重',       True),
    ('n4',           '档位4 极重',     True),
]

NOISE_SOURCES = {
    'wechat':       'wechat_worst_case_compressor.py:preset=mainstream_worst',
    'pimog':        'physical_moire.py:preset=screen_capture',
    'pimog_wechat': 'physical_moire.py:screen_capture -> wechat_worst_case_compressor.py:mainstream_worst',
}
for _k, _t in NOISE_TIERS.items():
    NOISE_SOURCES[_k] = f'NOISE_TIERS[{_k}] {_t["desc"]}  (标定残余 {_t["residue"]}, 频段std {_t["band_std"]})'


def apply_noise_variant(image, mask, bboxes, kind, rng):
    """按 kind 调用 generate_dataset 里已实现的噪声函数。"""
    if kind in NOISE_TIERS:
        return apply_noise_tier(image, mask, bboxes, kind, rng)
    elif kind == 'wechat':
        return add_wechat_noise(image), mask, bboxes
    elif kind == 'pimog':
        return add_pimog_noise(image, mask, bboxes, rng=rng)
    elif kind == 'pimog_wechat':
        img, mask, bboxes = add_pimog_noise(image, mask, bboxes, rng=rng)
        return add_wechat_noise(img), mask, bboxes
    else:
        raise ValueError(kind)


def load_carrier(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def parse_dataset_counts(spec):
    counts = {}
    if not spec:
        return counts
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition('=')
        counts[name.strip()] = int(value)
    return counts


def resolve_split_counts(total, n_train_avail, n_val_avail, val_ratio=0.2):
    if total <= 0:
        return n_train_avail, n_val_avail
    n_val = min(int(round(total * val_ratio)), n_val_avail)
    n_train = min(total - n_val, n_train_avail)
    if n_train + n_val < total:
        leftover = total - n_train - n_val
        extra_train = min(leftover, max(0, n_train_avail - n_train))
        n_train += extra_train
        leftover -= extra_train
        n_val += min(leftover, max(0, n_val_avail - n_val))
    return n_train, n_val


def _save_sample(dirs, stem, image, mask, bboxes):
    cv2.imwrite(os.path.join(dirs['images'], stem + '.png'), image)
    if mask is not None:
        cv2.imwrite(os.path.join(dirs['masks'], stem + '.png'), mask)
    with open(os.path.join(dirs['labels'], stem + '.txt'), 'w') as f:
        for bbox in bboxes:
            f.write(f'{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} '
                    f'{bbox[3]:.6f} {bbox[4]:.6f}\n')


def _make_dirs(out_root, ds_name, split):
    base = os.path.join(out_root, ds_name, split)
    dirs = {
        'images': os.path.join(base, 'images'),
        'masks': os.path.join(base, 'masks'),
        'labels': os.path.join(base, 'labels'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _probe_sizes(images_dir, files, max_probe=800):
    from PIL import Image
    shapes = []
    step = max(1, len(files) // max_probe)
    for f in files[::step][:max_probe]:
        try:
            with Image.open(os.path.join(images_dir, f)) as im:
                w, h = im.size
            shapes.append((h, w))
        except Exception:
            pass
    return shapes or [(SCREEN_H, SCREEN_W)]


def _pick_sources(files, images_dir, n_need, rng, cell_h, cell_w, max_try=30):
    srcs, names = [], []
    pool = list(range(len(files)))
    for _ in range(min(n_need, len(pool))):
        img = name = None
        for _t in range(max_try):
            if not pool:
                break
            i = pool.pop(int(rng.randint(0, len(pool))))
            cand = load_carrier(os.path.join(images_dir, files[i]))
            if cand is None:
                continue
            img, name = cand, files[i]
            if cand.shape[0] >= cell_h and cand.shape[1] >= cell_w:
                break
        if img is None:
            continue
        srcs.append(img)
        names.append(name)
    return srcs, names


def generate_for_dataset(info, split, out_root, n_take, alpha, rng,
                         locator_pattern, channel_mode, log,
                         noise_variants=None):
    variants = NOISE_VARIANTS if noise_variants is None else noise_variants
    images_dir = info.get(f'{split}_images')
    if images_dir is None:
        return {}
    files = _list_images(images_dir)
    if not files:
        return {}

    ds_name = info['name']
    clean_dirs = _make_dirs(os.path.join(out_root, 'clean'), ds_name, split)
    noisy_dirs = _make_dirs(os.path.join(out_root, 'noisy'), ds_name, split)

    shapes = _probe_sizes(images_dir, files)
    cols, rows = pick_canvas_grid(shapes, SCREEN_W, SCREEN_H)
    n_need = cols * rows
    cell_w, cell_h = SCREEN_W // cols, SCREEN_H // rows
    cov = np.mean([(w >= cell_w and h >= cell_h) for h, w in shapes])
    log(f'  [{ds_name}/{split}] grid {cols}x{rows}  cell {cell_w}x{cell_h}  '
        f'1:1-fit {cov * 100:.0f}%  ({n_need} src/canvas)')

    n_samples = n_take if n_take > 0 else len(files)
    n_done = 0
    t0 = time.time()
    noise_hist = {}

    solo_order = rng.permutation(len(files)).tolist()
    solo_at = 0

    for i in range(n_samples):
        if n_need == 1:
            if solo_at >= len(solo_order):
                solo_order = rng.permutation(len(files)).tolist()
                solo_at = 0
            fi = solo_order[solo_at]
            solo_at += 1
            cand = load_carrier(os.path.join(images_dir, files[fi]))
            if cand is None:
                continue
            srcs, src_names = [cand], [files[fi]]
        else:
            srcs, src_names = _pick_sources(files, images_dir, n_need, rng, cell_h, cell_w)
        if not srcs:
            continue
        canvas = build_canvas(srcs, rng, SCREEN_W, SCREEN_H, grid=(cols, rows))
        wm_id = rng.randint(0, 16 ** 5 - 1)

        # ── 1) 干净水印图 ──
        clean_img, bboxes, mask, _ = generate_one_sample(
            wm_id, FIX_FG_MATRIX, locator_pattern, alpha, rng,
            carrier_img=canvas, apply_noise=False, channel_mode=channel_mode)

        if n_need == 1:
            stem = f'{ds_name}_{os.path.splitext(src_names[0])[0]}'
        else:
            stem = f'{ds_name}_t{i:06d}'

        _save_sample(clean_dirs, stem, clean_img, mask, bboxes)

        # ── 2) 噪声变体各加一遍 ──
        for noise_tag, noise_cn, need_rng in variants:
            # pimog 每次调用需要独立 rng 状态以获得不同的透视参数
            noise_rng = np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
            noisy_img, mask_n, bboxes_n = apply_noise_variant(
                clean_img.copy(), mask.copy(), [list(b) for b in bboxes],
                noise_tag, noise_rng)
            noisy_stem = f'{stem}_{noise_tag}'
            _save_sample(noisy_dirs, noisy_stem, noisy_img, mask_n, bboxes_n)
            noise_hist[noise_tag] = noise_hist.get(noise_tag, 0) + 1

        n_done += 1
        if (i + 1) % 20 == 0 or (i + 1) == n_samples:
            dt = time.time() - t0
            log(f'  [{ds_name}/{split}] {i + 1}/{n_samples}  '
                f'{dt / (i + 1):.2f}s/img  ({len(variants)} noisy variants each)')

    return {'n': n_done, 'noise_hist': noise_hist,
            'src_per_canvas': n_need, 'grid': [cols, rows],
            'cell': [cell_w, cell_h], 'fit_ratio': float(cov)}


def main():
    parser = argparse.ArgumentParser(
        description='多数据集载体 → 拼贴画布 → clean + 3-noise 变体')
    parser.add_argument('--data_root', default='/data1/lpl/datasets')
    parser.add_argument('--output_root', default='/data1/lpl/datasets_labeled_3noise')
    parser.add_argument('--dataset_counts', default='coco_minator_dataset=2000,document_ds=2000,bcgd=0')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.032)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--channel_mode', default='b', choices=['b', 'yw'])
    parser.add_argument('--wechat_preset', default='mainstream_worst')
    parser.add_argument('--noise_kinds', default='n1,n2,n3,n4',
                        help='逗号分隔的噪声变体; 可选 wechat,pimog,pimog_wechat,'
                             'n1,n2,n3,n4 (n1..n4 = 标定过的非几何噪声档位,'
                             '每个 clean 样本各出一份)')
    args = parser.parse_args()

    known = {t: (cn, rngf) for t, cn, rngf in NOISE_VARIANTS}
    variants = []
    for tag in [s.strip() for s in args.noise_kinds.split(',') if s.strip()]:
        if tag not in known:
            raise SystemExit(f'--noise_kinds: 未知变体 {tag!r}, 可选 {sorted(known)}')
        variants.append((tag, known[tag][0], known[tag][1]))

    log = print
    log('=' * 60)
    log('prepare_multids_3noise: tile canvas -> clean + noise variants')
    log('=' * 60)
    log(f'carriers : {args.data_root}')
    log(f'output   : {args.output_root}')
    log(f'alpha    : {args.alpha}  channel={args.channel_mode}')
    log(f'canvas   : tile (NO resize) -> {SCREEN_W}x{SCREEN_H}')
    log(f'noise    : {len(variants)} variants per clean sample:')
    for tag, cn, _ in variants:
        log(f'  {tag:14s} {cn:12s}  ← {NOISE_SOURCES[tag]}')
    log(f'wechat   : {args.wechat_preset}  (wechat_worst_case_compressor.py)')
    log(f'photo    : screen_capture  (physical_moire.py)')

    import generate_dataset as gd
    gd._get_wechat_compressor(args.wechat_preset)

    datasets = discover_datasets(args.data_root)
    if not datasets:
        raise RuntimeError(f'No datasets found in {args.data_root}')

    locator_pattern = (np.load(LOCATOR_PATTERN_PATH)
                       if os.path.exists(LOCATOR_PATTERN_PATH)
                       else FIX_FG_MATRIX[0].copy())

    want_counts = parse_dataset_counts(args.dataset_counts)
    log(f'counts   : {want_counts}  (val_ratio={args.val_ratio}, 0=ALL)')

    rng = np.random.RandomState(args.seed)
    summary = {}
    plan = {}
    for info in datasets:
        ds_name = info['name']
        n_train_avail = len(_list_images(info.get('train_images', '')))
        n_val_avail = len(_list_images(info.get('val_images', '')))

        if ds_name in want_counts:
            n_train, n_val = resolve_split_counts(
                want_counts[ds_name], n_train_avail, n_val_avail, args.val_ratio)
        elif want_counts:
            log(f'[{ds_name}] not in --dataset_counts, skipped')
            continue
        else:
            n_train, n_val = n_train_avail, n_val_avail

        plan[ds_name] = {'train': n_train, 'val': n_val,
                         'train_avail': n_train_avail, 'val_avail': n_val_avail}
        log(f'[{ds_name}] plan train={n_train}/{n_train_avail}  '
            f'val={n_val}/{n_val_avail}')

        counts = {}
        for split, n_take in (('train', n_train), ('val', n_val)):
            counts[split] = generate_for_dataset(
                info, split, args.output_root, n_take, args.alpha, rng,
                locator_pattern, args.channel_mode, log,
                noise_variants=variants)
        summary[ds_name] = counts
        log(f'[{ds_name}] generated '
            f'train={counts.get("train", {}).get("n", 0)} '
            f'val={counts.get("val", {}).get("n", 0)}')

    meta = {
        'source_root': args.data_root,
        'dataset_counts': want_counts,
        'val_ratio': args.val_ratio,
        'plan': plan,
        'alpha': args.alpha,
        'channel_mode': args.channel_mode,
        'canvas': {
            'size': [SCREEN_W, SCREEN_H],
            'method': 'tile_no_resize',
        },
        'template': 'diagonal_stripe_45deg_period4_width2',
        'noise': '3fixed(wechat,pimog,pimog_wechat)',
        'noise_sources': NOISE_SOURCES,
        'noise_variants': {tag: cn for tag, cn, _ in NOISE_VARIANTS},
        'wechat_preset': args.wechat_preset,
        'locator_positions': [list(p) for p in get_locator_positions()],
        'num_locator_blocks': len(get_locator_positions()),
        'screen_size': [SCREEN_W, SCREEN_H],
        'per_dataset_counts': summary,
    }
    os.makedirs(args.output_root, exist_ok=True)
    meta_path = os.path.join(args.output_root, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log('\nDone. Per-dataset counts:')
    total_clean = 0
    total_noisy = 0
    for name, counts in summary.items():
        tn = counts.get('train', {}).get('n', 0)
        vn = counts.get('val', {}).get('n', 0)
        total_clean += tn + vn
        for split in ('train', 'val'):
            for v in counts.get(split, {}).get('noise_hist', {}).values():
                total_noisy += v
        log(f'  {name:24s} clean train={tn:6d}  val={vn:6d}')
    ratio = (f'{total_noisy / total_clean:.1f}x' if total_clean else 'n/a')
    log(f'\nTotal: {total_clean} clean -> {total_noisy} noisy ({ratio})')
    log(f'clean -> {os.path.join(args.output_root, "clean")}')
    log(f'noisy -> {os.path.join(args.output_root, "noisy")}')


if __name__ == '__main__':
    main()
