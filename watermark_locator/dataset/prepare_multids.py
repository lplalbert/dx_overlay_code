"""
多数据集 → 拼贴画布 + 干净/加噪两套标注

流程 (对每个数据集分别处理, 不跨集混拼):
    1. 把源图**无缩放**拼成 1920x1080 画布
         - 源图 >= 1920x1080 (bcgd):     1:1 随机裁剪
         - 源图 <  1920x1080 (coco/document): 原生分辨率网格平铺后裁切
       (禁止 resize: 640x480 放大 3 倍会把纹理低通成糊的, 与真实截屏不符)
    2. 加 v1 水印 → **干净水印图**     → <output_root>/clean/
    3. 对同一张干净图施加 pair 噪声 → **带噪声图** → <output_root>/noisy/
       noisy 的 mask/bboxes 随几何噪声同步变换, 与 clean 分开存

输出布局 (multi_dataset.py 可直接发现):
    <output_root>/clean/<ds_name>/<split>/{images,masks,labels}/
    <output_root>/noisy/<ds_name>/<split>/{images,masks,labels}/

噪声 (pair, 4 选 2):
    wechat     = wechat_worst_case_compressor.py  真 JPEG q60 4:2:0
    pimog      = physical_moire.py                屏-摄拍照模拟
    tile_crop  = 3x3 平铺+旋转+裁剪 (几何, 同步标签)
    identity   = 无噪声

用法:
    # 每个数据集总样本数 (0 = 用尽全部载体)
    python prepare_multids.py --dataset_counts coco_minator_dataset=2000,document_ds=2000,bcgd=0

    # 只重做某一侧
    python prepare_multids.py --stage clean
    python prepare_multids.py --stage noisy
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
    apply_pair_noise,
    build_canvas,
    pick_canvas_grid,
    crop_tile_1x1,
    get_locator_positions,
    SCREEN_W, SCREEN_H,
)
from generate_locator_pattern import FIX_FG_MATRIX
from multi_dataset import discover_datasets, _list_images

LOCATOR_PATTERN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'locator_pattern.npy')

NOISE_SOURCES = {
    'wechat': 'wechat_worst_case_compressor.py:preset=mainstream_worst',
    'pimog': 'physical_moire.py:preset=screen_capture',
    'tile_crop': 'generate_dataset.add_tile_rotate_crop_noise',
    'identity': 'none',
}


def load_carrier(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


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
    """把总样本数拆到 train/val (默认 8:2)。total<=0 表示用尽全部载体。"""
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
    """只读文件头拿 (H, W), 用于选拼贴网格。"""
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
    """抽 n_need 张源图; 优先挑能 1:1 裁出 cell 的 (避免大面积边缘补齐)。

    单次调用内**不放回**抽取 (同一张画布里不重复平铺同一源图)。
    跨画布仍是有放回 —— 那边 stem 用 t{i:06d} 序号命名, 不会互相覆盖。
    """
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
                break  # 够大, 直接用
        if img is None:
            continue
        srcs.append(img)
        names.append(name)
    return srcs, names


def generate_for_dataset(info, split, out_root, n_take, alpha, rng,
                         locator_pattern, channel_mode, stage, log):
    images_dir = info.get(f'{split}_images')
    if images_dir is None:
        return {}

    files = _list_images(images_dir)
    if not files:
        return {}

    ds_name = info['name']
    do_clean = stage in ('both', 'clean')
    do_noisy = stage in ('both', 'noisy')
    clean_dirs = _make_dirs(os.path.join(out_root, 'clean'), ds_name, split) if do_clean else None
    noisy_dirs = _make_dirs(os.path.join(out_root, 'noisy'), ds_name, split) if do_noisy else None

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

    # n_need==1 时 stem 用源文件名命名 -> 必须让每个源图恰好被用一次。
    # 有放回随机抽会让约 1/e 的载体永远抽不中, 同名输出互相覆盖 (优惠券收集)。
    # 洗牌后顺序取, 取完一轮再洗, 就能吃满全部载体。
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
            carrier_img=canvas, apply_noise=False,
            channel_mode=channel_mode,
        )

        if n_need == 1:
            stem = f'{ds_name}_{os.path.splitext(src_names[0])[0]}'
        else:
            stem = f'{ds_name}_t{i:06d}'

        if do_clean:
            _save_sample(clean_dirs, stem, clean_img, mask, bboxes)

        # ── 2) 由干净图施加 pair 噪声 ──
        noise_types = ['identity', 'identity']
        if do_noisy:
            noisy_img, mask_n, bboxes_n, noise_types = apply_pair_noise(
                clean_img.copy(), mask.copy(), list(bboxes), rng)
            _save_sample(noisy_dirs, stem, noisy_img, mask_n, bboxes_n)
            key = '+'.join(noise_types)
            noise_hist[key] = noise_hist.get(key, 0) + 1

        n_done += 1
        if (i + 1) % 20 == 0 or (i + 1) == n_samples:
            dt = time.time() - t0
            log(f'  [{ds_name}/{split}] {i + 1}/{n_samples}  '
                f'{dt / (i + 1):.2f}s/img  noise={noise_types}')

    return {'n': n_done, 'noise_hist': noise_hist,
            'src_per_canvas': n_need, 'grid': [cols, rows],
            'cell': [cell_w, cell_h], 'fit_ratio': float(cov)}


def main():
    parser = argparse.ArgumentParser(
        description='多数据集载体 → 拼贴画布 → 干净/加噪两套标注')
    parser.add_argument('--data_root', default='/data1/lpl/datasets')
    parser.add_argument('--output_root', default='/data1/lpl/datasets_labeled')
    parser.add_argument('--dataset_counts', default='',
                        help='每个数据集总样本数, 如 '
                             'coco_minator_dataset=2000,document_ds=2000,bcgd=0 '
                             '(0=全部). 优先于 --per_split.')
    parser.add_argument('--per_split', type=int, default=200,
                        help='每 (数据集, split) 样本数, 0=全部. '
                             '仅在未给 --dataset_counts 时生效')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.032)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--channel_mode', default='b', choices=['b', 'yw'])
    parser.add_argument('--stage', default='both', choices=['both', 'clean', 'noisy'],
                        help='both=干净+加噪; clean=只出干净; noisy=只出加噪')
    parser.add_argument('--wechat_preset', default='mainstream_worst')
    args = parser.parse_args()

    log = print
    log('=' * 60)
    log('prepare_multids: tile canvas -> clean + noisy labeled set')
    log('=' * 60)
    log(f'carriers : {args.data_root}')
    log(f'output   : {args.output_root}')
    log(f'alpha    : {args.alpha}  channel={args.channel_mode}  stage={args.stage}')
    log(f'canvas   : tile (NO resize) -> {SCREEN_W}x{SCREEN_H}')
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
    if want_counts:
        log(f'counts   : {want_counts}  (val_ratio={args.val_ratio}, 0=ALL)')
    else:
        log(f'per_split: {args.per_split or "ALL"}')

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
        elif args.per_split > 0:
            n_train = min(args.per_split, n_train_avail)
            n_val = min(args.per_split, n_val_avail)
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
                locator_pattern, args.channel_mode, args.stage, log)
        summary[ds_name] = counts
        log(f'[{ds_name}] generated '
            f'train={counts.get("train", {}).get("n", 0)} '
            f'val={counts.get("val", {}).get("n", 0)}')

    meta = {
        'source_root': args.data_root,
        'dataset_counts': want_counts or None,
        'per_split': None if want_counts else args.per_split,
        'val_ratio': args.val_ratio,
        'plan': plan,
        'alpha': args.alpha,
        'channel_mode': args.channel_mode,
        'stage': args.stage,
        'canvas': {
            'size': [SCREEN_W, SCREEN_H],
            'method': 'tile_no_resize',
            'note': 'src>=canvas -> 1:1 random crop; src<canvas -> native-res grid tile then crop',
        },
        'template': 'diagonal_stripe_45deg_period4_width2',
        'noise': 'pair(identity,wechat,tile_crop,pimog)',
        'noise_sources': NOISE_SOURCES,
        'wechat_preset': args.wechat_preset,
        'locator_positions': [list(p) for p in get_locator_positions()],
        'num_locator_blocks': len(get_locator_positions()),
        'screen_size': [SCREEN_W, SCREEN_H],
        'per_dataset_counts': summary,
    }
    os.makedirs(args.output_root, exist_ok=True)
    meta_path = os.path.join(args.output_root, 'metadata.json')
    # 分批跑 (如只补 bcgd) 时合并而非覆盖, 否则后跑的会把先跑的溯源抹掉
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                prev = json.load(f)
            for k in ('plan', 'per_dataset_counts'):
                if isinstance(prev.get(k), dict):
                    merged = dict(prev[k])
                    merged.update(meta.get(k) or {})
                    meta[k] = merged
        except (json.JSONDecodeError, OSError):
            pass
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log('\nDone. Per-dataset counts:')
    for name, counts in summary.items():
        tn = counts.get('train', {}).get('n', 0)
        vn = counts.get('val', {}).get('n', 0)
        log(f'  {name:24s} train={tn:6d}  val={vn:6d}')
    if args.stage in ('both', 'noisy'):
        log('\nNoise histogram (noisy side):')
        for name, counts in summary.items():
            hist = {}
            for split in ('train', 'val'):
                for k, v in counts.get(split, {}).get('noise_hist', {}).items():
                    hist[k] = hist.get(k, 0) + v
            log(f'  {name:24s} {hist}')
    log(f'\nclean -> {os.path.join(args.output_root, "clean")}')
    log(f'noisy -> {os.path.join(args.output_root, "noisy")}')


if __name__ == '__main__':
    main()
