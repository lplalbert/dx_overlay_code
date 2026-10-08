#!/usr/bin/env python3
"""clean 树 → 3 噪声树（v2：96 码字检测，无 mask）。

与 v1 ``watermark_locator/dataset/prepare_noise_from_clean.py`` 同构，差别只有两点：

  1. v2 只有 ``images/`` ``labels/`` ``meta/``，没有 ``masks/``
  2. 标签是 v2 的 YOLO 行 ``cls cx cy bw h``；pimog 的残差形变会**重写**它

噪声（与 v1 逐字相同）::

  wechat        add_wechat_noise —— 真 JPEG q60 4:2:0，**非几何**，标签不变
  pimog         add_pimog_noise  —— 屏-摄全链路（摩尔纹/曝光/PSF/CFA/ISP），
                                    残差形变 0.5~5px 经 content-warp 同步标签
  pimog_wechat  pimog → wechat

顺序也照 v1：先 pimog（几何）后 wechat（非几何），因为"拍照后微信压缩"
物理上就是先拍屏再发图。

用法::

    # 单进程
    python prepare_noise_v2.py

    # 多卡分片（/data1 是 IO 瓶颈，别开太多）
    CUDA_VISIBLE_DEVICES=0 python prepare_noise_v2.py --shard 0/4 &
    CUDA_VISIBLE_DEVICES=1 python prepare_noise_v2.py --shard 1/4 &
    CUDA_VISIBLE_DEVICES=2 python prepare_noise_v2.py --shard 2/4 &
    CUDA_VISIBLE_DEVICES=3 python prepare_noise_v2.py --shard 3/4 &
    wait
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(V2_ROOT))
for p in (os.path.join(REPO, 'watermark_locator', 'dataset'), V2_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_dataset import add_pimog_noise, add_wechat_noise  # noqa: E402

# 与 v1 逐字相同：标签区分噪声种，盐保证跨进程可复现（不能用 hash()）
NOISE_VARIANTS = [
    ('wechat',       '单独微信压缩',   False),
    ('pimog',        '单独模拟拍照',   True),
    ('pimog_wechat', '拍照后微信压缩', True),
]
NOISE_SALT = {'wechat': 11, 'pimog': 23, 'pimog_wechat': 37}

DEFAULT_CLEAN_ROOT = '/data1/lpl/datasets_v2/clean'
DEFAULT_OUT_ROOT = '/data1/lpl/datasets_v2/noisy'
V2_ROOT_DIR = '/data1/lpl/datasets_v2'


# ───────────────────────── 小工具 ─────────────────────────

def load_rows(label_path):
    """YOLO 标签 ``cls cx cy bw bh``（归一化）→ ``[[cls,cx,cy,bw,bh], ...]``。"""
    rows = []
    if not os.path.exists(label_path):
        return rows
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                rows.append([int(p[0])] + [float(x) for x in p[1:5]])
    return rows


def save_sample(out_dir, split, stem, image, rows, meta):
    cv2.imwrite(os.path.join(out_dir, 'images', split, stem + '.png'), image)
    with open(os.path.join(out_dir, 'labels', split, stem + '.txt'), 'w') as f:
        for r in rows:
            f.write(f'{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n')
    with open(os.path.join(out_dir, 'meta', split, stem + '.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False)


def collect_clean_stems(clean_root):
    """``clean/images/{train,val}/*.png`` → ``[(split, stem), ...]``。"""
    items = []
    for split in ('train', 'val'):
        d = os.path.join(clean_root, 'images', split)
        if not os.path.isdir(d):
            continue
        for stem in sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))):
            items.append((split, stem))
    return items


def apply_noise_variant(image, rows, kind, rng):
    if kind == 'wechat':
        return add_wechat_noise(image), rows
    if kind == 'pimog':
        img, _, rows = add_pimog_noise(image, None, rows, rng=rng)
        return img, rows
    if kind == 'pimog_wechat':
        img, _, rows = add_pimog_noise(image, None, rows, rng=rng)
        return add_wechat_noise(img), rows
    raise ValueError(kind)


# ───────────────────────── 主流程 ─────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description='v2 clean → 3 噪声树')
    p.add_argument('--clean_root', default=DEFAULT_CLEAN_ROOT)
    p.add_argument('--out_root', default=DEFAULT_OUT_ROOT)
    p.add_argument('--shard', default='0/1', help='i/n 分片')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help='只跑 8 个样本，写到 /data1/tmp_vis/v2_noise_smoke')
    args = p.parse_args(argv)

    shard_i, shard_n = (int(x) for x in args.shard.split('/'))
    assert 0 <= shard_i < shard_n, args.shard

    out_root = '/data1/tmp_vis/v2_noise_smoke' if args.smoke else args.out_root
    print('=' * 70)
    print(f'prepare_noise_v2  shard {shard_i}/{shard_n}  '
          f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "default")}')
    print('=' * 70)
    print(f'clean_root : {args.clean_root}')
    print(f'out_root   : {out_root}   smoke={args.smoke}')
    for tag, cn, _ in NOISE_VARIANTS:
        print(f'  {tag:14s} {cn}')

    items = collect_clean_stems(args.clean_root)
    if not items:
        raise FileNotFoundError(f'no clean images under {args.clean_root}/images')
    items = items[shard_i::shard_n]
    if args.smoke:
        items = items[:8]
    print(f'this shard  : {len(items)} clean → {len(items) * 3} noisy')

    for split in ('train', 'val'):
        for sub in ('images', 'labels', 'meta'):
            os.makedirs(os.path.join(out_root, sub, split), exist_ok=True)

    # 预热：wechat preset 固定 mainstream_worst（此后 add_wechat_noise() 不带
    # preset，正好命中同一份缓存）；moire 懒加载，先碰一次免得计时里算它。
    import generate_dataset as gd
    gd._get_wechat_compressor('mainstream_worst')
    gd._get_moire_sim()

    t0 = time.time()
    n_skip = 0
    hist = {}
    nbox_before, nbox_after = [], []

    for idx, (split, stem) in enumerate(items):
        img = cv2.imread(os.path.join(args.clean_root, 'images', split, stem + '.png'))
        if img is None:
            print(f'  [skip] unreadable {stem}')
            continue
        rows = load_rows(os.path.join(args.clean_root, 'labels', split, stem + '.txt'))
        nbox_before.append(len(rows))

        # 把水印身份带进 noisy meta：下游要靠"解出来的 ID == watermark_id"
        # 来验几何（数框/查形状查不出偏移符号写反）。
        src_meta = {}
        smp = os.path.join(args.clean_root, 'meta', split, stem + '.json')
        if os.path.exists(smp):
            try:
                with open(smp) as f:
                    src_meta = json.load(f)
            except Exception as exc:                     # noqa: BLE001
                print(f'  [warn] bad meta {smp}: {exc}')

        for tag, cn, need_rng in NOISE_VARIANTS:
            noisy_stem = f'{stem}_{tag}'
            out_png = os.path.join(out_root, 'images', split, noisy_stem + '.png')
            if os.path.exists(out_png) and not args.overwrite:
                n_skip += 1
                continue

            # 与 v1 同一个种子公式（跨进程确定性）
            rng = np.random.RandomState(
                (args.seed * 1_000_003 + idx * 17 + NOISE_SALT[tag] * 1_003) & 0x7FFFFFFF)
            noisy_img, noisy_rows = apply_noise_variant(
                img.copy(), [list(r) for r in rows], tag, rng)
            nbox_after.append(len(noisy_rows))
            save_sample(out_root, split, noisy_stem, noisy_img, noisy_rows, {
                'stem': noisy_stem, 'source_stem': stem, 'split': split,
                'tree': 'noisy', 'noise': [tag], 'noise_cn': cn,
                'window': [img.shape[1], img.shape[0]],
                'n_boxes': len(noisy_rows), 'n_boxes_source': len(rows),
                'geometric': bool(need_rng),
                # 水印身份从 clean meta 透传，供解码验几何
                **{k: src_meta[k] for k in
                   ('watermark_id', 's', 'screen', 'interval_px',
                    'symbol_interval_px', 'alpha', 'stripe', 'sequence16', 'symbols')
                   if k in src_meta},
            })
            hist[tag] = hist.get(tag, 0) + 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(items):
            dt = time.time() - t0
            rate = dt / (idx + 1)
            print(f'  [{shard_i}/{shard_n}] {idx + 1}/{len(items)}  '
                  f'{rate:.2f}s/clean  eta {rate * (len(items) - idx - 1) / 60:.1f}min')

    dt = time.time() - t0
    print(f'\nshard {shard_i}/{shard_n} done in {dt / 60:.1f}min')
    print(f'  generated: {sum(hist.values())} noisy   skipped(existed): {n_skip}')
    print(f'  hist: {hist}')
    if nbox_before:
        print(f'  boxes: source mean {np.mean(nbox_before):.1f}  '
              f'noisy mean {np.mean(nbox_after) if nbox_after else float("nan"):.1f}'
              f'  (pimog 会丢掉形变后 <2px 的框)')

    # stage1 = clean + noisy。只由 shard 0 收尾时写，免得并行写坏；
    # 写在 out_root 的父目录，冒烟不会污染正式目录。
    if shard_i == 0:
        write_stage1(args.clean_root, out_root,
                     os.path.dirname(os.path.abspath(out_root)))
    return 0


def write_stage1(clean_root, noisy_root, dest_dir):
    """写 ``<dest_dir>/stage1.yaml``（clean + noisy 两棵树一起训）。

    ``dest_dir`` 取 out_root 的父目录：正式跑就是 /data1/lpl/datasets_v2，
    冒烟跑就是冒烟树的父目录 —— 免得冒烟把正式目录里的 yaml 改指临时树。
    """
    root = os.path.abspath(dest_dir)
    os.makedirs(root, exist_ok=True)

    def rel(p):
        p = os.path.abspath(p)
        return os.path.relpath(p, root) if p.startswith(root + os.sep) else p

    lines = [
        '# Auto-generated: stage-1 base training set = clean + noisy (3 变体).',
        '# 两个 image 目录一起给 ultralytics —— check_det_dataset 原生支持 list。',
        f'path: {root}',
        'train:',
        f'  - {rel(os.path.join(clean_root, "images", "train"))}',
        f'  - {rel(os.path.join(noisy_root, "images", "train"))}',
        'val:',
        f'  - {rel(os.path.join(clean_root, "images", "val"))}',
        f'  - {rel(os.path.join(noisy_root, "images", "val"))}',
        '',
        'names:',
        '  0: codeword',
        '',
    ]
    path = os.path.join(root, 'stage1.yaml')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nstage1 yaml: {path}')


if __name__ == '__main__':
    raise SystemExit(main())
