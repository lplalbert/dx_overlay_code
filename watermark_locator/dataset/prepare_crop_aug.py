#!/usr/bin/env python
"""在三噪声数据集上做随机裁剪增广, 打破定位块的位置先验。

背景
----
原数据集里 6 个定位块恒定落在 1920x1080 的固定网格坐标上, 模型很容易学成
"只看这 6 个相对位置"。评测时随机截取 25%-70% 面积, 定位块在窗口内的位置
变得任意, 全图 F1 0.997 -> 裁剪 F1 0.79, 根因就是这个位置先验。

本脚本只做**裁剪**, 不重新生成水印/噪声。对每张图随机取一个面积占比
25%-70% 的窗口, 同步裁 images / masks / labels, 写到另一个目录。

GT 随裁剪变换 + 残缺保留规则
----------------------------
原图 GT (绝对 xyxy)
    -> 平移 (-x0, -y0) 到裁剪图坐标系
    -> 与裁剪窗求交 (裁掉窗外部分)
完全落在窗内              -> 保留 (框取裁剪后的 xyxy)
部分落在窗内 (残缺定位块) -> **可见面积 >= KEEP_AREA_RATIO x 原面积
                              且可见宽 >= MIN_KEEP_SIDE 且可见高 >= MIN_KEEP_SIDE**
                              -> 保留裁剪后的框 (网络就该学会检出残缺的回字形)
                              否则 -> 抛弃 (mask 对应像素一并抹掉,
                                        两条任务的监督保持一致)
完全在窗外                -> 抛弃

保留下来的框按裁剪图自身的 cw/ch 重新归一化写成 YOLO `cls cx cy w h`。
mask 直接按窗口切片 (二值图, 等价于 INTER_NEAREST)。

输出布局与 `multi_dataset.build_multi_dataset` 完全兼容:
    OUT_ROOT/{ds}/{train,val}/{images,masks,labels}/
只需把 config 里的 `data_root` 换成本脚本的 OUT_ROOT 即可接入训练。

用法:
    python watermark_locator/dataset/prepare_crop_aug.py                    # noisy 三噪声树
    TREE=clean python watermark_locator/dataset/prepare_crop_aug.py         # clean 干净树
    SMOKE=1 TREE=clean python watermark_locator/dataset/prepare_crop_aug.py # 冒烟
"""
import json
import os
import sys

import cv2
import numpy as np

REPO = '/data1/lpl/dx_overlay_code'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'watermark_locator'))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator', 'dataset'))
from generate_dataset import SCREEN_W, SCREEN_H  # noqa: E402

TREE = os.environ.get('TREE', 'noisy')   # noisy (3 噪声) | clean (干净)
SRC_ROOT = f'/data1/lpl/datasets_labeled_3noise/{TREE}'
OUT_ROOT = f'/data1/lpl/datasets_labeled_3noise_cropped/{TREE}'

SEED = 20260926
AREA_LO, AREA_HI = 0.25, 0.70  # 裁剪窗面积占比范围
AR_RANGE = (0.5, 2.0)          # 裁剪窗宽高比范围, 避免退化成细条

# 残缺定位块保留阈值 (超过则保留裁剪框, 否则连 mask 一起抛弃)
KEEP_AREA_RATIO = 0.5          # 可见面积 / 原框面积
MIN_KEEP_SIDE = 40             # 可见宽、高各自下限 (px)

SMOKE = os.environ.get('SMOKE', '') == '1'
N_SMOKE_PER_SPLIT = 4


# ───────────────────────── 工具 ─────────────────────────

def label_file_to_boxes(path, w=SCREEN_W, h=SCREEN_H):
    """YOLO 标签 `cls cx cy w h` (归一化) -> 绝对 xyxy (float)。"""
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            _cls, cx, cy, bw, bh = (float(x) for x in p[:5])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def sample_crop_window(ratio_lo, ratio_hi, rng):
    """按面积占比随机取裁剪窗 (x0, y0, cw, ch, real_ratio)。

    宽高比在可行范围 [r*W/H, (W/H)/r] 与 AR_RANGE 之交集内采样,
    保证面积精确落在目标占比上、且不超出画布。
    """
    H, W = SCREEN_H, SCREEN_W
    r = rng.uniform(ratio_lo, ratio_hi)
    ar_lo = max(AR_RANGE[0], r * W / H)    # 保 ch <= H
    ar_hi = min(AR_RANGE[1], (W / H) / r)  # 保 cw <= W
    if ar_lo >= ar_hi:                     # 退化: 取可行中点
        ar = 0.5 * (max(AR_RANGE[0], r * W / H) + min(AR_RANGE[1], (W / H) / r))
    else:
        ar = rng.uniform(ar_lo, ar_hi)
    area = r * W * H
    cw = int(round(np.sqrt(area * ar)))
    ch = int(round(cw / ar))
    cw = max(1, min(cw, W))
    ch = max(1, min(ch, H))
    x0 = int(rng.randint(0, W - cw + 1))
    y0 = int(rng.randint(0, H - ch + 1))
    return x0, y0, cw, ch, (cw * ch) / float(W * H)


def transform_gt_keep(gt_abs, x0, y0, cw, ch):
    """GT 随裁剪变换 -> (kept_boxes, dropped_boxes)。

    kept_boxes   : 裁剪图坐标系下的 xyxy (float), 已按保留规则过滤
    dropped_boxes: 被抛弃的**可见部分** xyxy (裁剪图坐标系), 用于抹 mask
    """
    kept, dropped = [], []
    for (x1, y1, x2, y2) in gt_abs:
        orig = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if orig <= 0:
            continue
        ix1, iy1 = max(x1, x0), max(y1, y0)
        ix2, iy2 = min(x2, x0 + cw), min(y2, y0 + ch)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        if iw <= 0 or ih <= 0:
            continue                                    # 完全在窗外
        box = (ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0)
        vis_ratio = (iw * ih) / orig
        if vis_ratio >= KEEP_AREA_RATIO and iw >= MIN_KEEP_SIDE and ih >= MIN_KEEP_SIDE:
            kept.append(box)
        else:
            dropped.append(box)                         # 残缺过小 -> 抛弃
    return kept, dropped


def boxes_to_yolo(boxes, cw, ch, cls=0):
    """裁剪图坐标系 xyxy -> YOLO `cls cx cy w h` (按 cw/ch 归一化) 行。"""
    lines = []
    for (x1, y1, x2, y2) in boxes:
        cx = (x1 + x2) / 2.0 / cw
        cy = (y1 + y2) / 2.0 / ch
        bw = (x2 - x1) / float(cw)
        bh = (y2 - y1) / float(ch)
        lines.append(f'{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    return lines


def erase_boxes_in_mask(mask, boxes):
    """把被抛弃的定位块像素从 mask 里抹掉, 保持两条任务监督一致。"""
    h, w = mask.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        a, b = int(np.floor(x1)), int(np.floor(y1))
        c, d = int(np.ceil(x2)), int(np.ceil(y2))
        a, b = max(0, a), max(0, b)
        c, d = min(w, c), min(h, d)
        if c > a and d > b:
            mask[b:d, a:c] = 0
    return mask


# ───────────────────────── 主流程 ─────────────────────────

def collect_items():
    items = []
    for ds in sorted(os.listdir(SRC_ROOT)):
        for split in ('train', 'val'):
            img_dir = os.path.join(SRC_ROOT, ds, split, 'images')
            if not os.path.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith('.png'):
                    continue
                stem = os.path.splitext(fname)[0]
                items.append({
                    'ds': ds, 'split': split, 'stem': stem,
                    'img': os.path.join(img_dir, fname),
                    'mask': os.path.join(SRC_ROOT, ds, split, 'masks', stem + '.png'),
                    'lab': os.path.join(SRC_ROOT, ds, split, 'labels', stem + '.txt'),
                })
    return items


def new_stat():
    return {'n_img': 0, 'n_box_in': 0, 'n_kept': 0,
            'n_drop_small': 0, 'n_drop_outside': 0,
            'n_img_zero_kept': 0, 'area_sum': 0.0}


def main():
    items = collect_items()
    if SMOKE:
        # 每个 (ds, split) 只取前 N 张
        seen, picked = {}, []
        for it in items:
            k = (it['ds'], it['split'])
            if seen.get(k, 0) >= N_SMOKE_PER_SPLIT:
                continue
            seen[k] = seen.get(k, 0) + 1
            picked.append(it)
        items = picked
        # 冒烟输出写到临时目录, 不污染正式输出
        out_root = f'/data1/tmp_vis/crop_aug_smoke_{TREE}'
    else:
        out_root = OUT_ROOT

    print(f'SRC      : {SRC_ROOT}')
    print(f'OUT      : {out_root}')
    print(f'items    : {len(items)}')
    print(f'area     : [{AREA_LO}, {AREA_HI}]   ar={AR_RANGE}')
    print(f'keep rule: vis_area/orig >= {KEEP_AREA_RATIO} 且 vis_w,h >= {MIN_KEEP_SIDE}px')
    print(f'seed     : {SEED}   smoke={SMOKE}')

    rng = np.random.RandomState(SEED)
    stats = {}                      # (ds, split) -> dict
    total = new_stat()

    for i, it in enumerate(items, 1):
        key = (it['ds'], it['split'])
        st = stats.setdefault(key, new_stat())

        img = cv2.imread(it['img'])
        if img is None:
            print(f'  [skip] unreadable image {it["img"]}')
            continue
        mask = cv2.imread(it['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        gt_abs = label_file_to_boxes(it['lab'])
        x0, y0, cw, ch, real_r = sample_crop_window(AREA_LO, AREA_HI, rng)

        crop = img[y0:y0 + ch, x0:x0 + cw].copy()
        mcrop = mask[y0:y0 + ch, x0:x0 + cw].copy()
        kept, dropped = transform_gt_keep(gt_abs, x0, y0, cw, ch)
        if dropped:
            mcrop = erase_boxes_in_mask(mcrop, dropped)

        # 统计
        st['n_img'] += 1
        st['n_box_in'] += len(gt_abs)
        st['area_sum'] += real_r
        st['n_kept'] += len(kept)
        st['n_drop_small'] += len(dropped)
        st['n_drop_outside'] += len(gt_abs) - len(kept) - len(dropped)
        if not kept:
            st['n_img_zero_kept'] += 1

        # 写盘
        sub = os.path.join(out_root, it['ds'], it['split'])
        cv2.imwrite(os.path.join(sub, 'images', it['stem'] + '.png'), crop)
        cv2.imwrite(os.path.join(sub, 'masks', it['stem'] + '.png'), mcrop)
        lines = boxes_to_yolo(kept, cw, ch)
        with open(os.path.join(sub, 'labels', it['stem'] + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))

        if i % 500 == 0 or i == len(items):
            print(f'  {i}/{len(items)}  last=({it["stem"]}) area={real_r:.3f} '
                  f'kept={len(kept)} drop={len(dropped)}')

    summarize(stats, out_root, items)


def summarize(stats, out_root, items):
    print('\n' + '=' * 88)
    print(f'{"ds/split":<34s} {"图":>6s} {"原框":>7s} {"保留":>7s} {"残缺抛弃":>8s} '
          f'{"窗外":>6s} {"零保留图":>8s} {"均面积":>7s}')
    print('-' * 88)
    tot = {'n_img': 0, 'n_box_in': 0, 'kept': 0, 'drop': 0,
           'out': 0, 'zero': 0, 'area': 0.0}
    detail = {}
    for (ds, split) in sorted(stats):
        st = stats[(ds, split)]
        mean_a = st['area_sum'] / st['n_img'] if st['n_img'] else 0.0
        print(f'{ds + "/" + split:<34s} {st["n_img"]:>6d} {st["n_box_in"]:>7d} '
              f'{st["n_kept"]:>7d} {st["n_drop_small"]:>8d} {st["n_drop_outside"]:>6d} '
              f'{st["n_img_zero_kept"]:>8d} {mean_a:>7.3f}')
        detail[f'{ds}/{split}'] = {
            'n_img': st['n_img'], 'n_box_in': st['n_box_in'],
            'n_kept': st['n_kept'], 'n_drop_small': st['n_drop_small'],
            'n_drop_outside': st['n_drop_outside'],
            'n_img_zero_kept': st['n_img_zero_kept'],
            'mean_area_ratio': round(mean_a, 4),
        }
        tot['n_img'] += st['n_img']
        tot['n_box_in'] += st['n_box_in']
        tot['kept'] += st['n_kept']
        tot['drop'] += st['n_drop_small']
        tot['out'] += st['n_drop_outside']
        tot['zero'] += st['n_img_zero_kept']
        tot['area'] += st['area_sum']
    mean_a = tot['area'] / tot['n_img'] if tot['n_img'] else 0.0
    print('-' * 88)
    print(f'{"合计":<34s} {tot["n_img"]:>6d} {tot["n_box_in"]:>7d} {tot["kept"]:>7d} '
          f'{tot["drop"]:>8d} {tot["out"]:>6d} {tot["zero"]:>8d} {mean_a:>7.3f}')
    print('=' * 88)

    meta = {
        'src_root': SRC_ROOT, 'out_root': out_root, 'tree': TREE,
        'seed': SEED,
        'area_range': [AREA_LO, AREA_HI], 'ar_range': list(AR_RANGE),
        'keep_area_ratio': KEEP_AREA_RATIO, 'min_keep_side': MIN_KEEP_SIDE,
        'gt_policy': ('平移+裁剪; 可见面积>=keep_area_ratio 且可见宽高>=min_keep_side '
                      '-> 保留裁剪框, 否则连 mask 像素一并抛弃; 窗外 -> 抛弃'),
        'smoke': SMOKE,
        'totals': {'n_img': tot['n_img'], 'n_box_in': tot['n_box_in'],
                   'n_kept': tot['kept'], 'n_drop_small': tot['drop'],
                   'n_drop_outside': tot['out'], 'n_img_zero_kept': tot['zero'],
                   'mean_area_ratio': round(mean_a, 4)},
        'per_split': detail,
    }
    meta_path = os.path.join(out_root, 'crop_aug_meta.json')
    os.makedirs(out_root, exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'\nmeta: {meta_path}')


if __name__ == '__main__':
    out_root = (f'/data1/tmp_vis/crop_aug_smoke_{TREE}' if SMOKE else OUT_ROOT)
    # 输出子目录先建好
    for ds in sorted(os.listdir(SRC_ROOT)):
        for split in ('train', 'val'):
            for sub in ('images', 'masks', 'labels'):
                os.makedirs(os.path.join(out_root, ds, split, sub), exist_ok=True)
    main()
