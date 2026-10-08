#!/usr/bin/env python3
"""复用 v2 clean 树 → 只加 3 种噪声 (wechat / pimog / pimog_wechat)。

这是三数据集流水线的**第二步**::

    ① generate_dataset_v2.py --no_noise   →  clean    (无噪声)
    ② prepare_noise_from_clean_v2.py      →  noisy    (三噪声)      ← 本脚本
    ③ prepare_crop_aug_v2.py              →  cropped  (不同尺寸裁剪)

与 ``generate_dataset_v2.py`` 的 ``apply_pair_noise`` 路线不同：那里是从
[identity, wechat, tile_crop, pimog] 随机挑两种组合，标签是**同一张 clean 图**
重新渲染出来的，噪声之间没有同 stem 配对。这里直接读 clean 树，对**同一张图**
加三种确定的噪声，于是 ``train_000123`` / ``train_000123_wechat`` /
``train_000123_pimog`` / ``train_000123_pimog_wechat`` 是同一载体 + 同一水印 +
同一取景的四胞胎，噪声是唯一变量。

噪声来源 (与 v1 ``prepare_noise_from_clean.py`` 逐字一致)
--------------------------------------------------------
============= ==================== ==========
变体          含义                 几何
============= ==================== ==========
wechat        单独微信压缩          非几何
pimog         单独模拟拍照           形变
pimog_wechat  拍照后微信压缩         形变
============= ==================== ==========

``wechat`` 是真 JPEG q60 4:2:0，不改几何，标签**原样照抄**。
``pimog`` 有残差形变，标签经 ``_sync_labels_to_warp`` 跟着网格采样走，
过小的框 (``x_max-x_min <= 2``) 会被丢掉。

输出
----
::

    output_dir/
        images/{train,val}/*_{wechat,pimog,pimog_wechat}.png
        labels/{train,val}/*.txt
        meta/{train,val}/*.json      继承 clean 的 meta，追加 noise / derived_from
        watermark.yaml
        manifest.json

用法::

    # 冒烟
    python prepare_noise_from_clean_v2.py \
        --clean_root /tmp/v2_ds/clean --output_dir /tmp/v2_ds/noisy --smoke

    # 单卡
    python prepare_noise_from_clean_v2.py \
        --clean_root /data1/lpl/datasets_v2/clean \
        --output_dir /data1/lpl/datasets_v2/noisy

    # 多卡并行 (pimog 是 GPU 密集, 6 个分片)
    for i in 0 1 2 3 4 5; do
      CUDA_VISIBLE_DEVICES=$((i % 2)) python prepare_noise_from_clean_v2.py \
          --clean_root ... --output_dir ... --shard $i/6 &
    done; wait
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import generate_dataset_v2 as G  # noqa: E402  (顺带校验模板是 v2 那一份)
import predict_interval as PI  # noqa: E402  (解码门槛)
from generate_dataset import add_pimog_noise, add_wechat_noise  # noqa: E402

# 与 v1 prepare_noise_from_clean.py 完全一致：变体名 / 说明 / 是否几何
NOISE_VARIANTS: Tuple[Tuple[str, str, bool], ...] = (
    ('wechat', '单独微信压缩', False),
    ('pimog', '单独模拟拍照', True),
    ('pimog_wechat', '拍照后微信压缩', True),
)
NOISE_SALT = {'wechat': 11, 'pimog': 23, 'pimog_wechat': 37}

SPLITS = ('train', 'val')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ───────────────────────── 工具 ─────────────────────────

def collect_clean_stems(clean_root: str) -> List[Tuple[str, str]]:
    """遍历 clean 树，返回 ``[(split, stem), ...]``。

    v2 clean 树是扁平布局 ``clean/{images,labels,meta}/{split}/``，
    没有 v1 的 ``<ds_name>/<split>/`` 那一层。
    """
    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f'clean_root not found: {clean_root}')
    out: List[Tuple[str, str]] = []
    for split in SPLITS:
        img_dir = os.path.join(clean_root, 'images', split)
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith(IMG_EXTS):
                out.append((split, os.path.splitext(fn)[0]))
    return out


def load_bboxes(label_path: str) -> List[List[float]]:
    """YOLO ``cls cx cy w h`` → ``[[cls, cx, cy, w, h], ...]``。"""
    rows: List[List[float]] = []
    if not os.path.isfile(label_path):
        return rows
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                rows.append([int(float(p[0]))] + [float(x) for x in p[1:5]])
    return rows


def save_bboxes(label_path: str, bboxes: Sequence[Sequence[float]]) -> None:
    with open(label_path, 'w') as f:
        for b in bboxes:
            f.write(f'{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n')


def apply_noise_variant(image: np.ndarray,
                        bboxes: Optional[List[List[float]]],
                        kind: str,
                        rng: np.random.RandomState):
    """对同一张图加一种噪声。返回 ``(image, bboxes)``。"""
    if kind == 'wechat':
        return add_wechat_noise(image), bboxes
    if kind == 'pimog':
        out, _mask, boxes = add_pimog_noise(image, None, bboxes, rng=rng)
        return out, boxes
    if kind == 'pimog_wechat':
        out, _mask, boxes = add_pimog_noise(image, None, bboxes, rng=rng)
        return add_wechat_noise(out), boxes
    raise ValueError(f'unknown noise variant: {kind}')


# ───────────────────────── 自检 ─────────────────────────

def selftest(clean_root: str, carrier_paths: Sequence[str]) -> int:
    """加噪前后几何必须对得上 —— 用**解码**兜底，不看框数。

    标签几何写反时图像一个像素都不变，"框数量对/形状对"自检照样全绿。
    所以这里拿加噪前后的标签框当"完美检测"，跑一遍格点拟合 + 解码，
    要求解出的 ``watermark_id`` 等于 meta 里的真值。
    """
    print('=== prepare_noise_from_clean_v2 selftest ===')
    tally = {}
    verdicts = {}
    ok = True

    stems = collect_clean_stems(clean_root) if os.path.isdir(clean_root) else []
    tmp_dir = None
    if stems:
        split, stem = stems[0]
        img_path = os.path.join(clean_root, 'images', split, stem + '.png')
        lab_path = os.path.join(clean_root, 'labels', split, stem + '.txt')
        meta_path = os.path.join(clean_root, 'meta', split, stem + '.json')
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f'cannot read {img_path}')
        bboxes = load_bboxes(lab_path)
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
    else:
        import tempfile
        stem = '<synthetic>'      # 统计里要有名字，别 NameError
        tmp_dir = tempfile.mkdtemp(prefix='v2noise_selftest_')
        rng = np.random.RandomState(11)
        img, bboxes, meta = G.make_sample(12345, rng, list(carrier_paths),
                                          apply_noise=False)
        print(f'  (clean_root 为空，临时生成一张自检样本 -> {tmp_dir})')

    print(f'  clean: {img.shape[1]}x{img.shape[0]}  boxes={len(bboxes)}  '
          f'meta.id={meta.get("watermark_id")}  s={meta.get("s")}')

    def _decode(image, boxes, tag):
        rows = [[int(b[0]), b[1], b[2], b[3], b[4]] for b in boxes]
        if len(rows) < 8:
            print(f'  [{tag}] 只有 {len(rows)} 框，跳过解码')
            return None
        xyxy, conf = PI.load_yolo_labels_from_rows(
            rows, image.shape[1], image.shape[0])
        return PI.run(image, xyxy, conf, do_decode=True)

    # 未加噪基线先分级：几何坏是 clean 树/标签的问题（硬 FAIL）；
    # 其余非 ok 只是让这个样本退出"加噪后解码"的判读 —— 载体本身会把
    # ±8/255 调制抹掉（实测 clean 树 150 张里 2 张，train_000036:
    # ncw=96、s_err 0.026%、rms 0.35 却 id=None），那不是加噪的锅。
    want_id = int(meta.get('watermark_id', -1))
    base = _decode(img, bboxes, 'clean')
    bv, bd = PI.judge_decode(base, want_id if want_id >= 0 else None, meta)
    tally[bv] = tally.get(bv, 0) + 1
    if bv == 'geom':
        print(f'  [FAIL] clean 基线几何就对不上：{bd}')
        ok = False
    judgeable = (bv == 'ok')
    if judgeable:
        print(f'  [OK]   clean 基线 {bd}')
    else:
        print(f'  [WARN] clean 基线 {bd}')
        print(f'         → 本样本退出加噪后的解码判读（标签门槛照跑）')
        verdicts.setdefault(bv, []).append(f'{stem}/clean')

    for kind, desc, geom in NOISE_VARIANTS:
        rng = np.random.RandomState(
            (7 * 1_000_003 + 17 + NOISE_SALT[kind] * 1_003) & 0x7FFFFFFF)
        t0 = time.time()
        out_img, out_boxes = apply_noise_variant(
            img.copy(), [list(b) for b in bboxes], kind, rng)
        dt = time.time() - t0
        if out_img.shape != img.shape:
            print(f'  [FAIL] {kind}: 形状变了 {img.shape} -> {out_img.shape}')
            ok = False
            continue

        if not geom:   # wechat 非几何：标签必须逐位不变
            same = (len(out_boxes) == len(bboxes) and all(
                np.allclose(a[1:], b[1:], atol=0, rtol=0)
                for a, b in zip(out_boxes, bboxes)))
            if not same:
                print(f'  [FAIL] {kind}: 标签被改了（非几何噪声不该动标签）')
                ok = False
                continue
            print(f'  [OK]   {kind}: {desc}  标签逐位不变  {dt:.2f}s')
        else:
            moved = (len(out_boxes) != len(bboxes) or any(
                abs(a[1] - b[1]) > 1e-9 or abs(a[2] - b[2]) > 1e-9
                for a, b in zip(out_boxes, bboxes)))
            print(f'  [OK]   {kind}: {desc}  框 {len(bboxes)}->{len(out_boxes)}  '
                  f'{"形变同步" if moved else "形变未动框"}  {dt:.2f}s')

        res = _decode(out_img, out_boxes, kind)
        rv, rd = PI.judge_decode(
            res, want_id if want_id >= 0 else None, meta)
        tally[rv] = tally.get(rv, 0) + 1
        # 加噪的**本意**就是降低可读性，所以 wipe/floor 是报告项不是门槛。
        # 硬门槛只有 geom —— 那意味着 _sync_labels_to_warp 的几何同步坏了
        # （框被挪到错位，s / interval_px 对不上）。
        if rv == 'geom':
            print(f'  [FAIL] {kind}: {rd}  —— 标签几何同步坏了')
            ok = False
            verdicts.setdefault(rv, []).append(f'{stem}/{kind}')
        elif rv == 'ok':
            print(f'  [OK]   {kind}: {rd}')
        else:
            print(f'  [WARN] {kind}: {rd}')
            print(f'         采集退化下 ID 是报告项'
                  + ('（基线本就不 ok，更不作判读）' if not judgeable else ''))
            verdicts.setdefault(rv, []).append(f'{stem}/{kind}')

        if tmp_dir:
            cv2.imwrite(os.path.join(tmp_dir, f'{kind}.png'), out_img)

    print('\n--- 解码分级统计 ---')
    for v in ('ok', 'floor', 'wipe', 'misdecode', 'geom'):
        if tally.get(v):
            print(f'  {v:10} {tally[v]}')
    # 这里**故意不设解码命中率门槛**。加噪的本意就是压低可读性，
    # "加噪后解不出"和"标签几何同步坏了"在解码这一层是同一个现象
    # （都表现为 wipe），区分不了。所以：
    #   * geom 逐张硬卡 —— 尺度/角度/非刚体同步错误，s_err / ip_err 会爆
    #   * 解码只作报告 —— 刚体平移型的同步错误要靠**大批量**验收集的
    #     命中率门槛才抓得到（几何错系统性、载体抹调制散发性），
    #     小样本自检的比率没有统计意义。
    # 实测本样本：wechat / pimog / pimog_wechat 三个都 wipe，但 s_err<=0.05%、
    # ip_err<=0.03px —— 几何同步是对的，是噪声把 ±8/255 调制压掉了。
    if tally.get('geom'):
        print(f'  [FAIL] 有 {tally["geom"]} 次几何对不上 —— 标签同步坏了')
    for v, lst in verdicts.items():
        if v in ('wipe', 'misdecode'):
            print(f'  发出 {v}: {lst}')
    print('=== selftest:', 'PASS' if ok else 'FAIL', '===')
    return 0 if ok else 1


# ───────────────────────── 生成 ─────────────────────────

def _one(args_tuple):
    (split, stem, clean_root, kind, seed, out_root) = args_tuple
    rng = np.random.RandomState(seed)
    img_path = os.path.join(clean_root, 'images', split, stem + '.png')
    lab_path = os.path.join(clean_root, 'labels', split, stem + '.txt')
    meta_path = os.path.join(clean_root, 'meta', split, stem + '.json')

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f'cannot read {img_path}')
    bboxes = load_bboxes(lab_path)

    t0 = time.time()
    out_img, out_boxes = apply_noise_variant(img, bboxes, kind, rng)
    dt = time.time() - t0

    new_stem = f'{stem}_{kind}'
    cv2.imwrite(os.path.join(out_root, 'images', split, new_stem + '.png'), out_img)
    save_bboxes(os.path.join(out_root, 'labels', split, new_stem + '.txt'), out_boxes)

    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta['derived_from'] = stem
    meta['noise'] = [kind]
    meta['noise_sec'] = round(dt, 3)
    meta['n_boxes'] = len(out_boxes)
    if kind != 'wechat':
        # pimog 有残差形变；interval_px 仍是名义值，不是拟合值
        meta['geometry'] = 'warped'
    with open(os.path.join(out_root, 'meta', split, new_stem + '.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False)
    return split, kind, dt, len(out_boxes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description='v2 三噪声数据集：clean -> wechat/pimog/pimog_wechat')
    p.add_argument('--clean_root', type=str, required=True)
    p.add_argument('--output_dir', type=str, default=None, help='省略时仅自检')
    p.add_argument('--smoke', action='store_true', help='每 split 只取 4 张')
    p.add_argument('--shard', type=str, default=None, help='i/n 分片，跨 GPU 并行')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--seed', type=int, default=20260930)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--carrier_root', type=str, default='/data1/lpl/datasets')
    args = p.parse_args(argv)

    carrier_paths = G.discover_carriers(args.carrier_root)
    print(f'carriers: {len(carrier_paths)} images')

    if args.selftest:
        return selftest(args.clean_root, carrier_paths)

    if not args.output_dir:
        p.error('--output_dir is required unless --selftest')
    stems = collect_clean_stems(args.clean_root)
    if not stems:
        raise RuntimeError(f'no clean samples under {args.clean_root}')
    if args.smoke:
        keep = []
        for split in SPLITS:
            keep += [s for s in stems if s[0] == split][:4]
        stems = keep
        print(f'smoke: 取 {len(stems)} 张')

    shard_i, shard_n = 0, 1
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split('/'))
    tasks = []
    for i, (split, stem) in enumerate(stems):
        if i % shard_n != shard_i:
            continue
        for kind, _desc, _g in NOISE_VARIANTS:
            out_png = os.path.join(args.output_dir, 'images', split,
                                   f'{stem}_{kind}.png')
            if os.path.exists(out_png) and not args.overwrite:
                continue
            seed = (args.seed * 1_000_003 + i * 17
                    + NOISE_SALT[kind] * 1_003) & 0x7FFFFFFF
            tasks.append((split, stem, args.clean_root, kind, seed, args.output_dir))

    if not tasks:
        print('nothing to do (全部已存在；用 --overwrite 重跑)')
        return 0

    for split in SPLITS:
        for sub in ('images', 'labels', 'meta'):
            os.makedirs(os.path.join(args.output_dir, sub, split), exist_ok=True)

    print(f'shard {shard_i}/{shard_n}: {len(tasks)} jobs')
    t0 = time.time()
    times: Dict[str, List[float]] = {k: [] for k, _d, _g in NOISE_VARIANTS}
    counts: Dict[str, int] = {k: 0 for k, _d, _g in NOISE_VARIANTS}
    n_boxes_hist: List[int] = []
    for i, t in enumerate(tasks, 1):
        _split, kind, dt, nb = _one(t)
        times[kind].append(dt)
        counts[kind] += 1
        n_boxes_hist.append(nb)
        if i % 20 == 0 or i == len(tasks):
            print(f'  {i}/{len(tasks)}  {time.time() - t0:.0f}s  {counts}  '
                  f'平均框 {np.mean(n_boxes_hist):.1f}')

    if shard_i == 0:
        with open(os.path.join(args.output_dir, 'watermark.yaml'), 'w') as f:
            f.write(
                '# Auto-generated YOLO dataset config (v2, 96 codewords) - 三噪声\n'
                f'path: {os.path.abspath(args.output_dir)}\n'
                'train: images/train\n'
                'val: images/val\n'
                '\n'
                'names:\n'
                '  0: codeword\n')
        manifest = {
            'generator': 'watermark_locator/v2/dataset/prepare_noise_from_clean_v2.py',
            'derived_from': os.path.abspath(args.clean_root),
            'variants': [
                {'name': k, 'desc': d, 'geometric': g, 'salt': NOISE_SALT[k]}
                for k, d, g in NOISE_VARIANTS
            ],
            'counts': counts,
            'n_boxes_mean': float(np.mean(n_boxes_hist)) if n_boxes_hist else None,
            'noise_sec_mean': {k: (float(np.mean(v)) if v else None)
                               for k, v in times.items()},
            'seed': args.seed,
        }
        with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f'\nwrote {args.output_dir}  {counts}  '
          f'{sum(n_boxes_hist)} boxes  {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
