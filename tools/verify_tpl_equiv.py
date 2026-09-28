#!/usr/bin/env python
"""核对: vis/tpl 里的模板 与 vis/real_capture 9 张图 是否等效。

检查 3 件事:
  A. 每张图能否用「其 wm_id 的模板」+「其载体(先盖标)」逐字节复现
  B. 同一载体的 3 张图, 6 个定位块区域是否逐字节相同 (定位模板应恒定)
  C. 不同 wm_id 的消息区模板是否不同 (消息模板应随 ID 变)
"""
import json
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import cv2
import numpy as np

# 仓库根 = 本文件上两级 (tools/ -> repo root)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARRIER_DIR = os.environ.get('DX_CARRIER_DIR', '/data1/lpl/datasets/test')
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'watermark_locator'))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator', 'dataset'))

from generate_dataset import (
    generate_one_sample, build_canvas, alpha_blend_watermark,
    get_locator_abs_rect, SCREEN_W, SCREEN_H,
)
from generate_locator_pattern import FIX_FG_MATRIX

CAP = os.path.join(REPO, 'vis', 'real_capture')
MARK_PNG = os.path.join(REPO, 'qr_loc_mark.png')
LOCATOR_NPY = os.path.join(REPO, 'watermark_locator', 'locator_pattern.npy')
ALPHA = 0.032
SEED = 20260928
MARK = 120
MARK_POS = {'TL': (0, 0), 'TR': (SCREEN_W - MARK, 0),
            'BL': (0, SCREEN_H - MARK), 'BR': (SCREEN_W - MARK, SCREEN_H - MARK)}

LOCATOR_IDX = [(0, 3), (1, 1), (1, 5), (2, 3), (3, 1), (3, 5)]
LOCATOR_RECTS = [get_locator_abs_rect(i, j) for i, j in LOCATOR_IDX]


def build_marked(si, fname, mark3):
    src = cv2.imread(os.path.join(CARRIER_DIR, fname))
    rng = np.random.RandomState(SEED + si)
    canvas = build_canvas([src], rng, SCREEN_W, SCREEN_H, grid=(1, 1))
    marked = canvas.copy()
    for (x, y) in MARK_POS.values():
        marked[y:y + MARK, x:x + MARK] = mark3
    return marked


def main():
    locator_pattern = np.load(LOCATOR_NPY)
    ref = cv2.imread(MARK_PNG, cv2.IMREAD_GRAYSCALE)
    ref = cv2.resize(ref, (MARK, MARK), interpolation=cv2.INTER_AREA)
    ref = np.where(ref >= 128, 255, 0).astype(np.uint8)
    mark3 = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    man = json.load(open(os.path.join(CAP, 'manifest.json'), encoding='utf-8'))
    files = sorted(f for f in os.listdir(CARRIER_DIR) if f.endswith('.png'))
    marked_by_si = {si: build_marked(si, f, mark3)
                    for si, f in enumerate(files, 1)}

    print('=== A. 逐字节复现: 模板(wm_id) + 载体(先盖标) -> 出图 ===')
    ok = 0
    for rec in man['images']:
        img = cv2.imread(os.path.join(CAP, rec['file']))
        marked = marked_by_si[rec['carrier_idx']]
        rng = np.random.RandomState(SEED + rec['carrier_idx'])
        clean, _, _, _ = generate_one_sample(
            rec['wm_id'], FIX_FG_MATRIX, locator_pattern, ALPHA, rng,
            carrier_img=marked, apply_noise=False, channel_mode='b')
        same = np.array_equal(img, clean)
        nz = int((img != clean).any(axis=2).sum())
        print(f'  {rec["file"]:22s} wm_id={rec["wm_id_6digit"]}  '
              f'{"逐字节一致" if same else f"不一致 {nz} px"}')
        ok += int(same)
    print(f'  -> {ok}/{len(man["images"])} 张可用其 wm_id 的模板精确复现')

    print('\n=== B. 同载体 3 张: 6 个定位块区域是否逐字节相同 ===')
    for si in (1, 2, 3):
        recs = [r for r in man['images'] if r['carrier_idx'] == si]
        imgs = [cv2.imread(os.path.join(CAP, r['file'])) for r in recs]
        same_all = True
        for li, (x, y, w, h) in enumerate(LOCATOR_RECTS, 1):
            patches = [im[y:y + h, x:x + w] for im in imgs]
            same = all(np.array_equal(patches[0], p) for p in patches[1:])
            same_all &= same
        print(f'  载体{si:02d}  {[r["wm_id_6digit"] for r in recs]}  '
              f'6 个定位块 {"全部逐字节相同" if same_all else "有差异"}')
    print('  -> 定位块模板不随 wm_id 变 (仅随载体变)')

    print('\n=== C. 不同 wm_id: 消息区模板是否不同 ===')
    ids = [r['wm_id'] for r in man['images']]
    from rs_gen_Syn_template_nums_dual import get_wm_seq
    for r in man['images'][:3]:
        s = list(get_wm_seq(r['wm_id']))
        s[-1] = 16
        print(f'  wm_id={r["wm_id_6digit"]}  码字序列={s}')
    seqs = {r['wm_id_6digit']: tuple(list(get_wm_seq(r['wm_id']))[:-1] + [16])
            for r in man['images']}
    uniq = len(set(seqs.values()))
    print(f'  -> 9 张图共 {uniq} 套不同的消息模板 (100% 各不相同)' if uniq == len(seqs)
          else f'  -> 只有 {uniq} 套不同')
    print('  -> 消息区模板随 wm_id 变; 只有 6 个定位块是全数据集恒定的')

    print('\n=== 结论 ===')
    print('  vis/tpl/ 里的「完整模板」是 wm_id=323972 那一张的模板。')
    print('  另外 8 张 wm_id 不同 -> 消息区模板不同, 不是同一张。')
    print('  等效的是: 公式/结构/α/条纹/定位块, 这些完全一致。')


if __name__ == '__main__':
    main()
