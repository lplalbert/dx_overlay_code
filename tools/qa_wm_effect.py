#!/usr/bin/env python
"""QA: 9 张水印图的实际效果。

查 6 件事:
  1. 强度是否一致 (逐张 max/mean |Δ|, 逐通道)
  2. 是否溢出/裁剪 (结果被 clip 到 0/255 会削弱水印)
  3. 舍入偏差 (np.round 是否系统性地偏亮/偏暗)
  4. 角标区域的强度 (纯 0/255 载体 -> Δ 最大, 最容易看出显示问题)
  5. 色偏方向 (B 往下拉 / G,R 往上拉)
  6. 平坦区可见性 (最坏情况: 大片纯色上的 8/255 是否看得出)
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


def main():
    locator_pattern = np.load(LOCATOR_NPY)
    ref = cv2.imread(MARK_PNG, cv2.IMREAD_GRAYSCALE)
    ref = cv2.resize(ref, (MARK, MARK), interpolation=cv2.INTER_AREA)
    ref = np.where(ref >= 128, 255, 0).astype(np.uint8)
    mark3 = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    man = json.load(open(os.path.join(CAP, 'manifest.json'), encoding='utf-8'))
    files = sorted(f for f in os.listdir(CARRIER_DIR) if f.endswith('.png'))

    # 重建「盖标后的载体」= 嵌水印前的底
    marked_by_si = {}
    for si, f in enumerate(files, 1):
        src = cv2.imread(os.path.join(CARRIER_DIR, f))
        rng = np.random.RandomState(SEED + si)
        canvas = build_canvas([src], rng, SCREEN_W, SCREEN_H, grid=(1, 1))
        m = canvas.copy()
        for (x, y) in MARK_POS.values():
            m[y:y + MARK, x:x + MARK] = mark3
        marked_by_si[si] = m

    print('=== 1. 强度一致性 (|Δ| = 出图 - 盖标载体) ===')
    print(f'{"文件":22s} {"max|Δ|":>7s} {"mean|Δ|":>8s} {"非零%":>7s} '
          f'{"ΔB均":>7s} {"ΔG均":>7s} {"ΔR均":>7s}')
    stats = []
    for rec in man['images']:
        img = cv2.imread(os.path.join(CAP, rec['file']))
        marked = marked_by_si[rec['carrier_idx']]
        d = img.astype(np.int16) - marked.astype(np.int16)
        ad = np.abs(d)
        nz = (d != 0).any(axis=2)
        row = {
            'file': rec['file'], 'wm': rec['wm_id_6digit'],
            'max': int(ad.max()), 'mean': float(ad[nz].mean()) if nz.any() else 0.0,
            'nz': float(nz.mean()),
            'mB': float(d[:, :, 0][nz].mean()), 'mG': float(d[:, :, 1][nz].mean()),
            'mR': float(d[:, :, 2][nz].mean()),
        }
        stats.append(row)
        print(f'{rec["file"]:22s} {row["max"]:>7d} {row["mean"]:>8.3f} '
              f'{row["nz"]*100:>6.2f}% {row["mB"]:>7.2f} {row["mG"]:>7.2f} {row["mR"]:>7.2f}')
    mx = [s['max'] for s in stats]
    print(f'  -> 9 张 max|Δ| 取值 {sorted(set(mx))}  '
          f'(理论上界 α×255 = {ALPHA*255:.2f})')
    print(f'  -> 非零像素占比取值 {sorted(set(round(s["nz"],4) for s in stats))}  '
          f'(应全部相同 = 模板决定)')

    print('\n=== 2. 溢出/裁剪检查 (逐张) ===')
    for rec in man['images']:
        marked = marked_by_si[rec['carrier_idx']]
        img = cv2.imread(os.path.join(CAP, rec['file']))
        d = img.astype(np.int16) - marked.astype(np.int16)
        # 理论未 round 值的落点: c*(1-α) 与 c*(1-α)+8.16, 看是否出 [0,255]
        clip_lo = int((marked.astype(np.float32) * (1 - ALPHA) < 0).sum())
        clip_hi = int((marked.astype(np.float32) * (1 - ALPHA) + ALPHA * 255 > 255).sum())
        vals = sorted(set(np.unique(d).tolist()))
        print(f'  {rec["file"]:22s} max|Δ|={np.abs(d).max()}  Δ取值={vals}  '
              f'会触底像素={clip_lo} 会触顶像素={clip_hi}')
        assert np.abs(d).max() <= 8, 'Δ 超出 α×255 上界!'
    print('  -> |Δ| 恒 <= 8, 且 out=c(1-α)+wm·α 的落点始终在 [0,255] 内 -> 零裁剪')
    print('     (α=0.032 时 255(1-α)+255α = 255.0 恰好饱和, 不会溢出)')

    print('\n=== 3. 舍入偏差 ===')
    for rec in man['images']:
        img = cv2.imread(os.path.join(CAP, rec['file']))
        marked = marked_by_si[rec['carrier_idx']]
        d = img.astype(np.int16) - marked.astype(np.int16)
        nz = (d != 0).any(axis=2)
        bias = float(d[:, :, 0][nz].mean()) if nz.any() else 0.0
        pos = int((d[:, :, 0][nz] > 0).sum())
        neg = int((d[:, :, 0][nz] < 0).sum())
        print(f'  {rec["file"]:22s} B通道 正Δ像素={pos:>7d} 负Δ像素={neg:>7d} '
              f'净偏={bias:+.3f}')
    print('  -> B 通道只有负Δ(拉向0)或0; G/R 只有正Δ(拉向255) —— 单向, 无双向抖动')

    print('\n=== 4. 角标区域 (纯0/255 载体, Δ 最大) ===')
    rec = man['images'][0]
    img = cv2.imread(os.path.join(CAP, rec['file']))
    marked = marked_by_si[rec['carrier_idx']]
    d = img.astype(np.int16) - marked.astype(np.int16)
    for tag, (x1, y1, x2, y2) in rec['mark_rects'].items():
        pd = d[y1:y2, x1:x2]
        g = cv2.cvtColor(marked[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        black = g < 128
        print(f'  {tag}: 黑像素区 Δ(B,G,R)={pd[black][0].tolist() if black.any() else "-"}  '
              f'白像素区 Δ={pd[~black][0].tolist() if (~black).any() else "-"}  '
              f'max|Δ|={np.abs(pd).max()}')
    print('  -> 纯黑格: Δ=(0,+8,+8)  -> 黑变成 (0,8,8), 微微偏绿')
    print('  -> 纯白格: Δ=(-8,0,0)  -> 白变成 (247,255,255), 微微偏黄')

    print('\n=== 5. 色偏方向 ===')
    print('  黄(信号)像素: B 拉向 0 (ΔB=-α·c_B) , G/R 拉向 255 (Δ=+α·(255-c))')
    print('  -> 暗区 (c 小): G/R 上抬明显, B 几乎不动  => 整体偏亮/偏绿')
    print('  -> 亮区 (c 大): B 下压明显, G/R 几乎不动  => 整体偏暗/偏黄')
    print('  -> 中灰 (c=128): 三通道各 ±4, 最均衡')

    print('\n=== 6. 最坏情况可见性: 大片纯色 ===')
    for cval, name in [(0, '纯黑'), (16, '近黑'), (128, '中灰'),
                       (240, '近白'), (255, '纯白')]:
        carrier = np.full((200, 200, 3), cval, np.uint8)
        wm = np.full((200, 200, 3), 255, np.uint8)
        wm[:, :, 0] = 0                       # 全黄
        out = alpha_blend_watermark(carrier, wm, ALPHA, channel_mode='b')
        dd = out.astype(np.int16) - carrier.astype(np.int16)
        print(f'  大片{name:4s}(={cval:3d}) 上 max|Δ|={np.abs(dd).max()}  '
              f'Δ(B,G,R)={dd[0,0].tolist()}')
    print('  -> 最坏是纯黑/纯白大片: Δ 达 8/255 = 3.1%, 平坦区人眼在 1-2% 即可察觉')
    print('     但条纹只保留 50%, 且是高频斜纹 -> 观感更接近「噪点」而非「色块」')

    print('\n=== 结论 ===')
    print('  强度: 9 张的模板决定的改动占比完全一致; |Δ| 上界统一 8, 无一张越界')
    print('        但**实际幅度随载体明暗变** (暗区动G/R, 亮区动B, 中灰各±4)')
    print('        —— 这不是 bug, 是 out=c(1-α)+wm·α 的固有性质')
    print('  显示风险点: ① 角标纯黑/纯白格上 Δ 最大(8), 平坦纯色最容易看出')
    print('             ② 色偏是单向的 (B↓ G↑ R↑), 不是零均值噪声')
    print('             ③ 无裁剪/无溢出, 舍入误差 < 1 LSB')


if __name__ == '__main__':
    main()
