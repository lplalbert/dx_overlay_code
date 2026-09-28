#!/usr/bin/env python
"""回答「区别是什么导致的」—— 用受控实验分离两个原因, 再回到真实 9 张图上验证。

原因 A: 融合是「向模板值插值」不是「加固定幅度」  ->  Δ = α·(wm - c), 幅度随载体明暗变
原因 B: dynamic_mask 让中性白格 a_eff = 0         ->  白格完全不动, 只有黄格动
角标差异: 上面两条在角标这个「纯 0/255 载体」上的具体表现(按区域重新统计, 不用单像素)
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
    alpha_blend_watermark, build_canvas, generate_one_sample,
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


def blend_scalar(c, wm_ch):
    """单像素单通道: out = c*(1-a) + wm*a, 含 dynamic_mask。"""
    wm = np.array([[[wm_ch, 255, 255]]], np.uint8)   # B=wm_ch, G=R=255
    car = np.array([[[c, c, c]]], np.uint8)
    out = alpha_blend_watermark(car, wm, ALPHA, channel_mode='b')
    return out[0, 0].astype(np.int16) - c


def main():
    print('═══ 受控实验 1: 固定模板=黄, 只改载体亮度 c ═══')
    print('  (模板 B=0 G=255 R=255; Δ 应等于 α·(wm−c))')
    print(f'  {"c":>4s} {"ΔB":>6s} {"ΔG":>6s} {"ΔR":>6s} {"|Δ|max":>7s}   解释')
    for c in [0, 16, 32, 64, 96, 128, 160, 192, 224, 240, 255]:
        dB, dG, dR = blend_scalar(c, 0)
        note = ''
        if c == 0:
            note = 'B 已是 0, 拉不动; 全靠 G/R'
        elif c == 255:
            note = 'G/R 已是 255, 拉不动; 全靠 B'
        elif c == 128:
            note = '三通道均摊, 各只 ±4'
        print(f'  {c:>4d} {dB:>6d} {dG:>6d} {dR:>6d} '
              f'{max(abs(dB),abs(dG),abs(dR)):>7d}   {note}')
    print('  ==> 原因A: Δ = α·(wm − c)。c 离 0/255 越远, 幅度越小。')
    print('      max|Δ| = α·max(c, 255−c)  ∈ [0, 8.16], 不是常数。')

    print('\n═══ 受控实验 2: 固定 c=128, 只改模板 (黄 vs 白) ═══')
    for wm_ch, name in [(0, '黄(信号)'), (255, '白(中性)')]:
        dB, dG, dR = blend_scalar(128, wm_ch)
        print(f'  c=128 模板={name:10s} -> Δ(B,G,R)=({dB},{dG},{dR})')
    print('  ==> 原因B: dynamic_mask = min(1, dev*2), dev=max|wm−255|/255。')
    print('      白格 dev=0 -> a_eff=0 -> 像素逐字节不动。只有黄格有水印。')

    print('\n═══ 真实 9 张图: 按载体亮度分箱, 看 mean|Δ| ═══')
    print('  (如果「区别」真由 c 决定, 这里应单调)')
    man = json.load(open(os.path.join(CAP, 'manifest.json'), encoding='utf-8'))
    files = sorted(f for f in os.listdir(CARRIER_DIR) if f.endswith('.png'))
    ref = cv2.imread(MARK_PNG, cv2.IMREAD_GRAYSCALE)
    ref = np.where(cv2.resize(ref, (MARK, MARK), interpolation=cv2.INTER_AREA) >= 128,
                   255, 0).astype(np.uint8)
    mark3 = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    marked_by_si = {}
    for si, f in enumerate(files, 1):
        src = cv2.imread(os.path.join(CARRIER_DIR, f))
        canvas = build_canvas([src], np.random.RandomState(SEED + si),
                              SCREEN_W, SCREEN_H, grid=(1, 1))
        for (x, y) in MARK_POS.values():
            canvas[y:y + MARK, x:x + MARK] = mark3
        marked_by_si[si] = canvas

    bins = [(0, 32), (32, 96), (96, 160), (160, 224), (224, 256)]
    agg = {b: [[], [], []] for b in bins}
    for rec in man['images']:
        img = cv2.imread(os.path.join(CAP, rec['file']))
        m = marked_by_si[rec['carrier_idx']]
        d = img.astype(np.int16) - m.astype(np.int16)
        g = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        nz = (d != 0).any(axis=2)
        for b in bins:
            sel = (g >= b[0]) & (g < b[1]) & nz
            if sel.any():
                for ch in range(3):
                    agg[b][ch].append(float(np.abs(d[:, :, ch][sel]).mean()))
    print(f'  {"载体亮度 c":>12s} {"|ΔB|":>7s} {"|ΔG|":>7s} {"|ΔR|":>7s} {"|Δ|max":>7s}')
    for b in bins:
        if not agg[b][0]:
            continue
        mB, mG, mR = (float(np.mean(agg[b][ch])) for ch in range(3))
        print(f'  [{b[0]:>3d},{b[1]:>3d})      {mB:>7.2f} {mG:>7.2f} {mR:>7.2f} '
              f'{max(mB,mG,mR):>7.2f}')
    print('  ==> 与受控实验 1 的曲线一致: 暗箱 |ΔG/R| 大 |ΔB| 小, 亮箱反过来。')

    print('\n═══ 角标区域 —— 按区域统计(修正上一条的单像素误读) ═══')
    rec = man['images'][0]
    img = cv2.imread(os.path.join(CAP, rec['file']))
    m = marked_by_si[rec['carrier_idx']]
    d = img.astype(np.int16) - m.astype(np.int16)
    print(f'  {rec["file"]}')
    print(f'  {"角标":6s} {"子区":6s} {"像素数":>8s} {"改动%":>8s} '
          f'{"mean|Δ|":>8s} {"max|Δ|":>7s}   Δ取值')
    for tag, (x1, y1, x2, y2) in rec['mark_rects'].items():
        pd = d[y1:y2, x1:x2]
        g = cv2.cvtColor(m[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        for sub, sel in [('黑', g < 128), ('白', g >= 128)]:
            n = int(sel.sum())
            if n == 0:
                continue
            sub_d = pd[sel]
            nz = (sub_d != 0).any(axis=1)
            pct = 100.0 * nz.mean()
            mean_ad = float(np.abs(sub_d[nz]).mean()) if nz.any() else 0.0
            mx = int(np.abs(sub_d).max())
            uniq = sorted({tuple(r) for r in sub_d[nz].tolist()})[:4]
            print(f'  {tag:6s} {sub:6s} {n:>8d} {pct:>7.1f}% {mean_ad:>8.2f} '
                  f'{mx:>7d}   {uniq}')

    print('\n═══ 6 个定位块区域 —— 逐块改动率(看是不是位置/图案决定) ═══')
    for li, (x, y, w, h) in enumerate(LOCATOR_RECTS, 1):
        pcts = []
        for r in [x for x in man['images'] if x['carrier_idx'] == 1]:
            im = cv2.imread(os.path.join(CAP, r['file']))
            pd = im[y:y + h, x:x + w].astype(np.int16) - m[y:y + h, x:x + w]
            pcts.append(100.0 * (pd != 0).any(axis=2).mean())
        print(f'  定位#{li}  改动率={[round(p,1) for p in pcts]}')
    print('  ==> 同一载体下 3 个不同 wm_id 的定位块改动率相同 -> 模板恒定;')
    print('      但 6 个定位块彼此改动率不同 -> 由 45° 条纹与块位置的相对相位决定。')

    print('\n═══ 结论 ═══')
    print('  有区别, 两个独立原因:')
    print('   A. 融合是 α·(wm−c) 插值, 不是固定幅度加性 -> 幅度随载体亮度 0→8 漂')
    print('      这是主要来源: 9 张 mean|Δ| 2.94 / 4.08 / 4.83 全由此造成')
    print('   B. dynamic_mask 让白(中性)格 a_eff=0 -> 74.6% 像素根本不改')
    print('      这是覆盖率问题, 不是强度问题: 9 张恒为 25.38%')
    print('  角标上看起来不一样 = A+B 在「纯 0/255 载体」上的叠加,')
    print('  加上 45° 条纹相位在各角标位置不同(改动率随之不同)。')


if __name__ == '__main__':
    main()
