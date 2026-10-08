"""极性翻转试验 —— 拍后 rect 一半图振幅是负的, 判定是通道序翻转还是物理抹除。

现象 (decode_real_capture.py)
----------------------------
拍前 PNG  : 9/9 解对, 交付振幅中位 7.68 (设计 255*alpha = 8.16)  -> 生成端没问题
拍后 rect : 0 解对, 交付振幅中位 -0.20, 一半图是 -8.5 / -8.2 / -7.2 ...

负振幅不是"信号没了", 是**信号反着在**。G-B 取反会让匹配滤波器把 17 个模板
全挑反, argmax 落到最差的那个 —— 于是符号层整层读错。这跟"物理抹除"是两回事。

本脚本对每张 rect 两个极性各解一遍:
    原极性: gb = G - B
    反极性: 交换 B/G 通道 -> gb = B - G = -(G - B)
看反极性能不能把那些图救回已知集。

判据: 已知集 {153364, 163462, 323972, 513074, 697521, 710750, 728982, 780033, 820170}。
再看 pz / pzvx 两组里同一 ID 会不会重复出现 —— 两组是同 9 块屏各拍一遍,
同一 ID 在两组都读出来, 基本不可能是巧合 (16^5 种 ID)。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_cross_tier import ALPHA, Decoder, LOCATOR_NPY, selftest  # noqa: E402
from decode_real_capture import amp_of  # noqa: E402

CAP = '/data1/lpl/dx_overlay_code/vis/real_capture'
KNOWN = {153364, 163462, 323972, 513074, 697521, 710750, 728982, 780033, 820170}
GATE = 3          # slot_ok >= 3 采信 (前面扫出来的"解错=0"最小门)
OUT = os.path.join(os.environ.get('DX_OUT_DIR', '.'), 'polarity_decode.json')


def main():
    dec = Decoder(np.load(LOCATOR_NPY))
    selftest(dec)

    files = sorted(glob.glob(os.path.join(CAP, 'detect_out', 'rect', '*.png')))
    print(f'\nrect 图 {len(files)} 张, 每张两个极性')
    print(f'{"file":44s} {"src":>5s} | {"原极性 amp/id/门":^26s} | {"反极性 amp/id/门":^26s}')
    rows = []
    tally = {}
    for p in files:
        img = cv2.imread(p)
        name = os.path.basename(p)
        src = 'pzvx' if name.startswith('pzvx') else 'pz'
        variants = {}
        for pol, flip in (('原', False), ('反', True)):
            im = img.copy()
            if flip:
                im[:, :, 0], im[:, :, 1] = img[:, :, 1], img[:, :, 0].copy()
            r = dec.decode(im)
            amp, _ = amp_of(dec, im)
            gate = r['decode_ok'] and r['n_slot_ok'] >= GATE
            in_set = r['decode_ok'] and r['wm_id'] in KNOWN
            variants[pol] = {'amp': amp, 'id': r['wm_id'], 'slot_ok': r['n_slot_ok'],
                             'gate': gate, 'in_set': in_set, 'erase': r['n_erase']}
            if gate and in_set:
                tally.setdefault(r['wm_id'], set()).add(src)

        def fmt(v):
            if not v['gate']:
                return f"{v['amp']:>6.2f}  拒(s={v['slot_ok']:2d})"
            mark = '*' if v['in_set'] else ' '
            return f"{v['amp']:>6.2f} {v['id']:6d}{mark}(s={v['slot_ok']:2d})"
        print(f'{name[:44]:44s} {src:>5s} | {fmt(variants["原"]):^26s} | '
              f'{fmt(variants["反"]):^26s}')
        rows.append({'file': name, 'src': src,
                     **{f'orig_{k}': v for k, v in variants['原'].items()},
                     **{f'flip_{k}': v for k, v in variants['反'].items()}})

    print('\n  * = 解出的 ID 落在已知集 9 个里')
    print(f'  采信门: decode_ok 且 slot_ok >= {GATE}\n')

    n_o = sum(1 for r in rows if r['orig_gate'] and r['orig_in_set'])
    n_f = sum(1 for r in rows if r['flip_gate'] and r['flip_in_set'])
    n_both = sum(1 for r in rows if r['orig_gate'] and r['orig_in_set']
                 and r['flip_gate'] and r['flip_in_set'])
    n_any = sum(1 for r in rows if (r['orig_gate'] and r['orig_in_set'])
                or (r['flip_gate'] and r['flip_in_set']))
    print(f'  原极性落已知集 {n_o}/{len(rows)}')
    print(f'  反极性落已知集 {n_f}/{len(rows)}')
    print(f'  两极性都落    {n_both}')
    print(f'  至少一个极性落 {n_any}/{len(rows)}')

    print('\n  跨组重复 (pz 与 pzvx 是同 9 块屏各拍一遍, 同 ID 两组都读出=强证据):')
    for wid in sorted(tally):
        print(f'    {wid:6d}  出现在 {sorted(tally[wid])}')
    both = [w for w, s in tally.items() if len(s) > 1]
    print(f'  -> 两组都读出的 ID: {both}')

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f'\n明细已写 {OUT}')


if __name__ == '__main__':
    main()
