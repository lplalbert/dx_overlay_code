"""包络参考 vs 条纹参考 —— 判定实拍塌掉的是"位块"还是"4px 条纹载波"。

位尺寸实测纠正
--------------
`FIX_FG_MATRIX` 是 (16, **64**), 每码字 64 位 = **8x8**, 不是 64x64。
`gen_wm_block(block_size=64)` -> 每位 64x64 -> 512x512, 缩到 160x135:
    位宽 = 160/8 = **20 px**   位高 = 135/8 = **17 px**
(此前算的 2.5px 是拿 160/(512/8) 去除, 把模板空间的 px/位 当成了位数, 错。)

那被抹的是什么
--------------
`stripe_mask(angle=45, period=4, stripe_width=2)` 在画布空间是 **4px 周期的 45 度梳**,
水印只落在梳齿上。我的匹配滤波参考 `refs = (keep & yellow)*8.16` 里**带着这条梳**。
拍照链(对焦/重采样/摩尔纹)抹 4px 梳比抹 20px 位块容易得多 —— 梳一没, 带梳参考的
相关就塌, 表现为"振幅塌零/反号、符号不认同", 看着像符号层被抹。

本脚本做 A/B: 同一批图两种参考
    梳参考  : keep & yellow        (精确匹配, 对梳被抹不鲁棒)
    包络参考: yellow only          (不管梳, 只看位块的黄/白身份)
包络参考是"梳包络"的失配滤波器 —— 梳占空比 50%, 能量减半, 但对梳被抹免疫。

判据: 拍前 PNG 两种都该全对(梳完好); 拍后 rect 若包络参考明显更好, 说明
实拍塌的是梳, 不是位块 —— 那么改格式/改解码都还有救, 不必降到存在性检测。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_cross_tier as D  # noqa: E402
from decode_cross_tier import (  # noqa: E402
    ALPHA, MSG_H, MSG_W, Decoder, LOCATOR_NPY, VOTE_MIN, selftest, xcorr)
from generate_dataset import stripe_mask  # noqa: E402
from generate_locator_pattern import FIX_FG_MATRIX  # noqa: E402
from decode_cross_tier import build_yellow_masks, resize_tmpl, slot_cells  # noqa: E402

CAP = '/data1/lpl/dx_overlay_code/vis/real_capture'
KNOWN = {153364, 163462, 323972, 513074, 697521, 710750, 728982, 780033, 820170}
GATE = 3
OUT = os.path.join(os.environ.get('DX_OUT_DIR', '.'), 'envelope_decode.json')


class EnvDecoder(Decoder):
    """参考不含条纹梳 —— 只用黄/白位块掩码。"""

    def __init__(self, locator_pattern, mode='envelope'):
        self.mode = mode
        self.yel = build_yellow_masks(locator_pattern)
        self.keep = stripe_mask(D.SCREEN_W, D.SCREEN_H, angle=45.0, period=4,
                                stripe_width=2)
        self.cells = slot_cells()
        self.refs = []
        for slot in range(16):
            per = []
            for (x, y) in self.cells[slot]:
                k = self.keep[y:y + MSG_H, x:x + MSG_W]
                if mode == 'envelope':
                    per.append(np.stack(
                        [t.astype(np.float32) * (ALPHA * 255.0) for t in self.yel]))
                else:
                    per.append(np.stack(
                        [(k & t).astype(np.float32) * (ALPHA * 255.0)
                         for t in self.yel]))
            self.refs.append(np.stack(per))


def run(dec, files, label, rows, key):
    print(f'\n--- {label} ---')
    print(f'{"file":42s} {"src":>5s} {"amp":>7s} {"解出":>8s} {"s_ok":>4s} {"对?":>4s}')
    tally = {}
    n_set = 0
    for p in files:
        img = cv2.imread(p)
        name = os.path.basename(p)
        src = ('png' if name.startswith('wm')
               else 'pzvx' if name.startswith('pzvx') else 'pz')
        r = dec.decode(img)
        # 交付振幅用各自的参考算, 才可比
        gb = (img[:, :, 1].astype(np.float32) - img[:, :, 0].astype(np.float32))
        amps = []
        for slot in range(16):
            for ci, (x, y) in enumerate(dec.cells[slot]):
                o = gb[y:y + MSG_H, x:x + MSG_W]
                sc = [xcorr(o, dec.refs[slot][ci][t]) for t in range(17)]
                t = int(np.argmax(sc))
                m = dec.refs[slot][ci][t] / (ALPHA * 255.0)
                den = float((m * m).sum())
                if den > 0:
                    amps.append(float((o * m).sum() / den))
        amp = float(np.median(amps)) if amps else float('nan')

        gt = None
        if name.startswith('wm') and '_c' in name:
            gt = int(name[2:name.index('_c')])
        gate = r['decode_ok'] and r['n_slot_ok'] >= GATE
        inset = gate and r['wm_id'] in KNOWN
        if gt is not None:
            okmark = 'Y' if (gate and r['wm_id'] == gt) else '.'
        else:
            okmark = '*' if inset else ('.' if gate else ' ')
        if inset:
            n_set += 1
            tally.setdefault(r['wm_id'], set()).add(src)
        ids = f'{r["wm_id"]:6d}' if r['decode_ok'] else '     -'
        print(f'{name[:42]:42s} {src:>5s} {amp:>7.2f} {ids:>8s} '
              f'{r["n_slot_ok"]:>4d} {okmark:>4s}')
        rows.append({'file': name, 'src': src, 'mode': key,
                     'id': r['wm_id'], 'ok': r['decode_ok'],
                     'slot': r['n_slot_ok'], 'amp': amp,
                     'gate': gate, 'in_set': inset, 'gt': gt})
    print(f'  -> 落已知集 {n_set}/{len(files)}   跨组重复 ID: '
          f'{sorted((w, sorted(s)) for w, s in tally.items())}')


def main():
    # 自检: 包络参考也必须过灰底 8/8 + 死输入 3/3, 否则不许读实拍
    for mode in ('stripe', 'envelope'):
        d = EnvDecoder(np.load(LOCATOR_NPY), mode=mode)
        print(f'\n########## 参考模式 = {mode} ##########')
        selftest(d)

    png = sorted(glob.glob(os.path.join(CAP, 'wm*.png')))
    rect = sorted(glob.glob(os.path.join(CAP, 'detect_out', 'rect', '*.png')))

    rows = []
    for mode, label in (('stripe', '梳参考 (现用, 精确匹配)'),
                        ('envelope', '包络参考 (不含梳, 对梳被抹免疫)')):
        d = EnvDecoder(np.load(LOCATOR_NPY), mode=mode)
        run(d, png, f'{label} — 拍前 PNG', rows, mode)
        run(d, rect, f'{label} — 拍后 rect', rows, mode)

    print('\n' + '=' * 70)
    print('对照 (落已知集 / 总数):')
    for mode in ('stripe', 'envelope'):
        for grp, sel in (('拍前 PNG', lambda r: r['src'] == 'png'),
                         ('拍后 rect', lambda r: r['src'] != 'png')):
            v = [r for r in rows if r['mode'] == mode and sel(r)]
            n = sum(1 for r in v if r['gt'] is not None and r['gate']
                    and r['id'] == r['gt'])
            ns = sum(1 for r in v if r['gt'] is None and r['in_set'])
            hit = n if any(r['gt'] is not None for r in v) else ns
            print(f'  {mode:8s} {grp:9s}  {hit}/{len(v)}')

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f'\n明细已写 {OUT}')


if __name__ == '__main__':
    main()
