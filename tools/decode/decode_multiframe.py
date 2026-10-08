"""多帧分数级融合 —— 直接测"多帧聚合"这条路值多少。

背景
----
单帧实拍: 拍前 9/9 -> 拍后 5/18。剩下 13 张里, 最好的一张 (pzvx_13547)
有 4 个可信槽, 只差 1 个槽就够 RS 的信息量 (15,5) 需要的 5 个。多帧一叠很可能就进。

pz 与 pzvx 是同 9 块屏各拍一遍 (各 9 张)。所以每块屏有两个独立样本。
不做图像对齐 —— 透视校正残差会让平均糊掉。改在**匹配滤波分数层**融合:
对每个码字格, 把两帧的 17 个模板相关分加起来再 argmax。这是检测级融合的标准做法,
对配准误差免疫, 而且天然是相干累积 (信号加, 噪声按 sqrt 加)。

配对: 同一块屏的两帧内容相同, 用降采样灰度签名的最近邻配。
配对质量会打印出来, 配错的对融合会变差 —— 那本身也是信息。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_cross_tier import (  # noqa: E402
    ALPHA, MSG_H, MSG_W, Decoder, LOCATOR_NPY, VOTE_MIN, selftest, xcorr)
from decode_real_capture import amp_of  # noqa: E402

CAP = '/data1/lpl/dx_overlay_code/vis/real_capture'
RECT = os.path.join(CAP, 'detect_out', 'rect')
KNOWN = {153364, 163462, 323972, 513074, 697521, 710750, 728982, 780033, 820170}
GATE = 3
OUT = os.path.join(os.environ.get('DX_OUT_DIR', '.'), 'multiframe_decode.json')


def sig(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (48, 27)).astype(np.float32).ravel()


def cell_scores(dec, img):
    """每格 17 个模板的相关分, 形状 (16, 6, 17)。"""
    gb = (img[:, :, 1].astype(np.float32) - img[:, :, 0].astype(np.float32))
    out = np.zeros((16, 6, 17), np.float64)
    for slot in range(16):
        for ci, (x, y) in enumerate(dec.cells[slot]):
            o = gb[y:y + MSG_H, x:x + MSG_W]
            for t in range(17):
                out[slot, ci, t] = xcorr(o, dec.refs[slot][ci][t])
    return out


def decide(dec, fused):
    """分数 -> 符号 -> 6 路投票 -> RS。与 decode_cross_tier 的拒识口径一致。"""
    votes = np.zeros(16, np.int64)
    agree = np.zeros(16, np.int64)
    gap_med = np.zeros(16, np.float64)
    for slot in range(16):
        preds, gaps = [], []
        for ci in range(6):
            sc = fused[slot, ci]
            order = np.argsort(sc)[::-1]
            t = int(order[0])
            preds.append(t)
            gaps.append(sc[t] - sc[order[1]])
        preds = np.array(preds)
        gaps = np.array(gaps)
        cnt = np.bincount(preds, minlength=17)
        win = int(np.argmax(cnt))
        votes[slot] = win
        agree[slot] = int(cnt[win])
        sup = gaps[preds == win]
        gap_med[slot] = float(np.median(sup)) if sup.size else 0.0

    from decode_cross_tier import _rs_codec, LOCATOR_CODEWORD_INDEX
    hard = [s for s in range(15) if agree[s] <= 1 or gap_med[s] <= 0.0]
    erase_pos = sorted(14 - s for s in hard)
    wm_id, ok = -1, False
    if _rs_codec is not None and len(erase_pos) <= 10:
        try:
            rs_cw = [int(votes[s]) for s in range(14, -1, -1)]
            dec5 = [int(e) for e in _rs_codec.decode(
                rs_cw, erase_pos=erase_pos or None)[0]]
            wm_id = 0
            for i, v in enumerate(reversed(dec5)):
                wm_id += v * (16 ** i)
            ok = True
        except Exception:
            ok = False
    n_slot_ok = int(((agree >= VOTE_MIN) & (gap_med > 0.0)).sum())
    return {'wm_id': wm_id, 'decode_ok': ok, 'n_slot_ok': n_slot_ok,
            'n_erase': len(erase_pos), 'gap_p50': float(np.median(gap_med))}


def main():
    dec = Decoder(np.load(LOCATOR_NPY))
    selftest(dec)

    pz = sorted(glob.glob(os.path.join(RECT, 'pz_*.png')))
    pzvx = sorted(glob.glob(os.path.join(RECT, 'pzvx_*.png')))
    print(f'\npz {len(pz)} 张, pzvx {len(pzvx)} 张')

    imgs = {p: cv2.imread(p) for p in pz + pzvx}
    sigs = {p: sig(im) for p, im in imgs.items()}

    # 配对: 每个 pz 找最像的 pzvx
    pairs = []
    used = set()
    for a in pz:
        best, bs = None, -1e18
        for b in pzvx:
            if b in used:
                continue
            s = -float(np.abs(sigs[a] - sigs[b]).mean())
            if s > bs:
                bs, best = s, b
        used.add(best)
        pairs.append((a, best, bs))
    pairs.sort(key=lambda t: t[2], reverse=True)
    print('\n配对 (相似度 = 降采样灰度平均绝对差的负数, 越大越像):')
    for a, b, s in pairs:
        print(f'  {os.path.basename(a)[:38]:38s} <-> '
              f'{os.path.basename(b)[:38]:38s}  {s:.3f}')

    # 单帧结果作对照
    print('\n单帧 vs 2 帧分数融合:')
    print(f'{"pz":38s} {"pzvx":38s} | {"单帧(pz / pzvx)":^34s} | {"2帧融合":^22s}')
    rows = []
    tally1, tally2 = {}, {}
    for a, b, _ in pairs:
        ra = dec.decode(imgs[a])
        rb = dec.decode(imgs[b])
        sc = cell_scores(dec, imgs[a]) + cell_scores(dec, imgs[b])
        rf = decide(dec, sc)
        ga = ra['decode_ok'] and ra['n_slot_ok'] >= GATE
        gb_ = rb['decode_ok'] and rb['n_slot_ok'] >= GATE
        gf = rf['decode_ok'] and rf['n_slot_ok'] >= GATE

        def f(r, g, tag):
            if g and r['wm_id'] in KNOWN:
                tally1.setdefault(r['wm_id'], set()).add(tag)
                return f'{r["wm_id"]:6d}*(s={r["n_slot_ok"]:2d})'
            return f'{"-":>6s} (s={r["n_slot_ok"]:2d})'
        sa = f(ra, ga, 'a')
        sb = f(rb, gb_, 'b')
        if gf and rf['wm_id'] in KNOWN:
            tally2.setdefault(rf['wm_id'], set())
            tally2[rf['wm_id']].add(os.path.basename(a)[:12])
            sf = f'{rf["wm_id"]:6d}*(s={rf["n_slot_ok"]:2d})'
        else:
            sf = f'{"-":>6s} (s={rf["n_slot_ok"]:2d})'
        print(f'{os.path.basename(a)[:38]:38s} {os.path.basename(b)[:38]:38s} | '
              f'{sa:^16s} {sb:^16s} | {sf:^22s}')
        rows.append({'pz': os.path.basename(a), 'pzvx': os.path.basename(b),
                     'a_id': ra['wm_id'], 'a_slot': ra['n_slot_ok'], 'a_gate': ga,
                     'b_id': rb['wm_id'], 'b_slot': rb['n_slot_ok'], 'b_gate': gb_,
                     'f_id': rf['wm_id'], 'f_slot': rf['n_slot_ok'], 'f_gate': gf,
                     'f_erase': rf['n_erase']})

    n1 = len({w for w, s in tally1.items()})
    print(f'\n  单帧 (两个极性组各自算) 落已知集的 ID 数: {n1}  {sorted(tally1)}')
    print(f'  2 帧融合 落已知集的 ID 数: {len(tally2)}  {sorted(tally2)}')
    n_pair_ok = sum(1 for r in rows if r['f_gate'] and r['f_id'] in KNOWN)
    print(f'  按对算: 融合后可读 {n_pair_ok}/{len(rows)} 对')

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f'\n明细已写 {OUT}')


if __name__ == '__main__':
    main()
