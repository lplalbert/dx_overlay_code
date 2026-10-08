"""色偏校正后再解 —— 用 4 个黑白回字角标当该图的中性灰参考。

为什么
------
实拍 rect 图拒识 13 张, 交付振幅中位 -0.20 (设计 +8.16), 一半反号一半塌零。
极性翻转试过了: 能量回来 (amp 13) 但 slot_ok 全是 1, 一条也救不出。
所以不是通道序换, 是**真实拍照的白平衡/色偏把结构化能量注进 G−B 这同一通道**,
把 ±8.16 的水印淹了。

角标是纯黑白三环回 (qr_loc_mark.png), 本来 G−B ≡ 0。拍完 G−B ≠ 0,
那就是色偏在该图上的直接测量。四个角标贴死画布四角:
    TL (0,0)  TR (1800,0)  BL (0,960)  BR (1800,960),  各 120x120

白平衡是逐通道乘性: G'=g·G, B'=b·B
    G'-B' = (g-b)·(G+B)/2 + (g+b)·(G-B)/2
中性像素 (G=B) 上只有第一项, 所以在角标的黑(0,0,0)和白(255,255,255)上
各测一个点, 就把泄漏项的斜率和截距定下来。校正:
    GB_corr = (G-B) - (a + c·L),   L=(G+B)/2
扣掉的是"中性像素本该有的 G−B", 留下的才是水印 + 与亮度无关的色度。

预期: 校正后交付振幅回到 +8 附近, 拒识的图里能救回一部分。
如果一点救不回, 说明色偏不是主要杀手, 焦点/摩尔纹/重采样才是。
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
RECT = os.path.join(CAP, 'detect_out', 'rect')
KNOWN = {153364, 163462, 323972, 513074, 697521, 710750, 728982, 780033, 820170}
GATE = 3
MARKS = [(0, 0), (1800, 0), (0, 960), (1800, 960)]   # 左上角, 各 120x120
MSZ = 120
OUT = os.path.join(os.environ.get('DX_OUT_DIR', '.'), 'wbcorr_decode.json')


def fit_neutral(img):
    """在 4 个角标上拟合 G−B = a + c·L 的中性线。返回 (a, c, 诊断)。"""
    B = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    gb = G - B
    L = (G + B) * 0.5
    # 黑环 / 白环分开: 角标是三环回, 内外环白、中间环黑之类。
    # 不去猜环的拓扑 —— 直接在角标区域里按亮度分位数取两端点。
    xs, ys = [], []
    dbg = []
    for (mx, my) in MARKS:
        gbs = gb[my:my + MSZ, mx:mx + MSZ].ravel()
        Ls = L[my:my + MSZ, mx:mx + MSZ].ravel()
        lo = Ls <= np.percentile(Ls, 25)     # 黑环
        hi = Ls >= np.percentile(Ls, 75)     # 白环
        if lo.sum() < 50 or hi.sum() < 50:
            continue
        x_lo, y_lo = float(Ls[lo].mean()), float(gbs[lo].mean())
        x_hi, y_hi = float(Ls[hi].mean()), float(gbs[hi].mean())
        xs += [x_lo, x_hi]
        ys += [y_lo, y_hi]
        dbg.append((mx, my, x_lo, y_lo, x_hi, y_hi))
    if len(xs) < 4:
        return 0.0, 0.0, dbg
    x = np.array(xs, np.float64)
    y = np.array(ys, np.float64)
    # 最小二乘拟 y = a + c x
    xm, ym = x.mean(), y.mean()
    c = float(((x - xm) * (y - ym)).sum() / max(((x - xm) ** 2).sum(), 1e-9))
    a = float(ym - c * xm)
    return a, c, dbg


def correct(img):
    B = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    L = (G + B) * 0.5
    a, c, dbg = fit_neutral(img)
    gb = (G - B) - (a + c * L)
    out = img.copy()
    # 只动 G, 保持 L 大致不变: G_new = L + gb/2, B_new = L - gb/2
    out[:, :, 1] = np.clip(L + gb * 0.5, 0, 255).astype(np.uint8)
    out[:, :, 0] = np.clip(L - gb * 0.5, 0, 255).astype(np.uint8)
    return out, a, c, dbg


def main():
    dec = Decoder(np.load(LOCATOR_NPY))
    selftest(dec)

    # 先在拍前 PNG 上验证校正不会破坏好样本 (它色偏≈0, 应当几乎不变)
    print('\n=== 校正的无害性: 拍前 PNG (本来 9/9) ===')
    for p in sorted(glob.glob(os.path.join(CAP, 'wm*.png'))):
        img = cv2.imread(p)
        im2, a, c, _ = correct(img)
        r0, r1 = dec.decode(img), dec.decode(im2)
        a0, _ = amp_of(dec, img)
        a1, _ = amp_of(dec, im2)
        print(f'  {os.path.basename(p)[:28]:28s} a={a:+6.2f} c={c:+.4f}  '
              f'id {r0["wm_id"]:6d}-> {r1["wm_id"]:6d}  '
              f'amp {a0:5.2f}->{a1:5.2f}  slot {r0["n_slot_ok"]:2d}->{r1["n_slot_ok"]:2d}')

    files = sorted(glob.glob(os.path.join(RECT, '*.png')))
    print(f'\n=== 拍后 rect {len(files)} 张: 原始 vs 色偏校正 ===')
    print(f'{"file":40s} {"src":>5s} {"a":>7s} {"c":>7s} | '
          f'{"原始 amp/id/门":^24s} | {"校正 amp/id/门":^24s}')
    rows = []
    tally = {}
    for p in files:
        img = cv2.imread(p)
        name = os.path.basename(p)
        src = 'pzvx' if name.startswith('pzvx') else 'pz'
        im2, a, c, _ = correct(img)
        r0, r1 = dec.decode(img), dec.decode(im2)
        a0, _ = amp_of(dec, img)
        a1, _ = amp_of(dec, im2)
        g0 = r0['decode_ok'] and r0['n_slot_ok'] >= GATE
        g1 = r1['decode_ok'] and r1['n_slot_ok'] >= GATE
        s0 = g0 and r0['wm_id'] in KNOWN
        s1 = g1 and r1['wm_id'] in KNOWN
        if s1:
            tally.setdefault(r1['wm_id'], set()).add(src)

        def fmt(r, amp, gate, inset):
            if not gate:
                return f'{amp:>6.2f}  拒(s={r["n_slot_ok"]:2d})'
            return f'{amp:>6.2f} {r["wm_id"]:6d}{"*" if inset else " "}(s={r["n_slot_ok"]:2d})'
        print(f'{name[:40]:40s} {src:>5s} {a:>+7.2f} {c:>+7.4f} | '
              f'{fmt(r0, a0, g0, s0):^24s} | {fmt(r1, a1, g1, s1):^24s}')
        rows.append({'file': name, 'src': src, 'a': a, 'c': c,
                     'orig_id': r0['wm_id'], 'orig_amp': a0,
                     'orig_slot': r0['n_slot_ok'], 'orig_gate': g0, 'orig_set': s0,
                     'corr_id': r1['wm_id'], 'corr_amp': a1,
                     'corr_slot': r1['n_slot_ok'], 'corr_gate': g1, 'corr_set': s1})

    n0 = sum(1 for r in rows if r['orig_set'])
    n1 = sum(1 for r in rows if r['corr_set'])
    print(f'\n  原始  落已知集 {n0}/{len(rows)}')
    print(f'  校正  落已知集 {n1}/{len(rows)}')
    print('\n  校正后跨组重复:')
    for wid in sorted(tally):
        print(f'    {wid:6d}  {sorted(tally[wid])}')

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f'\n明细已写 {OUT}')


if __name__ == '__main__':
    main()
