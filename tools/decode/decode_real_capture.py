"""实拍链独立解码 —— 用我自己的匹配滤波器, 不用对方的解码器。

为什么要这个
------------
对方的实拍端到端结果是 0/18, 但**对照臂(拍前无噪声 PNG)只有 2/9**。
无噪声 PNG 本该接近 100% —— 我自己的 clean 树是 60/60。对照臂低成这样,
说明 0/18 里混着他们解码器的锅, 还不能全算在拍照链头上。

所以用**另一把经过自检的尺子**独立测同一批图:
  A. 拍前 PNG (vis/real_capture/wm*.png)      —— 对照臂, 期望接近 100%
  B. 拍后 rect (vis/real_capture/detect_out/rect/*.png) —— 实拍+微信链

顺带量第三件事: **实际交付的水印振幅**。设计是 Δ(G−B)=255α=8.16,
但早前记过"实拍 α 实际只交付 3~5 不是 8"。每格做一元回归
amp = <G−B, m>/<m,m>, m=该胜出模板的信号格, 中位数就是交付振幅。
如果拍前 PNG 就已经掉到 3~5, 那是生成端的锅, 不是拍照链。

自检硬门: 灰底合成 8 个已知 wm_id 必须解回, 3 个死输入必须被拒。
不过就是脚本错了, 一律不读实拍数。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_cross_tier import Decoder, ALPHA, LOCATOR_NPY  # noqa: E402
from decode_cross_tier import selftest  # noqa: E402

CAP = '/data1/lpl/dx_overlay_code/vis/real_capture'
OUT = os.path.join(os.environ.get('DX_OUT_DIR', '.'), 'real_capture_decode.json')


def amp_of(dec, img_bgr):
    """交付振幅: 每格对胜出模板做一元回归 <G−B, m>/<m,m>, 取中位。

    设计值 = 255*ALPHA = 8.16 (只在 keep&yellow 的信号格上)。
    """
    from decode_cross_tier import MSG_H, MSG_W, xcorr
    gb = (img_bgr[:, :, 1].astype(np.float32)
          - img_bgr[:, :, 0].astype(np.float32))
    amps = []
    for slot in range(16):
        for ci, (x, y) in enumerate(dec.cells[slot]):
            o = gb[y:y + MSG_H, x:x + MSG_W]
            if o.shape[:2] != (MSG_H, MSG_W):
                continue
            sc = [xcorr(o, dec.refs[slot][ci][t]) for t in range(17)]
            t = int(np.argmax(sc))
            m = dec.refs[slot][ci][t] / (ALPHA * 255.0)   # 还原成 0/1 掩码
            den = float((m * m).sum())
            if den > 0:
                amps.append(float((o * m).sum() / den))
    return float(np.median(amps)) if amps else float('nan'), len(amps)


def load_manifest():
    d = json.load(open(os.path.join(CAP, 'manifest.json')))
    recs = d.get('samples') or d.get('images') or d.get('items') or []
    if isinstance(recs, dict):
        recs = list(recs.values())
    return d, recs


def main():
    dec = Decoder(np.load(LOCATOR_NPY))
    selftest(dec)                     # 硬门

    man, recs = load_manifest()
    # 拍前 PNG 的真值: 从文件名 wm<id>_c<n>.png 拿, 不依赖 manifest 结构
    def gt_from_name(fn):
        b = os.path.basename(fn)
        if b.startswith('wm') and '_c' in b:
            return int(b[2:b.index('_c')])
        return None

    groups = [
        ('拍前 PNG (无噪声对照)', sorted(glob.glob(os.path.join(CAP, 'wm*.png'))),
         'clean_rect'),
        ('拍后 rect (实拍+微信)', sorted(glob.glob(
            os.path.join(CAP, 'detect_out', 'rect', '*.png'))), 'photo'),
    ]

    rows = []
    for gname, files, kind in groups:
        print(f'\n=== {gname}  n={len(files)} ===')
        print(f'{"file":44s} {"gt":>8s} {"解出":>8s} {"对?":>4s} '
              f'{"slot_ok":>7s} {"cell_ok":>7s} {"erase":>5s} '
              f'{"gap":>7s} {"amp":>6s}')
        n_ok = n_bad = n_fail = 0
        amps = []
        for p in files:
            img = cv2.imread(p)
            if img is None:
                print(f'  {os.path.basename(p)[:44]:44s}  **读不出**')
                continue
            if img.shape[0] != 1080 or img.shape[1] != 1920:
                # rect 应该是 1920x1080; 不是就跳过并说明
                print(f'  {os.path.basename(p)[:44]:44s}  尺寸 {img.shape[1]}x{img.shape[0]} '
                      f'**不是 1920x1080, 跳过**')
                continue
            r = dec.decode(img)
            amp, ncell = amp_of(dec, img)
            amps.append(amp)
            gt = gt_from_name(p)
            if not r['decode_ok']:
                v = '不出'
                n_fail += 1
            elif gt is not None and r['wm_id'] == gt:
                v = ' Y '
                n_ok += 1
            elif gt is None:
                v = ' ? '
                n_fail += 1
            else:
                v = '**N**'
                n_bad += 1
            gts = f'{gt:d}' if gt is not None else '?'
            print(f'  {os.path.basename(p)[:44]:44s} '
                  f'{gts:>8s} '
                  f'{r["wm_id"]:>8d} {v:>6s} {r["n_slot_ok"]:>7d} '
                  f'{r["n_cell_ok"]:>7d} {r["n_erase"]:>5d} '
                  f'{r["gap_p50"]:>7.4f} {amp:>6.2f}')
            rows.append({'group': gname, 'file': os.path.basename(p), 'gt': gt,
                         'wm_id': r['wm_id'], 'ok': r['decode_ok'], 'verdict': v,
                         'n_slot_ok': r['n_slot_ok'], 'n_cell_ok': r['n_cell_ok'],
                         'n_erase': r['n_erase'], 'gap_p50': r['gap_p50'],
                         'amp': amp})
        den = n_ok + n_bad + n_fail
        print(f'  -> 解对 {n_ok}  解错 {n_bad}  解不出 {n_fail}   '
              f'命中率 {(n_ok / den * 100) if den else 0:.1f}%')
        if amps:
            a = [x for x in amps if np.isfinite(x)]
            print(f'  -> 交付振幅 中位 {np.median(a):.2f}  '
                  f'p25 {np.percentile(a, 25):.2f}  p75 {np.percentile(a, 75):.2f}'
                  f'   (设计 255*alpha = {255 * ALPHA:.2f})')

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f'\n明细已写 {OUT}')


if __name__ == '__main__':
    main()
