"""从 cross_tier_decode.json 找拒识门的操作点 —— 只读 JSON, 不重跑解码。

问题: 宽松口径把"有点信但不满 4/6"的槽当符号交给 RS, 解码率上去了,
但**解错也出来了 (6 条)**。严格口径解错=0 但解码率低。

中间有没有一条门, 既挡住全部解错又不误伤解对?
判据候选: slot_ok (6 路投票 >= 4 的槽数) —— 策略无关的复制一致度。
对 k = 0..16 扫 "slot_ok >= k 才采信", 报每档的解码率 / 解错率 / 误伤。
"""
import os
import json

P = os.path.join(os.environ.get('DX_OUT_DIR', '.'), 'cross_tier_decode.json')
TAGS = ('clean', 'n1', 'n2', 'n3', 'n4')

def main():
    rows = json.load(open(P))
    by = {}
    for r in rows:
        by.setdefault(r['tag'], []).append(r)

    # clean 是锚, 看它自己在各 k 下会不会被误拒
    print('clean 锚点在各 k 下的存活 (n=60, 期望全存活):')
    for k in (0, 1, 2, 3, 4, 6, 8, 10, 13):
        v = by['clean']
        kept = sum(1 for r in v if r['decode_ok'] and r['n_slot_ok'] >= k)
        print(f'  slot_ok>={k:2d}  存活 {kept:2d}/60')
    print()

    hdr = (f'{"k":>3s} | ' + ' | '.join(f'{t:^21s}' for t in TAGS[1:])
           + ' | 全局')
    print(hdr)
    print(f'{"":>3s} | ' + ' | '.join(f'{"解对  解错  率":^21s}' for _ in TAGS[1:])
          + ' | 解错合计')
    print('-' * len(hdr))

    best = None
    for k in range(0, 13):
        cells = []
        tot_ok = tot_bad = tot_den = tot_lost = 0
        for t in TAGS[1:]:
            v = by[t]
            ok = sum(1 for r in v if r['decode_ok'] and r['wm_id'] != 0xFFFFFFFF
                     and r['n_slot_ok'] >= k and r['verdict'] == '解对')
            bad = sum(1 for r in v if r['verdict'] == '解错' and r['n_slot_ok'] >= k)
            # 被门挡掉的解对 = 误伤
            lost = sum(1 for r in v if r['verdict'] == '解对' and r['n_slot_ok'] < k)
            den = len(v)
            tot_ok += ok
            tot_bad += bad
            tot_den += den
            tot_lost += lost
            rate = ok / den * 100
            cells.append(f'{ok:3d}  {bad:3d}  {rate:5.1f}%')
        mark = ''
        if tot_bad == 0:
            mark = '  <- 解错=0'
            if best is None:
                best = (k, tot_ok, tot_lost)
        print(f'{k:3d} | ' + ' | '.join(f'{c:^21s}' for c in cells)
              + f' | {tot_bad:3d}{mark}')

    print()
    if best:
        k, ok, lost = best
        print(f'最小的"解错=0"门: slot_ok >= {k}')
        print(f'  保留解对 {ok},  误伤解对 {lost}')
    else:
        print('没有一条 k 能让解错归零')

    print('\n被误读的样本长什么样 (解错的 slot_ok 分布):')
    for t in TAGS[1:]:
        bad = [r for r in by[t] if r['verdict'] == '解错']
        if not bad:
            continue
        print(f'  {t}: slot_ok = {[r["n_slot_ok"] for r in bad]}  '
              f'erase = {[r["n_erase"] for r in bad]}')

    print('\n解对的 slot_ok 分布 (看误伤会不会发生):')
    for t in TAGS[1:]:
        ok = sorted(r['n_slot_ok'] for r in by[t] if r['verdict'] == '解对')
        if not ok:
            continue
        print(f'  {t}: n={len(ok):2d}  slot_ok 最小={ok[0]:2d}  '
              f'p25={ok[len(ok)//4]:2d}  中位={ok[len(ok)//2]:2d}  最大={ok[-1]:2d}')

    print('\n解不出的 slot_ok 分布 (看门能不能把它们救回来):')
    for t in TAGS[1:]:
        f = sorted(r['n_slot_ok'] for r in by[t] if r['verdict'] == '解不出')
        if not f:
            continue
        print(f'  {t}: n={len(f):2d}  slot_ok 最小={f[0]:2d}  '
              f'中位={f[len(f)//2]:2d}  最大={f[-1]:2d}')


if __name__ == '__main__':
    main()
