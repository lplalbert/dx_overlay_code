"""跨档自洽解码 —— 匹配滤波器, 检测器不在环, 纯 CPU。

为什么要这个
------------
残余(零偏移相关)不预测可解码性, 这已被证伪(残余高的档解码率可以是 0)。
所以档位上界必须按**解码率**定, 不能按残余定。本脚本就是那把尺子。

口径
----
* 不用任何训练模型。直接拿生成时的同一套模板渲染做匹配滤波 —— 测的是
  **信号层物理上还剩多少**, 检测器的锅不在里面。
* clean 与 n1..n4 同 stem 共用一个 wm_id (prepare_multids_3noise.py:226-248),
  所以 `decode(clean)` 是真值锚, 逐档比对。
* 解不出 / 解错 分开: 解错是**自信的错 ID**, 比解不出危险得多。

几何 (从 generate_dataset.generate_one_sample 反推, 已核对)
------------------------------------------------------------
块 4x6, 每块 320x270; 每块 2x2 码字格, 每格 160x135 → **96 个码字格**。
wm_seq 共 16 个值 (15 个 RS 码字倒序 + 1 个定位索引), 切成 4 组 x4。
组 k 铺在 k=(j+(i%2)*2)%4 的 6 个块里, **每个值重复 6 次** (6 路冗余)。
组内重排 msgs_reorder=[a,c,b,d]:
    slot%4 == 0 -> 象限(0,0)   == 1 -> 象限(1,0)
    slot%4 == 2 -> 象限(0,1)   == 3 -> 象限(1,1)
slot 15 (组3 第4格) = 定位图案, 位置恰是 get_locator_positions() 那 6 个。

自检 (必须先过)
--------------
用 generate_one_sample 在纯灰底上造已知 wm_id 的图, 跑匹配滤波, 必须解回同一个 id。
不过就是脚本错了, 不是数据的问题。这条不过一律不读 n1..n4 的数。
"""
import os
import random
import sys

import cv2
import numpy as np

REPO = os.environ.get('DX_OVERLAY_REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator'))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator', 'dataset'))

from generate_dataset import (  # noqa: E402
    SCREEN_W, SCREEN_H, BLOCK_ROWS, BLOCK_COLS, BLOCK_W, BLOCK_H,
    MSG_W, MSG_H, alpha_blend_watermark, gen_rect_tp, gen_wm_block,
    generate_one_sample, stripe_mask,
)
from generate_locator_pattern import FIX_FG_MATRIX  # noqa: E402
from utils import rs_encode, _rs_codec, LOCATOR_CODEWORD_INDEX  # noqa: E402

ALPHA = 0.032
ROOT = os.environ.get('DX_DATASET_ROOT', '/data1/lpl/datasets_labeled_4tier')
TAGS = ('clean', 'n1', 'n2', 'n3', 'n4')
N_STEM = 60
LOCATOR_NPY = os.path.join(REPO, 'watermark_locator', 'locator_pattern.npy')

# 6 路投票: 至少几路同向才算这个槽位读出来了
VOTE_MIN = 4
# 16 个槽位里至少几个读出来才不算 floor (15 个 RS 槽全要, 定位槽不要求)
SLOT_FLOOR = 13


# ───────────────────── 模板 / 参考 ─────────────────────

def resize_tmpl(img, w, h):
    """与 generate_one_sample 里的完全一致: NEAREST + 阈值锁死 0/255。"""
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    out[:, :, :3] = np.where(out[:, :, :3] >= 128, 255, 0).astype(np.uint8)
    return out


def build_yellow_masks(locator_pattern):
    """17 个码字模板 -> 160x135 的"黄格"布尔图 (True=信号格)。"""
    ext = np.vstack([FIX_FG_MATRIX, locator_pattern.reshape(1, 64)])
    yel = []
    for idx in range(17):
        tmpl = gen_wm_block(ext[idx], block_size=64, v_tp_fn=gen_rect_tp,
                            channel_mode='b')
        small = resize_tmpl(tmpl, MSG_W, MSG_H)
        yel.append(small[:, :, 0] == 0)   # B=0 -> 黄(信号)
    return yel


def slot_cells():
    """16 个槽位 -> 各自的 6 个码字格左上角 (x, y)。"""
    cells = [[] for _ in range(16)]
    quad = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
    for i in range(BLOCK_ROWS):
        for j in range(BLOCK_COLS):
            k = (j + (i % 2) * 2) % 4
            for p in range(4):
                bi, bj = quad[p]
                x = j * BLOCK_W + bj * MSG_W
                y = i * BLOCK_H + bi * MSG_H
                cells[k * 4 + p].append((x, y))
    return cells


def xcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    s = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / s) if s > 0 else 0.0


# ───────────────────── 匹配滤波 ─────────────────────

class Decoder:
    def __init__(self, locator_pattern):
        self.yel = build_yellow_masks(locator_pattern)
        self.keep = stripe_mask(SCREEN_W, SCREEN_H, angle=45.0, period=4,
                                stripe_width=2)
        self.cells = slot_cells()
        # 每个槽位的参考: (6, 17, MSG_H, MSG_W) float32, 只算一次
        self.refs = []
        for slot in range(16):
            per = []
            for (x, y) in self.cells[slot]:
                k = self.keep[y:y + MSG_H, x:x + MSG_W]
                per.append(np.stack(
                    [(k & t).astype(np.float32) * (ALPHA * 255.0)
                     for t in self.yel]))
            self.refs.append(np.stack(per))          # (6,17,H,W)

    def decode(self, img_bgr):
        gb = (img_bgr[:, :, 1].astype(np.float32)
              - img_bgr[:, :, 0].astype(np.float32))
        votes = np.zeros(16, dtype=np.int64)
        agree = np.zeros(16, dtype=np.int64)          # 胜出符号拿到几路
        gap_med = np.zeros(16, dtype=np.float64)
        n_cell_ok = 0
        cell_scores = []
        cell_gaps = []
        for slot in range(16):
            preds = []
            gaps = []
            for ci, (x, y) in enumerate(self.cells[slot]):
                o = gb[y:y + MSG_H, x:x + MSG_W]
                sc = [xcorr(o, self.refs[slot][ci][t]) for t in range(17)]
                order = np.argsort(sc)[::-1]
                t = int(order[0])
                preds.append(t)
                gaps.append(sc[t] - sc[order[1]])     # top1-top2 分差
                cell_scores.append(sc[t])
                cell_gaps.append(gaps[-1])
            preds = np.array(preds)
            gaps = np.array(gaps)
            cnt = np.bincount(preds, minlength=17)
            win = int(np.argmax(cnt))
            votes[slot] = win
            agree[slot] = int(cnt[win])
            # 支持胜出符号的那些格的分差中位 —— 平票(分差 0)不构成证据
            sup = gaps[preds == win]
            gap_med[slot] = float(np.median(sup)) if sup.size else 0.0
            if (agree[slot] >= VOTE_MIN) and (gap_med[slot] > 0.0):
                n_cell_ok += int(agree[slot])

        # ── 拒识策略 ──
        # 无证据 = 单票(agree<=1) 或 平票(gap<=0)。这类槽作 RS 的 erasure,
        # 其余按多数票当符号送 RS —— RS(15,5) 本来就能容 5 个错, 把"有点信但
        # 不满 4/6"的槽硬擦掉反而是浪费 (实测: 那样 n2 的 f 会冲到 11 > 10,
        # RS 直接拒解)。erasure 预算 10 个 (已实测 f=10 OK / f=11 FAIL)。
        hard = [s for s in range(15) if agree[s] <= 1 or gap_med[s] <= 0.0]
        hard.sort(key=lambda s: (agree[s], gap_med[s]))   # 最不可信优先
        erased = hard

        rs_cw = [int(votes[s]) for s in range(14, -1, -1)]
        erase_pos = sorted(14 - s for s in erased)        # 槽 s -> rs_cw 下标 14-s
        wm_id, ok = -1, False
        # 无证据槽超过 erasure 预算就直接拒 —— 不能截断到 10 个硬解:
        # 全平图会凑成 f=10 + 5 个 0, 仍是合法全零码字, 又把幻觉通道打开。
        if _rs_codec is not None and len(erase_pos) <= 10:
            try:
                # decode 返回 3 元组 (解出的 5 个信息符号, 整段码字, 纠错位)
                dec = [int(e) for e in _rs_codec.decode(
                    rs_cw, erase_pos=erase_pos or None)[0]]
                # rs_encode 里 data = ans[::-1] 是大端 hex, 累加要倒回小端
                wm_id = 0
                for i, v in enumerate(reversed(dec)):
                    wm_id += v * (16 ** i)
                ok = True
            except Exception:
                ok = False

        n_slot_ok = int(((agree >= VOTE_MIN) & (gap_med > 0.0)).sum())
        return {
            'wm_id': wm_id,
            'decode_ok': ok,
            'votes': votes,
            'agree': agree,
            'n_slot_ok': n_slot_ok,
            'n_cell_ok': n_cell_ok,
            'n_erase': len(erase_pos),
            'loc_slot_ok': bool(votes[15] == LOCATOR_CODEWORD_INDEX),
            'score_p50': float(np.median(cell_scores)),
            'gap_p50': float(np.median(cell_gaps)),
        }


# ───────────────────── 自检 ─────────────────────

def selftest(dec):
    """灰底合成 -> 必须解回已知 wm_id。不过就不许读真实数据。"""
    rng = np.random.RandomState(0)
    gray = np.full((SCREEN_H, SCREEN_W, 3), 128, np.uint8)
    pat = np.load(LOCATOR_NPY)
    ids = [0, 1, 0x12345, 0xABCDE, 0xFFFFF, int(rng.randint(0, 16 ** 5 - 1)),
           int(rng.randint(0, 16 ** 5 - 1)), int(rng.randint(0, 16 ** 5 - 1))]
    print('自检 (灰底合成, 已知 wm_id):')
    bad = 0
    for wid in ids:
        img, _, _, _ = generate_one_sample(
            wid, FIX_FG_MATRIX, pat, ALPHA, np.random.RandomState(1),
            carrier_img=gray, apply_noise=False, channel_mode='b')
        r = dec.decode(img)
        flag = 'OK' if (r['decode_ok'] and r['wm_id'] == wid) else '**FAIL**'
        if flag != 'OK':
            bad += 1
        print(f'  wm_id=0x{wid:05X} -> 解出 0x{r["wm_id"]:05X} ok={r["decode_ok"]} '
              f'slot_ok={r["n_slot_ok"]}/16 loc={r["loc_slot_ok"]} {flag}')
    if bad:
        raise SystemExit(f'自检失败 {bad}/{len(ids)} —— 脚本自身有错, 不读真实数据')
    print(f'  自检通过 {len(ids)}/{len(ids)}')

    # 幻觉通道回归: 全零码字是合法 RS 码字, decode([0]*15) 会"成功"解出 id=0。
    # 死输入(纯平图 / 纯噪声)必须被拒识, 不能吐一个看起来合法的 ID。
    print('\n  幻觉通道回归 (死输入必须被拒):')
    dead = {
        '纯平 128': np.full((SCREEN_H, SCREEN_W, 3), 128, np.uint8),
        '纯平 0': np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8),
        '纯噪声': np.random.RandomState(7).randint(
            0, 256, (SCREEN_H, SCREEN_W, 3)).astype(np.uint8),
    }
    bad = 0
    for name, img in dead.items():
        r = dec.decode(img)
        # 期望: decode_ok=False (拿不到 ID)。绝不能是 ok=True 且 id=0
        halluc = r['decode_ok'] and r['wm_id'] == 0
        flag = '**幻觉**' if halluc else ('OK' if not r['decode_ok'] else '**解出了**')
        if flag != 'OK':
            bad += 1
        print(f'    {name:10s} -> ok={r["decode_ok"]} id=0x{r["wm_id"]:05X} '
              f'erase={r["n_erase"]:2d} slot_ok={r["n_slot_ok"]:2d} {flag}')
    if bad:
        raise SystemExit(f'幻觉通道回归失败 {bad}/3 —— 有死输入被解成了合法 ID')
    print('    幻觉通道已封 3/3\n')


# ───────────────────── 主流程 ─────────────────────

def find_stems():
    """从 noisy/n1 抽 stem, 覆盖 3 个数据集、train/val。"""
    by_ds = {}
    for ds in sorted(os.listdir(os.path.join(ROOT, 'noisy'))):
        got = []
        for split in ('train', 'val'):
            d = os.path.join(ROOT, 'noisy', ds, split, 'images')
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith('_n1.png'):
                    got.append((ds, split, f[:-len('_n1.png')]))
        if got:
            by_ds[ds] = got
    rnd = random.Random(0)
    # 按数据集分层抽样
    names = sorted(by_ds)
    per = max(1, N_STEM // max(1, len(names)))
    out = []
    for ds in names:
        pool = by_ds[ds]
        out.extend(rnd.sample(pool, min(per, len(pool))))
    return out


def stem_path(ds, split, stem, tag):
    base = os.path.join(ROOT, 'clean' if tag == 'clean' else 'noisy',
                        ds, split, 'images')
    name = f'{stem}.png' if tag == 'clean' else f'{stem}_{tag}.png'
    return os.path.join(base, name)


def main():
    print(f'档位 {TAGS}   抽样目标 {N_STEM} stem')
    dec = Decoder(np.load(LOCATOR_NPY))
    selftest(dec)

    stems = find_stems()
    print(f'实际抽到 {len(stems)} stem\n')

    per_tag = {t: [] for t in TAGS}
    rows = []
    for (ds, split, stem) in stems:
        ref_id = None
        for tag in TAGS:
            p = stem_path(ds, split, stem, tag)
            img = cv2.imread(p)
            if img is None:
                continue
            r = dec.decode(img)
            if tag == 'clean':
                ref_id = r['wm_id'] if r['decode_ok'] else None
                verdict = '锚'
            elif ref_id is None:
                verdict = '无锚'
            elif not r['decode_ok']:
                verdict = '解不出'
            elif r['wm_id'] == ref_id:
                verdict = '解对'
            else:
                verdict = '解错'
            per_tag[tag].append((verdict, r))
            rows.append((ds, stem, tag, verdict, r))
        print(f'  [{ds}] {stem[:28]}', flush=True)

    # ── 汇总 ──
    print('\n' + '=' * 78)
    print(f'{"档":6s} {"n":>3s} {"解对":>5s} {"解错":>5s} {"解不出":>6s} '
          f'{"无锚":>5s} {"命中率":>7s} {"slot_ok中位":>11s} {"cell_ok中位":>11s} '
          f'{"erase中位":>9s} {"相关p50":>8s} {"分差p50":>8s} {"loc槽对":>7s}')
    print('-' * 96)
    for tag in TAGS:
        v = per_tag[tag]
        if not v:
            continue
        n = len(v)
        c_ok = sum(1 for x in v if x[0] == '解对')
        c_bad = sum(1 for x in v if x[0] == '解错')
        c_fail = sum(1 for x in v if x[0] == '解不出')
        c_na = sum(1 for x in v if x[0] in ('锚', '无锚'))
        denom = c_ok + c_bad + c_fail
        rate = (c_ok / denom * 100) if denom else float('nan')
        sk = float(np.median([x[1]['n_slot_ok'] for x in v]))
        ck = float(np.median([x[1]['n_cell_ok'] for x in v]))
        er = float(np.median([x[1]['n_erase'] for x in v]))
        sc = float(np.median([x[1]['score_p50'] for x in v]))
        gp = float(np.median([x[1]['gap_p50'] for x in v]))
        lo = sum(1 for x in v if x[1]['loc_slot_ok']) / n * 100
        print(f'{tag:6s} {n:3d} {c_ok:5d} {c_bad:5d} {c_fail:6d} {c_na:5d} '
              f'{rate:6.1f}% {sk:11.1f} {ck:11.1f} {er:9.1f} {sc:8.4f} {gp:8.4f} '
              f'{lo:6.0f}%')

    print('\n  读法:')
    print('    命中率 = 解对 / (解对+解错+解不出)  —— 分母不含"锚"/"无锚"')
    print('    解错 是自信的错 ID, 单列; 它比解不出危险')
    print(f'    slot_ok = 拒识后仍可信的槽位数 (满分 16, 15 个 RS 槽 + 1 定位槽)')
    print(f'             门 = 6 路投票 >= {VOTE_MIN} 且 top1-top2 分差 > 0 (平票不算证据)')
    print('    cell_ok = 通过门的 6 路里同向的总格数 (满分 96) —— 相当于对方口径的 ncw')
    print('    erase   = 被拒识送去 RS(erasure) 的 RS 槽数; RS(15,5) 最多容 10 个')
    print('    分差    = top1-top2; 压到 0 就是纯噪声里挑 argmax, 此时必须拒')

    # 解错明细
    bad = [r for r in rows if r[3] == '解错']
    if bad:
        print(f'\n  解错明细 ({len(bad)} 条):')
        for (ds, stem, tag, _, r) in bad:
            print(f'    {tag:5s} {stem[:34]:34s} 解出 0x{r["wm_id"]:05X} '
                  f'slot_ok={r["n_slot_ok"]}')

    n_anch = sum(1 for (d, s, t, v, r) in rows if t == 'clean' and r['decode_ok'])
    n_cln = sum(1 for (d, s, t, v, r) in rows if t == 'clean')
    print(f'\n  clean 锚点: {n_anch}/{n_cln} 解出 '
          f'(锚失败的 stem, 其下各档一律记"无锚")')

    out = os.path.join(os.environ.get('DX_OUT_DIR', '.'), 'cross_tier_decode.json')
    import json
    with open(out, 'w') as f:
        json.dump([{'ds': d, 'stem': s, 'tag': t, 'verdict': v,
                    'wm_id': r['wm_id'], 'decode_ok': r['decode_ok'],
                    'n_slot_ok': r['n_slot_ok'], 'n_cell_ok': r['n_cell_ok'],
                    'n_erase': r['n_erase'], 'score_p50': r['score_p50'],
                    'gap_p50': r['gap_p50'], 'loc': r['loc_slot_ok']}
                   for (d, s, t, v, r) in rows], f, indent=1)
    print(f'明细已写 {out}')


if __name__ == '__main__':
    main()
