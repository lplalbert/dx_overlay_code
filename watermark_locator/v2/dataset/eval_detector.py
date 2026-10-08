#!/usr/bin/env python3
"""拿**训练出来的检测器**端到端解 ID：检测 → 格点 → 间距 → 解码 → 对真值。

和 ``verify_decode.py`` 的分工
-----------------------------
``verify_decode.py`` 走 ``--labels``（真值框），验的是**标签几何对不对** ——
"偏移符号写反时图像完全不变，数框/查形状永远绿"，只有解码抓得到。那道门已经
过验（PASS）。

本脚本走 ``PI.detect``（**检测器输出**），验的是**训练出来的模型行不行**：
它找到的框够不够把码字读出来、读出来之后能不能解出 ID。这才是这次两阶段
训练（stage1 clean+noisy → stage2 cropped 微调）的验收。

判级定义仍然只有一份：``predict_interval.judge_decode``。两个脚本共用同一把
尺，否则同一批数据一个过另一个不过。五档 ok / floor / wipe / misdecode /
geom 含义不变。

但**几何硬卡在这里和判级脱钩**：``judge_decode`` 的判序是 floor 先于 geom，
所以 ncw<16 的样本即使 s_err 已经 25% 也只报 ``floor``（实测 epoch8 的
``train_000003`` 就是）。s / interval_px 来自格点拟合，和读出几个码字无关，
所以本脚本拿同一套容差直接卡 s_err / ip_err，``floor`` 也照样算几何硬失败。
``judge_decode`` 本身不动 —— 判级定义只有一份。

但**门槛多一道**，只有检测器这侧才有这个坑
----------------------------------------
真值框下 ncw 基本恒为 96，`floor` 很少见。检测器下框是找出来的，找不到就
ncw < 16 全落 ``floor`` —— 而 ``floor`` **不进分母**（它是"证据不够判"，
不是"解错了"）。于是检测器什么都不找时，分母为 0、命中率 NaN，
``den == 0`` 的那道"分母空就放行"会让门槛**空过**。

所以加第三道门：**informative 比例**（ncw ≥ floor 的样本占比）。
它卡的是"检测器到底给出证据了没有"，和"给了证据之后解对没有"分开看。
默认 0.90，这个数是**待实测修订**的下限，不是实测值 —— 首次跑完按分布改。

容差
----
检测器框心有抖动，格点拟合方差比真值框大。``s_tol`` / ``ip_tol`` 默认仍取
严格值（clean 的物理精度），但**逐张报告 s_err / ip_err 分布**，判 geom 之前
先看是不是贴着阈值的连续分布 —— 这是上次 12 张假 geom 的教训（真值框 + pimog
形变，容差比形变残余还窄）。真重采样错误是 2x~3.84x，s_err 100%+，宽松到 5%
照样抓得住。

用法::

    # 训练完先小样本看分布
    python eval_detector.py \\
        --weights ../runs/detect/output/v2_vv1/yolo_finetune/weights/best.pt \\
        --tree /data1/lpl/datasets_v2/clean --limit 200

    # 再全量验
    python eval_detector.py --weights .../best.pt \\
        --tree /data1/lpl/datasets_v2/clean \\
        --tree /data1/lpl/datasets_v2/cropped
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.dirname(HERE)
if V2_ROOT not in sys.path:
    sys.path.insert(0, V2_ROOT)

import predict_interval as PI  # noqa: E402

# 只有这些档计入"分母"（数据量够、能给出信息）
INFORMATIVE = ('ok', 'wipe', 'misdecode')


def make_detector(weights, conf=0.25, iou=0.45, imgsz=1920,
                  max_det=512, device=None):
    """→ 闭包 ``img_bgr -> (xyxy, conf)``，与 :func:`PI.detect` 同契约。

    唯一的差别是**权重只载一次**：``PI.detect`` 每次调用都 ``YOLO(weights)``,
    批量评估几千张会载几千次模型。letterbox 语义完全照搬（``imgsz=1920`` 对
    1920×1080 的图 r=1.0，只 pad 不重采样）。
    """
    from ultralytics import YOLO
    if not os.path.exists(weights):
        raise FileNotFoundError(f'weights not found: {weights}')
    model = YOLO(weights)

    def _run(image_bgr):
        r = model.predict(source=image_bgr, imgsz=imgsz, conf=conf, iou=iou,
                          max_det=max_det, device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        c = r.boxes.conf.cpu().numpy().astype(np.float32)
        return xyxy, c

    return _run


def iter_samples(tree):
    for split in ('train', 'val'):
        d = os.path.join(tree, 'images', split)
        if not os.path.isdir(d):
            continue
        for stem in sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))):
            yield {
                'split': split, 'stem': stem, 'tree': tree,
                'img': os.path.join(d, stem + '.png'),
                'meta': os.path.join(tree, 'meta', split, stem + '.json'),
            }


def load_meta(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:                                  # noqa: BLE001
        return {}


def noise_tag(meta):
    noise = meta.get('noise') or []
    if isinstance(noise, str):
        noise = [noise]
    return '+'.join(str(x) for x in noise) if noise else '(clean)'


def evaluate(it, meta, det, args):
    """→ (verdict, detail, ncw, n_det, s_err, ip_err, geom_bad)

    ``geom_bad`` 和 ``verdict == 'geom'`` **不是一回事**，必须分开：

    ``judge_decode`` 的判序是 ok → floor → geom → wipe/misdecode，
    ``floor``（ncw < 16）排在 ``geom`` **前面**。于是低召回 + 真几何错的样本
    会落进 ``floor``，硬卡几何的那道门根本不触发。实测 epoch8 的
    ``train_000003``：ncw=15 恰好卡在 floor 下面，s_err 却是 25.15%
    （0.75× 节距塌缩还在），``verdict`` 报 ``floor``，几何错被藏起来了。

    ``s`` / ``interval_px`` 来自格点拟合，和解出几个码字**无关**，所以几何
    该不该卡也不该受 floor 影响。``geom_bad`` 就是拿同一套容差直接卡
    s_err / ip_err，不管 verdict 是什么 —— 只加在检测器这侧，不改
    ``judge_decode`` 的判级定义（那仍然只有一份，``verify_decode.py`` 共用）。
    """
    wid = meta.get('watermark_id')
    img = cv2.imread(it['img'])
    if img is None:
        return 'geom', f'图读不出 {it["img"]}', 0, 0, float('nan'), float('nan')

    xyxy, conf = det(img)
    res = PI.run(img, xyxy, conf, do_decode=True)

    noise = meta.get('noise') or []
    if isinstance(noise, str):
        noise = [noise]
    warped = any('pimog' in str(x) for x in noise)
    s_tol = args.s_tol_warp if warped else args.s_tol
    ip_tol = args.ip_tol_warp if warped else args.ip_tol

    verdict, detail = PI.judge_decode(res, wid, meta, floor=args.min_codewords,
                                      s_tol=s_tol, ip_tol=ip_tol)

    # 分布要逐张记，判 geom 前先看是不是贴着阈值
    s_err = ip_err = float('nan')
    s_got = (res or {}).get('s')
    s_true = meta.get('s')
    if s_got is not None and s_true:
        s_err = abs(float(s_got) - float(s_true)) / float(s_true)
    ig = (res or {}).get('interval_px') or ()
    it_ = meta.get('interval_px') or ()
    if len(ig) == 2 and len(it_) == 2:
        ip_err = max(abs(float(a) - float(b)) for a, b in zip(ig, it_))

    # 几何硬卡独立于 verdict：s_err / ip_err 是 NaN 时无法判，不作数
    geom_bad = ((s_err == s_err and s_err >= s_tol) or
                (ip_err == ip_err and ip_err >= ip_tol))

    return verdict, detail, int((res or {}).get('n_codewords') or 0), \
        len(xyxy), s_err, ip_err, geom_bad


def main(argv=None):
    p = argparse.ArgumentParser(description='检测器端到端解 ID 验训练结果')
    p.add_argument('--weights', required=True, help='训练出的 best.pt')
    p.add_argument('--tree', action='append', required=True,
                   help='数据集根（含 images/ meta/），可给多次')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--imgsz', type=int, default=1920)
    p.add_argument('--max-det', type=int, default=512)
    p.add_argument('--device', default=None)
    p.add_argument('--limit', type=int, default=0, help='每棵树只验前 N 张，0=全部')
    p.add_argument('--min_codewords', type=int, default=PI.DECODE_FLOOR)
    p.add_argument('--min_clean_rate', type=float, default=0.95,
                   help='clean 解码命中率下限（ok / (ok+wipe+misdecode)）')
    p.add_argument('--min_informative', type=float, default=0.90,
                   help='ncw >= floor 的样本占比下限。**待实测修订**，'
                        '不是实测值。缺了这道门，检测器什么都不找时分母为 0，'
                        '命中率门槛会空过。')
    p.add_argument('--s_tol', type=float, default=0.01)
    p.add_argument('--ip_tol', type=float, default=1.0)
    p.add_argument('--s_tol_warp', type=float, default=0.05,
                   help='pimog 派生样本：形变残余 0.5~5px，meta 是形变前真值')
    p.add_argument('--ip_tol_warp', type=float, default=6.0)
    p.add_argument('--json', help='逐样本结果 JSON 输出')
    args = p.parse_args(argv)

    items = []
    for t in args.tree:
        if not os.path.isdir(t):
            raise FileNotFoundError(f'tree not found: {t}')
        items.extend(iter_samples(t))
    if args.limit:
        items = items[:args.limit] if len(args.tree) == 1 else items[:args.limit]
    if not items:
        raise FileNotFoundError('no images')

    det = make_detector(args.weights, conf=args.conf, iou=args.iou,
                        imgsz=args.imgsz, max_det=args.max_det,
                        device=args.device)

    print(f'weights : {args.weights}')
    print(f'trees   : {args.tree}')
    print(f'N       : {len(items)}   判级 = predict_interval.judge_decode')
    print(f'        conf={args.conf} iou={args.iou} imgsz={args.imgsz} '
          f'max_det={args.max_det}  floor @ <{args.min_codewords} 码字')
    print(f'        容差 clean s<{args.s_tol} ip<{args.ip_tol} | '
          f'warped s<{args.s_tol_warp} ip<{args.ip_tol_warp}')
    print()
    print(f'{"split":<6s} {"stem":<34s} {"噪声":<14s} {"检出":>4s} {"码字":>4s} '
          f'{"s_err%":>8s} {"ip_err":>7s} {"判级":<10s} 判定')

    groups = defaultdict(lambda: defaultdict(int))
    geom_failures = []
    clean_miss = []
    s_errs, ip_errs = [], []
    rows_out = []

    for it in items:
        meta = load_meta(it['meta'])
        wid = meta.get('watermark_id')
        if wid is None:
            print(f'  [skip] {it["stem"]}: meta 无 watermark_id')
            continue

        tag = noise_tag(meta)
        verdict, detail, ncw, n_det, s_err, ip_err, geom_bad = \
            evaluate(it, meta, det, args)
        if ncw < args.min_codewords and verdict in INFORMATIVE:
            verdict = 'floor'
        if s_err == s_err:
            s_errs.append(s_err)
        if ip_err == ip_err:
            ip_errs.append(ip_err)

        groups[tag][verdict] += 1
        rows_out.append({'stem': it['stem'], 'tree': it['tree'], 'tag': tag,
                         'want_id': wid, 'verdict': verdict, 'detail': detail,
                         'ncw': ncw, 'n_det': n_det, 'geom_bad': bool(geom_bad),
                         's_err': s_err, 'ip_err': ip_err})

        if geom_bad:
            geom_failures.append(it['stem'])
            # verdict 可能是 floor —— 判级说"证据不够"，但几何错照样算硬失败
            out = ('FAIL(几何)' if verdict == 'geom'
                   else f'FAIL(几何·被{verdict}藏)')
        elif tag == '(clean)' and verdict in ('wipe', 'misdecode'):
            clean_miss.append((it['stem'], verdict))
            out = '载体抹调制' if verdict == 'wipe' else '解错!'
        elif verdict in ('wipe', 'misdecode'):
            out = f'{verdict}(未卡)'
        else:
            out = 'OK' if verdict == 'ok' else verdict

        print(f'{it["split"]:<6s} {it["stem"]:<34s} {tag:<14s} {n_det:4d} '
              f'{ncw:4d} {s_err * 100:8.3f} {ip_err:7.3f} {verdict:<10s} {out}')

    # ── 汇总 ─────────────────────────────────────────────────────
    print('\n' + '=' * 104)
    print(f'{"噪声":<14s} {"图":>5s} {"ok":>5s} {"wipe":>5s} {"misdec":>7s} '
          f'{"floor":>6s} {"geom":>5s} {"inform":>7s} {"命中率*":>8s}')
    print('-' * 104)
    stat = {}
    for tag in sorted(groups):
        st = groups[tag]
        n = sum(st.values())
        den = sum(st.get(k, 0) for k in INFORMATIVE)
        okn = st.get('ok', 0)
        rate = (okn / den * 100) if den else float('nan')
        inf = den / n * 100 if n else float('nan')
        stat[tag] = dict(n=n, den=den, ok=okn, rate=rate / 100 if den else float('nan'),
                         informative=inf / 100 if n else float('nan'))
        print(f'{tag:<14s} {n:5d} {okn:5d} {st.get("wipe", 0):5d} '
              f'{st.get("misdecode", 0):7d} {st.get("floor", 0):6d} '
              f'{st.get("geom", 0):5d} {inf:6.1f}% {rate:7.1f}%')
    print('=' * 104)
    print('*命中率 = ok / (ok+wipe+misdecode)，floor 不进分母；'
          'inform = (ok+wipe+misdecode)/图，检测器给出证据的比例')

    if s_errs:
        a = np.array(s_errs) * 100
        b = np.array(ip_errs)
        print(f'\ns_err%   n={len(a)}  p50={np.median(a):.4f}  '
              f'p90={np.percentile(a, 90):.4f}  max={a.max():.4f}  '
              f'(阈值 clean {args.s_tol * 100:.1f} / warp {args.s_tol_warp * 100:.1f})')
        print(f'ip_err   n={len(b)}  p50={np.median(b):.3f}  '
              f'p90={np.percentile(b, 90):.3f}  max={b.max():.3f} px  '
              f'(阈值 clean {args.ip_tol} / warp {args.ip_tol_warp})')
        print('  → 判 geom 前先看这里：贴着阈值的连续分布 = 容差问题，'
              '大出量级 = 真几何错')

    # ── 门槛 ─────────────────────────────────────────────────────
    clean = stat.get('(clean)')
    print()
    print(f'geom 门槛（逐张，独立于判级）: 超容差 {len(geom_failures)} 张'
          f'{"  -> OK" if not geom_failures else "  -> FAIL"}')
    if geom_failures:
        print(f'  几何硬失败 stems: {geom_failures[:20]}')
        print('    含被 floor 藏起来的（ncw<16 但 s_err/ip_err 已超容差）—— '
              's 来自格点拟合，和读出几个码字无关')

    info_gate = True
    rate_gate = True
    if clean:
        print(f'clean informative 门槛: {clean["informative"] * 100:.1f}%  '
              f'门槛 >= {args.min_informative * 100:.0f}%'
              f'（检测器给出证据的比例，防空过）')
        info_gate = clean['informative'] >= args.min_informative
        print(f'clean 解码门槛: {clean["ok"]}/{clean["den"]} = '
              f'{clean["rate"] * 100:.1f}%  门槛 >= {args.min_clean_rate * 100:.0f}%')
        rate_gate = (clean['den'] == 0) or (clean['rate'] >= args.min_clean_rate)
        if clean_miss:
            print(f'  clean 未解对 ({len(clean_miss)}): '
                  f'{[f"{s}:{v}" for s, v in clean_miss[:12]]}')
            print('    wipe=载体抹调制；misdecode=RS 纠到错 ID（推理会给自信错答案）')
    else:
        print('clean 门槛: 未适用（没有 (clean) 样本）')

    ok = (not geom_failures) and info_gate and rate_gate
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'weights': args.weights, 'trees': args.tree,
                       'rows': rows_out, 'stat': stat}, f, ensure_ascii=False)
        print(f'\n逐样本结果 → {args.json}')
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
