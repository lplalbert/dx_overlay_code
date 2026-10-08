#!/usr/bin/env python3
"""端到端解 ID 验几何：把生成出来的标签真拿去格点+解码，比对 watermark_id。

为什么要这个脚本
----------------
"标签偏移写反时**图像完全不变**，数框/查形状永远绿" —— 这是踩过的坑。
数据生成链上有多处会平移标签：

  * ``generate_dataset_v2.frame_to_window``   1:1 裁剪 / 贴装
  * ``prepare_crop_aug_v2.crop_to_window``    裁剪窗平移 + 贴装平移

任何一处符号反了，输出的图和框数都完全正常，只有**解出来的 ID**会不对。
注意间距是平移不变量，整体平移不改框心差 —— 所以连 ``interval_px`` 都照样
准，不能拿它当几何正确性的证据。

判级：一份定义，别各写各的
--------------------------
逐张分级交给 ``predict_interval.judge_decode()``（本脚本和各 selftest 共用同一
份，否则同一批数据一个脚本过另一个不过）。五档：

========  ==========================================================
ok        解对了
floor     ncw < 16，数据量不足（16 符号投票 + RS(15,5) 的下限）
wipe      ncw 够、几何也对，但解不出 → 载体把 ±8/255 调制抹了
misdecode ncw 够、几何也对，但解**错**（RS 纠到错码字）—— 比 None 更危险
geom      s / interval_px 对不上 → 真是几何错
========  ==========================================================

三棵树的门槛不是同一把尺
------------------------
**geom —— 逐张硬卡，对所有树都成立。** s/间距对不上就是尺度被重采样改了。

**clean —— 看比率，不看单张。** 门槛 ``--min_clean_rate``。
  * 几何错是**系统性**的：偏移符号写反会让同批命中率塌到 ~0%，一抓一个准。
  * 载体抹调制是**散发**的：只挂一两张。实测 ``train_000036`` 96 码字、
    格点 96/96 内点、间距误差 0.0013%、框尺寸精确 160s×135s，仍 ``id=None``；
    同一 id 换 6 个别的载体 6/6 解出、换 6 个别的 id 也全解出 —— 不是编码也不是
    几何，是那一张载体的纹理抹了调制（见 wm-alpha-0032-chroma-limits）。
    逐张硬卡会把它误报成几何错误。
  * 所以解码这道门不能删（只有它抓得到偏移反号），只能按比率用。

**noisy —— 只报告，不设命中率。** 加噪的本意就是压低可读性，"加噪后解不出"
和"标签几何同步坏了"在解码层是同一个现象，解码分辨不了。几何仍然靠 geom
逐张卡；刚体平移型的标签错误（``_sync_labels_to_warp`` 偏移反号）只能靠
**大批量**收集的命中率看趋势，小样本自检的比率没有统计意义。

**cropped —— 条件化逐张，只对 clean 派生样本生效。**

    基线解不对 ⇒ 这张裁剪结果不携带几何信息，只报不卡（否则误报）
    基线解得对 ⇒ 同载体同 id，裁剪后必须也解对，否则只能赖裁剪 → FAIL

**为什么只限 clean 派生**：这个条件化的前提是"裁剪不该毁信号"，对**边缘样本
不成立**。实测 40 张 wechat 派生的裁剪：源命中 5、裁后命中 2 —— 裁剪吃掉了 5 个
里的 3 个边缘命中，而标签已证严格等于源标签平移（800/800，最大偏差 9.6e-04 px）。
所以那是**证据量效应**（可见码字变少 + amodal 框外沿盖在载体垫上），不是标签错。
clean 派生有充足裕度（裁后命中率 98.7%），条件化在那里才有意义。

对 noisy 派生的裁剪样本，几何靠两件事验，都不依赖解码：

  * ``geom`` 门槛（warp-aware 容差）
  * **平移恒等式**：裁剪标签必须逐字节等于源标签平移 —— 这条能抓到偏移符号
    写反，且与信噪比无关。见 ``/tmp/diag_amodal.py``（800/800 PASS）。

**为什么"id 不是 None"绝不能当通过**：``train_000133`` ncw=12 时 RS ``nfix=5``
纠出个**错 ID**（535912，真值 50612）。推理时它会给出一个看起来自信的错答案，
比报 None 危险得多。所以一律和真值比。

用法::

    python verify_decode.py --tree /data1/lpl/datasets_v2/clean --limit 200
    python verify_decode.py --tree /data1/lpl/datasets_v2/noisy \\
        --baseline_tree /data1/lpl/datasets_v2/clean
    python verify_decode.py --tree /data1/lpl/datasets_v2/cropped \\
        --baseline_tree /data1/lpl/datasets_v2/clean
"""

import argparse
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.dirname(HERE)
if V2_ROOT not in sys.path:
    sys.path.insert(0, V2_ROOT)

import predict_interval as PI  # noqa: E402

# 只有这些档计入"分母"（数据量够、能给出信息）
INFORMATIVE = ('ok', 'wipe', 'misdecode')


def iter_samples(tree):
    for split in ('train', 'val'):
        d = os.path.join(tree, 'images', split)
        if not os.path.isdir(d):
            continue
        for stem in sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))):
            yield {
                'split': split, 'stem': stem,
                'img': os.path.join(d, stem + '.png'),
                'lab': os.path.join(tree, 'labels', split, stem + '.txt'),
                'meta': os.path.join(tree, 'meta', split, stem + '.json'),
            }


def load_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                rows.append([int(p[0])] + [float(x) for x in p[1:5]])
    return rows


def load_meta(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:                                  # noqa: BLE001
        return {}


def evaluate(it, meta, rows, img):
    """→ (verdict, detail, ncw)。判级定义只有一份：``PI.judge_decode``。

    几何容差按**物理不确定度**取，不是拍脑袋：

    * pimog 拍照模拟是残余 0.5~5 px 的形变场，``meta['interval_px']`` 是**形变前**
      的真值，拿它当参照本身就只有这个精度。裁剪后框数变少、空间跨度变窄，
      格点拟合方差随之变大 —— 实测 12 张被判 ``geom`` 的样本，源图 ip_err
      0.02~0.56 px，裁后 1.07~1.90 px，是**连续分布贴着 1 px 阈值**（p50=0.28
      p90=1.34 max=3.39），不是双峰。
    * 而裁剪标签已单独验过严格等于源标签平移（800/800，最大偏差 9.6e-04 px），
      所以那些 ``geom`` 只能是拟合方差，不是标签错。
    * 真正的重采样错误是 2x~3.84x，s_err 是 100%+，宽松到 5% 照样抓得住。

    所以 warped 样本的容差放宽到形变量级，clean 保持严格。
    """
    wid = meta.get('watermark_id')
    h, w = img.shape[:2]
    xyxy, conf = PI.load_yolo_labels_from_rows(rows, w, h)
    res = PI.run(img, xyxy, conf, do_decode=True)
    noise = meta.get('noise') or []
    if isinstance(noise, str):
        noise = [noise]
    warped = any('pimog' in str(x) for x in noise)
    s_tol = 0.05 if warped else 0.01
    ip_tol = 6.0 if warped else 1.0
    verdict, detail = PI.judge_decode(res, wid, meta, s_tol=s_tol, ip_tol=ip_tol)
    return verdict, detail, int((res or {}).get('n_codewords') or 0)


def main(argv=None):
    p = argparse.ArgumentParser(description='端到端解 ID 验几何')
    p.add_argument('--tree', required=True,
                   help='数据集根（含 images/ labels/ meta/）')
    p.add_argument('--baseline_tree', action='append', default=None,
                   help='未裁剪的源树，可给多次。cropped 是 clean ∪ noisy 两棵树'
                        '裁出来的，基线必须两棵都给，否则 noisy 派生的样本会错拿 '
                        'clean 的同名 stem 当基线。给它才启用条件化门槛。')
    p.add_argument('--limit', type=int, default=0, help='只验前 N 张，0=全部')
    p.add_argument('--min_codewords', type=int, default=PI.DECODE_FLOOR,
                   help='低于此码字数的样本一律 floor，不进分母。'
                        '默认 16 = PI.DECODE_FLOOR。')
    p.add_argument('--min_clean_rate', type=float, default=0.95,
                   help='无噪声样本解码命中率下限。几何错系统性（塌到 ~0%%），'
                        '载体抹调制散发 —— 卡比率不逐张卡。')
    args = p.parse_args(argv)

    items = list(iter_samples(args.tree))
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise FileNotFoundError(f'no images under {args.tree}')

    base_roots = [os.path.abspath(b) for b in (args.baseline_tree or [])]
    for b in base_roots:
        if not os.path.isdir(b):
            raise FileNotFoundError(f'baseline_tree not found: {b}')

    print(f'tree     : {args.tree}')
    print(f'baseline : {base_roots or "(none)"}')
    print(f'N        : {len(items)}   判级 = predict_interval.judge_decode  '
          f'floor @ <{args.min_codewords} 码字  '
          f'clean 命中率门槛 {args.min_clean_rate * 100:.0f}%')
    print(f'\n{"split":<6s} {"stem":<34s} {"噪声":<14s} {"框":>4s} {"码字":>4s} '
          f'{"判级":<10s} {"基线":<10s} 判定')

    groups = {}          # noise tag -> {verdict: n}
    geom_failures = []   # 逐张硬卡：尺度/间距被改了
    cond_failures = []   # 基线 ok 但裁剪/加噪后不 ok
    clean_miss = []      # clean 里 ncw 够却没解对（wipe/misdecode）
    base_cache = {}

    for it in items:
        meta = load_meta(it['meta'])
        wid = meta.get('watermark_id')
        if wid is None:
            print(f'  [skip] {it["stem"]}: meta 无 watermark_id')
            continue

        noise = meta.get('noise') or []
        if isinstance(noise, str):
            noise = [noise]
        tag = '+'.join(str(x) for x in noise) if noise else '(clean)'
        is_clean = (tag == '(clean)')

        img = cv2.imread(it['img'])
        if img is None:
            print(f'  [skip] unreadable {it["img"]}')
            continue
        rows = load_rows(it['lab'])

        verdict, detail, ncw = evaluate(it, meta, rows, img)
        if ncw < args.min_codewords and verdict in INFORMATIVE:
            verdict = 'floor'      # 与 judge_decode 的地板一致，只是可调

        # ── 基线：条件化门槛的参照 ────────────────────────────────
        # cropped 的 stem 与源 stem **同名**（裁剪脚本不加后缀），所以基线就是
        # 在源树里按 stem 精确找。cropped 由 clean ∪ noisy 两棵树裁出，两棵都要给。
        base_v = None
        if base_roots:
            key = it['stem']
            base_v = base_cache.get(key)
            if base_v is None:
                src = meta.get('source_stem') or meta.get('derived_from')
                cands = [c for c in (src, it['stem']) if c]
                base_v = 'missing'
                for base_root in base_roots:
                    if base_v != 'missing':
                        break
                    for c in cands:
                        for split in ('train', 'val'):
                            ip = os.path.join(base_root, 'images', split,
                                              f'{c}.png')
                            if not os.path.exists(ip):
                                continue
                            bmeta = load_meta(os.path.join(
                                base_root, 'meta', split, f'{c}.json'))
                            bimg = cv2.imread(ip)
                            if bimg is None:
                                continue
                            brows = load_rows(os.path.join(
                                base_root, 'labels', split, f'{c}.txt'))
                            base_v, _, _ = evaluate(it, bmeta, brows, bimg)
                            break
                        if base_v != 'missing':
                            break
                base_cache[key] = base_v

        # ── 判定 ─────────────────────────────────────────────────
        st = groups.setdefault(tag, {})
        st[verdict] = st.get(verdict, 0) + 1

        if verdict == 'geom':
            ok = False
            verdict_out = 'FAIL(几何)'
            geom_failures.append(it['stem'])
        elif is_clean:
            # clean 派生：基线（未裁剪源）有充足裕度，裁剪不该毁信号 → 逐张条件化。
            # 但 floor（裁后码字数 < 16）是"证据不够判"，不是"裁错了"，不能算失败。
            if base_roots and base_v == 'ok' and verdict in ('wipe', 'misdecode'):
                ok = False
                cond_failures.append((it['stem'], verdict))
                verdict_out = f'FAIL(条件 {verdict})'
            elif verdict in ('wipe', 'misdecode'):
                ok = False
                clean_miss.append((it['stem'], verdict))
                verdict_out = '载体抹调制' if verdict == 'wipe' else '解错!'
            else:
                ok = True
                verdict_out = 'OK' if verdict == 'ok' else verdict
        elif base_roots:
            # noisy 派生：**只报告**，不作门槛。
            # 实测 40 张 wechat：源命中 5，裁后命中 2 —— 裁剪吃掉了 5 个里 3 个
            # 边缘命中，而标签已证严格等于源平移（800/800）。所以"基线 ok ⇒
            # 裁后必须 ok"这个前提对**边缘样本不成立**：裁剪减少证据、amodal
            # 框外沿又盖在载体垫上，本来就在解码边缘的样本会掉下去。这是
            # 证据量效应，不是标签错。几何靠上面的 geom 门槛 + 平移恒等式验。
            ok = True
            verdict_out = (f'{verdict}(未卡'
                           f'{",基线ok" if base_v == "ok" else ""})')
        else:
            ok = True
            verdict_out = f'{verdict}(未卡)'

        print(f'{it["split"]:<6s} {it["stem"]:<34s} {tag:<14s} {len(rows):4d} '
              f'{ncw:4d} {verdict:<10s} {str(base_v) if base_v else "-":<10s} '
              f'{verdict_out}')

    # ── 汇总 ─────────────────────────────────────────────────────
    print('\n' + '=' * 100)
    print(f'{"噪声":<14s} {"图":>5s} {"ok":>5s} {"wipe":>5s} {"misdec":>7s} '
          f'{"floor":>6s} {"geom":>5s} {"命中率*":>8s}')
    print('-' * 100)
    for tag in sorted(groups):
        st = groups[tag]
        n = sum(st.values())
        den = sum(st.get(k, 0) for k in INFORMATIVE)
        okn = st.get('ok', 0)
        rate = (okn / den * 100) if den else float('nan')
        print(f'{tag:<14s} {n:5d} {okn:5d} {st.get("wipe", 0):5d} '
              f'{st.get("misdecode", 0):7d} {st.get("floor", 0):6d} '
              f'{st.get("geom", 0):5d} {rate:7.1f}%')
    print('=' * 100)
    print('*命中率 = ok / (ok+wipe+misdecode)，floor 不进分母')

    clean = groups.get('(clean)', {})
    den_c = sum(clean.get(k, 0) for k in INFORMATIVE)
    ok_c = clean.get('ok', 0)
    rate_c = (ok_c / den_c) if den_c else float('nan')
    print(f'\ngeom 门槛（逐张，所有树）: 超容差 {len(geom_failures)} 张'
          f'{"  -> OK" if not geom_failures else "  -> FAIL"}')
    if geom_failures:
        print(f'  几何硬失败 stems: {geom_failures[:20]}')

    print(f'clean 解码门槛（比率）: {ok_c}/{den_c} = {rate_c * 100:.1f}%  '
          f'门槛 >= {args.min_clean_rate * 100:.0f}%')
    if clean_miss:
        print(f'  clean 未解对 ({len(clean_miss)})：'
              f'{[f"{s}:{v}" for s, v in clean_miss[:12]]}')
        print(f'    wipe=载体抹调制（散发，与几何无关）；'
              f'misdecode=RS 纠到错 ID（危险，推理会给自信错答案）')
    if base_roots:
        print(f'条件化门槛（基线 ok ⇒ 本树必须 ok）: 失败 {len(cond_failures)} 张'
              f'{"  -> OK" if not cond_failures else "  -> FAIL"}')
        if cond_failures:
            print(f'  条件失败 stems: {[f"{s}:{v}" for s, v in cond_failures[:20]]}')
    else:
        print('条件化门槛: 未启用（缺 --baseline_tree），noisy/cropped 只报告')

    # clean 树本身没有样本时（纯 noisy 树）clean 门槛不适用，只看 geom + 条件化
    clean_gate = (den_c == 0) or (rate_c >= args.min_clean_rate)
    ok = (not geom_failures and clean_gate
          and (not base_roots or not cond_failures))
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
