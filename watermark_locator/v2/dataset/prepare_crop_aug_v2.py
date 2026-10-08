#!/usr/bin/env python3
"""不同尺寸裁剪增广 —— 三数据集流水线的**第三步**，也是微调集。

    ① generate_dataset_v2.py --no_noise   →  clean    (无噪声)
    ② prepare_noise_from_clean_v2.py      →  noisy    (三噪声)
    ③ prepare_crop_aug_v2.py              →  cropped  (不同尺寸裁剪)  ← 本脚本

为什么要裁剪
------------
①② 的画面永远铺满 1920×1080，96 个码字的格点相对窗原点是**固定相位**。
网络很快学成"只在那 96 个固定位置找框"。v1 的实测是全图 F1 0.997 → 随机裁剪
F1 0.79，根因就是这个位置先验。裁剪把格点相位变成任意值，并逼网络学会检出
残缺码字。

为什么必须 pad 回 1920×1080（本脚本与 v1 ``prepare_crop_aug.py`` 的**唯一**实质区别）
----------------------------------------------------------------------------------
v1 把裁剪图按**原生尺寸**写盘，靠 loader 里的 ``SCALE_MODE=pad_native`` 在读入时
补回 1920×1080。v2 走 ultralytics 的 YOLO yaml，没有自己的 loader，于是这个 pad
只能在**生成时**做。

不做的后果是致命的 —— ``BaseDataset.load_image`` 会把图缩放到 ``max(h,w)==imgsz``，
**早于** ``LetterBox(scaleup=False)`` 执行，所以 ``scaleup=False`` 根本拦不住。实测
(ultralytics 8.4.157, imgsz=1920, train/val × rect 真假 × batch 1/3 全部一致)::

    源 500×400   -> 亮块(真值 160×135) 614×519   放大 3.84×
    源 961×600   -> 亮块              320×270   放大 2.00×
    源 1920×1080 -> 亮块              160×135   尺度 1.000  (唯一不被缩放的)

一旦被缩放，条纹周期 4 px（**渲染像素周期**）就被双线性重采样成 4k px 的平滑波，
码字 160s×135s 这把"像素间距尺"也就没了 —— 正是 DESIGN.md §12.4 严禁的那件事。
所以裁剪图**必须在写盘前就 1:1 贴回 1920×1080 载体画布**，全程零重采样。

GT 变换 + 残缺保留规则（**amodal 全框**）
----------------------------------------
::

    原图 GT (归一化于 1920x1080)
      -> 像素 xyxy
      -> 与裁剪窗求交，算可见比例
      -> 可见面积 >= KEEP_AREA_RATIO x 原面积
         且可见宽 >= KEEP_SIDE_RATIO x 原宽
         且可见高 >= KEEP_SIDE_RATIO x 原高      -> 保留
         否则                                   -> 丢弃
      -> 标注写**完整的原框**（不是裁剪后的框），平移 (+px-x0, +py-y0) 进画布

为什么要写**完整框**而不是裁剪后的框
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
裁缺码字的**裁剪框中心是偏的**，实测会把格点拟合直接拉垮：

====== ============== ============== =========
trial   全框拟合       仅完整框        渲染真值
====== ============== ============== =========
1       s=1.230        s=1.35590      1.35509
        rms=101.19     rms=0.27
4       s=1.35590      s=1.35590      1.35509
        rms=19.53      rms=0.23
====== ============== ============== =========

rms 从 0.2 px 劣化到 17~101 px，s 最多偏 9%，ID 当场解不出来。RANSAC 的
tol=3 px 挡不住这种偏移（裁缺 30% 的框中心就偏 30 px，仍会被收成内点）。

而保留规则要求可见宽高各 >= 50%，所以**框中心一定落在可见部分内** —— 于是写
完整框既拿到精确中心，又不会让 YOLO 的正样本格落到载体垫上。三个附带好处：

* 所有框都是 160s×135s，``majority_size``（间距初值）不被裁剪尺寸污染
* 残缺码字仍是正样本，网络在裁剪边缘不会学成"不响应"
* 格点间距这把尺子在裁剪后依旧精确，端到端解码才过得去

完整框超出 1920×1080 画布时**整条丢弃**（不裁剪标注，否则又把偏移引回来），
计数在 ``info['n_out_of_canvas']``。

输出
----
::

    output_dir/
        images/{train,val}/*.png     恒 1920x1080，r = 1.0，条纹恒 4 px
        labels/{train,val}/*.txt     YOLO `0 cx cy w h`
        meta/{train,val}/*.json      追加 crop / paste / n_dropped / truncated
        watermark.yaml
        manifest.json

用法::

    python prepare_crop_aug_v2.py --selftest
    python prepare_crop_aug_v2.py \\
        --src_root /data1/lpl/datasets_v2/clean \\
        --src_root /data1/lpl/datasets_v2/noisy \\
        --output_dir /data1/lpl/datasets_v2/cropped
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import generate_dataset_v2 as G  # noqa: E402
import predict_interval as PI  # noqa: E402
import prepare_noise_from_clean_v2 as N  # noqa: E402

SPLITS = ('train', 'val')

AREA_LO, AREA_HI = 0.25, 0.70   # 裁剪窗面积占源图比例
AR_RANGE = (0.5, 2.0)           # 裁剪窗宽高比，避免退化成细条
KEEP_AREA_RATIO = 0.5           # 可见面积 / 原框面积
KEEP_SIDE_RATIO = 0.5           # 可见宽、高 / 原框宽、高（= 8x8 格里保住 4x4）
MIN_KEEP_SIDE = 4               # 再加一道绝对下限，挡掉数值噪声

# 端到端解码的硬门槛下限：可见码字数低于它时 16 符号投票本身就不可靠，
# 解不出 ID 是数据量问题不是几何问题。实测 12 张 x 10 裁剪：
#   ncw>=16 → ID 解对 81/82 (99%)   ncw>=32 → 57/57 (100%)
#   ncw<=10 → 只有 ~80%，且拟合 s 也开始飘
DECODE_FLOOR = 16


# ───────────────────────── 裁剪窗 ─────────────────────────

def sample_crop_window(rng: np.random.RandomState,
                       src_w: int, src_h: int,
                       area_lo: float = AREA_LO, area_hi: float = AREA_HI,
                       ar_range: Tuple[float, float] = AR_RANGE
                       ) -> Tuple[int, int, int, int]:
    """采样一个完全落在源图里的裁剪窗 → ``(x0, y0, cw, ch)``。"""
    area = float(rng.uniform(area_lo, area_hi)) * src_w * src_h
    # AR 的可行域还要受"放得进源图"约束，否则反复重采样
    ar_lo, ar_hi = ar_range[0], ar_range[1]
    ar_hi = ar_range[1]
    for _ in range(64):
        ar = float(rng.uniform(ar_lo, ar_hi))
        cw = int(round(np.sqrt(area * ar)))
        ch = int(round(area / max(cw, 1)))
        if 8 <= cw <= src_w and 8 <= ch <= src_h:
            x0 = int(rng.randint(0, src_w - cw + 1))
            y0 = int(rng.randint(0, src_h - ch + 1))
            return x0, y0, cw, ch
        ar_hi = max(ar_lo, ar_hi * 0.92)
        ar_lo = min(ar_hi, ar_lo * 1.08)
    # 保底：居中 50% 面积窗
    cw, ch = src_w // 2, src_h // 2
    return src_w // 4, src_h // 4, cw, ch


def crop_to_window(img: np.ndarray,
                   labels: Sequence[Sequence[float]],
                   rng: np.random.RandomState,
                   carrier_paths: Optional[Sequence[str]] = None,
                   carrier_cache: Optional[dict] = None,
                   area_lo: float = AREA_LO,
                   area_hi: float = AREA_HI,
                   ar_range: Tuple[float, float] = AR_RANGE,
                   keep_amodal_in_canvas: bool = False
                   ) -> Tuple[np.ndarray, List[List[float]], dict]:
    """1:1 裁剪 + 1:1 贴回 1920×1080 载体画布。**全程零重采样**。

    返回 ``(out_img, out_labels, info)``；``out_labels`` 归一化于 (WINDOW_W, WINDOW_H)。
    """
    src_h, src_w = img.shape[:2]
    win_w, win_h = G.WINDOW_W, G.WINDOW_H
    x0, y0, cw, ch = sample_crop_window(rng, src_w, src_h,
                                        area_lo, area_hi, ar_range)
    crop = img[y0:y0 + ch, x0:x0 + cw]

    canvas = G._carrier_canvas(rng, carrier_paths, carrier_cache, win_w, win_h)

    # 先算保留集，**再**定贴装位置 —— amodal 全框必须整条落进画布，
    # 否则只能整条丢（裁短就把偏移引回来了）。所以贴装可行区间取决于
    # 每个保留框"戳出裁剪窗多远"(stick)。
    kept_raw = []        # (cls, ax0, ay0, ow, oh)
    sticks = []          # (左, 上, 右, 下) 戳出量
    box_sizes: List[List[float]] = []    # 保留框的像素宽高（应恒 = 160s x 135s）
    truncated = 0
    dropped_small = 0
    dropped_out = 0
    for cls, cx, cy, bw, bh in labels:
        ax0 = cx * src_w - bw * src_w / 2.0
        ay0 = cy * src_h - bh * src_h / 2.0
        ax1 = cx * src_w + bw * src_w / 2.0
        ay1 = cy * src_h + bh * src_h / 2.0
        ow, oh = ax1 - ax0, ay1 - ay0
        ix0, iy0 = max(ax0, x0), max(ay0, y0)
        ix1, iy1 = min(ax1, x0 + cw), min(ay1, y0 + ch)
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw <= 0 or ih <= 0:
            dropped_out += 1
            continue
        vis = (iw * ih) / max(ow * oh, 1e-9)
        if not (vis >= KEEP_AREA_RATIO
                and iw >= KEEP_SIDE_RATIO * ow and ih >= KEEP_SIDE_RATIO * oh
                and iw >= MIN_KEEP_SIDE and ih >= MIN_KEEP_SIDE):
            dropped_small += 1
            continue
        if iw < ow - 1e-6 or ih < oh - 1e-6:
            truncated += 1
        kept_raw.append((cls, ax0, ay0, ow, oh))
        sticks.append((max(0.0, x0 - ax0), max(0.0, y0 - ay0),
                       max(0.0, ax1 - (x0 + cw)), max(0.0, ay1 - (y0 + ch))))
        box_sizes.append([float(ow), float(oh)])

    # 可行贴装区间：全框要落进 1920x1080
    #   nx0 = ax0 - x0 + px >= 0        ->  px >= stick_left
    #   nx1 = ax1 - x0 + px <= win_w    ->  px <= (win_w - cw) - stick_right
    sl = max((s[0] for s in sticks), default=0.0)
    sr = max((s[2] for s in sticks), default=0.0)
    st = max((s[1] for s in sticks), default=0.0)
    sb = max((s[3] for s in sticks), default=0.0)
    px_lo, px_hi = sl, (win_w - cw) - sr
    py_lo, py_hi = st, (win_h - ch) - sb
    # 默认**全区间均匀采样**，和旧版逐位一致（保证磁盘上那棵树可复现）。
    # keep_amodal_in_canvas=True 才收窄到可行区间 —— 代价是贴装分布略偏
    # 离边，收益是裁缺框一个不丢。可复现性优先，所以默认关。
    # 两条分支都走 rng.randint，**各消耗同样多的随机数** —— 于是开不开
    # 开关，后续样本能逐对比较；顺带免掉 float 取整的边界麻烦。
    paste_policy = 'uniform'
    if keep_amodal_in_canvas and px_lo <= px_hi and py_lo <= py_hi:
        plo = min(max(int(np.ceil(px_lo)), 0), win_w - cw)
        phi = min(max(int(np.floor(px_hi)), 0), win_w - cw)
        qlo = min(max(int(np.ceil(py_lo)), 0), win_h - ch)
        qhi = min(max(int(np.floor(py_hi)), 0), win_h - ch)
        if plo <= phi and qlo <= qhi:
            px = int(rng.randint(plo, phi + 1))
            py = int(rng.randint(qlo, qhi + 1))
            paste_policy = 'amodal_feasible'
        else:
            px = int(rng.randint(0, win_w - cw + 1))
            py = int(rng.randint(0, win_h - ch + 1))
    else:
        px = int(rng.randint(0, win_w - cw + 1))
        py = int(rng.randint(0, win_h - ch + 1))
    canvas[py:py + ch, px:px + cw] = crop

    # 第二遍：贴装已定，按 amodal 全框写标签，放不进画布的整条丢
    kept: List[List[float]] = []
    kept_sizes: List[List[float]] = []
    out_of_canvas = 0
    for (cls, ax0, ay0, ow, oh), sz in zip(kept_raw, box_sizes):
        nx0 = ax0 - x0 + px
        ny0 = ay0 - y0 + py
        nx1, ny1 = nx0 + ow, ny0 + oh
        if nx0 < 0 or ny0 < 0 or nx1 > win_w or ny1 > win_h:
            # 完整框放不进画布。裁短它就又把偏移引回来，只能整条丢
            out_of_canvas += 1
            continue
        kept.append([cls,
                     (nx0 + nx1) / 2.0 / win_w,
                     (ny0 + ny1) / 2.0 / win_h,
                     (nx1 - nx0) / win_w,
                     (ny1 - ny0) / win_h])
        kept_sizes.append(sz)
    box_sizes = kept_sizes

    info = {
        'crop': [int(x0), int(y0), int(cw), int(ch)],
        'crop_area_ratio': float(cw * ch) / float(src_w * src_h),
        'paste': [int(px), int(py)],
        'paste_policy': paste_policy,
        'src': [int(src_w), int(src_h)],
        'window': [int(win_w), int(win_h)],
        'n_in': len(labels),
        'n_kept': len(kept),
        'n_truncated': int(truncated),
        'n_dropped_small': int(dropped_small),
        'n_dropped_out': int(dropped_out),
        'n_out_of_canvas': int(out_of_canvas),
        'box_wh_mean': ([float(np.mean(np.asarray(box_sizes, np.float64)[:, 0])),
                         float(np.mean(np.asarray(box_sizes, np.float64)[:, 1]))]
                        if box_sizes else None),
        'box_wh_std': ([float(np.std(np.asarray(box_sizes, np.float64)[:, 0])),
                        float(np.std(np.asarray(box_sizes, np.float64)[:, 1]))]
                       if box_sizes else None),
        'label_policy': 'amodal_full_box',
        # 码字格位在画布上的像素间距：裁剪只做平移，不改间距
        'pad_mode': 'carrier_canvas',
        'resample': False,
    }
    return canvas, kept, info


# ───────────────────────── 自检 ─────────────────────────

def selftest(carrier_paths: Sequence[str], n: int = 3,
             keep_amodal_in_canvas: bool = False) -> int:
    """几何必须靠**解码**兜底，不看框数。

    门槛：拿裁剪后写下的标签框当"完美检测"，跑格点拟合 + 解码，
    必须解出 meta 里的 ``watermark_id``。另外必须逐张断言输出是 1920×1080 ——
    否则 ultralytics 会在 loader 里把图放大，4 px 条纹当场报废。
    """
    print('=== prepare_crop_aug_v2 selftest ===')
    ok = True
    rng = np.random.RandomState(20260930)

    for k in range(n):
        img, labels, meta = G.make_sample(
            54321 + k, rng, list(carrier_paths), apply_noise=False)
        src_h, src_w = img.shape[:2]
        print(f'\n  -- 样本 {k}: 源 {src_w}x{src_h}  s={meta["s"]:.3f}  '
              f'框 {len(labels)}  id={meta["watermark_id"]}')

        # 基线：未裁剪必须先解对
        xyxy, conf = PI.load_yolo_labels_from_rows(
            [[int(b[0]), b[1], b[2], b[3], b[4]] for b in labels], src_w, src_h)
        base = PI.run(img, xyxy, conf, do_decode=True)
        if base.get('id') is None or int(base['id']) != int(meta['watermark_id']):
            print(f'  [FAIL] 未裁剪基线解不出正确 ID (id={base.get("id")})')
            ok = False
            continue
        print(f'  [OK]   未裁剪基线 ID={base["id"]}  s={base["s"]:.4f}')

        # 一次随机裁剪 + 一次强制大裁剪（保证硬门槛真被执行到，
        # 而不是全落在"数据量不足"的 WARN 分支里）
        # 三种裁剪：随机 / 强制大 / 强制宽。
        # 宽裁剪（ar 偏大）时 cw 接近 1920，贴装余量 win_w-cw 变小，
        # amodal 框才容易出画布 —— 不加这一档，出画布那条路自检永远踩不到。
        trials = ((AREA_LO, AREA_HI, AR_RANGE),
                  (0.60, 0.85, AR_RANGE),
                  (0.25, 0.80, (1.2, 2.0)))
        for trial, (alo, ahi, ar) in enumerate(trials):
            out, kept, info = crop_to_window(
                img, labels, rng, carrier_paths, {}, alo, ahi, ar,
                keep_amodal_in_canvas=keep_amodal_in_canvas)

            # 硬门槛 1：输出必须是 1920x1080，否则 loader 会放大
            if out.shape[:2] != (G.WINDOW_H, G.WINDOW_W):
                print(f'  [FAIL] 输出形状 {out.shape[:2]} != '
                      f'({G.WINDOW_H}, {G.WINDOW_W})')
                ok = False
                continue
            if info['resample'] is not False:
                print(f'  [FAIL] resample 标志不是 False')
                ok = False
                continue

            # 硬门槛 2：保留框的像素尺寸必须**逐个**等于源图上的 160s x 135s。
            #        amodal 标注下裁缺框也写完整尺寸，所以方差必须是 0 ——
            #        有任何缩放/裁短都会在这里露馅。
            b = np.asarray(labels, np.float64)
            sw, sh = b[:, 3].mean() * src_w, b[:, 4].mean() * src_h
            if not kept:
                print(f'  [WARN] 没有保留框，跳过尺度比对')
            else:
                # 每个保留框的尺寸都必须**等于源图里某个框的尺寸**。
                # 注意 96 个码字本来就差 ±1 px（rounded_boundary 取整到偶边界），
                # 所以不能要求方差为 0，只能要求"尺寸集合是源图尺寸集合的子集"。
                kb = np.asarray(kept, np.float64)
                kw, kh = kb[:, 3] * G.WINDOW_W, kb[:, 4] * G.WINDOW_H
                src_wh = np.stack([b[:, 3] * src_w, b[:, 4] * src_h], axis=1)
                worst = 0.0
                for w_i, h_i in zip(kw, kh):
                    d = np.hypot(src_wh[:, 0] - w_i, src_wh[:, 1] - h_i).min()
                    worst = max(worst, float(d))
                if worst > 0.02:
                    print(f'  [FAIL] 保留框尺寸偏离源图最远 {worst:.3f} px '
                          f'—— 裁剪不该改尺度 (均值 {sw:.1f}x{sh:.1f} -> '
                          f'{kw.mean():.1f}x{kh.mean():.1f})')
                    ok = False
                    continue

            if (keep_amodal_in_canvas
                    and info['n_out_of_canvas'] != 0
                    and info['n_truncated'] > 0):
                print(f'  [FAIL] keep_amodal_in_canvas 开着却丢了 '
                      f'{info["n_out_of_canvas"]} 个 amodal 框')
                ok = False
                continue
            print(f'  [OK]   裁剪 {info["crop"]} ({info["crop_area_ratio"]:.2f}) '
                  f'贴 {info["paste"]}  框 {info["n_in"]}->{info["n_kept"]} '
                  f'(残缺 {info["n_truncated"]} / 小 {info["n_dropped_small"]} '
                  f'/ 窗外 {info["n_dropped_out"]} '
                  f'/ 出画布 {info["n_out_of_canvas"]})')

            # 硬门槛 3：端到端解码。
            # 解码本身有数据量下限（16 符号投票 + RS(15,5)）：实测 ncw>=16 才
            # 稳定 99%，ncw<=10 不可靠。所以下限以上是**硬门槛**，以下是报告项 ——
            # 否则会把"裁得太狠"误报成几何错误。
            if len(kept) < 8:
                print(f'  [WARN] 保留 {len(kept)} 框 < 8，裁剪太狠，跳过解码')
                continue
            xyxy2, conf2 = PI.load_yolo_labels_from_rows(
                [[int(b[0]), b[1], b[2], b[3], b[4]] for b in kept],
                G.WINDOW_W, G.WINDOW_H)
            res = PI.run(out, xyxy2, conf2, do_decode=True)
            ncw = res.get('n_codewords') or 0
            _got, _want = res.get('id'), int(meta['watermark_id'])
            if ncw < DECODE_FLOOR:
                flag = 'OK  ' if _got == _want else 'WARN'
                print(f'  [{flag}] ncw={ncw} < {DECODE_FLOOR}，数据量不足：'
                      f'ID={_got} (真值 {_want}) —— 不作硬门槛')
                continue
            if _got is None or int(_got) != _want:
                print(f'  [FAIL] 裁剪后解出 ID={_got} != 真值 {_want}  '
                      f'(codewords={ncw})')
                ok = False
            else:
                print(f'  [OK]   裁剪后 ID={res["id"]}  s={res["s"]:.4f}  '
                      f'内点 {int(res["interval"]["n_inliers"])}  '
                      f'codewords={res["n_codewords"]}')
                # 硬门槛 4：拟合出的 s 必须等于渲染时的 s。
                # 这是"像素间距这把尺子没被重采样毁掉"的直接证据 ——
                # 裁剪只平移，格点间距在画布上必须还是 160s x 135s。
                if abs(float(res['s']) - float(meta['s'])) > 0.01 * float(meta['s']):
                    print(f'  [FAIL] 拟合 s={res["s"]:.4f} != 渲染 s={meta["s"]:.4f} '
                          f'(间距被改了)')
                    ok = False

    print('\n=== selftest:', 'PASS' if ok else 'FAIL', '===')
    return 0 if ok else 1


# ───────────────────────── 生成 ─────────────────────────

def collect_sources(src_roots: Sequence[str]
                    ) -> List[Tuple[str, str, str, str]]:
    """``[(src_root, split, stem, tag), ...]``。

    多棵树的 stem 不冲突（clean 无后缀，noisy 带 ``_wechat`` 等），直接平铺进
    同一个输出树即可；冲突时报错，不静默覆盖。
    """
    seen: Dict[Tuple[str, str], str] = {}
    out: List[Tuple[str, str, str, str]] = []
    for root in src_roots:
        tag = os.path.basename(os.path.normpath(root)) or 'src'
        for split, stem in N.collect_clean_stems(root):
            key = (split, stem)
            if key in seen:
                raise RuntimeError(
                    f'stem 冲突: {stem!r} 同时出现在 {seen[key]} 与 {root}')
            seen[key] = root
            out.append((root, split, stem, tag))
    return out


def _one(args_tuple):
    (src_root, split, stem, seed, out_root,
     carrier_paths, area_lo, area_hi, ar_lo, ar_hi) = args_tuple
    rng = np.random.RandomState(seed)
    img = cv2.imread(os.path.join(src_root, 'images', split, stem + '.png'))
    if img is None:
        raise RuntimeError(f'cannot read {src_root}/images/{split}/{stem}.png')
    labels = N.load_bboxes(os.path.join(src_root, 'labels', split, stem + '.txt'))

    cache: dict = {}
    out_img, kept, info = crop_to_window(
        img, labels, rng, carrier_paths, cache,
        area_lo, area_hi, (ar_lo, ar_hi))

    cv2.imwrite(os.path.join(out_root, 'images', split, stem + '.png'), out_img)
    N.save_bboxes(os.path.join(out_root, 'labels', split, stem + '.txt'), kept)

    meta = {}
    mp = os.path.join(src_root, 'meta', split, stem + '.json')
    if os.path.isfile(mp):
        with open(mp) as f:
            meta = json.load(f)
    meta['derived_from'] = stem
    meta['crop'] = info
    meta['n_boxes'] = len(kept)
    with open(os.path.join(out_root, 'meta', split, stem + '.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False)
    return split, info, len(kept)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description='v2 裁剪增广集：1:1 裁剪 + 1:1 贴回 1920x1080（绝不 resize）')
    p.add_argument('--src_root', action='append', default=None,
                   help='源树，可给多次（clean + noisy）')
    p.add_argument('--output_dir', type=str, default=None, help='省略时仅自检')
    p.add_argument('--carrier_root', type=str, default='/data1/lpl/datasets')
    p.add_argument('--carrier_path', type=str, default=None)
    p.add_argument('--seed', type=int, default=20260930)
    p.add_argument('--smoke', action='store_true', help='每 split 只取 6 张')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--area_lo', type=float, default=AREA_LO)
    p.add_argument('--area_hi', type=float, default=AREA_HI)
    p.add_argument('--ar_lo', type=float, default=AR_RANGE[0])
    p.add_argument('--ar_hi', type=float, default=AR_RANGE[1])
    p.add_argument('--jobs', type=int, default=1,
                   help='纯 CPU，可开；/data1 是 IO 瓶颈别开太大')
    p.add_argument('--keep_amodal_in_canvas', action='store_true',
                   help='贴装收窄到可行区间，裁缺框一个不丢（默认关）')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args(argv)

    if args.carrier_path and os.path.exists(args.carrier_path):
        carrier_paths = [args.carrier_path]
    else:
        carrier_paths = G.discover_carriers(args.carrier_root)
    print(f'carriers: {len(carrier_paths)} images')

    if args.selftest:
        return selftest(carrier_paths, n=3,
                        keep_amodal_in_canvas=args.keep_amodal_in_canvas)

    if not args.output_dir:
        p.error('--output_dir is required unless --selftest')

    if not args.src_root:
        p.error('--src_root is required unless --selftest')
    items = collect_sources(args.src_root)
    if not items:
        raise RuntimeError(f'no source images under {args.src_root}')
    if args.smoke:
        keep = []
        for split in SPLITS:
            keep += [it for it in items if it[1] == split][:6]
        items = keep
        print(f'smoke: 取 {len(items)} 张')

    tasks = []
    for i, (src_root, split, stem, _tag) in enumerate(items):
        out_png = os.path.join(args.output_dir, 'images', split, stem + '.png')
        if os.path.exists(out_png) and not args.overwrite:
            continue
        seed = (args.seed * 1_000_003 + i * 31 + 7) & 0x7FFFFFFF
        tasks.append((src_root, split, stem, seed, args.output_dir,
                      carrier_paths, args.area_lo, args.area_hi,
                      args.ar_lo, args.ar_hi))

    if not tasks:
        print('nothing to do (全部已存在；用 --overwrite 重跑)')
        return 0

    for split in SPLITS:
        for sub in ('images', 'labels', 'meta'):
            os.makedirs(os.path.join(args.output_dir, sub, split), exist_ok=True)

    print(f'{len(tasks)} jobs')
    t0 = time.time()
    counts = {'train': 0, 'val': 0}
    ratios: List[float] = []
    kept_hist: List[int] = []
    trunc_hist: List[int] = []
    n_in_hist: List[int] = []

    def _consume(res):
        split, info, nb = res
        counts[split] += 1
        ratios.append(info['crop_area_ratio'])
        kept_hist.append(nb)
        trunc_hist.append(info['n_truncated'])
        n_in_hist.append(info['n_in'])
        done = sum(counts.values())
        if done % 20 == 0 or done == len(tasks):
            print(f'  {done}/{len(tasks)}  {time.time() - t0:.0f}s  {counts}  '
                  f'面积比 {np.mean(ratios):.2f}  框 '
                  f'{np.mean(n_in_hist):.1f}->{np.mean(kept_hist):.1f}  '
                  f'残缺 {np.mean(trunc_hist):.1f}')

    if args.jobs > 1:
        import multiprocessing as mp
        # carrier_paths 里是路径，spawn 传得动；不要把 cache 传过去
        with mp.get_context('spawn').Pool(args.jobs) as pool:
            for res in pool.imap_unordered(_one, tasks, chunksize=1):
                _consume(res)
    else:
        for t in tasks:
            _consume(_one(t))

    with open(os.path.join(args.output_dir, 'watermark.yaml'), 'w') as f:
        f.write(
            '# Auto-generated YOLO dataset config (v2, 96 codewords) - 裁剪增广\n'
            '# 每张图都已 1:1 贴回 1920x1080：ultralytics 算出 r=1.0，只 pad 不缩放\n'
            f'path: {os.path.abspath(args.output_dir)}\n'
            'train: images/train\n'
            'val: images/val\n'
            '\n'
            'names:\n'
            '  0: codeword\n')

    manifest = {
        'generator': 'watermark_locator/v2/dataset/prepare_crop_aug_v2.py',
        'src_roots': [os.path.abspath(r) for r in args.src_root],
        'window': [G.WINDOW_W, G.WINDOW_H],
        'pad_mode': 'carrier_canvas',
        'resample': False,
        'why_pad': ('ultralytics BaseDataset.load_image 会把 max(h,w) 缩到 imgsz，'
                    '早于 LetterBox(scaleup=False)；小图存盘会被放大并重采样掉 '
                    '4px 条纹，所以必须在生成时贴回 1920x1080'),
        'area_range': [args.area_lo, args.area_hi],
        'ar_range': [args.ar_lo, args.ar_hi],
        'keep': {'area_ratio': KEEP_AREA_RATIO,
                 'side_ratio': KEEP_SIDE_RATIO,
                 'min_side_px': MIN_KEEP_SIDE},
        'counts': counts,
        'crop_area_ratio_mean': float(np.mean(ratios)) if ratios else None,
        'n_boxes_in_mean': float(np.mean(n_in_hist)) if n_in_hist else None,
        'n_boxes_kept_mean': float(np.mean(kept_hist)) if kept_hist else None,
        'n_truncated_mean': float(np.mean(trunc_hist)) if trunc_hist else None,
        'seed': args.seed,
    }
    with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f'\nwrote {args.output_dir}  {counts}  '
          f'{sum(n_in_hist)}->{sum(kept_hist)} boxes  {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
