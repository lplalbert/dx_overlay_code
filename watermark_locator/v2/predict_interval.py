#!/usr/bin/env python3
"""v2 推理：检测 → 格点 → **固定间距** → 水印 ID。

三层分工（DESIGN.md §5 / §11）：

1. **YOLO 只负责找出 96 个码字块**。框宽当不了尺子：IoU/CIoU 约束不到 0.5% 的
   精度，偏差在平均后仍然存在。所以检测头输出的框只当"这里有码字"用。
2. **间距在这里量**。把框中心交给 :func:`lattice.fit_lattice` 做确定性格点共识，
   得到 s / 间距 / 原点 / 倾角。单点定位散布 σ_c ≈ 2 px，96 点在 12 列上平均后
   σ_a = σ_c·√12/(M·√N) ≈ 0.06 px ≈ 0.04%（N=96, M=12）。这就是
   "固定间距先验"的用法：框宽给粗尺度（2–5%），格点给精密间距（0.02%）。
3. **ID 解码走** :func:`lattice.decode_patches`：格内条纹/非条纹对比 +
   4 相位搜索 + 每槽 6 份副本相干累积 + RS(15,5)。全程确定性，不学习、不改网络。

用法::

    # 走检测器
    python predict_interval.py --image shot.png --weights vv2/output/v2_vv2/yolo_single/weights/best.pt
    # 跳过检测（有现成 YOLO 标签），直接量间距/解 ID
    python predict_interval.py --image shot.png --labels shot.txt
    # 自检：不需要权重，用真值框验证"格点 → 间距 → ID"这一段
    python predict_interval.py --selftest

输出的 ``interval_px`` 是**码字间距** = 160s × 135s；``symbol_interval_px`` 是
同槽副本间距 = 640s × 540s。
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lattice  # noqa: E402

WINDOW_W, WINDOW_H = 1920, 1080


# ════════════════════════════════════════════════════════════════════
# 1. 检测
# ════════════════════════════════════════════════════════════════════

def detect(image_bgr, weights, conf=0.25, iou=0.45, imgsz=1920,
           max_det=512, device=None):
    """YOLO 推理 → (xyxy (N,4) float32, conf (N,) float32)。

    ``imgsz=1920`` 与训练一致：对 1920×1080 的图 letterbox r=1.0，**只 pad 不
    重采样**。缩小 imgsz 会把 4 px 的条纹周期下采样成 1.3 px 而混叠 —— 那是
    渲染像素周期，不是屏幕分辨率周期。
    """
    from ultralytics import YOLO
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f'weights not found: {weights}\n'
            f'  先跑 vv1/train.py 或 vv2/train.py，或传 --labels 跳过检测。')
    model = YOLO(weights)
    r = model.predict(source=image_bgr, imgsz=imgsz, conf=conf, iou=iou,
                      max_det=max_det, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    c = r.boxes.conf.cpu().numpy().astype(np.float32)
    return xyxy, c


def load_yolo_labels(path, img_w, img_h):
    """YOLO 标签 ``cls cx cy w h``（归一化）→ (xyxy, conf)。

    用来在没训练出权重时评估"格点 → 间距 → ID"这一段，
    或者对照真值框算检测的上限。
    """
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                rows.append([float(v) for v in p[:5]])
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    a = np.asarray(rows, np.float64)
    cx, cy, w, h = a[:, 1] * img_w, a[:, 2] * img_h, a[:, 3] * img_w, a[:, 4] * img_h
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return xyxy.astype(np.float32), np.ones(len(a), np.float32)


# ════════════════════════════════════════════════════════════════════
# 2. 格点 → 间距
# ════════════════════════════════════════════════════════════════════

def majority_size(xyxy, rel=0.12):
    """框宽/框高初值 = **覆盖面积最大的同尺寸簇**的中位数。

    码字框的尺寸高度一致（同一个 s 下就是 160s×135s），误报框尺寸任意。
    按**个数**取主簇是错的：20 个真码字 + 30 个 30×30 小误报时，误报簇反而更大，
    hint 被拉到 30px，格点拟合的 ``pitch_hint`` 一旦跑偏迭代就收不回来
    （实测间距算成 32.8px，误差 79%）。

    按**覆盖面积** n·median(w·h) 取簇才是对的：码字框铺满水印区域，
    误报框只占画面的零头 —— "占画面最多的那一档尺寸"就是我们要的尺度。
    20×155×131 ≈ 4.1e5 ≫ 30×30×30 = 2.7e4。
    """
    a = np.asarray(xyxy, np.float64).reshape(-1, 4)
    if a.shape[0] == 0:
        return float(lattice.CODEWORD_W), float(lattice.CODEWORD_H)
    w = a[:, 2] - a[:, 0]
    h = a[:, 3] - a[:, 1]
    area = np.maximum(w, 1.0) * np.maximum(h, 1.0)
    best_score, best_m = -1.0, np.ones(a.shape[0], dtype=bool)
    for i in range(a.shape[0]):
        m = (np.abs(w - w[i]) <= rel * w[i]) & (np.abs(h - h[i]) <= rel * h[i])
        score = float(area[m].sum())
        if score > best_score:
            best_score, best_m = score, m
    return float(np.median(w[best_m])), float(np.median(h[best_m]))


def estimate_lattice(xyxy, theta_span=6.0, tol=3.0):
    """框中心 → :class:`lattice.LatticeFit` + 内点掩码。

    ``pitch_hint`` 用框宽/框高的**主簇**中位数（:func:`majority_size`）。
    框宽当不了精密尺子（§5），当初值正好 —— 会被迭代精修掉，但初值本身
    不能被误报拖偏。``theta_span=6`` 覆盖 tile_crop 的 ±5° 旋转与屏摄残余
    单应：5° 在 1920 px 上漂 167 px ≈ 1 个 pitch，不搜角会全盘皆输。
    """
    xyxy = np.asarray(xyxy, np.float64)
    if xyxy.shape[0] < 2:
        raise ValueError(f'need ≥2 boxes to fit a lattice, got {xyxy.shape[0]}')
    centers = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0,
                        (xyxy[:, 1] + xyxy[:, 3]) / 2.0], axis=1)
    hint = majority_size(xyxy)
    fit = lattice.fit_lattice(centers, pitch_hint=hint, tol=tol,
                              theta_span=theta_span)
    return fit, fit.inliers, hint


def collect_codewords(image_bgr, xyxy, conf, fit, inliers=None, min_size=8):
    """内点框 → 每个格位切一个码字 patch。

    返回 ``(patches, boxes)``，``boxes[i] = (col, row, x0, y0, x1, y1)``，
    ``(x0,y0)`` 是 **patch 在原图里的整数原点** —— 条纹相位就是
    ``(x0+y0+φ) % 4``（见 :func:`lattice.stripe_phase_of`），所以这里必须是
    切图时真正用的那个原点，不能用框中心或别的近似。

    同一格位被多个框命中时只留置信度最高的那个。

    **不要按 0..11 / 0..7 过滤格位**：``fit.nearest`` 的原点是拟合出来的任意
    原点，不是网格左上角，相对下标可以是负的也可以 ≥12。``slot_of`` 对行列都是
    4 周期，而 :func:`lattice.align_and_decode_scores` 会把 16 种原点偏移全试一遍，
    所以下标本身无所谓 —— 按范围过滤只会白白丢掉一整圈码字（实测 96 框丢到剩 54）。
    """
    h, w = image_bgr.shape[:2]
    xyxy = np.asarray(xyxy, np.float64)
    conf = np.asarray(conf, np.float64)
    best = {}
    for i in range(xyxy.shape[0]):
        if inliers is not None and not bool(inliers[i]):
            continue
        cx = (xyxy[i, 0] + xyxy[i, 2]) / 2.0
        cy = (xyxy[i, 1] + xyxy[i, 3]) / 2.0
        j, k, rx, ry = fit.nearest(cx, cy)
        x0, y0, x1, y1 = fit.cell_window(j, k)
        # 出界的码字不完整，切出来按比例分格会歪，直接丢掉
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        if (x1 - x0) < min_size or (y1 - y0) < min_size:
            continue
        key = (int(j), int(k))
        if key not in best or conf[i] > best[key][0]:
            best[key] = (float(conf[i]), x0, y0, x1, y1)

    order = sorted(best)                      # (col,row) 行优先，输出可复现
    patches, boxes = [], []
    for (j, k) in order:
        _, x0, y0, x1, y1 = best[(j, k)]
        patches.append(image_bgr[y0:y1, x0:x1])
        boxes.append((int(j), int(k), int(x0), int(y0), int(x1), int(y1)))
    return patches, boxes


# ════════════════════════════════════════════════════════════════════
# 3. 一条流水线
# ════════════════════════════════════════════════════════════════════

def run(image_bgr, xyxy, conf, theta_span=6.0, tol=3.0, do_decode=True,
        stripe_offset=None):
    """检测框 → 间距 + (可选) 水印 ID。返回一个 dict。"""
    fit, inliers, hint = estimate_lattice(xyxy, theta_span=theta_span, tol=tol)
    info = lattice.read_interval(fit)
    info['n_detections'] = int(np.asarray(xyxy).shape[0])
    info['pitch_hint'] = [float(hint[0]), float(hint[1])]

    out = {
        'interval': info,
        'interval_px': info['interval_px'],
        'symbol_interval_px': info['symbol_interval_px'],
        's': info['s'],
        'theta_deg': info['theta_deg'],
        'id': None,
        'decode': None,
        'n_codewords': 0,
    }
    if not do_decode:
        return out

    patches, boxes = collect_codewords(image_bgr, xyxy, conf, fit, inliers)
    out['n_codewords'] = len(patches)
    if len(patches) < 8:
        out['decode'] = {'error': f'only {len(patches)} codewords, need ≥8'}
        return out

    # |theta| 大时轴对齐切片不再是码字本身，解码会跟着歪。
    # 间距估计不受影响（已把角搜进去了），这里只对 ID 把关。
    if abs(info['theta_deg']) > 1.0:
        out['decode'] = {
            'error': f'theta={info["theta_deg"]:+.2f}° too large for the '
                     f'axis-aligned decode crop (|θ|>1°); '
                     f'interval above is still valid'}
        return out

    dec = lattice.decode_patches(patches, boxes, stripe_offset=stripe_offset)
    dec['n_codewords'] = len(patches)
    dec['symbols_expected'] = lattice.SUB_ROWS * lattice.SUB_COLS
    out['decode'] = dec
    out['id'] = dec['id']
    return out


# ════════════════════════════════════════════════════════════════════
# 4. 报告 / 可视化
# ════════════════════════════════════════════════════════════════════

# 解码下限：16 符号投票 + RS(15,5) 的数据量门槛，属于**解码器**的性质，
# 所以放在这里而不是数据集脚本里。实测 12 张 x 10 裁剪：
#   ncw>=32 -> 57/57 (100%)   ncw>=16 -> 81/82 (99%)
#   ncw>=10 -> 84/87 (97%)    ncw>= 6 -> 85/109 (78%)
DECODE_FLOOR = 16


def judge_decode(res, want_id, meta, floor=DECODE_FLOOR, s_tol=0.01, ip_tol=1.0):
    """把一次 :func:`run` 的结果分级 —— "解不出 ID" 不止一种成因。

    实测 clean 树 150 张里 6 张解不对，**没有一张是几何错**：

    ==========  ==========================================================
    verdict     含义
    ==========  ==========================================================
    ok          解对了
    floor       ncw < floor，数据量不足（16 符号投票 + RS(15,5) 的下限）
    wipe        ncw 够、几何也对，但解不出 → 载体把 ±8/255 调制抹了
    misdecode   ncw 够、几何也对，但解**错**（RS 纠到错码字）—— 更严重
    geom        s / interval_px 对不上 → 真是几何错，逐张硬卡
    ==========  ==========================================================

    为什么解码门槛看**比率**而几何门槛逐张硬卡：

    * 几何错是**系统性**的 —— 同批塌到 0%，比率门槛一抓一个准
    * wipe / misdecode 是**散发**的 —— 挂一两张，逐张硬卡会误报
    * **偏移符号写反只有解码抓得到**：所有框平移 2*取景偏移，被
      ``LatticeFit`` 吸进 ``tx,ty``，``s``/``rms`` 照样漂亮。所以解码
      这道门不能删，只能按比率用。

    实测对照 ``clean/train_000036``：ncw=96、``n_obs`` 6-6、
    ``n_empty_slots``=0、s_err 0.026%、rms 0.35，``id`` 仍是 None ——
    96 个码字全读出来了且 16 槽全覆盖，是**符号读错**不是没读到。
    同批 ``train_000133`` 更糟：ncw=12 时 RS ``nfix=5`` 纠出个**错 ID**
    （535912，真值 50612）—— 所以"id 不是 None"绝不能当通过，
    必须和真值比。
    """
    if res is None:
        return 'geom', 'run() 返回 None'
    got = res.get('id')
    ncw = int(res.get('n_codewords') or 0)
    s_got = res.get('s')
    s_true = (meta or {}).get('s')
    s_err = abs(float(s_got) - float(s_true)) / float(s_true) if (
        s_got is not None and s_true) else float('nan')
    ip_got = res.get('interval_px') or ()
    ip_true = (meta or {}).get('interval_px') or ()
    ip_err = float('nan')
    if len(ip_got) == 2 and len(ip_true) == 2:
        ip_err = max(abs(float(a) - float(b)) for a, b in zip(ip_got, ip_true))

    def _txt(extra=''):
        tag = '解不出' if got is None else f'解错 {got} != {want_id}'
        return (f'{tag} ncw={ncw} s_err={s_err * 100:.3f}% '
                f'ip_err={ip_err:.3f}px{extra}')

    geom_ok = (s_err == s_err and s_err < s_tol and        # s_err != s_err -> NaN
               (ip_err != ip_err or ip_err < ip_tol))

    if got is not None and want_id is not None and int(got) == int(want_id):
        return 'ok', (f'ID={got} ncw={ncw} s_err={s_err * 100:.3f}% '
                      f'ip_err={ip_err:.3f}px')
    if ncw < floor:
        return 'floor', (f'ncw={ncw} < {floor} 数据量不足；' + _txt())
    if not geom_ok:
        return 'geom', (f'几何对不上 s_err={s_err * 100:.3f}% '
                        f'ip_err={ip_err:.3f}px ncw={ncw} id={got}')
    return ('wipe' if got is None else 'misdecode'), _txt(' 但几何正常')


def report(res, title='v2'):
    iv = res['interval']
    print(f'=== {title} · 固定间距 ===')
    print(f'  检测框        : {iv["n_detections"]}  '
          f'(pitch hint {iv["pitch_hint"][0]:.1f}×{iv["pitch_hint"][1]:.1f} px)')
    print(f'  格点内点      : {iv["n_inliers"]}   rms {iv["rms_px"]:.2f} px   '
          f'θ {iv["theta_deg"]:+.2f}°   origin ({iv["origin"][0]:.2f}, '
          f'{iv["origin"][1]:.2f})')
    print(f'  尺度          : s = {res["s"]:.5f}')
    print(f'  码字间距      : {res["interval_px"][0]:.3f} × {res["interval_px"][1]:.3f} px'
          f'   (= 160s × 135s)')
    print(f'  符号间距      : {res["symbol_interval_px"][0]:.3f} × '
          f'{res["symbol_interval_px"][1]:.3f} px   (= 640s × 540s)')

    d = res['decode']
    if d is None:
        print('=== 水印 ID ===\n  (跳过)')
        return
    if 'error' in d:
        print(f'=== 水印 ID ===\n  跳过：{d["error"]}')
        return
    seq = ' '.join(f'{v:X}' if v >= 0 else '-' for v in d['sequence'])
    print(f'=== 水印 ID ===')
    print(f'  码字          : {d["n_codewords"]}/{d["symbols_expected"]}'
          f'   空槽 {d["n_empty_slots"]}   原点偏移 shift={d["shift"]}')
    print(f'  条纹相位      : φ={d["stripe_phase"]}   evidence={d["evidence"]:.4g}')
    print(f'  序列 seq16    : {seq}')
    if d['id'] is None:
        print(f'  ID            : 解码失败 (RS 不收敛)')
    else:
        print(f'  ID            : 0x{d["id"]:05X} ({d["id"]})   '
              f'RS 纠错 nfix={d["nfix"]}')


def draw(image_bgr, xyxy, res, path):
    """把检测框、格点窗、解码符号画到图上。"""
    img = image_bgr.copy()
    for (x0, y0, x1, y1) in np.asarray(xyxy, np.float64):
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0), 1)
    d = res.get('decode') or {}
    # 格点窗：按间距把 8×12 格铺回图上，与解码用的 cell_window 一致
    s = res['s']
    cw = int(round(lattice.CODEWORD_W * s))
    ch = int(round(lattice.CODEWORD_H * s))
    tx, ty = res['interval']['origin']
    th = math.radians(res['theta_deg'])
    ct, st = math.cos(th), math.sin(th)
    for k in range(lattice.SUB_ROWS):
        for j in range(lattice.SUB_COLS):
            cx = tx + ct * j * lattice.CODEWORD_W * s - st * k * lattice.CODEWORD_H * s
            cy = ty + st * j * lattice.CODEWORD_W * s + ct * k * lattice.CODEWORD_H * s
            x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
            cv2.rectangle(img, (x0, y0), (x0 + cw, y0 + ch), (255, 120, 0), 1)
    syms = d.get('symbols') or []
    if syms and 'error' not in d:
        # symbols 与 collect_codewords 的输出同序，但 collect 会丢掉出界/重复格，
        # 所以这里只在能对上时写字 —— 画图是辅助，不追求 100% 标注。
        pass
    txt = f's={s:.4f}  interval={res["interval_px"][0]:.2f}x{res["interval_px"][1]:.2f}px'
    if res.get('id') is not None:
        txt += f'  ID=0x{res["id"]:05X}'
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# ════════════════════════════════════════════════════════════════════
# 5. CLI
# ════════════════════════════════════════════════════════════════════

def main(argv=None):
    p = argparse.ArgumentParser(
        description='v2 间距估计 + 水印 ID（检测 → 格点 → 固定间距）')
    p.add_argument('--image', help='输入图 (BGR，1:1 屏幕截图/取景)')
    p.add_argument('--weights', help='YOLO 权重 best.pt；给了 --labels 时可省')
    p.add_argument('--labels', help='YOLO 标签 txt（归一化），跳过检测器')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--imgsz', type=int, default=1920)
    p.add_argument('--max-det', type=int, default=512)
    p.add_argument('--device', default=None)
    p.add_argument('--theta-span', type=float, default=6.0,
                   help='格点拟合的角度搜索范围 (度)，默认 ±6')
    p.add_argument('--tol', type=float, default=3.0, help='格点内点判定阈值 (px)')
    p.add_argument('--no-decode', action='store_true', help='只量间距，不解 ID')
    p.add_argument('--stripe-offset', type=int, default=None,
                   help='(dx+dy)%4，已知取景偏移就填；默认搜 0..3')
    p.add_argument('--save', help='可视化输出路径 (png)')
    p.add_argument('--json', help='结果 JSON 输出路径')
    p.add_argument('--selftest', action='store_true',
                   help='不需要权重：生成样本，用真值框验证 格点→间距→ID')
    p.add_argument('--carriers', default='/data1/lpl/datasets',
                   help='--selftest 自然载体目录')
    args = p.parse_args(argv)

    if args.selftest:
        return selftest(args.carriers)

    if not args.image:
        p.error('--image is required (or use --selftest)')
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'cannot read image: {args.image}')

    if args.labels:
        xyxy, conf = load_yolo_labels(args.labels, img.shape[1], img.shape[0])
        print(f'检测: {len(xyxy)} 框  ← labels {args.labels}')
    else:
        if not args.weights:
            p.error('--weights is required unless --labels or --selftest')
        xyxy, conf = detect(img, args.weights, conf=args.conf, iou=args.iou,
                            imgsz=args.imgsz, max_det=args.max_det,
                            device=args.device)
        print(f'检测: {len(xyxy)} 框  conf≥{args.conf}  ← {args.weights}')

    if len(xyxy) < 2:
        print('检测框不足 2 个，无法拟合格点。')
        return 2

    res = run(img, xyxy, conf, theta_span=args.theta_span, tol=args.tol,
              do_decode=not args.no_decode, stripe_offset=args.stripe_offset)
    report(res, title=os.path.basename(args.image))
    if args.save:
        draw(img, xyxy, res, args.save)
        print(f'\n可视化 → {args.save}')
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False, default=float)
        print(f'JSON → {args.json}')
    return 0 if res['id'] is not None or args.no_decode else 1


# ════════════════════════════════════════════════════════════════════
# 6. 自检：不需要权重
# ════════════════════════════════════════════════════════════════════

def selftest(carriers=None):
    """用真值框（跳过检测器）验证 **格点 → 间距 → ID** 这一段。

    检测器那一段要等训练完才能验，这里验的是后面所有的确定性算法：
    间距精度、倾角搜索、相位搜索、相干累积、RS 纠错。
    """
    sys.path.insert(0, os.path.join(HERE, 'dataset'))
    import generate_dataset_v2 as G

    rng = np.random.RandomState(11)
    wid = 0x1E240
    ok = True
    ones = lambda n: np.ones(n, np.float32)        # noqa: E731

    print('— 真值框 → 格点 → 间距+ID (硬门槛：无噪声) —')
    for s_true in (0.67, 1.0, 1.33, 2.0):
        img, labels, meta = G.make_sample(
            wid, rng, G.discover_carriers(carriers) if carriers else None,
            apply_noise=False, s_min=s_true * 0.97, s_max=s_true * 1.03)
        xyxy, conf = load_yolo_labels_from_rows(labels, img.shape[1], img.shape[0])
        res = run(img, xyxy, conf, theta_span=6.0)
        e_w = abs(res['interval_px'][0] - meta['interval_px'][0]) / meta['interval_px'][0]
        e_h = abs(res['interval_px'][1] - meta['interval_px'][1]) / meta['interval_px'][1]
        d = res['decode'] or {}
        hit = (d.get('id') == wid)
        good = (e_w < 2e-3 and e_h < 2e-3 and hit)
        ok &= good
        print(f'  s≈{s_true:.2f}: 框 {len(xyxy)}  内点 {res["interval"]["n_inliers"]}  '
              f'间距 {res["interval_px"][0]:.3f}×{res["interval_px"][1]:.3f} '
              f'(真值 {meta["interval_px"][0]:.3f}×{meta["interval_px"][1]:.3f}, '
              f'误差 {max(e_w, e_h) * 100:.3f}%)  θ {res["theta_deg"]:+.2f}°  '
              f'ID {"0x%05X" % d["id"] if hit else "FAIL"}  {"OK" if good else "FAIL"}')

    print('— 含采集退化 (报告，不作门槛) —')
    # 这里不卡指标，原因要写清楚：
    #   * wechat   真 JPEG，非几何 —— 只是把 ±4 灰阶的调制压掉；
    #   * pimog    屏摄残余形变（physical_moire），**网格本身不再是刚性点阵**，
    #              框尺寸会飘 ±3%，刚性点阵拟合 rms 到 9px 是物理事实，不是 bug；
    #   * tile_crop 3x3 平铺+旋转，出界框被 clip 成 12px 宽的碎片，标签自己就脏。
    # 真部署是"整屏截图/拍照"，上面三条是训练增广的极限压力，不该拿来卡推理。
    for s_true in (1.0, 1.33):
        img, labels, meta = G.make_sample(
            wid, rng, G.discover_carriers(carriers) if carriers else None,
            apply_noise=True, s_min=s_true * 0.97, s_max=s_true * 1.03)
        xyxy, conf = load_yolo_labels_from_rows(labels, img.shape[1], img.shape[0])
        res = run(img, xyxy, conf, theta_span=6.0)
        e_w = abs(res['interval_px'][0] - meta['interval_px'][0]) / meta['interval_px'][0]
        d = res['decode'] or {}
        hit = (d.get('id') == wid)
        print(f'  s≈{s_true:.2f} noise={meta["noise"]}: 框 {len(xyxy)}  '
              f'内点 {res["interval"]["n_inliers"]}  间距误差 {e_w * 100:.3f}%  '
              f'θ {res["theta_deg"]:+.2f}°  '
              f'ID {"0x%05X" % d["id"] if hit else d.get("error", "miss")}  '
              f'{"ID-OK" if hit else "id-miss"}')

    print('— 间距精度 vs 检测框数（模拟漏检/误报）—')
    img, labels, meta = G.make_sample(
        wid, rng, G.discover_carriers(carriers) if carriers else None,
        apply_noise=False, s_min=0.97, s_max=1.03)
    xyxy, conf = load_yolo_labels_from_rows(labels, img.shape[1], img.shape[0])
    gt = meta['interval_px'][0]
    for n_keep, n_spur in ((96, 0), (60, 10), (36, 20), (20, 30)):
        n_keep = min(n_keep, len(xyxy))
        keep = np.sort(rng.choice(len(xyxy), size=n_keep, replace=False))
        a = xyxy[keep]
        if n_spur:
            sp = np.stack([
                rng.uniform(0, img.shape[1] - 40, n_spur),
                rng.uniform(0, img.shape[0] - 40, n_spur)], axis=1)
            sp = np.concatenate([sp, sp + 30], axis=1).astype(np.float32)
            a = np.concatenate([a, sp], axis=0)
        try:
            r2 = run(img, a, ones(len(a)), do_decode=False)
            err = abs(r2['interval_px'][0] - gt) / gt * 100
            good = err < 0.15
            print(f'  保留 {n_keep} / 误报 {n_spur}: 间距 {r2["interval_px"][0]:.3f} px  '
                  f'误差 {err:.3f}%  内点 {r2["interval"]["n_inliers"]}  '
                  f'{"OK" if good else "FAIL"}')
        except Exception as e:                      # noqa: BLE001
            good = False
            print(f'  保留 {n_keep} / 误报 {n_spur}: 异常 {e}  FAIL')
        ok &= good

    print('— 条纹相位：已知 Φ vs 4 相位搜索 —')
    # ``stripe_offset`` 就是全局 Φ：图像里真条纹 = ((X+Y+Φ)%4)<2，
    # 与切片原点无关（见 lattice.stripe_phase_of 的推导）。
    # 直接渲染的图 Φ=0，所以这里能钉死：只有 φ=0 能解出 ID。
    flat = np.full((WINDOW_H, WINDOW_W, 3), 128, np.uint8)
    img = G.alpha_blend_watermark(flat, G.render_template(wid, WINDOW_W, WINDOW_H),
                                  G.ALPHA)
    rows = G.to_yolo(G.codeword_boxes(WINDOW_W, WINDOW_H), WINDOW_W, WINDOW_H)
    xyxy, _ = load_yolo_labels_from_rows(rows, WINDOW_W, WINDOW_H)
    hit = {}
    for phi in (0, 1, 2, 3):
        d = (run(img, xyxy, np.ones(len(xyxy), np.float32), stripe_offset=phi)
             ['decode'] or {})
        hit[phi] = (d.get('id') == wid)
    d0 = run(img, xyxy, np.ones(len(xyxy), np.float32), stripe_offset=None)['decode'] or {}
    n_hit = sum(hit.values())
    good = (n_hit == 1 and hit[0] and d0.get('id') == wid
            and d0.get('stripe_phase') == 0)
    ok &= good
    print(f'  固定 Φ=0..3 → ID 命中 {["%d:%s" % (k, "Y" if v else "n") for k, v in hit.items()]}'
          f'  (应只有 Φ=0 命中={n_hit == 1})')
    print(f'  搜索 → φ={d0.get("stripe_phase")} (期望 0)  '
          f'ID {"0x%05X" % d0["id"] if d0.get("id") == wid else "FAIL"}  '
          f'{"OK" if good else "FAIL"}')

    print('— 取景偏移下相位跟着走 —')
    # 从更大的画面里按 (dx,dy) 1:1 裁进窗内 → Φ = (dx+dy) % 4
    BH, BW = WINDOW_H + 4, WINDOW_W + 4
    big = np.full((BH, BW, 3), 128, np.uint8)
    big = G.alpha_blend_watermark(big, G.render_template(wid, BW, BH), G.ALPHA)
    for dx, dy in ((1, 0), (0, 1), (2, 3), (3, 3)):
        crop = big[dy:dy + WINDOW_H, dx:dx + WINDOW_W].copy()
        rows = G.to_yolo(G.codeword_boxes(BW, BH), BW, BH)
        sh = []
        for cls, cx, cy, bw, bh in rows:            # 大画面坐标 → 窗坐标
            px, py = cx * BW - dx, cy * BH - dy
            sh.append([cls, px / WINDOW_W, py / WINDOW_H,
                       bw * BW / WINDOW_W, bh * BH / WINDOW_H])
        xy, _ = load_yolo_labels_from_rows(sh, WINDOW_W, WINDOW_H)
        m = ((xy[:, 0] >= 0) & (xy[:, 1] >= 0)
             & (xy[:, 2] <= WINDOW_W) & (xy[:, 3] <= WINDOW_H))
        xy = xy[m]
        d = run(crop, xy, ones(len(xy)), stripe_offset=None)['decode'] or {}
        want = (dx + dy) % 4
        good = (d.get('id') == wid and d.get('stripe_phase') == want)
        ok &= good
        print(f'  裁剪偏移 ({dx},{dy}) → Φ 应为 {want}: '
              f'搜索 φ={d.get("stripe_phase")}  '
              f'ID {"0x%05X" % d["id"] if d.get("id") == wid else "FAIL"}  '
              f'{"OK" if good else "FAIL"}')

    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


def load_yolo_labels_from_rows(rows, img_w, img_h):
    """``to_yolo`` 的输出（已归一化）→ (xyxy, conf)。"""
    a = np.asarray(rows, np.float64).reshape(-1, 5)
    cx, cy = a[:, 1] * img_w, a[:, 2] * img_h
    w, h = a[:, 3] * img_w, a[:, 4] * img_h
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return xyxy.astype(np.float32), np.ones(len(a), np.float32)


if __name__ == '__main__':
    sys.exit(main())
