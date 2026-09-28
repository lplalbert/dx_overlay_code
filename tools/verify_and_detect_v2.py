#!/usr/bin/env python
"""v2 校验 + 检测:
1) 120x120 贴边角标, 逐字节 = 参考图缩放
2) 先标后水印 -> 角标区域的水印振幅完整 (max|Δ| ≈ 8, 与别处一致)
3) 跑 vv2, 报 6 个定位块各自 OK/MISS (重点看 #6, 它被 BR 标盖了 66.7%)
"""
import json
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import cv2
import numpy as np

# 仓库根 = 本文件上两级 (tools/ -> repo root)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARRIER_DIR = os.environ.get('DX_CARRIER_DIR', '/data1/lpl/datasets/test')
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'watermark_locator'))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator', 'dataset'))

from generate_dataset import generate_one_sample, build_canvas, SCREEN_W, SCREEN_H
from generate_locator_pattern import FIX_FG_MATRIX

OUT = os.path.join(REPO, 'vis', 'real_capture')
MARK_PNG = os.path.join(REPO, 'qr_loc_mark.png')
LOCATOR_NPY = os.path.join(REPO, 'watermark_locator', 'locator_pattern.npy')
VV2_CKPT = os.path.join(REPO, 'runs/detect/output/v1_vv2_yolo_v3',
                        'yolo_crop_aug_finetune/weights/best.pt')
ALPHA = 0.032
SEED = 20260928
MARK = 120
LOC = [(1120, 135, 160, 135), (480, 405, 160, 135), (1760, 405, 160, 135),
       (1120, 675, 160, 135), (480, 945, 160, 135), (1760, 945, 160, 135)]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def run_vv2(model, img, device, conf=0.25):
    """native 尺度推理 (绝对 640/1920), 避免 letterbox 尺度漂移。"""
    s = 640.0 / SCREEN_W
    w, h = int(round(img.shape[1] * s)), int(round(img.shape[0] * s))
    small = cv2.resize(img, (w, h))
    pw, ph = (w + 31) // 32 * 32, (h + 31) // 32 * 32
    canvas = np.full((ph, pw, 3), 114, np.uint8)
    canvas[:h, :w] = small
    r = model.predict(source=canvas, imgsz=max(pw, ph), conf=conf,
                      device=device, verbose=False)[0]
    boxes, confs = [], []
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append((int(round(x1 / s)), int(round(y1 / s)),
                          int(round(x2 / s)), int(round(y2 / s))))
            confs.append(float(b.conf[0]))
    return boxes, confs


def main():
    # 参考标
    ref = cv2.imread(MARK_PNG, cv2.IMREAD_GRAYSCALE)
    ref = cv2.resize(ref, (MARK, MARK), interpolation=cv2.INTER_AREA)
    ref = np.where(ref >= 128, 255, 0).astype(np.uint8)
    ref3 = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    man = json.load(open(os.path.join(OUT, 'manifest.json'), encoding='utf-8'))
    locator_pattern = np.load(LOCATOR_NPY)
    files = sorted(f for f in os.listdir(CARRIER_DIR) if f.endswith('.png'))

    canvases, markeds = {}, {}
    for si, fname in enumerate(files, 1):
        src = cv2.imread(os.path.join(CARRIER_DIR, fname))
        rng = np.random.RandomState(SEED + si)
        canvas = build_canvas([src], rng, SCREEN_W, SCREEN_H, grid=(1, 1))
        canvases[si] = canvas
        marked = canvas.copy()
        for (x, y) in [(0, 0), (SCREEN_W - MARK, 0),
                       (0, SCREEN_H - MARK), (SCREEN_W - MARK, SCREEN_H - MARK)]:
            marked[y:y + MARK, x:x + MARK] = ref3
        markeds[si] = marked

    # ── 1/2: 贴边角标 + 水印振幅 ──
    print('=== 1) 角标校验 ===')
    bad = 0
    for rec in man['images']:
        img = cv2.imread(os.path.join(OUT, rec['file']))
        assert img.shape == (SCREEN_H, SCREEN_W, 3)
        assert f'wm{rec["wm_id_6digit"]}_' in rec['file']
        for tag, (x1, y1, x2, y2) in rec['mark_rects'].items():
            # 贴边: 至少一边坐标为 0 或到边
            assert x1 == 0 or x2 == SCREEN_W, (tag, rec['mark_rects'][tag])
            assert y1 == 0 or y2 == SCREEN_H, (tag, rec['mark_rects'][tag])
            assert (x2 - x1, y2 - y1) == (MARK, MARK)
        # 角标区域有水印改动 (先标后水印 -> 不为 0)
        marked = markeds[rec['carrier_idx']]
        rng = np.random.RandomState(SEED + rec['carrier_idx'])
        clean, _, _, _ = generate_one_sample(
            rec['wm_id'], FIX_FG_MATRIX, locator_pattern, ALPHA, rng,
            carrier_img=marked, apply_noise=False, channel_mode='b')
        assert np.array_equal(img, clean), rec['file']   # 出图=复现结果
        for tag, (x1, y1, x2, y2) in rec['mark_rects'].items():
            d = np.abs(img[y1:y2, x1:x2].astype(int) - marked[y1:y2, x1:x2].astype(int))
            if d.max() == 0:
                print(f'  X {rec["file"]} 角标 {tag} 上无水印改动 (被遮挡?)')
                bad += 1
    print(f'  四角标 120x120 贴边 OK; 角标区域均有水印改动'
          if bad == 0 else f'  FAIL {bad}')

    print('\n=== 2) 水印振幅 (max|Δ|, 理论上限 8.16) ===')
    rec = man['images'][0]
    marked = markeds[rec['carrier_idx']]
    rng = np.random.RandomState(SEED + rec['carrier_idx'])
    clean, _, _, _ = generate_one_sample(
        rec['wm_id'], FIX_FG_MATRIX, locator_pattern, ALPHA, rng,
        carrier_img=marked, apply_noise=False, channel_mode='b')
    d = np.abs(clean.astype(int) - marked.astype(int)).max(axis=2)
    for tag, (x1, y1, x2, y2) in rec['mark_rects'].items():
        print(f'  角标 {tag} 区域   max|Δ|={d[y1:y2, x1:x2].max()}')
    for li, (x, y, w, h) in enumerate(LOC, 1):
        print(f'  定位块 #{li} 区域  max|Δ|={d[y:y+h, x:x+w].max()}'
              + ('   <- 被 BR 标盖 66.7%' if li == 6 else ''))

    # ── 3) vv2 检测 ──
    print('\n=== 3) vv2 检测 (conf=0.25, native 尺度) ===')
    import torch
    from ultralytics import YOLO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(VV2_CKPT)

    rows = []
    for rec in man['images']:
        img = cv2.imread(os.path.join(OUT, rec['file']))
        boxes, confs = run_vv2(model, img, device)
        gt = [tuple(r) for r in rec['gt_locator_xyxy']]
        used, hits = set(), []
        for gi, g in enumerate(gt, 1):
            best, bi = -1.0, -1
            for i, p in enumerate(boxes):
                if i in used:
                    continue
                v = iou(g, p)
                if v > best:
                    best, bi = v, i
            ok = best >= 0.5 and bi >= 0
            if ok:
                used.add(bi)
            hits.append((gi, ok, best, confs[bi] if bi >= 0 else -1.0))
        tp = sum(1 for h in hits if h[1])
        fp = len(boxes) - tp
        fn = 6 - tp
        mark = ''.join('O' if h[1] else 'X' for h in hits)
        print(f'  {rec["file"]}  [{mark}]  tp={tp} fp={fp} fn={fn}  '
              f'conf=' + ','.join(f'{c:.2f}' for c in confs))
        for gi, ok, best, c in hits:
            flag = '  <== BR 标覆盖 66.7%' if gi == 6 else ''
            print(f'      #{gi} {"OK " if ok else "MISS"}  iou={best:.2f}  '
                  f'conf={c:.2f}{flag}')
        rows.append({'file': rec['file'], 'tp': tp, 'fp': fp, 'fn': fn,
                     'per_block': {str(h[0]): h[1] for h in hits}})

    tp = sum(r['tp'] for r in rows); fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rr = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rr / (p + rr) if p + rr else 0.0
    print(f'\n  ALL  tp={tp} fp={fp} fn={fn}  P={p:.3f} R={rr:.3f} F1={f1:.3f}')
    for gi in range(1, 7):
        n_ok = sum(1 for r in rows if r['per_block'][str(gi)])
        print(f'    定位块 #{gi}: {n_ok}/{len(rows)} 命中'
              + ('   (被 BR 标盖 66.7%)' if gi == 6 else ''))

    with open(os.path.join(OUT, 'clean_detect_vv2.json'), 'w', encoding='utf-8') as f:
        json.dump({'conf': 0.25, 'mode': 'native', 'ckpt': VV2_CKPT,
                   'rows': rows,
                   'overall': {'tp': tp, 'fp': fp, 'fn': fn,
                               'precision': p, 'recall': rr, 'f1': f1}},
                  f, ensure_ascii=False, indent=2)
    print(f'\n  结果: {OUT}/clean_detect_vv2.json')


if __name__ == '__main__':
    main()
