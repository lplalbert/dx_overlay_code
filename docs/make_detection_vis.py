#!/usr/bin/env python
"""生成定位块检测效果图 + 验证集指标。

1) 测试图检测框可视化 (/data1/lpl/datasets/test, 3 图 x 3 噪声)
   每个 (样本, 噪声) 输出 4 张独立图 —— 不拼接:
     ① clean + GT   ② noisy + GT   ③ vv1 检测   ④ vv2 检测
2) 验证集 box-level 评测 (IoU 0.5 贪心匹配): tp/fp/fn / P / R / F1

权重取 v2 续训当前 best。输出:
  vis/report/detection/    原始效果图
  docs/figures/detection/  下采样副本 (宽 1280, 给文档插图)
  vis/report/val_metrics.json
"""
import json
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

import cv2
import numpy as np
import torch

REPO = '/data1/lpl/dx_overlay_code'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'watermark_locator'))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator', 'dataset'))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator', 'v1', 'vv1_unet', 'unet'))

from generate_dataset import (
    generate_one_sample, add_wechat_noise, add_pimog_noise,
    build_canvas, SCREEN_W, SCREEN_H,
)
from generate_locator_pattern import FIX_FG_MATRIX
from unet.unet_model import UNet

ALPHA = 0.032
SEED = 20260924
TEST_DIR = '/data1/lpl/datasets/test'
VAL_ROOT = '/data1/lpl/datasets_labeled_3noise/noisy'
OUT = os.path.join(REPO, 'vis/report/detection')
OUT_DOC = os.path.join(REPO, 'docs/figures/detection')
METRICS_PATH = os.path.join(REPO, 'vis/report/val_metrics.json')
LOCATOR_NPY = os.path.join(REPO, 'watermark_locator/locator_pattern.npy')
VV1_CKPT = os.path.join(REPO, 'output/v1_vv1_unet_v2/best_model_noisy_3noise_finetune.pth')
VV2_CKPT = os.path.join(REPO, 'runs/detect/output/v1_vv2_yolo_v2',
                        'yolo_noisy_3noise_finetune/weights/best.pt')
VV1_IN = (540, 960)

GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

NOISES = [
    ('wechat',       '单独微信压缩',   False),
    ('pimog',        '单独模拟拍照',   True),
    ('pimog_wechat', '拍照后微信压缩', True),
]


# ───────────────────────── 工具 ─────────────────────────

def apply_noise(image, mask, bboxes, kind, rng):
    if kind == 'wechat':
        image = add_wechat_noise(image)
    elif kind == 'pimog':
        image, mask, bboxes = add_pimog_noise(image, mask, bboxes, rng=rng)
    elif kind == 'pimog_wechat':
        image, mask, bboxes = add_pimog_noise(image, mask, bboxes, rng=rng)
        image = add_wechat_noise(image)
    else:
        raise ValueError(kind)
    return image, mask, bboxes


def norm_to_xyxy(b, w, h):
    _cls, cx, cy, bw, bh = b
    return (int(round((cx - bw / 2) * w)), int(round((cy - bh / 2) * h)),
            int(round((cx + bw / 2) * w)), int(round((cy + bh / 2) * h)))


def label_file_to_boxes(path, w, h):
    """YOLO 标签 `cls cx cy w h` (归一化) -> 绝对 xyxy 列表。"""
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            boxes.append(norm_to_xyxy([float(x) for x in parts[:5]], w, h))
    return boxes


def draw_boxes(img, boxes, color, thickness=3, tag='', confs=None):
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = tag
        if confs is not None:
            label = f'{tag} {confs[i]:.2f}' if tag else f'{confs[i]:.2f}'
        if label:
            cv2.putText(out, label, (x1 + 4, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def add_header(img, line1, line2):
    hdr = 56
    out = np.full((img.shape[0] + hdr, img.shape[1], 3), 24, np.uint8)
    out[hdr:] = img
    cv2.putText(out, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (190, 190, 190), 1, cv2.LINE_AA)
    return out


def save_both(img, path):
    """原图存 vis/report, 下采样副本存 docs/figures (宽 1280)。"""
    cv2.imwrite(path, img)
    h, w = img.shape[:2]
    tw = 1280
    if w > tw:
        img = cv2.resize(img, (tw, int(round(h * tw / w))), interpolation=cv2.INTER_AREA)
    doc_path = os.path.join(OUT_DOC, os.path.basename(path))
    cv2.imwrite(doc_path, img)


def mask_to_boxes(mask, min_area=200):
    m = (mask > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area and w > 4 and h > 4:
            boxes.append((x, y, x + w, y + h))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def match_boxes(gt, pred, thr=0.5):
    """贪心 IoU 匹配 -> (tp, fp, fn)。"""
    used = set()
    tp = 0
    for g in gt:
        best, best_i = -1, -1
        for i, p in enumerate(pred):
            if i in used:
                continue
            v = iou(g, p)
            if v > best:
                best, best_i = v, i
        if best >= thr and best_i >= 0:
            used.add(best_i)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ───────────────────────── 模型 ─────────────────────────

def load_vv1(device):
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    ckpt = torch.load(VV1_CKPT, map_location='cpu', weights_only=False)
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


def run_vv1(model, img_bgr, device):
    h, w = VV1_IN
    x = cv2.resize(img_bgr, (w, h)).astype(np.float32) / 255.0
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    with torch.no_grad():
        logits = model(t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    small = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(small, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_NEAREST)
    return mask


def load_vv2(device):
    from ultralytics import YOLO
    return YOLO(VV2_CKPT)


def run_vv2(model, img_bgr, device):
    r = model.predict(source=img_bgr, imgsz=640, conf=0.25, device=device, verbose=False)[0]
    boxes, confs = [], []
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
            confs.append(float(b.conf[0]))
    return boxes, confs


# ───────────────────────── 测试图可视化 ─────────────────────────

def run_test_vis(vv1, vv2, device):
    locator_pattern = np.load(LOCATOR_NPY)
    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')))
    assert len(files) == 3, f'expect 3 test images, got {len(files)}: {files}'

    summary = []
    for si, fname in enumerate(files, 1):
        src = cv2.imread(os.path.join(TEST_DIR, fname))
        assert src is not None, fname
        name = os.path.splitext(fname)[0]
        stem = f'{si:02d}_{name}'

        rng = np.random.RandomState(SEED + si)
        canvas = build_canvas([src], rng, SCREEN_W, SCREEN_H, grid=(1, 1))
        wm_id = int(rng.randint(0, 16 ** 5 - 1))
        clean, bboxes, mask, _ = generate_one_sample(
            wm_id, FIX_FG_MATRIX, locator_pattern, ALPHA, rng,
            carrier_img=canvas, apply_noise=False, channel_mode='b')
        gt_clean = [norm_to_xyxy(b, SCREEN_W, SCREEN_H) for b in bboxes]

        save_both(add_header(
            draw_boxes(clean, gt_clean, GREEN, 3, 'GT'),
            f'{si}  CLEAN watermark + GT labels     wm_id=0x{wm_id:05X}   alpha={ALPHA}',
            f'green = GT locator boxes   count={len(gt_clean)}'),
            os.path.join(OUT, f'{stem}_clean.png'))
        print(f'[{stem}]  wm_id=0x{wm_id:05X}')

        for ni, (noise_tag, noise_cn, need_rng) in enumerate(NOISES, 1):
            rng_n = np.random.RandomState(SEED + si * 100 + ni)
            noisy, mask_n, bboxes_n = apply_noise(
                clean.copy(), mask.copy(), [list(b) for b in bboxes],
                noise_tag, rng_n)
            gt_noisy = [norm_to_xyxy(b, SCREEN_W, SCREEN_H) for b in bboxes_n]

            save_both(add_header(
                draw_boxes(noisy, gt_noisy, GREEN, 3, 'GT'),
                f'{si}  NOISY [{noise_cn}] + GT labels',
                f'green = GT locator boxes   count={len(gt_noisy)}'),
                os.path.join(OUT, f'{stem}_{noise_tag}_noisy.png'))

            mask_pred = run_vv1(vv1, noisy, device)
            v1_boxes = mask_to_boxes(mask_pred)
            tp1, fp1, fn1 = match_boxes(gt_noisy, v1_boxes)
            save_both(add_header(
                draw_boxes(noisy, v1_boxes, MAGENTA, 3, 'pred'),
                f'{si}  vv1 U-Net [{noise_cn}] detected labels',
                f'magenta = predicted boxes   GT={len(gt_noisy)} pred={len(v1_boxes)}   '
                f'tp={tp1} fp={fp1} fn={fn1}'),
                os.path.join(OUT, f'{stem}_{noise_tag}_vv1.png'))

            v2_boxes, v2_confs = run_vv2(vv2, noisy, device)
            tp2, fp2, fn2 = match_boxes(gt_noisy, v2_boxes)
            save_both(add_header(
                draw_boxes(noisy, v2_boxes, MAGENTA, 3, 'pred', v2_confs),
                f'{si}  vv2 YOLOv8 [{noise_cn}] detected labels',
                f'magenta = predicted boxes   GT={len(gt_noisy)} pred={len(v2_boxes)}   '
                f'tp={tp2} fp={fp2} fn={fn2}'),
                os.path.join(OUT, f'{stem}_{noise_tag}_vv2.png'))

            print(f'  {noise_tag:12s}  vv1 tp/fp/fn={tp1}/{fp1}/{fn1}'
                  f'   vv2 tp/fp/fn={tp2}/{fp2}/{fn2}')
            summary.append({
                'stem': stem, 'noise': noise_tag, 'noise_cn': noise_cn,
                'wm_id': f'0x{wm_id:05X}',
                'gt': len(gt_noisy),
                'vv1': {'tp': tp1, 'fp': fp1, 'fn': fn1},
                'vv2': {'tp': tp2, 'fp': fp2, 'fn': fn2,
                        'confs': [round(c, 3) for c in v2_confs]},
            })
    return summary


# ───────────────────────── 验证集评测 ─────────────────────────

def iter_val():
    for ds in sorted(os.listdir(VAL_ROOT)):
        img_dir = os.path.join(VAL_ROOT, ds, 'val', 'images')
        lab_dir = os.path.join(VAL_ROOT, ds, 'val', 'labels')
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = os.path.splitext(fname)[0]
            yield (os.path.join(img_dir, fname),
                   os.path.join(lab_dir, stem + '.txt'))


def run_val_eval(vv1, vv2, device, max_images=0):
    """box-level 评测 (IoU 0.5)。max_images=0 表示全量。"""
    acc = {
        'vv1': {'tp': 0, 'fp': 0, 'fn': 0},
        'vv2': {'tp': 0, 'fp': 0, 'fn': 0},
    }
    per_ds = {}
    n = 0
    for img_path, lab_path in iter_val():
        if max_images and n >= max_images:
            break
        img = cv2.imread(img_path)
        if img is None:
            continue
        gt = label_file_to_boxes(lab_path, SCREEN_W, SCREEN_H)

        v1_boxes = mask_to_boxes(run_vv1(vv1, img, device))
        tp, fp, fn = match_boxes(gt, v1_boxes)
        acc['vv1']['tp'] += tp
        acc['vv1']['fp'] += fp
        acc['vv1']['fn'] += fn

        v2_boxes, _ = run_vv2(vv2, img, device)
        tp, fp, fn = match_boxes(gt, v2_boxes)
        acc['vv2']['tp'] += tp
        acc['vv2']['fp'] += fp
        acc['vv2']['fn'] += fn

        ds = os.path.relpath(img_path, VAL_ROOT).split(os.sep)[0]
        d = per_ds.setdefault(ds, {'n': 0, 'gt': 0,
                                   'vv1': {'tp': 0, 'fp': 0, 'fn': 0},
                                   'vv2': {'tp': 0, 'fp': 0, 'fn': 0}})
        d['n'] += 1
        d['gt'] += len(gt)
        for k, boxes in (('vv1', v1_boxes), ('vv2', v2_boxes)):
            tp, fp, fn = match_boxes(gt, boxes)
            d[k]['tp'] += tp
            d[k]['fp'] += fp
            d[k]['fn'] += fn

        n += 1
        if n % 100 == 0:
            print(f'  val {n} ...')

    out = {'n_images': n, 'overall': acc, 'per_dataset': {}}
    for k in ('vv1', 'vv2'):
        tp, fp, fn = acc[k]['tp'], acc[k]['fp'], acc[k]['fn']
        p, r, f1 = prf(tp, fp, fn)
        acc[k].update({'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)})
    for ds, d in per_ds.items():
        row = {'n': d['n'], 'gt': d['gt']}
        for k in ('vv1', 'vv2'):
            tp, fp, fn = d[k]['tp'], d[k]['fp'], d[k]['fn']
            p, r, f1 = prf(tp, fp, fn)
            row[k] = {**d[k], 'precision': round(p, 4), 'recall': round(r, 4),
                      'f1': round(f1, 4)}
        out['per_dataset'][ds] = row
    return out


# ───────────────────────── 主流程 ─────────────────────────

def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_DOC, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    print(f'vv1 ckpt: {VV1_CKPT}  exists={os.path.isfile(VV1_CKPT)}')
    print(f'vv2 ckpt: {VV2_CKPT}  exists={os.path.isfile(VV2_CKPT)}')

    vv1 = load_vv1(device)
    vv2 = load_vv2(device)

    print('\n===== 测试图检测框可视化 =====')
    test_summary = run_test_vis(vv1, vv2, device)

    print('\n===== 验证集 box-level 评测 (IoU 0.5) =====')
    val_metrics = run_val_eval(vv1, vv2, device)
    for k in ('vv1', 'vv2'):
        m = val_metrics['overall'][k]
        print(f"  {k}: P={m['precision']}  R={m['recall']}  F1={m['f1']}  "
              f"tp/fp/fn={m['tp']}/{m['fp']}/{m['fn']}")

    payload = {
        'alpha': ALPHA,
        'vv1_ckpt': VV1_CKPT,
        'vv2_ckpt': VV2_CKPT,
        'iou_threshold': 0.5,
        'test_vis': test_summary,
        'val_metrics': val_metrics,
    }
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\nmetrics: {METRICS_PATH}')
    print(f'images : {OUT}')
    print(f'doc    : {OUT_DOC}')


if __name__ == '__main__':
    main()
