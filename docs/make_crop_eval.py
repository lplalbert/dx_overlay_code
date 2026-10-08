#!/usr/bin/env python
"""验证集小规模评测: 随机截取 25%-70% 面积后的定位块识别。

**关键 1 —— GT 必须随裁剪窗口一起变换**, 否则统计全是错的:
    原图 GT (绝对 xyxy)
        -> 平移 (-x0, -y0) 到裁剪图坐标系
        -> 与裁剪窗求交 (裁掉窗外部分)
    完全落在窗内   -> gt_full     参与 P/R/F1 统计
    部分落在窗内   -> gt_partial  ignore 区域:
                        既不算漏检 (残缺定位块不要求检出),
                        命中它的预测也不算误检 (避免惩罚"它确实在那儿")
    完全在窗外     -> 丢弃

**关键 2 —— 推理必须保持定位块的绝对尺度**, 否则测的是尺度失配不是遮挡:
    定位块在屏幕上恒为 160x135 px。训练时 1920x1080 -> 960x540,
    网输入里定位块固定是 80x67。若把裁剪图**拉伸**到 960x540,
    定位块会随裁剪面积放大 1.4~2.7 倍, U-Net 直接失灵。

    native 模式 (默认, 正确): 不做 padding, 按训练的绝对比例缩放后直接前向。
        vv1 U-Net 是全卷积, 裁剪图 x0.5 后吃任意尺寸;
        vv2 先手动 x(640/1920)=1/3, 再贴到 stride 对齐的小画布,
        杜绝 ultralytics letterbox 的二次缩放。
        这是"原生分辨率只截到屏幕一部分"的真实部署方式。
    canvas 模式 (对照): 裁剪图按原生分辨率贴到 1920x1080 (其余填 114)。
        尺度对了, 但内容/灰的硬直角边界本身是强角点伪影, 会污染结果。
    stretch 模式 (对照): 裁剪图直接拉伸到网输入尺寸 —— 即"随手裁完就喂"
        会发生的事, 用来量化尺度失配单独造成了多少损失。

匹配口径: IoU >= 0.5 贪心匹配 gt_full; 剩余预测若与任一 gt_partial 的
IoU >= 0.3 则忽略, 否则计误报。

阶梯设计: 5 个面积占比区间 x 60 张。同一批 60 张基图配对比较
(每阶梯独立随机裁剪窗口), 这样阶梯间的差异可归因于裁剪大小而非底图内容。

用法:
    CUDA_VISIBLE_DEVICES=2 python docs/make_crop_eval.py              # native
    MODE=canvas  CUDA_VISIBLE_DEVICES=2 python docs/make_crop_eval.py # 对照
    MODE=stretch CUDA_VISIBLE_DEVICES=2 python docs/make_crop_eval.py # 对照
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

from generate_dataset import SCREEN_W, SCREEN_H
from unet.unet_model import UNet

VAL_ROOT = '/data1/lpl/datasets_labeled_3noise/noisy'
MODE = os.environ.get('MODE', 'native')          # native (正确) | canvas | stretch
PAD_FILL = 114                                    # YOLO letterbox 标准填充值
_OUT_SUFFIX = {'native': '', 'canvas': '_canvas', 'stretch': '_stretch'}[MODE]
OUT_JSON = os.path.join(REPO, f'vis/report/crop_eval{_OUT_SUFFIX}.json')
OUT_VIS = os.path.join(REPO, f'vis/report/crop_eval{_OUT_SUFFIX}')
VV1_CKPT = os.path.join(REPO, 'output/v1_vv1_unet_v2/best_model_noisy_3noise_finetune.pth')
VV2_CKPT = os.path.join(REPO, 'runs/detect/output/v1_vv2_yolo_v2',
                        'yolo_noisy_3noise_finetune/weights/best.pt')
VV1_IN = (540, 960)

SEED = 20260925
N_PER_TIER = 60
IOU_MATCH = 0.5
IOU_IGNORE = 0.3
FULL_TOL = 0.98          # 面积保留 >= 98% 视为完全可见 (容忍 1px 取整)
AR_RANGE = (0.5, 2.0)    # 裁剪窗宽高比范围, 避免退化成细条
N_VIS_PER_TIER = 2       # 每阶梯存几张示例

# 面积占比阶梯 [lo, hi)
TIERS = [
    ('25-34%', 0.25, 0.34),
    ('34-43%', 0.34, 0.43),
    ('43-52%', 0.43, 0.52),
    ('52-61%', 0.52, 0.61),
    ('61-70%', 0.61, 0.70),
]

GREEN = (0, 255, 0)        # gt_full
YELLOW = (0, 255, 255)     # gt_partial (ignore)
MAGENTA = (255, 0, 255)    # vv1 预测
CYAN = (255, 255, 0)       # vv2 预测


# ───────────────────────── 工具 ─────────────────────────

def label_file_to_boxes(path, w=SCREEN_W, h=SCREEN_H):
    """YOLO 标签 `cls cx cy w h` (归一化) -> 绝对 xyxy (int)。"""
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            _cls, cx, cy, bw, bh = (float(x) for x in p[:5])
            boxes.append((int(round((cx - bw / 2) * w)), int(round((cy - bh / 2) * h)),
                          int(round((cx + bw / 2) * w)), int(round((cy + bh / 2) * h))))
    return boxes


def sample_crop_window(ratio_lo, ratio_hi, rng):
    """按面积占比随机取裁剪窗 (x0, y0, cw, ch)。

    宽高比在可行范围 [r*W/H, (W/H)/r] 与 AR_RANGE 之交集内采样,
    保证面积精确落在目标占比上、且不超出画布。
    """
    H, W = SCREEN_H, SCREEN_W
    r = rng.uniform(ratio_lo, ratio_hi)
    ar_lo = max(AR_RANGE[0], r * W / H)   # 保 ch <= H
    ar_hi = min(AR_RANGE[1], (W / H) / r)  # 保 cw <= W
    if ar_lo >= ar_hi:                     # 退化: 取可行中点
        ar = 0.5 * (max(AR_RANGE[0], r * W / H) + min(AR_RANGE[1], (W / H) / r))
    else:
        ar = rng.uniform(ar_lo, ar_hi)
    area = r * W * H
    cw = int(round(np.sqrt(area * ar)))
    ch = int(round(cw / ar))
    cw = max(1, min(cw, W))
    ch = max(1, min(ch, H))
    x0 = int(rng.randint(0, W - cw + 1))
    y0 = int(rng.randint(0, H - ch + 1))
    return x0, y0, cw, ch, (cw * ch) / float(W * H)


def split_gt_after_crop(gt_abs, x0, y0, cw, ch):
    """GT 随裁剪变换 -> (gt_full, gt_partial), 均为裁剪图坐标系下的 xyxy。"""
    full, partial = [], []
    for (x1, y1, x2, y2) in gt_abs:
        orig = max(0, x2 - x1) * max(0, y2 - y1)
        if orig <= 0:
            continue
        ix1, iy1 = max(x1, x0), max(y1, y0)
        ix2, iy2 = min(x2, x0 + cw), min(y2, y0 + ch)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            continue                                    # 完全在窗外
        box = (ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0)
        if inter >= FULL_TOL * orig:
            full.append((x1 - x0, y1 - y0, x2 - x0, y2 - y0))  # 未裁剪的原始框
        else:
            partial.append(box)
    return full, partial


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def match_with_ignore(gt_full, gt_partial, preds):
    """(tp, fp, fn, n_ignored, n_partial_hit)。"""
    used = set()
    tp = 0
    for g in gt_full:
        best, best_i = -1.0, -1
        for i, p in enumerate(preds):
            if i in used:
                continue
            v = iou(g, p)
            if v > best:
                best, best_i = v, i
        if best >= IOU_MATCH and best_i >= 0:
            used.add(best_i)
            tp += 1
    fn = len(gt_full) - tp

    n_ignored = 0
    for i, p in enumerate(preds):
        if i in used:
            continue
        if any(iou(p, g) >= IOU_IGNORE for g in gt_partial):
            n_ignored += 1
        # 否则 -> 误报 (fp)
    fp = len(preds) - tp - n_ignored

    n_partial_hit = sum(
        1 for g in gt_partial
        if any(iou(g, p) >= IOU_IGNORE for p in preds))
    return tp, fp, fn, n_ignored, n_partial_hit


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def draw_boxes(img, boxes, color, thickness=3, tag='', confs=None):
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = tag
        if confs is not None:
            label = f'{tag} {confs[i]:.2f}' if tag else f'{confs[i]:.2f}'
        if label:
            cv2.putText(out, label, (x1 + 4, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def add_header(img, line1, line2):
    hdr = 56
    out = np.full((img.shape[0] + hdr, img.shape[1], 3), 24, np.uint8)
    out[hdr:] = img
    cv2.putText(out, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (190, 190, 190), 1, cv2.LINE_AA)
    return out


def to_canvas(crop):
    """裁剪图按原生分辨率贴到 1920x1080 画布 (左上角), 其余填 PAD_FILL。

    这样定位块仍是 160x135, 走标准推理后网输入里固定 80x67 —— 与训练一致。
    预测坐标天然等于裁剪图坐标 (贴在 0,0)。
    """
    canvas = np.full((SCREEN_H, SCREEN_W, 3), PAD_FILL, np.uint8)
    h, w = crop.shape[:2]
    canvas[:h, :w] = crop
    return canvas


def clip_boxes(boxes, w, h):
    """把预测框钳制到裁剪图范围内 (canvas 模式下会伸进 padding 区)。

    只钳制坐标、不丢框, 这样 vv2 的 confs 下标依然对齐。
    """
    return [(min(max(x1, 0), w), min(max(y1, 0), h),
             min(max(x2, 0), w), min(max(y2, 0), h))
            for x1, y1, x2, y2 in boxes]


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
    """注意: 掩码放大回**裁剪图自身尺寸** (不是 1920x1080)。"""
    h0, w0 = img_bgr.shape[:2]
    h, w = VV1_IN
    x = cv2.resize(img_bgr, (w, h)).astype(np.float32) / 255.0
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    small = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(small, (w0, h0), interpolation=cv2.INTER_NEAREST)


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


# ───────────── native 模式: 定尺度、无大幅 padding ─────────────

def run_vv1_native(model, img_bgr, device):
    """U-Net 全卷积: 裁剪图 x0.5 后直接前向 (训练时 1920->960 同为 x0.5)。

    不 padding, 定位块在网输入里恒为 80x67 —— 与训练完全一致。
    """
    h0, w0 = img_bgr.shape[:2]
    w, h = max(1, int(round(w0 * 0.5))), max(1, int(round(h0 * 0.5)))
    x = cv2.resize(img_bgr, (w, h)).astype(np.float32) / 255.0
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    small = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(small, (w0, h0), interpolation=cv2.INTER_NEAREST)


def run_vv2_native(model, img_bgr, device):
    """手动 x1/3 后贴到 stride 对齐的小画布, 避免 letterbox 二次缩放。

    ultralytics 的 letterbox 会把长边缩到 imgsz; 若直接喂裁剪图, 定位块
    表观尺寸 = 160 * 640/max(cw,ch), 会随裁剪面积漂移 1.4~2.7 倍。
    这里先 x(640/1920)=1/3 (与训练同比例), 再 pad 到 32 的倍数,
    predict 看到的图长边=imgsz, letterbox 归一化后尺度严格不变。
    """
    h0, w0 = img_bgr.shape[:2]
    s = 640.0 / SCREEN_W
    w, h = max(1, int(round(w0 * s))), max(1, int(round(h0 * s)))
    small = cv2.resize(img_bgr, (w, h))
    pw = (w + 31) // 32 * 32
    ph = (h + 31) // 32 * 32
    canvas = np.full((ph, pw, 3), PAD_FILL, np.uint8)
    canvas[:h, :w] = small
    r = model.predict(source=canvas, imgsz=max(pw, ph), conf=0.25,
                      device=device, verbose=False)[0]
    boxes, confs = [], []
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append((int(round(x1 / s)), int(round(y1 / s)),
                          int(round(x2 / s)), int(round(y2 / s))))
            confs.append(float(b.conf[0]))
    return boxes, confs


# ───────────────────────── 主流程 ─────────────────────────

def collect_val_items():
    items = []
    for ds in sorted(os.listdir(VAL_ROOT)):
        img_dir = os.path.join(VAL_ROOT, ds, 'val', 'images')
        lab_dir = os.path.join(VAL_ROOT, ds, 'val', 'labels')
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = os.path.splitext(fname)[0]
            items.append({'ds': ds,
                          'img': os.path.join(img_dir, fname),
                          'lab': os.path.join(lab_dir, stem + '.txt'),
                          'stem': stem})
    return items


def new_acc():
    return {'n_img': 0, 'gt_full': 0, 'gt_partial': 0,
            'vv1': {'tp': 0, 'fp': 0, 'fn': 0, 'ign': 0, 'partial_hit': 0,
                    'img_all_found': 0, 'img_perfect': 0},
            'vv2': {'tp': 0, 'fp': 0, 'fn': 0, 'ign': 0, 'partial_hit': 0,
                    'img_all_found': 0, 'img_perfect': 0},
            'n_full_hist': {}}


def accumulate(acc, gt_full, gt_partial, r1, r2):
    acc['n_img'] += 1
    acc['gt_full'] += len(gt_full)
    acc['gt_partial'] += len(gt_partial)
    for k, (tp, fp, fn, ig, phit) in (('vv1', r1), ('vv2', r2)):
        acc[k]['tp'] += tp
        acc[k]['fp'] += fp
        acc[k]['fn'] += fn
        acc[k]['ign'] += ig
        acc[k]['partial_hit'] += phit
        if fn == 0:
            acc[k]['img_all_found'] += 1
        if fn == 0 and fp == 0:
            acc[k]['img_perfect'] += 1
    hist = acc['n_full_hist']
    hist[len(gt_full)] = hist.get(len(gt_full), 0) + 1


def finalize(acc):
    out = {k: v for k, v in acc.items() if k not in ('vv1', 'vv2')}
    for k in ('vv1', 'vv2'):
        d = dict(acc[k])
        p, r, f1 = prf(d['tp'], d['fp'], d['fn'])
        d.update({'precision': p, 'recall': r, 'f1': f1})
        n = acc['n_img']
        d['img_all_found_rate'] = round(d['img_all_found'] / n, 4) if n else 0.0
        d['img_perfect_rate'] = round(d['img_perfect'] / n, 4) if n else 0.0
        out[k] = d
    out['n_full_hist'] = dict(sorted(acc['n_full_hist'].items()))
    return out


def main():
    os.makedirs(OUT_VIS, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    print(f'vv1: {VV1_CKPT}')
    print(f'vv2: {VV2_CKPT}')
    print(f'tiers={len(TIERS)}  n_per_tier={N_PER_TIER}  '
          f'iou_match={IOU_MATCH}  iou_ignore={IOU_IGNORE}  ar={AR_RANGE}')
    _MODE_DESC = {
        'native': '定尺度无大幅 padding, 定位块尺度与训练一致 (真实部署)',
        'canvas': '定尺度 + 1920x1080 灰填充 (硬边界伪影对照)',
        'stretch': '拉伸到网输入尺寸 (尺度失配对照)',
    }
    print(f'MODE={MODE}  {_MODE_DESC[MODE]}')

    vv1 = load_vv1(device)
    vv2 = load_vv2(device)

    items = collect_val_items()
    print(f'val pool: {len(items)} images')

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(items))[:N_PER_TIER].tolist()
    base_items = [items[i] for i in order]     # 同一批基图, 跨阶梯配对

    results = {}
    for ti, (name, lo, hi) in enumerate(TIERS, 1):
        acc = new_acc()
        print(f'\n===== 阶梯 {ti}  面积占比 {name} =====')
        for ii, it in enumerate(base_items):
            img = cv2.imread(it['img'])
            if img is None:
                continue
            gt_abs = label_file_to_boxes(it['lab'])
            x0, y0, cw, ch, real_r = sample_crop_window(lo, hi, rng)
            crop = img[y0:y0 + ch, x0:x0 + cw].copy()
            gt_full, gt_partial = split_gt_after_crop(gt_abs, x0, y0, cw, ch)

            net_in = to_canvas(crop) if MODE == 'canvas' else crop
            if MODE == 'native':
                v1_boxes = mask_to_boxes(run_vv1_native(vv1, crop, device))
                v2_boxes, v2_confs = run_vv2_native(vv2, crop, device)
            else:
                v1_boxes = mask_to_boxes(run_vv1(vv1, net_in, device))
                v2_boxes, v2_confs = run_vv2(vv2, net_in, device)
            if MODE == 'canvas':          # 伸进 padding 的框裁回来 (只影响画图, 统计仍用原框)
                v1_draw = clip_boxes(v1_boxes, cw, ch)
                v2_draw = clip_boxes(v2_boxes, cw, ch)
            else:
                v1_draw, v2_draw = v1_boxes, v2_boxes
            r1 = match_with_ignore(gt_full, gt_partial, v1_boxes)
            r2 = match_with_ignore(gt_full, gt_partial, v2_boxes)
            accumulate(acc, gt_full, gt_partial, r1, r2)

            if ii < N_VIS_PER_TIER:
                tag = (f'{name}  area={real_r*100:.1f}%  window=({x0},{y0},{cw},{ch})  '
                       f'GT_full={len(gt_full)} GT_partial={len(gt_partial)}')
                base = draw_boxes(crop, gt_partial, YELLOW, 2, 'partial')
                base = draw_boxes(base, gt_full, GREEN, 3, 'GT')
                cv2.imwrite(os.path.join(OUT_VIS, f'{name}_{ii}_0gt.png'), add_header(
                    base, f'[{it["stem"]}]  {tag}',
                    'green = GT 完全可见   yellow = 部分可见 (ignore)'))
                cv2.imwrite(os.path.join(OUT_VIS, f'{name}_{ii}_1vv1.png'), add_header(
                    draw_boxes(crop, v1_draw, MAGENTA, 3, 'pred'),
                    f'[{it["stem"]}]  vv1 U-Net  {tag}',
                    f'magenta = 预测   tp/fp/fn={r1[0]}/{r1[1]}/{r1[2]}  忽略={r1[3]}'))
                cv2.imwrite(os.path.join(OUT_VIS, f'{name}_{ii}_2vv2.png'), add_header(
                    draw_boxes(crop, v2_draw, CYAN, 3, 'pred', v2_confs),
                    f'[{it["stem"]}]  vv2 YOLOv8  {tag}',
                    f'cyan = 预测   tp/fp/fn={r2[0]}/{r2[1]}/{r2[2]}  忽略={r2[3]}'))

            if (ii + 1) % 20 == 0:
                print(f'  {ii + 1}/{N_PER_TIER}  gt_full累计={acc["gt_full"]}  '
                      f'gt_partial累计={acc["gt_partial"]}')

        results[name] = finalize(acc)
        m = results[name]
        v1, v2 = m['vv1'], m['vv2']
        print(f'  合计  GT_full={m["gt_full"]}  GT_partial={m["gt_partial"]}  '
              f'全可见/图分布={m["n_full_hist"]}')
        print(f'  vv1  P={v1["precision"]}  R={v1["recall"]}  F1={v1["f1"]}  '
              f'tp/fp/fn/ign={v1["tp"]}/{v1["fp"]}/{v1["fn"]}/{v1["ign"]}')
        print(f'  vv2  P={v2["precision"]}  R={v2["recall"]}  F1={v2["f1"]}  '
              f'tp/fp/fn/ign={v2["tp"]}/{v2["fp"]}/{v2["fn"]}/{v2["ign"]}')

    # 汇总
    print('\n' + '=' * 96)
    print(f'{"阶梯":<10s} {"图":>3s} {"GT_full":>7s} {"GT_part":>7s} | '
          f'{"vv1 P":>6s} {"vv1 R":>6s} {"vv1 F1":>6s} | '
          f'{"vv2 P":>6s} {"vv2 R":>6s} {"vv2 F1":>6s} | {"vv1全中":>7s} {"vv2全中":>7s}')
    print('-' * 96)
    tot = new_acc()
    for name, _, _ in TIERS:
        m = results[name]
        v1, v2 = m['vv1'], m['vv2']
        print(f'{name:<10s} {m["n_img"]:>3d} {m["gt_full"]:>7d} {m["gt_partial"]:>7d} | '
              f'{v1["precision"]:>6.3f} {v1["recall"]:>6.3f} {v1["f1"]:>6.3f} | '
              f'{v2["precision"]:>6.3f} {v2["recall"]:>6.3f} {v2["f1"]:>6.3f} | '
              f'{v1["img_all_found_rate"]:>7.3f} {v2["img_all_found_rate"]:>7.3f}')
    print('=' * 96)

    payload = {
        'seed': SEED,
        'mode': MODE,
        'n_per_tier': N_PER_TIER,
        'n_base_images': N_PER_TIER,
        'paired_design': True,
        'iou_match': IOU_MATCH,
        'iou_ignore': IOU_IGNORE,
        'full_tol': FULL_TOL,
        'ar_range': list(AR_RANGE),
        'gt_policy': ('平移+裁剪: 完全可见=gt_full(统计); 部分可见=gt_partial(ignore, '
                      '命中不计误检/漏检); 窗外=丢弃'),
        'vv1_ckpt': VV1_CKPT,
        'vv2_ckpt': VV2_CKPT,
        'tiers': {name: results[name] for name, _, _ in TIERS},
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\nmetrics: {OUT_JSON}')
    print(f'vis    : {OUT_VIS}')


if __name__ == '__main__':
    main()
