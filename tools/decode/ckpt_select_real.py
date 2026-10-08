"""按实拍召回选 v5 权重 —— 不用 val fitness。

为什么
------
v4 的 best.pt 是按 val fitness 选的, 而部署点是 18 张实拍。
ckpt_sensitivity_curve 显示 v4 从 v3 接上后灵敏度一路掉, ep50 反而是最好点,
最终 best 却不是它。选错指标等于白训。

用法
----
  python ckpt_select_real.py <weights_dir>
      [--live-only]          只统计水印残余>0 的 6 张 (信号真在的那几张)
      [--model v3|v4|...]    对照组权重路径会自动找

口径与 draw_boxes.py 严格一致 (detect_at/eval_boxes 见 detect_eval.py):
  输入 vis/real_capture/detect_out/rect/*.png (18 张矫正图, 1920x1080)
  detect_at: max(w,h)=imgsz 缩放 -> pad 到 stride32 (填114) -> 框映回 /s
  conf=0.25, imgsz=640
  GT 六个固定定位块 160x135

输出按**实拍召回**排序, 不是按 val。
"""
import argparse
import glob
import os
import re
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import cv2
import numpy as np

REPO = os.environ.get('DX_OVERLAY_REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_eval import detect_at, eval_boxes

RECT = os.path.join(REPO, 'vis', 'real_capture', 'detect_out', 'rect')
GT_XYXY = [(1120, 135, 1280, 270), (480, 405, 640, 540), (1760, 405, 1920, 540),
           (1120, 675, 1280, 810), (480, 945, 640, 1080), (1760, 945, 1920, 1080)]
CONF = 0.25

# 水印残余 > 0 的 6 张 (wm_presence_control.py 测出)
LIVE = {
    'pz_微信图片_20260928113913_13539_11',
    'pz_微信图片_20260928113919_13542_11',
    'pz_微信图片_20260928113908_13537_11',
    'pzvx_微信图片_20260928113944_13552_11',
    'pzvx_微信图片_20260928113939_13547_11',
    'pzvx_微信图片_20260928113941_13549_11',
}

BASELINE = {
    'v3_best': os.path.join(REPO, 'runs', 'detect', 'output', 'v1_vv2_yolo_v3',
                           'yolo_crop_aug_finetune', 'weights', 'best.pt'),
    'v4_best': os.path.join(REPO, 'runs', 'detect', 'output', 'v1_vv2_yolo_v4',
                           'yolo_scale_aug_finetune', 'weights', 'best.pt'),
}


def find_ckpts(weights_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(weights_dir, '*.pt'))):
        stem = os.path.basename(p)[:-3]
        m = re.search(r'epoch(\d+)', stem)
        order = int(m.group(1)) if m else (10 ** 6 if stem == 'best' else 10 ** 6 - 1)
        out.append((order, stem, p))
    return [p for _, _, p in sorted(out)]


def evaluate(path, files, live_only):
    from ultralytics import YOLO
    m = YOLO(path)
    tot = np.zeros(3, int)
    for f in files:
        stem = os.path.basename(f)[:-4]
        if live_only and stem not in LIVE:
            continue
        img = cv2.imread(f)
        if img is None:
            continue
        tot += eval_boxes([tuple(b) for b in detect_at(m, img, 640, conf=CONF)[0]],
                          GT_XYXY)
    tp, fp, fn = (int(v) for v in tot)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1


def main():
    global CONF
    ap = argparse.ArgumentParser()
    ap.add_argument('weights_dir')
    ap.add_argument('--live-only', action='store_true')
    ap.add_argument('--conf', type=float, default=CONF)
    args = ap.parse_args()

    CONF = args.conf

    files = sorted(glob.glob(os.path.join(RECT, '*.png')))
    if not files:
        raise SystemExit(f'no real captures under {RECT}')
    n = len(LIVE) if args.live_only else len(files)
    print(f'实拍 {len(files)} 张, 本次评 {n} 张'
          f'{" (只水印残余>0)" if args.live_only else ""}, '
          f'imgsz=640 conf={CONF}\n')

    rows = []
    for tag, p in BASELINE.items():
        if os.path.exists(p):
            rows.append((tag, p))
    for p in find_ckpts(args.weights_dir):
        rows.append((os.path.basename(p)[:-3], p))

    print(f'  {"权重":22s} {"tp":>4s} {"fp":>4s} {"fn":>4s} '
          f'{"P":>7s} {"R":>7s} {"F1":>7s}   排名依据=实拍召回')
    print('  ' + '-' * 74)
    res = []
    for tag, p in rows:
        try:
            t = evaluate(p, files, args.live_only)
        except Exception as e:
            print(f'  {tag:22s}  **失败** {type(e).__name__}: {e}')
            continue
        res.append((tag, t, p))
    res.sort(key=lambda r: (-r[1][4], -r[1][5]))
    for i, (tag, t, p) in enumerate(res):
        mark = ' ←' if i == 0 else ''
        print(f'  {tag:22s} {t[0]:4d} {t[1]:4d} {t[2]:4d} '
              f'{t[3]:7.1%} {t[4]:7.1%} {t[5]:7.1%}   #{i + 1}{mark}')

    print('\n  读法: 只看"R"排出来的第一名。val fitness 高不代表实拍强 ——')
    print('        v4 的 best.pt 就是按 val 选的, 实拍反而不如 ep50。')


if __name__ == '__main__':
    main()
