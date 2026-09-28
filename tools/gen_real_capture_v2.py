#!/usr/bin/env python
"""真实拍照 + 微信压缩 验证用图生成  (v2: 贴边小角标, 先标后水印)

v2 改动 (Request 53 修正):
  1. 角标 180 -> 120
  2. 四角标**贴死边缘, 不留空隙** (含右下)
  3. **先盖回字标, 再叠水印** —— 水印压在标上, 振幅完整, 不被标遮挡

注意: 右下标 (1800,960)-(1920,1080) 覆盖定位块 #6 (1760,945)-(1920,1080)
      的 66.7% 面积。水印振幅虽完整, 但 #6 底下 2/3 的载体纹理变成黑白回字。
      实际影响用 vv2 跑一遍看 (check_detect.py)。

出图要求:
  - 叠水印 (alpha=0.032, channel_mode='b'), 不加任何模拟噪声
  - 水印序列随机, 六位十进制数写进文件名
  - 四角回字标, 样式 = qr_loc_mark.png
  - 输出是待拍照的纯净图: 不加 header / 不画框 / 不拼图
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

from generate_dataset import (
    generate_one_sample, build_canvas, SCREEN_W, SCREEN_H,
    get_locator_abs_rect,
)
from generate_locator_pattern import FIX_FG_MATRIX

ALPHA = 0.032
SEED = 20260928
N_IDS_PER_CARRIER = 3
ID_LO, ID_HI = 0, 999999          # 六位十进制, 落在 MAX_WATERMARK_ID=1048575 之内

TEST_DIR = CARRIER_DIR
OUT = os.path.join(REPO, 'vis', 'real_capture')
MARK_PNG = os.path.join(REPO, 'qr_loc_mark.png')
LOCATOR_NPY = os.path.join(REPO, 'watermark_locator', 'locator_pattern.npy')

MARK = 120          # 角标边长 (参考图 300x300, 缩到 120)
# 四角贴死边缘, 不留空隙
MARK_POS = {
    'TL': (0, 0),
    'TR': (SCREEN_W - MARK, 0),
    'BL': (0, SCREEN_H - MARK),
    'BR': (SCREEN_W - MARK, SCREEN_H - MARK),
}

LOCATOR_IDX = [(0, 3), (1, 1), (1, 5), (2, 3), (3, 1), (3, 5)]
LOCATOR_RECTS = [get_locator_abs_rect(i, j) for i, j in LOCATOR_IDX]


# ───────────────────────── 角标 ─────────────────────────

def load_mark(size):
    """读 qr_loc_mark.png 缩到 size, 二值化成纯 0/255 三通道。"""
    src = cv2.imread(MARK_PNG, cv2.IMREAD_GRAYSCALE)
    assert src is not None, MARK_PNG
    m = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    m = np.where(m >= 128, 255, 0).astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def stamp_marks(img, mark_bgr):
    """四角盖回字标 (先标后水印的"先"这一步)。返回盖标图 + 各角标框。"""
    out = img.copy()
    rects = {}
    h, w = mark_bgr.shape[:2]
    for tag, (x, y) in MARK_POS.items():
        out[y:y + h, x:x + w] = mark_bgr
        rects[tag] = (x, y, x + w, y + h)
    return out, rects


def report_overlap(mark_rects):
    """打印角标与定位块的重叠 —— 先标后水印不毁振幅, 但会改 #6 底下的纹理。"""
    for tag, (x1, y1, x2, y2) in mark_rects.items():
        for li, (lx, ly, lw, lh) in enumerate(LOCATOR_RECTS, 1):
            ix1, iy1 = max(x1, lx), max(y1, ly)
            ix2, iy2 = min(x2, lx + lw), min(y2, ly + lh)
            ov = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if ov:
                frac = ov / (lw * lh)
                print(f'  ! 角标 {tag} 覆盖定位块 #{li} 的 {frac*100:.1f}% '
                      f'({ov}/{lw*lh} px)')
    print('  (其余角标-定位块组合无重叠)')


# ───────────────────────── 主流程 ─────────────────────────

def main():
    os.makedirs(OUT, exist_ok=True)
    mark_bgr = load_mark(MARK)
    mark_rects = {k: (v[0], v[1], v[0] + MARK, v[1] + MARK)
                  for k, v in MARK_POS.items()}
    print(f'角标 {MARK}x{MARK}  贴边位置 {MARK_POS}')
    print(f'定位块 {LOCATOR_RECTS}')
    print('角标-定位块重叠:')
    report_overlap(mark_rects)

    locator_pattern = np.load(LOCATOR_NPY)
    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')))
    assert len(files) == 3, f'expect 3 carriers, got {files}'

    rng_ids = np.random.RandomState(SEED)
    all_ids = rng_ids.randint(ID_LO, ID_HI + 1, size=len(files) * N_IDS_PER_CARRIER)
    assert len(set(int(v) for v in all_ids)) == len(all_ids), 'wm_id 撞号'

    manifest = {
        'purpose': 'real photo capture + WeChat compression verification',
        'alpha': ALPHA, 'channel_mode': 'b', 'apply_noise': False,
        'order': 'mark first, then watermark (watermark is NOT occluded)',
        'canvas': [SCREEN_W, SCREEN_H],
        'mark': {
            'style_source': MARK_PNG, 'size': MARK,
            'positions': MARK_POS, 'flush_to_edge': True,
            'note': '四角贴死边缘不留缝。先盖标再叠水印, 水印振幅完整。'
                    '但 BR 标覆盖定位块 #6 66.7% 面积, #6 底下纹理变成黑白回字。',
        },
        'locator_rects_abs': [
            {'idx': k, 'rect': [r[0], r[1], r[2], r[3]]}
            for k, r in zip(LOCATOR_IDX, LOCATOR_RECTS)],
        'images': [],
    }

    k = 0
    for si, fname in enumerate(files, 1):
        src = cv2.imread(os.path.join(TEST_DIR, fname))
        assert src is not None, fname
        rng = np.random.RandomState(SEED + si)
        canvas = build_canvas([src], rng, SCREEN_W, SCREEN_H, grid=(1, 1))

        # 先盖角标 —— 之后水印叠在标上
        marked, mark_rects = stamp_marks(canvas, mark_bgr)

        for j in range(N_IDS_PER_CARRIER):
            wm_id = int(all_ids[k]); k += 1
            six = f'{wm_id:06d}'

            clean, bboxes, mask, _ = generate_one_sample(
                wm_id, FIX_FG_MATRIX, locator_pattern, ALPHA, rng,
                carrier_img=marked, apply_noise=False, channel_mode='b')

            name = f'wm{six}_c{si:02d}.png'
            cv2.imwrite(os.path.join(OUT, name), clean)

            gt_abs = []
            for b in bboxes:
                _cls, cx, cy, bw, bh = b
                gt_abs.append([int(round((cx - bw / 2) * SCREEN_W)),
                               int(round((cy - bh / 2) * SCREEN_H)),
                               int(round((cx + bw / 2) * SCREEN_W)),
                               int(round((cy + bh / 2) * SCREEN_H))])
            manifest['images'].append({
                'file': name, 'carrier': fname, 'carrier_idx': si,
                'wm_id': wm_id, 'wm_id_6digit': six, 'wm_id_hex5': f'{wm_id:05X}',
                'mark_rects': mark_rects,
                'gt_locator_xyxy': gt_abs,
                'gt_locator_yolo': [[round(v, 6) for v in b] for b in bboxes],
            })
            print(f'  {name}   carrier={fname}  wm_id={six} (0x{wm_id:05X})  '
                  f'GT={len(gt_abs)}')

    with open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    readme = f"""真实拍照 + 微信压缩验证图 (v2)
===============================

这批图 **已叠水印, 未加任何噪声**。噪声由你的真实拍照 / 微信压缩产生。

  水印强度  alpha = {ALPHA} (最大像素改动 8/255, 只落在 B 通道)
  画布      {SCREEN_W}x{SCREEN_H}
  角标      {MARK}x{MARK} 纯黑白三环回, 样式同 qr_loc_mark.png
  顺序      **先盖角标, 再叠水印** —— 水印压在标上, 振幅完整不被遮挡
  文件名    wm<六位十进制 wm_id>_c<载体序号>.png
  张数      {len(manifest['images'])}

角标位置 (左上角坐标, **四角贴死边缘, 不留空隙**):
  TL {MARK_POS['TL']}   TR {MARK_POS['TR']}
  BL {MARK_POS['BL']}   BR {MARK_POS['BR']}

已知代价
--------
BR 角标 (1800,960)-(1920,1080) 与定位块 #6 (1760,945)-(1920,1080) 重叠
14400 px = **#6 的 66.7%**。因为是先标后水印, #6 的水印振幅完整,
但底下 2/3 的载体纹理变成了黑白回字 (原本是截图内容)。
其余 5 个定位块与角标零重叠。

建议流程
--------
1. 逐张全屏显示, 手机拍照 (4 角都进画面, 别过曝/欠曝)
2. 拍到的照片过一次真实微信 (走「文件」或「图片」都行, 记下走哪条)
3. 用 4 个角标做透视校正, 校正回 {SCREEN_W}x{SCREEN_H}
4. 丢给 vv2 检测, 与 manifest.json 的 gt_locator_xyxy 对齐算 P/R

manifest.json 里每张图都有 wm_id / 六位数字 / 4 个角标框 / 6 个 GT 定位框。
"""
    with open(os.path.join(OUT, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(readme)

    print(f'\n输出目录: {OUT}')
    print(f'共 {len(manifest["images"])} 张 + manifest.json + README.txt')


if __name__ == '__main__':
    main()
