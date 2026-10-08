#!/usr/bin/env python3
"""v2 数据集生成器：载体 + v2 黄/白模板 + 成对噪声 → 96 码字 YOLO 检测数据集。

复用关系（import，不复制实现）
----------------------------
* ``dataset.generate_dataset.alpha_blend_watermark`` —— 混合公式逐位一致。
  0/255 模板下 dynamicMask ∈ {0,1}：白格完全不改，黄格 max|Δ| = 0.032×255 = 8.16。
* ``dataset.generate_dataset.apply_pair_noise`` —— identity / wechat / tile_crop /
  pimog 里随机挑两种组合，标签同步。
* ``dataset.generate_dataset.build_canvas`` —— 1:1 裁剪/平铺拼画布，**绝不 resize**。
* ``v2.generate_yellow_white_template`` —— v2 交替行排布（M1M2M1M2…/M3M4M3M4…）。

尺度怎么覆盖（DESIGN.md §12.4）
-----------------------------
条纹周期 4 px 是**渲染像素周期**，任何重采样都会毁掉它。所以尺度不能靠缩放增广，
只能靠**按目标分辨率原生渲染**：

    屏幕 (W, H) = (round(1920·s), round(1080·s))  →  码字 = (160·s, 135·s)

渲染完再用 1:1 裁剪/贴装把画面框进固定的 1920×1080 训练窗，**全程不重采样**。
于是 ultralytics 的 letterbox 对 1920×1080 输入算出 r = 1.0，只 pad 不缩放 ——
条纹恒 4 px、码字恒 160s×135s，s 不会被管线抹掉。

输出
----
::

    output_dir/
        images/{train,val}/*.png     1920×1080 BGR
        labels/{train,val}/*.txt     YOLO: `0 cx cy w h` 归一化，一类 `codeword`
        meta/{train,val}/*.json     sidecar（不进损失）：s / 间距 / 96 符号 / 噪声
        watermark.yaml              YOLO 数据集配置
        manifest.json               生成参数与统计

Examples::

    # 冒烟
    python generate_dataset_v2.py --num_samples 8 --output_dir /tmp/v2smoke --selftest

    # 正式
    python generate_dataset_v2.py --num_samples 4000 \\
        --carrier_root /data1/lpl/datasets --output_dir /data1/lpl/datasets_v2 --jobs 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, '..'))
_REPO_WM = os.path.abspath(os.path.join(_V2, '..'))
sys.path.insert(0, os.path.join(_REPO_WM, 'dataset'))
sys.path.insert(0, _V2)

from generate_dataset import (  # noqa: E402
    alpha_blend_watermark,
    apply_pair_noise,
    build_canvas,
)

# 模板生成器必须是 **v2 那一份**，不能靠 sys.path 碰运气。
# 仓库里有三份 generate_yellow_white_template.py：
#   dx_overlay_code/generate_yellow_white_template.py      ← 旧版，v1 交错排布
#   watermark_locator/v1/generate_yellow_white_template.py ← v1 交错排布
#   watermark_locator/v2/generate_yellow_white_template.py ← v2 交替行排布 (要用这个)
# 而 generate_dataset.py 会在 import 时把 `dataset/../..` 插到 sys.path[0]，
# 于是 `from generate_yellow_white_template import ...` 会**静默**拿到旧版，
# 排布差在 block_col=2,3 的两列上 —— 96 个码字里错 32 个，ID 靠 RS 侥幸才对。
# 所以这里按文件路径显式加载，并断言排布公式，杜绝再发生。
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'wm_template_v2', os.path.join(_V2, 'generate_yellow_white_template.py'))
_T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_T)
if _T.block_message_index(0, 2) != 0 or _T.block_message_index(1, 0) != 2:
    raise ImportError(f'{_T.__file__} 不是 v2 排布 (交替行)，拒绝使用')
build_rgb_templates = _T.build_rgb_templates
encode_watermark_sequence = _T.encode_watermark_sequence
rounded_boundary = _T.rounded_boundary
BLOCK_COLS, BLOCK_ROWS, GRID_SIZE = _T.BLOCK_COLS, _T.BLOCK_ROWS, _T.GRID_SIZE
import lattice  # noqa: E402

# ── 几何常量 (s=1 的基准屏) ──────────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080
WINDOW_W, WINDOW_H = 1920, 1080      # 训练窗固定；ultralytics 对它只 pad 不缩放
CODEWORD_W, CODEWORD_H = 160, 135    # s=1 下的码字
SUB_ROWS, SUB_COLS = BLOCK_ROWS * 2, BLOCK_COLS * 2      # 8×12 = 96
CELL_ROWS = CELL_COLS = GRID_SIZE                        # 8×8 格 / 码字

ALPHA = 0.032
MAX_DELTA = ALPHA * 255              # 8.16

# 真实屏幕分辨率（16:9），s = W/1920。外加 ±3% 抖动，避免网络只认离散尺寸。
SCREEN_SIZES = (
    (1280, 720), (1366, 768), (1600, 900),
    (1920, 1080), (2560, 1440), (3840, 2160),
)


# ───────────────────── 模板与标注 ─────────────────────

def render_template(wm_id: int, width: int, height: int,
                    stripe_angle: float = 45.0, stripe_period: int = 4,
                    stripe_width: int = 2, polarity: int = 0) -> np.ndarray:
    """渲染 v2 黄/白模板，返回 BGR (0/255)。

    ``stripe_width=2`` / ``period=4`` → 恰好 25% 像素写入（DESIGN.md §2）。
    模板只含 (255,255,255) 与 (0,255,255) 两种值，这是 alpha 契约的前提。
    """
    seq = encode_watermark_sequence(int(wm_id))
    rgba, _ = build_rgb_templates(
        seq, width, height,
        block_rows=BLOCK_ROWS, block_cols=BLOCK_COLS,
        polarity=polarity, alternate=False,
        pattern='diagonal', angle=stripe_angle,
        period=stripe_period, stripe_width=stripe_width,
    )
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def codeword_boxes(width: int, height: int) -> List[List[int]]:
    """v2 排布下 96 个码字框的**像素整数坐标** ``[col, row, x0, y0, x1, y1]``。

    ``build_rgb_templates`` 把 (width,height) 按 round-to-even 边界切成
    96 列 × 64 行格，一个码字 = 8×8 格 = (width/12, height/8)。
    边界与 ``lattice.codeword_cell_means`` 的切分口径一致。
    """
    xb = [rounded_boundary(width, c, SUB_COLS * CELL_COLS)
          for c in range(SUB_COLS * CELL_COLS + 1)]
    yb = [rounded_boundary(height, r, SUB_ROWS * CELL_ROWS)
          for r in range(SUB_ROWS * CELL_ROWS + 1)]
    out = []
    for i in range(SUB_ROWS):
        for j in range(SUB_COLS):
            out.append([j, i,
                        xb[j * CELL_COLS], yb[i * CELL_ROWS],
                        xb[j * CELL_COLS + CELL_COLS], yb[i * CELL_ROWS + CELL_ROWS]])
    return out


def to_yolo(boxes: Sequence[Sequence[int]], img_w: int, img_h: int,
            cls: int = 0) -> List[List[float]]:
    """像素框 → ``[[cls, cx, cy, w, h]]`` 归一化于 (img_w, img_h)。"""
    rows = []
    for _, _, x0, y0, x1, y1 in boxes:
        rows.append([cls,
                     (x0 + x1) / 2.0 / img_w,
                     (y0 + y1) / 2.0 / img_h,
                     (x1 - x0) / float(img_w),
                     (y1 - y0) / float(img_h)])
    return rows


# ───────────────────── 尺度与取景 ─────────────────────

def sample_screen(rng: np.random.RandomState,
                  s_min: float = 0.5, s_max: float = 2.2) -> Tuple[float, int, int]:
    """采样屏幕分辨率 → (s, W, H)。以真实屏幕为主 + ±3% 抖动。"""
    w, h = SCREEN_SIZES[int(rng.randint(len(SCREEN_SIZES)))]
    s = (w / SCREEN_W + h / SCREEN_H) / 2.0
    s *= float(rng.uniform(0.97, 1.03))
    s = min(max(s, s_min), s_max)
    return s, int(round(SCREEN_W * s)), int(round(SCREEN_H * s))


def frame_to_window(img: np.ndarray, labels: List[List[float]],
                    rng: np.random.RandomState,
                    carrier_paths: Optional[Sequence[str]] = None,
                    carrier_cache: Optional[dict] = None
                    ) -> Tuple[np.ndarray, List[List[float]]]:
    """把屏幕画面框进固定 (WINDOW_W, WINDOW_H) 训练窗，**全程 1:1 不重采样**。

    * 画面比窗大 → 随机 1:1 裁剪（真截屏只截了一块）
    * 画面比窗小 → 贴进一张 1:1 拼出来的同尺寸载体（水印只占一块，周围是负样本）
    * 一样大     → 原样

    只保留**完整落在窗内**的框；返回的 labels 仍是归一化坐标。
    """
    h, w = img.shape[:2]
    if (h, w) == (WINDOW_H, WINDOW_W):
        return img, _keep_inside(labels, w, h, 0, 0)

    if h >= WINDOW_H and w >= WINDOW_W:
        y0 = int(rng.randint(0, h - WINDOW_H + 1))
        x0 = int(rng.randint(0, w - WINDOW_W + 1))
        out = img[y0:y0 + WINDOW_H, x0:x0 + WINDOW_W].copy()
        # 窗坐标 = 屏坐标 − (x0,y0)：out[i,j] = img[y0+i, x0+j]
        return out, _keep_inside(labels, w, h, -x0, -y0)

    # 贴装：先 1:1 拼一张满窗载体，再把水印画面盖上去
    canvas = _carrier_canvas(rng, carrier_paths, carrier_cache, WINDOW_W, WINDOW_H)
    y0 = int(rng.randint(0, WINDOW_H - h + 1))
    x0 = int(rng.randint(0, WINDOW_W - w + 1))
    canvas[y0:y0 + h, x0:x0 + w] = img
    # 窗坐标 = 屏坐标 + (x0,y0)：canvas[y0+i, x0+j] = img[i,j]
    return canvas, _keep_inside(labels, w, h, x0, y0)


def _keep_inside(labels: List[List[float]], src_w: int, src_h: int,
                 dx: int, dy: int) -> List[List[float]]:
    """把归一化标签平移 (dx,dy) 像素后，只留完整落在 (WINDOW_W,WINDOW_H) 内的。

    ``dx/dy`` = **窗坐标 − 屏坐标**，跟 :func:`frame_to_window` 的取景方式对齐：

      * 1:1 裁剪 ``out[i,j] = img[y0+i, x0+j]``  →  窗 = 屏 − (x0,y0)  →  dx=-x0
      * 贴装 ``canvas[y0+i, x0+j] = img[i,j]``    →  窗 = 屏 + (x0,y0)  →  dx=+x0

    这个符号以前搞反过：裁剪/贴装两条分支都写成了相反的偏移，框整体偏
    2×取景偏移（贴装最大 58px、裁剪最大 662px），自检只查了框的数量和图的
    形状所以没炸。现在用**标签框解出来的码字必须等于期望符号**来兜底，
    见 :func:`selftest` 的"标签几何"一节。
    """
    out = []
    for cls, cx, cy, bw, bh in labels:
        px = cx * src_w + dx
        py = cy * src_h + dy
        hw = bw * src_w / 2.0
        hh = bh * src_h / 2.0
        if (px - hw >= 0 and px + hw <= WINDOW_W and
                py - hh >= 0 and py + hh <= WINDOW_H):
            out.append([cls, px / WINDOW_W, py / WINDOW_H,
                        2 * hw / WINDOW_W, 2 * hh / WINDOW_H])
    return out


# ───────────────────── 载体池 ─────────────────────

def _list_images(directory: str) -> List[str]:
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith(exts))


def discover_carriers(root: str) -> List[str]:
    """收集 root 下的载体图路径（不读像素，只建索引）。

    兼容两种目录结构：``root/{train,val}/*`` 与 ``root/<name>/{train,val}/*``。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f'carrier_root not found: {root}')
    paths: List[str] = []
    for sub in ('train', 'val'):
        paths += _list_images(os.path.join(root, sub))
    if not paths:
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(d):
                for sub in ('train', 'val'):
                    paths += _list_images(os.path.join(d, sub))
                paths += _list_images(d)
    if not paths:
        raise RuntimeError(f'no carrier images under {root}')
    return paths


def _carrier_canvas(rng: np.random.RandomState, paths: Optional[Sequence[str]],
                    cache: Optional[dict], w: int, h: int) -> np.ndarray:
    """1:1 拼一张 w×h 载体画布（绝不 resize）。没有载体时退化为中性灰。"""
    if not paths:
        return np.full((h, w, 3), 128, dtype=np.uint8)
    k = int(rng.randint(len(paths)))
    if cache is not None and k in cache:
        src = cache[k]
    else:
        src = cv2.imread(paths[k])
        if src is None:
            src = np.full((h, w, 3), 128, dtype=np.uint8)
        elif cache is not None:
            if len(cache) > 8:
                cache.clear()
            cache[k] = src
    return build_canvas([src], rng, target_w=w, target_h=h)


# ───────────────────── 单样本 ─────────────────────

def make_sample(wm_id: int, rng: np.random.RandomState,
                carrier_paths: Optional[Sequence[str]] = None,
                alpha: float = ALPHA,
                s_min: float = 0.5, s_max: float = 2.2,
                apply_noise: bool = True,
                stripe_width: int = 2,
                carrier_cache: Optional[dict] = None):
    """生成一个训练样本。

    物理顺序：**屏幕渲染 → alpha 融合 → 1:1 取景进训练窗 → 采集退化**。
    噪声放在最后，因为退化是施加在"最终看到的那张截图"上的。
    """
    s, W, H = sample_screen(rng, s_min, s_max)

    screen = _carrier_canvas(rng, carrier_paths, carrier_cache, W, H)
    template = render_template(wm_id, W, H, stripe_width=stripe_width)
    blended = alpha_blend_watermark(screen, template, alpha)

    px_boxes = codeword_boxes(W, H)
    labels = to_yolo(px_boxes, W, H)

    img, labels = frame_to_window(blended, labels, rng, carrier_paths, carrier_cache)

    noise_names: List[str] = []
    if apply_noise:
        img, _, labels, noise_names = apply_pair_noise(img, None, labels, rng)

    seq16 = list(encode_watermark_sequence(int(wm_id)))
    meta = {
        'watermark_id': int(wm_id),
        's': float(s),
        'screen': [W, H],
        'window': [WINDOW_W, WINDOW_H],
        'interval_px': [float(W) / SUB_COLS, float(H) / SUB_ROWS],
        'symbol_interval_px': [float(W) / SUB_COLS * 4, float(H) / SUB_ROWS * 4],
        'alpha': float(alpha),
        'stripe': {'angle': 45.0, 'period': 4, 'width': int(stripe_width)},
        'noise': [str(n) for n in noise_names],
        'n_boxes': len(labels),
        'sequence16': [int(v) for v in seq16],
        # 8×12 行优先的真符号格，评估/排障用（不进损失）
        'symbols': [int(v) for v in lattice.subblock_symbol_grid(seq16).ravel()],
    }
    return img, labels, meta


# ───────────────────── 自检 ─────────────────────

def selftest(carrier_paths: Optional[Sequence[str]] = None, n: int = 3) -> int:
    """几何、混合契约、端到端读回。全绿才允许跑大批量生成。"""
    rng = np.random.RandomState(7)
    ok = True

    print('— 码字框铺满画布 (无缝隙/重叠) —')
    for W, H in ((1920, 1080), (1280, 720), (2560, 1440), (1000, 563)):
        boxes = codeword_boxes(W, H)
        xs = sorted({b[2] for b in boxes} | {b[4] for b in boxes})
        ys = sorted({b[3] for b in boxes} | {b[5] for b in boxes})
        area = sum((b[4] - b[2]) * (b[5] - b[3]) for b in boxes)
        good = (len(boxes) == 96 and area == W * H and xs[0] == 0 and xs[-1] == W
                and ys[0] == 0 and ys[-1] == H
                and len(xs) == SUB_COLS + 1 and len(ys) == SUB_ROWS + 1)
        ok &= good
        cw = W / SUB_COLS
        ch = H / SUB_ROWS
        print(f'  {W}x{H}: 96 框, 覆盖 {area}/{W*H}, 切点 {len(xs)}x{len(ys)}, '
              f'码字均值 {cw:.1f}x{ch:.1f}  {"OK" if good else "FAIL"}')

    print('— alpha 混合契约 —')
    for W, H in ((1920, 1080), (1280, 720)):
        carrier = _carrier_canvas(rng, carrier_paths, None, W, H)
        tmpl = render_template(123456, W, H)
        out = alpha_blend_watermark(carrier, tmpl, ALPHA)
        diff = np.abs(out.astype(np.int16) - carrier.astype(np.int16))
        yellow = tmpl[:, :, 0] == 0            # B=0 → 黄(信号)
        white = ~yellow
        max_all = int(diff.max())
        max_white = int(diff[white].max()) if white.any() else 0
        changed = diff.max(axis=2) > 0
        yy, xx = np.indices(changed.shape)
        stripe = np.fmod((xx + yy).astype(np.float64), 4.0) < 2
        outside = int((changed & ~stripe).sum())
        vals = set(np.unique(tmpl).tolist())
        good = (max_all <= int(MAX_DELTA) and max_white == 0 and outside == 0
                and vals <= {0, 255} and abs(float(yellow.mean()) - 0.25) < 0.01)
        ok &= good
        print(f'  {W}x{H}: 模板取值 {sorted(vals)}  黄+条纹占比 {yellow.mean()*100:.2f}%  '
              f'max|Δ| {max_all}(≤{int(MAX_DELTA)})  白格 max|Δ| {max_white}  '
              f'条纹外改动 {outside}  {"OK" if good else "FAIL"}')

    print('— 端到端读回 [平坦载体]：几何 + 位序 + 混合 —')
    # 平坦载体用来查**几何/位序/混合**是否对，不能让载体纹理混淆结论。
    # 自然载体下的能力是另一回事，见下一节。
    wid = 0x1E240
    expect = [int(v) for v in lattice.subblock_symbol_grid(
        list(encode_watermark_sequence(wid))).ravel()]
    for s_true in (0.67, 1.0, 1.33, 2.0):
        W, H = int(round(SCREEN_W * s_true)), int(round(SCREEN_H * s_true))
        flat = np.full((H, W, 3), 128, np.uint8)
        img = alpha_blend_watermark(flat, render_template(wid, W, H), ALPHA)
        boxes = codeword_boxes(W, H)
        out = lattice.decode_patches([img[b[3]:b[5], b[2]:b[4]] for b in boxes],
                                     boxes, stripe_offset=None)
        n_ok = sum(1 for a, b in zip(out['symbols'], expect) if a == b)
        good = (n_ok == 96 and out['id'] == wid and out['nfix'] == 0
                and out['n_empty_slots'] == 0)
        ok &= good
        print(f'  s={s_true:.2f} ({W}x{H}): 逐码字 {n_ok}/96  相位{out["stripe_phase"]}  '
              f'ID 0x{wid:05X} → {out["id"]}  shift={out["shift"]}  nfix={out["nfix"]}  '
              f'{"OK" if good else "FAIL"}')

    print('— 端到端读回 [自然载体]：解码能力 —')
    # 自然载体纹理 std≈30–90，远大于水印的 ±4 灰阶。这里只考**能不能解出 ID**，
    # 逐码字正确率只是参考指标，不作硬门槛。
    if not carrier_paths:
        print('  (跳过，未提供载体)')
    else:
        for s_true in (0.67, 1.0, 1.33, 2.0):
            W, H = int(round(SCREEN_W * s_true)), int(round(SCREEN_H * s_true))
            boxes = codeword_boxes(W, H)
            n_hit, acc_sum, trials = 0, 0.0, 3
            for _ in range(trials):
                carrier = _carrier_canvas(rng, carrier_paths, None, W, H)
                img = alpha_blend_watermark(carrier, render_template(wid, W, H), ALPHA)
                out = lattice.decode_patches([img[b[3]:b[5], b[2]:b[4]] for b in boxes],
                                             boxes, stripe_offset=None)
                acc_sum += sum(1 for a, b in zip(out['symbols'], expect) if a == b) / 96.0
                n_hit += (out['id'] == wid)
            good = n_hit == trials
            ok &= good
            print(f'  s={s_true:.2f}: ID 恢复 {n_hit}/{trials}  '
                  f'逐码字均值 {acc_sum/trials*100:.1f}%  {"OK" if good else "FAIL"}')

    print('— 标签几何：样本自带的框必须解出 ID —')
    # 兜底 ``_keep_inside`` 的取景偏移符号。符号写反时**图像完全不变**，
    # 上面所有检查照样全绿，框却整体偏了 2×取景偏移（贴装最大 58px、
    # 1:1 裁剪最大 662px）—— 只有把框真拿去格点+解码才会露馅。
    # 这条在 s<1 (贴装) 和 s>1 (裁剪) 两条路径上都跑。
    import predict_interval as PI                      # noqa: E402
    for s_lo, s_hi in ((0.65, 0.72), (0.95, 1.05), (1.30, 1.36), (1.95, 2.10)):
        img, labels, meta = make_sample(wid, rng, carrier_paths, ALPHA,
                                        apply_noise=False, s_min=s_lo, s_max=s_hi)
        xyxy, conf = PI.load_yolo_labels_from_rows(labels, WINDOW_W, WINDOW_H)
        res = PI.run(img, xyxy, conf, theta_span=6.0)
        d = res['decode'] or {}
        e = abs(res['interval_px'][0] - meta['interval_px'][0]) / meta['interval_px'][0]
        hit = (d.get('id') == wid)
        nc = res['n_codewords']
        # 间距必须永远准。ID 在码字够多时是硬门槛；s≈2 的 1:1 裁剪只留下
        # 十几个码字（4K 屏被裁进 1920×1080 窗），是**数据构造**的稀疏角，
        # 真部署里抓整屏会有全部 96 个 —— 稀疏时只报不卡。
        hard = nc >= 20
        good = (e < 2e-3 and (hit or not hard))
        ok &= good
        print(f'  s={meta["s"]:.3f} 框 {len(xyxy)}/{meta["n_boxes"]}  '
              f'码字 {nc}  空槽 {d.get("n_empty_slots", "-")}  '
              f'间距误差 {e * 100:.3f}%  θ {res["theta_deg"]:+.2f}°  '
              f'ID {"0x%05X" % d["id"] if hit else ("miss" if nc else "-")}  '
              f'{"OK" if good else "FAIL"}'
              f'{"" if hard else "  (稀疏，ID 不作硬门槛)"}')

    print('— 完整样本（含噪声/取景）—')
    for k in range(n):
        img, labels, meta = make_sample(
            0x1E240, rng, carrier_paths, ALPHA, apply_noise=True)
        good = (img.shape[:2] == (WINDOW_H, WINDOW_W)
                and img.dtype == np.uint8 and 0 < len(labels) <= 2000)
        ok &= good
        print(f'  [{k}] {img.shape[1]}x{img.shape[0]}  框 {len(labels)}  '
              f's={meta["s"]:.3f}  screen={meta["screen"]}  noise={meta["noise"]}  '
              f'{"OK" if good else "FAIL"}')

    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


# ───────────────────── 批量生成 ─────────────────────

def _write_split(out_dir: str, split: str, idx: int,
                 img: np.ndarray, labels: List[List[float]], meta: dict) -> None:
    stem = f'{split}_{idx:06d}'
    cv2.imwrite(os.path.join(out_dir, 'images', split, stem + '.png'), img)
    with open(os.path.join(out_dir, 'labels', split, stem + '.txt'), 'w') as f:
        for cls, cx, cy, bw, bh in labels:
            f.write(f'{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
    with open(os.path.join(out_dir, 'meta', split, stem + '.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False)


def _one(args_tuple):
    """multiprocessing 入口：返回 (split, idx, img, labels, meta)。"""
    (idx, split, wm_id, seed, carrier_paths, alpha, s_min, s_max,
     apply_noise, stripe_width) = args_tuple
    rng = np.random.RandomState(seed)
    cache: dict = {}
    img, labels, meta = make_sample(
        wm_id, rng, carrier_paths, alpha, s_min, s_max, apply_noise,
        stripe_width, cache)
    return split, idx, img, labels, meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='v2 数据集生成器（96 码字检测）')
    p.add_argument('--num_samples', type=int, default=100)
    p.add_argument('--output_dir', type=str, default=None,
                   help='输出目录；--selftest 时可省略')
    p.add_argument('--carrier_root', type=str, default='/data1/lpl/datasets',
                   help='载体图根目录（只索引路径，按需读像素）')
    p.add_argument('--carrier_path', type=str, default=None,
                   help='单张载体图（优先于 carrier_root，冒烟用）')
    p.add_argument('--alpha', type=float, default=ALPHA)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--s_min', type=float, default=0.5)
    p.add_argument('--s_max', type=float, default=2.2)
    p.add_argument('--no_noise', action='store_true', help='不加噪声（冒烟）')
    p.add_argument('--stripe_width', type=int, default=2,
                   help='条纹占空 (period=4)。2 → 25% 像素写入')
    p.add_argument('--id_min', type=int, default=0)
    p.add_argument('--id_max', type=int, default=(1 << 20) - 1,
                   help='watermark ID 上界（20 bit）')
    p.add_argument('--jobs', type=int, default=1,
                   help='并行进程数。/data1 是 IO 瓶颈，别开太大')
    p.add_argument('--selftest', action='store_true', help='只跑自检后退出')
    args = p.parse_args(argv)

    if args.carrier_path:
        carrier_paths = [args.carrier_path] if os.path.exists(args.carrier_path) else []
    else:
        carrier_paths = discover_carriers(args.carrier_root)
    print(f'carriers: {len(carrier_paths)} images')

    if args.selftest:
        return selftest(carrier_paths)

    if not args.output_dir:
        p.error('--output_dir is required unless --selftest')
    out = args.output_dir
    for split in ('train', 'val'):
        for sub in ('images', 'labels', 'meta'):
            os.makedirs(os.path.join(out, sub, split), exist_ok=True)

    rng = np.random.RandomState(args.seed)
    n_val = int(round(args.num_samples * args.val_frac))
    n_train = args.num_samples - n_val

    tasks = []
    for idx in range(args.num_samples):
        split = 'val' if idx < n_val else 'train'
        k = idx if split == 'val' else idx - n_val
        wm_id = int(rng.randint(args.id_min, args.id_max + 1))
        seed = int(rng.randint(0, 2 ** 31 - 1))
        tasks.append((k, split, wm_id, seed, carrier_paths, args.alpha,
                      args.s_min, args.s_max, not args.no_noise, args.stripe_width))

    counts = {'train': 0, 'val': 0}
    stats = {'s': [], 'n_boxes': [], 'noise': {}}
    done = 0

    def _consume(res):
        nonlocal done
        split, idx, img, labels, meta = res
        _write_split(out, split, idx, img, labels, meta)
        counts[split] += 1
        stats['s'].append(meta['s'])
        stats['n_boxes'].append(meta['n_boxes'])
        for nname in meta['noise']:
            stats['noise'][nname] = stats['noise'].get(nname, 0) + 1
        done += 1
        if done % 20 == 0 or done == len(tasks):
            print(f'  {done}/{len(tasks)}  train={counts["train"]} val={counts["val"]}')

    if args.jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.jobs) as pool:
            for res in pool.imap_unordered(_one, tasks, chunksize=2):
                _consume(res)
    else:
        for t in tasks:
            _consume(_one(t))

    with open(os.path.join(out, 'watermark.yaml'), 'w') as f:
        f.write(f"""# Auto-generated YOLO dataset config (v2, 96 codewords)
path: {os.path.abspath(out)}
train: images/train
val: images/val

names:
  0: codeword
""")

    ss = np.asarray(stats['s'], dtype=np.float64)
    manifest = {
        'generator': 'watermark_locator/v2/dataset/generate_dataset_v2.py',
        'layout': 'v2-alternating-rows',
        'detection_unit': 'codeword (160x135 @ s=1, 1/4 of an M-block)',
        'num_classes': 1,
        'class_names': ['codeword'],
        'window': [WINDOW_W, WINDOW_H],
        'alpha': args.alpha,
        'stripe': {'angle': 45.0, 'period': 4, 'width': args.stripe_width},
        's_range': [args.s_min, args.s_max],
        's_min_obs': float(ss.min()) if ss.size else None,
        's_max_obs': float(ss.max()) if ss.size else None,
        's_mean': float(ss.mean()) if ss.size else None,
        'counts': counts,
        'n_boxes_mean': float(np.mean(stats['n_boxes'])) if stats['n_boxes'] else None,
        'noise_counts': stats['noise'],
        'seed': args.seed,
        'carriers': len(carrier_paths),
    }
    with open(os.path.join(out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nwrote {out}")
    print(f"  train={counts['train']}  val={counts['val']}")
    if ss.size:
        print(f"  s: min={ss.min():.3f} mean={ss.mean():.3f} max={ss.max():.3f}")
    print(f"  noise: {stats['noise']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
