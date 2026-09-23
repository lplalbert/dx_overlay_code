"""
合成数据集生成器

- 斜条纹模板: 45°方向, 周期4, 宽度2
- 载体图像: 纯白图（后续可指定真实载体）
- Pair噪声: 从[identity, wechat, tile_crop, pimog]随机选2种组合
- 输出: vv1 (U-Net分割mask) + vv2 (YOLO bbox)

用法:
    python generate_dataset.py --num_samples 100 --output_dir ./data --alpha 0.016
"""

import argparse
import json
import math
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import rs_encode, checkerboard_locator_positions, LOCATOR_CODEWORD_INDEX

# ───────────────────── 常量 ─────────────────────

WATERMARK_GRID_SIZE = 8
WATERMARK_GRID_CELLS = WATERMARK_GRID_SIZE * WATERMARK_GRID_SIZE
CHANNEL_ENCODING_ALPHA = 254
MESSAGE_ROW, MESSAGE_COL = 2, 2

BLOCK_ROWS, BLOCK_COLS = 4, 6
SCREEN_W, SCREEN_H = 1920, 1080
BLOCK_H = SCREEN_H // BLOCK_ROWS
BLOCK_W = SCREEN_W // BLOCK_COLS
MSG_H = BLOCK_H // 2
MSG_W = BLOCK_W // 2


# ───────────────────── 斜条纹模板 ─────────────────────
# 学习自 generate_yellow_white_template.py 的 _stripe_mask()
# 关键: 全局屏幕坐标, 45°用 xx+yy, fmod 相位

def stripe_mask(width, height, angle=45.0, period=4, stripe_width=2):
    """
    生成全局斜条纹掩码 (与原生 dx_overlay.exe 一致)。

    45° → coordinate = xx + yy
    135° → coordinate = xx - yy
    0° → coordinate = yy
    90° → coordinate = xx
    其他角度 → (sx/scale)*xx + (sy/scale)*yy

    相位: fmod(coordinate, period) < stripe_width
    """
    if stripe_width == period:
        return np.ones((height, width), dtype=bool)
    if stripe_width == 0:
        return np.zeros((height, width), dtype=bool)

    angle = float(np.float32(angle))
    yy, xx = np.indices((height, width), dtype=np.float64)
    effective_angle = 0.0 if angle == 180.0 else angle

    if effective_angle == 45.0:
        coordinate = xx + yy
    elif effective_angle == 135.0:
        coordinate = xx - yy
    elif effective_angle == 0.0:
        coordinate = yy
    elif effective_angle == 90.0:
        coordinate = xx
    else:
        radians = effective_angle * math.pi / 180.0
        sx = math.sin(radians)
        sy = math.cos(radians)
        scale = max(abs(sx), abs(sy))
        coordinate = (sx / scale) * xx + (sy / scale) * yy

    phase = np.fmod(coordinate, float(period))
    phase[phase < 0] += period
    return phase < stripe_width


def gen_diagonal_stripe_tp(blockRow, period=4, width=2, angle_deg=45):
    """兼容接口: 生成局部块内的斜条纹 (实际应使用全局 stripe_mask)。"""
    mask = stripe_mask(blockRow, blockRow, angle_deg, period, width)
    return (mask * 125).astype(np.uint8)


def gen_rect_tp(blockRow):
    return np.ones((blockRow, blockRow), dtype=np.uint8) * 125


def gen_block_single_uv(k=1, kv=1, blockRow=32, ratio_u=10, ratio_v=8,
                        u_tp_fn=gen_rect_tp, v_tp_fn=None, channel_mode='cb'):
    """
    生成单个码字块的BGRA纹理。

    channel_mode:
      'cb' — YCbCr域嵌入 (Cb/Cr 色度偏移), 对应 HLSL shader 的 Cb 通道
      'b'  — RGB域嵌入 (B通道), 对应 generate_yellow_white_template.py 的黄/白编码
    """
    if v_tp_fn is None:
        v_tp_fn = lambda n: gen_diagonal_stripe_tp(n, period=4, width=2, angle_deg=45)

    if channel_mode == 'b':
        # B通道编码: 黄色(信号)=B:0, 白色(中性)=B:255, R=G=255
        is_signal = (ratio_u != 0) or (ratio_v != 0)
        b_val = 0 if is_signal else 255
        b = np.full((blockRow, blockRow), b_val, dtype=np.uint8)
        g = np.full((blockRow, blockRow), 255, dtype=np.uint8)
        r = np.full((blockRow, blockRow), 255, dtype=np.uint8)
        alpha = np.full((blockRow, blockRow), CHANNEL_ENCODING_ALPHA, dtype=np.uint8)
        return cv2.merge((b, g, r, alpha))

    # Cb通道编码: YCbCr域
    y = np.full((blockRow, blockRow), 128, dtype=np.int16)
    cr = np.full((blockRow, blockRow), 128, dtype=np.int16)
    cb = np.full((blockRow, blockRow), 128, dtype=np.int16)

    if ratio_u != 0:
        cr_delta = u_tp_fn(blockRow).astype(np.int16) * ratio_u // 10
        cr += cr_delta if k == 1 else -cr_delta

    if ratio_v != 0:
        cb_delta = v_tp_fn(blockRow).astype(np.int16) * ratio_v // 10
        cb += cb_delta if kv == 1 else -cb_delta

    y = np.clip(y, 0, 255).astype(np.uint8)
    cr = np.clip(cr, 0, 255).astype(np.uint8)
    cb = np.clip(cb, 0, 255).astype(np.uint8)
    alpha = np.full((blockRow, blockRow), CHANNEL_ENCODING_ALPHA, dtype=np.uint8)
    return cv2.merge((cb, cr, y, alpha))


def gen_wm_block(pattern_64, block_size=64, ratio_u=10, ratio_v=8,
                 type_val=0, inverse=False, v_tp_fn=None, channel_mode='cb'):
    """从64值pattern生成完整的8x8水印块。

    参考 generate_yellow_white_template.py 的黄色/白色编码:
    - "黄色"单元格 (信号): 有 Cb/Cr 色度差, 条纹掩码调制其显隐
    - "白色"单元格 (无信号): 色度恒为 128 (中性), 始终不变
    """
    seq = np.asarray(pattern_64).reshape(-1)
    assert seq.size == WATERMARK_GRID_CELLS

    def _yellow(k, kv):
        """黄色(信号)单元格: 有 chroma delta。"""
        return gen_block_single_uv(k, kv, block_size, ratio_u, ratio_v,
                                   v_tp_fn=v_tp_fn, channel_mode=channel_mode)

    def _white(k, kv):
        """白色(无信号)单元格: 中性 128, 无 chroma delta。"""
        return gen_block_single_uv(k, kv, block_size, 0, 0,
                                   v_tp_fn=v_tp_fn, channel_mode=channel_mode)

    if type_val == 0:
        if not inverse:
            t1 = _white(1, 1)
            t0 = _yellow(0, 0)
        else:
            t1 = _yellow(0, 1)
            t0 = _white(1, 0)
    else:
        if not inverse:
            t1 = _yellow(1, 0)
            t0 = _white(0, 1)
        else:
            t1 = _white(0, 0)
            t0 = _yellow(1, 1)

    img = np.empty((WATERMARK_GRID_SIZE * block_size,
                    WATERMARK_GRID_SIZE * block_size, 4), dtype=np.uint8)
    for j, bit in enumerate(seq):
        row_idx, col_idx = divmod(j, WATERMARK_GRID_SIZE)
        r0, r1 = row_idx * block_size, (row_idx + 1) * block_size
        c0, c1 = col_idx * block_size, (col_idx + 1) * block_size
        img[r0:r1, c0:c1, :] = t1 if bit == 1 else t0
    return img


# ───────────────────── 噪声模型 (参考fftmask/pair噪声) ─────────────────────

# 预计算DCT矩阵
_DCT_MAT = np.zeros((8, 8), dtype=np.float64)
for u in range(8):
    for x in range(8):
        alpha_u = math.sqrt(1.0 / 8.0) if u == 0 else math.sqrt(2.0 / 8.0)
        _DCT_MAT[u, x] = alpha_u * math.cos((2 * x + 1) * u * math.pi / 16.0)
_DCT_MAT_T = _DCT_MAT.T.copy()

_JPEG_ZIGZAG_IDX = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]


def _get_zigzag_mask(zigzag_keep):
    mask = np.zeros((8, 8), dtype=np.float64)
    for k in range(min(zigzag_keep, 64)):
        r, c = _JPEG_ZIGZAG_IDX[k]
        mask[r, c] = 1.0
    return mask


def _whole_plane_dct_hf_zero(plane, keep_ratio):
    h, w = plane.shape
    ph = int(math.ceil(h / 8.0) * 8)
    pw = int(math.ceil(w / 8.0) * 8)
    padded = np.zeros((ph, pw), dtype=np.float64)
    padded[:h, :w] = plane - 128.0
    nbh, nbw = ph // 8, pw // 8
    blocks = padded.reshape(nbh, 8, nbw, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    dct_coeff = np.einsum('ij,bjk,kl->bil', _DCT_MAT, blocks, _DCT_MAT_T)
    keep = max(1, int(64 * keep_ratio))
    mask = _get_zigzag_mask(keep)
    dct_coeff *= mask
    recon = np.einsum('ij,bjk,kl->bil', _DCT_MAT_T, dct_coeff, _DCT_MAT)
    recon = recon.reshape(nbh, nbw, 8, 8).transpose(0, 2, 1, 3).reshape(ph, pw)
    return np.clip(recon[:h, :w] + 128.0, 0.0, 255.0)


def _chroma_420_downsample(plane):
    h, w = plane.shape
    h_even = h + (h % 2)
    w_even = w + (w % 2)
    padded = np.zeros((h_even, w_even), dtype=plane.dtype)
    padded[:h, :w] = plane
    ch, cw = h_even // 2, w_even // 2
    return padded.reshape(ch, 2, cw, 2).mean(axis=(1, 3))


def _chroma_420_upsample(plane2, h, w):
    return np.repeat(np.repeat(plane2, 2, axis=0), 2, axis=1)[:h, :w]


def add_wechat_noise(image, zigzag_keep=21):
    """模拟微信JPEG压缩: 下采样→YCbCr→DCT高频清零→重建"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]

    # 下采样→上采样
    small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # YCbCr → DCT高频清零
    keep_ratio = zigzag_keep / 64.0
    img_f = image.astype(np.float64)
    b, g, r = img_f[..., 0], img_f[..., 1], img_f[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    cb2 = _chroma_420_downsample(cb)
    cr2 = _chroma_420_downsample(cr)

    yq = _whole_plane_dct_hf_zero(y, keep_ratio)
    cbq = _whole_plane_dct_hf_zero(cb2, keep_ratio)
    crq = _whole_plane_dct_hf_zero(cr2, keep_ratio)

    hh, ww = y.shape
    cb_up = _chroma_420_upsample(cbq, hh, ww)
    cr_up = _chroma_420_upsample(crq, hh, ww)

    out_b = yq + 1.772 * (cb_up - 128.0)
    out_g = yq - 0.344136 * (cb_up - 128.0) - 0.714136 * (cr_up - 128.0)
    out_r = yq + 1.402 * (cr_up - 128.0)
    out = np.stack([out_b, out_g, out_r], axis=-1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def add_tile_rotate_crop_noise(image, mask=None, bboxes=None,
                               angle_range=(-5, 5), max_shift=0.5):
    """循环平移+旋转: 3x3拼接→旋转→中心裁剪。同步变换mask和bboxes。"""
    h, w = image.shape[:2]
    is_gray = len(image.shape) == 2

    tiled = np.tile(image, (3, 3)) if is_gray else np.tile(image, (3, 3, 1))

    angle = np.random.uniform(angle_range[0], angle_range[1])
    center = (tiled.shape[1] / 2, tiled.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(tiled, M, (tiled.shape[1], tiled.shape[0]),
                             borderMode=cv2.BORDER_WRAP)

    crop_h, crop_w = h, w
    max_offset_x = min(int(w * max_shift), (3 * w - crop_w) // 2 - 1)
    max_offset_y = min(int(h * max_shift), (3 * h - crop_h) // 2 - 1)
    crop_cx = w + np.random.randint(-max_offset_x, max_offset_x + 1)
    crop_cy = h + np.random.randint(-max_offset_y, max_offset_y + 1)
    x0 = crop_cx - crop_w // 2
    y0 = crop_cy - crop_h // 2
    cropped = rotated[y0:y0 + crop_h, x0:x0 + crop_w]
    if cropped.shape[:2] != (h, w):
        cropped = cv2.resize(cropped, (w, h))

    # ── 同步变换 mask ──
    if mask is not None:
        tiled_m = np.tile(mask, (3, 3))
        rotated_m = cv2.warpAffine(tiled_m, M, (tiled.shape[1], tiled.shape[0]),
                                    borderMode=cv2.BORDER_WRAP)
        cropped_m = rotated_m[y0:y0 + crop_h, x0:x0 + crop_w]
        if cropped_m.shape[:2] != (h, w):
            cropped_m = cv2.resize(cropped_m, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cropped_m

    # ── 同步变换 bboxes (tile 3x3 → 9份拷贝) ──
    if bboxes is not None and len(bboxes) > 0:
        new_bboxes = []
        for bbox in bboxes:
            cls, cx_n, cy_n, bw_n, bh_n = bbox
            # 生成3x3 tile的9份拷贝
            for ti in range(3):
                for tj in range(3):
                    cx_px = cx_n * w + tj * w
                    cy_px = cy_n * h + ti * h
                    bw_px = bw_n * w
                    bh_px = bh_n * h
                    corners = np.array([
                        [cx_px - bw_px/2, cy_px - bh_px/2],
                        [cx_px + bw_px/2, cy_px - bh_px/2],
                        [cx_px + bw_px/2, cy_px + bh_px/2],
                        [cx_px - bw_px/2, cy_px + bh_px/2],
                    ], dtype=np.float32)
                    ones = np.ones((4, 1), dtype=np.float32)
                    corners_h = np.hstack([corners, ones])
                    rotated_corners = (M @ corners_h.T).T[:, :2]
                    rotated_corners[:, 0] -= x0
                    rotated_corners[:, 1] -= y0
                    x_min = np.clip(rotated_corners[:, 0].min(), 0, w)
                    y_min = np.clip(rotated_corners[:, 1].min(), 0, h)
                    x_max = np.clip(rotated_corners[:, 0].max(), 0, w)
                    y_max = np.clip(rotated_corners[:, 1].max(), 0, h)
                    if x_max - x_min > 2 and y_max - y_min > 2:
                        new_cx = (x_min + x_max) / 2 / w
                        new_cy = (y_min + y_max) / 2 / h
                        new_bw = (x_max - x_min) / w
                        new_bh = (y_max - y_min) / h
                        new_bboxes.append([cls, new_cx, new_cy, new_bw, new_bh])
        bboxes = new_bboxes

    return cropped, mask, bboxes


def add_pimog_noise(image, mask=None, bboxes=None):
    """PIMOG噪声: 透视+光照扭曲+摩尔纹+高斯噪声。同步变换mask和bboxes。"""
    h, w = image.shape[:2]
    img_f = image.astype(np.float64) / 255.0

    # 轻微透视 (随机四角偏移, ±30px ≈ 1.5% 宽度)
    src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst_pts = src_pts + np.random.uniform(-30, 30, src_pts.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_f = cv2.warpPerspective(img_f, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # ── 同步变换 mask ──
    if mask is not None:
        mask = cv2.warpPerspective(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ── 同步变换 bboxes ──
    if bboxes is not None and len(bboxes) > 0:
        new_bboxes = []
        for bbox in bboxes:
            cls, cx_n, cy_n, bw_n, bh_n = bbox
            cx_px, cy_px = cx_n * w, cy_n * h
            bw_px, bh_px = bw_n * w, bh_n * h
            corners = np.array([
                [cx_px - bw_px/2, cy_px - bh_px/2],
                [cx_px + bw_px/2, cy_px - bh_px/2],
                [cx_px + bw_px/2, cy_px + bh_px/2],
                [cx_px - bw_px/2, cy_px + bh_px/2],
            ], dtype=np.float32)
            ones = np.ones((4, 1), dtype=np.float32)
            corners_h = np.hstack([corners, ones])
            warped = (M @ corners_h.T).T
            warped = warped[:, :2] / warped[:, 2:3]  # 透视除法
            x_min = np.clip(warped[:, 0].min(), 0, w)
            y_min = np.clip(warped[:, 1].min(), 0, h)
            x_max = np.clip(warped[:, 0].max(), 0, w)
            y_max = np.clip(warped[:, 1].max(), 0, h)
            if x_max - x_min > 2 and y_max - y_min > 2:
                new_cx = (x_min + x_max) / 2 / w
                new_cy = (y_min + y_max) / 2 / h
                new_bw = (x_max - x_min) / w
                new_bh = (y_max - y_min) / h
                new_bboxes.append([cls, new_cx, new_cy, new_bw, new_bh])
        bboxes = new_bboxes

    # 光照扭曲
    a = 0.7 + random.random() * 0.2
    b = 1.1 + random.random() * 0.2
    direction = random.randint(1, 4)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64)
    if direction in (1, 3):
        t = Y / max(h - 1, 1)
    else:
        t = X / max(w - 1, 1)
    if direction in (3, 4):
        t = 1.0 - t
    val = a + (b - a) * t
    img_f *= val[..., np.newaxis] * 0.85

    # 摩尔纹
    theta = np.random.uniform(0, np.pi)
    cx, cy = np.random.uniform(0, w), np.random.uniform(0, h)
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    z1 = 0.5 + 0.5 * np.cos(2 * np.pi * dist)
    phase = np.cos(theta) * X + np.sin(theta) * Y
    z2 = 0.5 + 0.5 * np.cos(phase)
    moire = (np.minimum(z1, z2)) * 2 - 1
    img_f += moire[..., np.newaxis] * 0.15

    # 高斯噪声
    img_f += np.random.normal(0, 0.03, img_f.shape)

    return np.clip(np.round(img_f * 255), 0, 255).astype(np.uint8), mask, bboxes


def apply_pair_noise(image, mask=None, bboxes=None, rng=None):
    """Pair噪声: 从[identity, wechat, tile_crop, pimog]随机选2种组合。几何噪声同步变换mask和bboxes。"""
    noise_pool = ['identity', 'wechat', 'tile_crop', 'pimog']
    first = rng.choice(noise_pool) if rng else np.random.choice(noise_pool)
    second_pool = [n for n in noise_pool if n != first]
    second = rng.choice(second_pool) if rng else np.random.choice(second_pool)

    for noise_type in [first, second]:
        if noise_type == 'identity':
            pass
        elif noise_type == 'wechat':
            image = add_wechat_noise(image)  # 非几何，标签不变
        elif noise_type == 'tile_crop':
            image, mask, bboxes = add_tile_rotate_crop_noise(image, mask, bboxes)
        elif noise_type == 'pimog':
            image, mask, bboxes = add_pimog_noise(image, mask, bboxes)

    return image, mask, bboxes, [first, second]


# ───────────────────── alpha融合 ─────────────────────

def alpha_blend_watermark(carrier_bgr, watermark_bgra, alpha, channel_mode='cb'):
    """模拟HLSL shader的alpha融合。

    channel_mode='cb': YCbCr域嵌入, YCbCr→RGB 后混合
    channel_mode='b':  B通道嵌入, 直接 RGB 混合 (黄/白)
    """
    if channel_mode == 'b':
        wm_bgr = watermark_bgra[:, :, :3].astype(np.float32)
    else:
        cb = watermark_bgra[:, :, 0].astype(np.float32)
        cr = watermark_bgra[:, :, 1].astype(np.float32)
        y  = watermark_bgra[:, :, 2].astype(np.float32)

        cr_dev = cr - 128.0
        cb_dev = cb - 128.0

        r = y + 1.402 * cr_dev
        g = y - 0.714136 * cr_dev - 0.344136 * cb_dev
        b = y + 1.772 * cb_dev

        wm_rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.float32)
        wm_bgr = wm_rgb[:, :, ::-1]

    carrier = carrier_bgr.astype(np.float32)
    result = carrier * (1.0 - alpha) + wm_bgr * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ───────────────────── 定位块位置 ─────────────────────

def get_locator_positions():
    """返回定位块在block网格中的位置。"""
    positions = []
    for i in range(BLOCK_ROWS):
        for j in range(BLOCK_COLS):
            k = (j + (i % 2) * 2) % 4
            if k == 3:
                positions.append((i, j))
    return positions


def get_locator_abs_rect(i, j):
    x = j * BLOCK_W + MSG_W
    y = i * BLOCK_H + MSG_H
    return (x, y, MSG_W, MSG_H)


# ───────────────────── 数据集生成 ─────────────────────

def generate_one_sample(wm_id, fix_fg_matrix, locator_pattern, alpha, rng,
                        carrier_img=None, apply_noise=True, channel_mode='cb'):
    """生成一个训练样本。"""
    from rs_gen_Syn_template_nums_dual import get_wm_seq
    wm_seq = list(get_wm_seq(wm_id))
    wm_seq[-1] = LOCATOR_CODEWORD_INDEX

    ext_matrix = np.vstack([fix_fg_matrix, locator_pattern.reshape(1, 64)])

    # 码字模板 (均匀填充, 条纹在全局级别统一施加)
    templates = []
    for idx in range(17):
        tmpl = gen_wm_block(ext_matrix[idx], block_size=64, v_tp_fn=gen_rect_tp,
                            channel_mode=channel_mode)
        templates.append(tmpl)

    all_messages = [wm_seq[i*4:(i+1)*4] for i in range(4)]

    def resize_tmpl(img, w, h):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    wm_blocks = []
    for msgs in all_messages:
        block = np.empty((BLOCK_H, BLOCK_W, 4), dtype=np.uint8)
        block[:, :, :3] = 128
        block[:, :, 3] = CHANNEL_ENCODING_ALPHA
        a, b, c, d = msgs
        msgs_reorder = [a, c, b, d]
        for bi in range(MESSAGE_ROW):
            for bj in range(MESSAGE_COL):
                idx = (bi * MESSAGE_COL + bj) % 4
                img_type = msgs_reorder[idx]
                tmpl = resize_tmpl(templates[img_type], MSG_W, MSG_H)
                block[bi*MSG_H:(bi+1)*MSG_H, bj*MSG_W:(bj+1)*MSG_W, :] = tmpl
        wm_blocks.append(block)

    wm_full = np.empty((BLOCK_ROWS * BLOCK_H, BLOCK_W * BLOCK_COLS, 4), dtype=np.uint8)
    wm_full[:, :, :3] = 128
    wm_full[:, :, 3] = CHANNEL_ENCODING_ALPHA

    locator_mask = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)
    bboxes = []

    for i in range(BLOCK_ROWS):
        for j in range(BLOCK_COLS):
            k = (j + (i % 2) * 2) % 4
            wm_full[i*BLOCK_H:(i+1)*BLOCK_H, j*BLOCK_W:(j+1)*BLOCK_W, :] = wm_blocks[k]
            if k == 3:
                x, y, w, h = get_locator_abs_rect(i, j)
                locator_mask[y:y+h, x:x+w] = 255
                cx = (x + w / 2) / SCREEN_W
                cy = (y + h / 2) / SCREEN_H
                bboxes.append([0, cx, cy, w / SCREEN_W, h / SCREEN_H])

    # ── 全局斜条纹调制 (学习自 generate_yellow_white_template.py) ──
    # 条纹 keep 掩码只作用于"黄色"(信号)单元格; "白色"(中性)单元格始终不变
    keep = stripe_mask(SCREEN_W, SCREEN_H, angle=45.0, period=4, stripe_width=2)
    if channel_mode == 'b':
        # B通道: 黄色单元格 B=0, 条纹外恢复 B=255 (白色不受影响)
        wm_full[:, :, 0][~keep] = 255
    else:
        # Cb/Cr通道: 非条纹区域的色度偏移清零 (白色=128不受影响)
        for ch in range(2):
            delta = wm_full[:, :, ch].astype(np.float32) - 128.0
            delta[~keep] = 0.0
            wm_full[:, :, ch] = np.clip(delta + 128.0, 0, 255).astype(np.uint8)

    # alpha融合
    if carrier_img is not None:
        carrier = cv2.resize(carrier_img, (SCREEN_W, SCREEN_H))
    else:
        carrier = np.full((SCREEN_H, SCREEN_W, 3), 255, dtype=np.uint8)

    blended = alpha_blend_watermark(carrier, wm_full, alpha, channel_mode=channel_mode)

    # Pair噪声 (几何噪声同步变换mask和bboxes)
    noise_types = ['identity', 'identity']
    if apply_noise:
        blended, locator_mask, bboxes, noise_types = apply_pair_noise(
            blended, locator_mask, bboxes, rng)

    return blended, bboxes, locator_mask, noise_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--alpha", type=float, default=0.016)
    parser.add_argument("--carrier_dir", type=str, default=None,
                        help="载体图像目录 (默认纯白)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--no_noise", action="store_true", help="不加噪声")
    parser.add_argument("--channel_mode", type=str, default="cb", choices=["b", "cb"],
                        help="嵌入域: b=B通道(黄/白), cb=Cb/Cr通道(色度)")
    parser.add_argument("--carrier_path", type=str, default=None,
                        help="单张载体图像路径 (优先于 carrier_dir)")
    args = parser.parse_args()

    global SCREEN_W, SCREEN_H, BLOCK_H, BLOCK_W, MSG_H, MSG_W
    SCREEN_W, SCREEN_H = args.image_width, args.image_height
    BLOCK_H = SCREEN_H // BLOCK_ROWS
    BLOCK_W = SCREEN_W // BLOCK_COLS
    MSG_H = BLOCK_H // 2
    MSG_W = BLOCK_W // 2

    rng = np.random.RandomState(args.seed)

    out = args.output_dir
    dirs = {
        'vv1_images': os.path.join(out, 'vv1', 'images'),
        'vv1_masks': os.path.join(out, 'vv1', 'masks'),
        'vv2_images': os.path.join(out, 'vv2', 'images'),
        'vv2_labels': os.path.join(out, 'vv2', 'labels'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    from generate_locator_pattern import FIX_FG_MATRIX
    locator_path = os.path.join(os.path.dirname(__file__), '..', 'locator_pattern.npy')
    locator_pattern = np.load(locator_path) if os.path.exists(locator_path) else FIX_FG_MATRIX[0].copy()

    # 载体图像
    carrier_images = []
    if args.carrier_path and os.path.exists(args.carrier_path):
        img = cv2.imread(args.carrier_path)
        if img is not None:
            carrier_images.append(img)
            print(f"Loaded carrier: {args.carrier_path} {img.shape[1]}x{img.shape[0]}")
    elif args.carrier_dir and os.path.exists(args.carrier_dir):
        for f in os.listdir(args.carrier_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img = cv2.imread(os.path.join(args.carrier_dir, f))
                if img is not None:
                    carrier_images.append(img)
        print(f"Loaded {len(carrier_images)} carrier images")

    print(f"Generating {args.num_samples} samples (alpha={args.alpha})")
    print(f"Screen: {SCREEN_W}x{SCREEN_H}, Locator blocks: {len(get_locator_positions())}")

    for idx in range(args.num_samples):
        wm_id = rng.randint(0, 16**5 - 1)
        carrier = carrier_images[rng.randint(len(carrier_images))] if carrier_images else None

        blended, bboxes, mask, noise_types = generate_one_sample(
            wm_id, FIX_FG_MATRIX, locator_pattern, args.alpha, rng,
            carrier_img=carrier, apply_noise=not args.no_noise,
            channel_mode=args.channel_mode
        )

        name = f"{idx:06d}"
        cv2.imwrite(os.path.join(dirs['vv1_images'], f"{name}.png"), blended)
        cv2.imwrite(os.path.join(dirs['vv1_masks'], f"{name}.png"), mask)
        cv2.imwrite(os.path.join(dirs['vv2_images'], f"{name}.png"), blended)

        with open(os.path.join(dirs['vv2_labels'], f"{name}.txt"), 'w') as f:
            for bbox in bboxes:
                f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{args.num_samples}  noise={noise_types}")

    meta = {
        "num_samples": args.num_samples,
        "alpha": args.alpha,
        "channel_mode": args.channel_mode,
        "screen_size": [SCREEN_W, SCREEN_H],
        "template": "diagonal_stripe_45deg_period4_width2",
        "noise": "pair(identity,wechat,tile_crop,pimog)",
        "locator_positions": [list(p) for p in get_locator_positions()],
        "num_locator_blocks": len(get_locator_positions()),
    }
    with open(os.path.join(out, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {out}")


if __name__ == '__main__':
    main()
