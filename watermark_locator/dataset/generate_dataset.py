"""
合成数据集生成器

- 斜条纹模板: 45°方向, 周期4, 宽度2
- 载体图像: 纯白图（后续可指定真实载体）
- Pair噪声: 从[identity, wechat, tile_crop, pimog]随机选2种组合
  - wechat = wechat_worst_case_compressor.py (真 JPEG q60 4:2:0)
  - pimog  = physical_moire.py (屏-摄摩尔纹/曝光/PSF/CFA/ISP)
- 输出: vv1 (U-Net分割mask) + vv2 (YOLO bbox)

用法:
    python generate_dataset.py --num_samples 100 --output_dir ./data --alpha 0.032
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import LOCATOR_CODEWORD_INDEX

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
                        u_tp_fn=gen_rect_tp, v_tp_fn=None, channel_mode='b'):
    """
    生成单个码字块的BGRA纹理。

    channel_mode:
      'b' / 'yw' — 黄/白 RGB 编码 (对应 generate_yellow_white_template.py
                   的 TemplateColorMode::YellowWhiteRgb): 模板**只含 0 和 255**。
                   黄色(信号) = (B,G,R) = (0,255,255); 白色(中性) = (255,255,255)
    """
    if v_tp_fn is None:
        v_tp_fn = lambda n: gen_diagonal_stripe_tp(n, period=4, width=2, angle_deg=45)

    # 黄/白 RGB 编码: 模板只取 0 和 255 两个值
    #   黄色(信号) = (B,G,R)=(0,255,255)  即 B=0
    #   白色(中性) = (B,G,R)=(255,255,255) 即 B=255
    is_signal = (ratio_u != 0) or (ratio_v != 0)
    b_val = 0 if is_signal else 255
    b = np.full((blockRow, blockRow), b_val, dtype=np.uint8)
    g = np.full((blockRow, blockRow), 255, dtype=np.uint8)
    r = np.full((blockRow, blockRow), 255, dtype=np.uint8)
    alpha = np.full((blockRow, blockRow), CHANNEL_ENCODING_ALPHA, dtype=np.uint8)
    return cv2.merge((b, g, r, alpha))


def gen_wm_block(pattern_64, block_size=64, ratio_u=10, ratio_v=8,
                 type_val=0, inverse=False, v_tp_fn=None, channel_mode='b'):
    """从64值pattern生成完整的8x8水印块。

    参考 generate_yellow_white_template.py 的黄色/白色编码:
    - "黄色"单元格 (信号): B=0 (模板值 0), 条纹掩码调制其显隐
    - "白色"单元格 (无信号): B=255 (模板值 255), 始终不变
    模板只含 0 / 255 两个值。
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


# ───────────────────── 噪声模型 (pair噪声) ─────────────────────
# 微信压缩  ← wechat_worst_case_compressor.py  (真实 JPEG 量化表, q60 4:2:0)
# 拍照模拟  ← physical_moire.py                (屏-摄摩尔纹/曝光/PSF/CFA/ISP)
# 几何噪声 (tile_crop) 自实现, 同步变换 mask/bboxes。

_WECHAT_COMPRESSOR = None
_WECHAT_PRESET = 'mainstream_worst'
_MOIRE_SIM = None


def _get_wechat_compressor(preset=None):
    """懒加载 wechat_worst_case_compressor.WeChatWorstCaseCompressor。"""
    global _WECHAT_COMPRESSOR, _WECHAT_PRESET
    preset = preset or _WECHAT_PRESET
    if _WECHAT_COMPRESSOR is None or _WECHAT_PRESET != preset:
        from wechat_worst_case_compressor import WeChatWorstCaseCompressor
        _WECHAT_COMPRESSOR = WeChatWorstCaseCompressor.from_preset(preset)
        _WECHAT_PRESET = preset
    return _WECHAT_COMPRESSOR


def add_wechat_noise(image, preset=None, quality=None, blur_radius=None,
                     subsampling=None):
    """微信最坏情况压缩 — 复用 wechat_worst_case_compressor.py。

    流程 (与参考实现一致):
      1. 短边缩到 1280 (4:3 图缩到 1706x1279); 1920x1080 短边 1080 < 1280 → 不缩
      2. GaussianBlur radius=0.6
      3. 真 JPEG 编解码: IJG 标准亮度/色度量化表按 quality=60 缩放, 4:2:0 子采样

    quality / blur_radius / subsampling 给出时覆盖 preset 的对应项 (见 NOISE_TIERS)。

    非几何 (输出画布与输入一致), mask/bbox 无需改动。
    """
    from PIL import Image

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]

    if quality is None and blur_radius is None and subsampling is None:
        compressor = _get_wechat_compressor(preset)
    else:
        from wechat_worst_case_compressor import WeChatWorstCaseCompressor
        ov = {}
        if quality is not None:
            ov['quality'] = int(quality)
        if blur_radius is not None:
            ov['blur_radius'] = float(blur_radius)
        if subsampling is not None:
            ov['subsampling'] = int(subsampling)
        compressor = WeChatWorstCaseCompressor.from_preset(
            preset or 'mainstream_worst', **ov)
    pil_in = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_out = compressor.compress_to_image(pil_in)

    # 参考实现的缩放规则只对长边/短边超阈值的图生效; 万一触发了缩放,
    # 恢复回原画布尺寸以保证标签坐标系不变 (训练分辨率 1920x1080 下不会触发)。
    if pil_out.size != (w, h):
        resample = getattr(Image, 'Resampling', Image).LANCZOS
        pil_out = pil_out.resize((w, h), resample)

    return cv2.cvtColor(np.array(pil_out.convert('RGB')), cv2.COLOR_RGB2BGR)


def add_chroma_texture(image, sigma, corr=3.0, rng=None):
    """受控色度纹理层 — 只在 G-B 平面调制, 不动亮度。

    为什么需要它
    ------------
    实拍里水印(±4 级 G-B)死不死, 由**图案同频段(20px 格)的载体色度纹理**决定,
    不是色度总量: 同一台相机同一条流水线, VS Code 截图频段 std 5.1 → 水印活,
    花壁纸+密集图标 17.3 → 水印灭。
    而 physical_moire 的 sensor_noise_scale / chroma_scale / lattice_visibility_floor
    全部推不动这个频段 (实测最高只到 3.8~4.4), 合成载体自身也只有 0.3~5.7。

    做法: 相关长度 corr px 的高斯场, G 加 / B 减, 即纯 G-B 调制。
    标定 (/data1/tmp_vis/tex_layer.py, corr=3.0, 叠在 pimog->wechat 之后):
        sigma  0 -> 频段std 3.06, 水印残余 +0.039   实拍"活"  (3.1~5.1)
        sigma  2 -> 频段std 4.96, 水印残余 +0.025   活/死边界
        sigma  5 -> 频段std ~10,  水印残余 ~+0.017   实拍"临界"(13.8~17.3)
        sigma  8 -> 频段std 15.60, 水印残余 +0.012   实拍"临界"
        sigma 12 -> 频段std 23.05, 水印残余 +0.009   已低于可学下限
    档位上界卡在 sigma<=8: 再往上水印实际不可见, 但标签仍写"有定位块",
    等于教模型幻觉。

    必须加在压缩**之后** — 加在之前会被 JPEG 抹掉 (sigma=6 前置只剩 10.79, 后置 11.94)。

    非几何, mask/bbox 无需改动。
    """
    if sigma <= 0:
        return image
    h, w = image.shape[:2]
    r = np.random if rng is None else rng
    g = r.randn(h, w).astype(np.float32)
    if corr > 0:
        g = cv2.GaussianBlur(g, (0, 0), corr)
    g = g / max(float(g.std()), 1e-6) * float(sigma)
    out = image.astype(np.float32)
    out[:, :, 1] += g
    out[:, :, 0] -= g
    return np.clip(out, 0, 255).astype(np.uint8)


def add_pimog_noise_strong(image, mask=None, bboxes=None, rng=None,
                           **overrides):
    """强化版屏摄 — 关掉"绝对可见性下限"并加大非几何退化。

    preset=screen_capture 的默认值里 preserve_absolute_visibility=True
    会把内容可见度托在 70%~90% (EXTREME_VISIBILITY_FLOOR_RANGE), 相当于给噪声限幅;
    真实拍摄没有这回事。同时加大传感器噪声/色度/光学 PSF, 这几项是实拍里
    真正杀死 ±4 级 G-B 信号的成分 (微信 q60 单独只把残余从 0.20 压到 0.17, 几乎不掉)。

    仍走 _WarpCapturingMoire 的标签同步, 几何 warp 参数不变 (0%~5% 保持原样)。
    """
    import torch

    base = _get_moire_sim()
    ov = dict(preserve_absolute_visibility=False, sensor_noise_scale=2.5,
              chroma_scale=2.0, optical_psf_scale=2.0)
    ov.update(overrides)
    # _WarpCapturingMoire 吃 **overrides 且会抓 content-warp grid, 一次跑完
    sim = _WarpCapturingMoire(device=base.device, **ov)
    sim.core.eval()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
         .to(device=base.device, dtype=torch.float32).div_(255.0))
    seed = int(rng.randint(0, 2 ** 31 - 1)) if rng is not None else 777
    with torch.inference_mode():
        out = sim(t, generator=torch.Generator(
            device=base.device).manual_seed(seed))
    o = (out[0].permute(1, 2, 0).clamp(0, 1).mul(255).round()
         .to(torch.uint8).cpu().numpy())
    out_bgr = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)

    if sim.last_grid is not None and (mask is not None or bboxes is not None):
        mask, bboxes = _sync_labels_to_warp(
            mask, bboxes, sim.last_grid, image.shape[0], image.shape[1])
    return out_bgr, mask, bboxes


# ── 非几何噪声档位 ────────────────────────────────────────────────
# 标定: /data1/tmp_vis/tier_calib3.py —— 走 apply_noise_tier 真实代码路径,
#       4 载体 × 12 种子取中位。**不能用单种子读数**: pimog 的摩尔纹周期
#       EXTREME_LATTICE_TARGET_ALIAS_PERIOD_RANGE=(5,96)px 逐次随机,
#       同一档位不同种子的频段std 可差 4 倍。
# 实拍参考 (vis/real_capture/detect_out/rect, 18 张):
#     活   残余 +0.034~+0.069  频段std  3.1~ 5.1  信号/纹理 0.72~1.16
#     临界 残余 +0.004~+0.020  频段std 13.8~17.3  信号/纹理 0.19~0.24
#     死   残余 -0.048~-0.007  频段std  6.0~ 7.0  信号/纹理 0.47~0.59
#
# 档位只覆盖到"临界"下限 (残余 p50 >= 0.007): 更狠会让标签说谎 (图案已不可见)。
# 每个 clean 样本出 n1..n4 各一份 → 语料各占 25%, 且每个载体在每个严重度上都有配对样本。
NOISE_TIERS = {
    'n1': dict(
        desc='轻   pimog默认 -> w q60',
        residue='+0.0553', band_std=4.66, share=0.25),
    'n2': dict(
        desc='中   pimog默认 -> w q60 -> 纹理 sigma=2',
        residue='+0.0345', band_std=6.10, share=0.25),
    'n3': dict(
        desc='重   pimog强 -> w q46 -> 纹理 sigma=5',
        residue='+0.0143', band_std=10.46, share=0.25),
    'n4': dict(
        desc='极重 pimog极强 -> w q46 x2 -> 纹理 sigma=8',
        residue='+0.0067', band_std=15.98, share=0.25),
}

_PIMOG_XSTRONG = dict(sensor_noise_scale=4.0, chroma_scale=3.0,
                      optical_psf_scale=2.5, ambient_glare_scale=3.0)


def apply_noise_tier(image, mask, bboxes, tier, rng):
    """按非几何噪声档位跑一条完整链路 (相机 → 微信 → 色度纹理)。

    几何项 (透视 0%~5% / content_warp) 一律走原配置, 不在本函数里调。
    """
    if tier == 'n1':
        img, mask, bboxes = add_pimog_noise(image, mask, bboxes, rng=rng)
        return add_wechat_noise(img), mask, bboxes
    if tier == 'n2':
        img, mask, bboxes = add_pimog_noise(image, mask, bboxes, rng=rng)
        img = add_wechat_noise(img)
        return add_chroma_texture(img, 2.0, rng=rng), mask, bboxes
    if tier == 'n3':
        img, mask, bboxes = add_pimog_noise_strong(image, mask, bboxes, rng=rng)
        img = add_wechat_noise(img, quality=46)
        return add_chroma_texture(img, 5.0, rng=rng), mask, bboxes
    if tier == 'n4':
        img, mask, bboxes = add_pimog_noise_strong(
            image, mask, bboxes, rng=rng, **_PIMOG_XSTRONG)
        img = add_wechat_noise(add_wechat_noise(img, quality=46), quality=46)
        return add_chroma_texture(img, 8.0, rng=rng), mask, bboxes
    raise ValueError(f'unknown noise tier: {tier!r}')


def add_tile_rotate_crop_noise(image, mask=None, bboxes=None,
                               angle_range=(-5, 5), max_shift=0.5, rng=None):
    """循环平移+旋转: 3x3拼接→旋转→中心裁剪。同步变换mask和bboxes。"""
    r = np.random if rng is None else rng
    h, w = image.shape[:2]
    is_gray = len(image.shape) == 2

    tiled = np.tile(image, (3, 3)) if is_gray else np.tile(image, (3, 3, 1))

    angle = r.uniform(angle_range[0], angle_range[1])
    center = (tiled.shape[1] / 2, tiled.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(tiled, M, (tiled.shape[1], tiled.shape[0]),
                             borderMode=cv2.BORDER_WRAP)

    crop_h, crop_w = h, w
    max_offset_x = min(int(w * max_shift), (3 * w - crop_w) // 2 - 1)
    max_offset_y = min(int(h * max_shift), (3 * h - crop_h) // 2 - 1)
    crop_cx = w + int(r.randint(-max_offset_x, max_offset_x + 1))
    crop_cy = h + int(r.randint(-max_offset_y, max_offset_y + 1))
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


class _WarpCapturingMoire:
    """EfficientScreenMoireNoise 的薄封装: 额外记录 content-warp 采样网格。

    参考实现的 forward() 只返回图像; 训练需要 mask/bbox 与图像同步,
    因此这里复刻 _warp_content_with_projected_residual 并保存 grid
    (output[i,j] 从 input 的 grid[i,j] 处采样), 供标签反变换使用。
    其余摩尔纹/曝光/PSF/CFA/ISP 全部走原实现, 不做任何改动。
    """

    def __init__(self, device='cpu', **overrides):
        import torch
        import torch.nn.functional as F
        from physical_moire import EfficientScreenMoireNoise

        outer = self

        class _Capturing(EfficientScreenMoireNoise):
            def _warp_content_with_projected_residual(
                    self, image, projected_coordinates, is_extreme, generator):
                # 与 physical_moire.EfficientScreenMoireNoise 同实现, 仅多存 grid
                if self.content_warp_scale == 0.0:
                    outer.last_grid = None
                    return image
                screen_x, screen_y = projected_coordinates
                batch, _, height, width = image.shape
                centre_y = height // 2
                centre_x = width // 2
                horizontal_x, vertical_x = self._local_spatial_gradients(screen_x)
                horizontal_y, vertical_y = self._local_spatial_gradients(screen_y)
                pixel_x = torch.arange(
                    width, device=image.device, dtype=image.dtype
                ).view(1, 1, 1, width)
                pixel_y = torch.arange(
                    height, device=image.device, dtype=image.dtype
                ).view(1, 1, height, 1)
                relative_x = pixel_x - float(centre_x)
                relative_y = pixel_y - float(centre_y)

                def projective_residual(coordinate, horizontal, vertical):
                    centre = coordinate[:, :, centre_y, centre_x].view(batch, 1, 1, 1)
                    slope_x = horizontal[:, :, centre_y, centre_x].view(batch, 1, 1, 1)
                    slope_y = vertical[:, :, centre_y, centre_x].view(batch, 1, 1, 1)
                    affine = centre + slope_x * relative_x + slope_y * relative_y
                    return coordinate - affine

                residual_x = projective_residual(screen_x, horizontal_x, vertical_x)
                residual_y = projective_residual(screen_y, horizontal_y, vertical_y)
                residual_peak = (
                    residual_x.square() + residual_y.square()
                ).sqrt().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                profile = self._profile_ranges(is_extreme)
                target_pixels = self._uniform(
                    image, *profile["content_warp_pixels"], (batch, 1, 1, 1), generator
                ) * self.content_warp_scale
                displacement_x = residual_x * (target_pixels / residual_peak)
                displacement_y = residual_y * (target_pixels / residual_peak)

                identity_x = torch.linspace(
                    -1.0, 1.0, width, device=image.device, dtype=image.dtype
                ).view(1, 1, width).expand(batch, height, width)
                identity_y = torch.linspace(
                    -1.0, 1.0, height, device=image.device, dtype=image.dtype
                ).view(1, height, 1).expand(batch, height, width)
                grid_x = identity_x + (
                    2.0 * displacement_x[:, 0] / float(max(width - 1, 1))
                )
                grid_y = identity_y + (
                    2.0 * displacement_y[:, 0] / float(max(height - 1, 1))
                )
                grid = torch.stack((grid_x, grid_y), dim=-1)
                outer.last_grid = grid.detach()
                return F.grid_sample(
                    image, grid, mode="bilinear", padding_mode="border",
                    align_corners=True,
                )

        self.core = _Capturing(device=device, **overrides).eval()
        self.last_grid = None
        self.device = device

    def __call__(self, img, generator=None):
        self.last_grid = None
        return self.core(img, generator=generator)


def _get_moire_sim():
    """懒加载屏-摄模拟器 (screen_capture 全链路: 摩尔纹+曝光+PSF+CFA+ISP)。"""
    global _MOIRE_SIM
    if _MOIRE_SIM is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _MOIRE_SIM = _WarpCapturingMoire(device=device)
    return _MOIRE_SIM


def _bilinear_sample(plane, xs, ys):
    """在 2D 图上按浮点像素坐标采样 (越界用最近值填充)。"""
    h, w = plane.shape
    xs = np.clip(xs, 0.0, w - 1.0)
    ys = np.clip(ys, 0.0, h - 1.0)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xs - x0
    fy = ys - y0
    return (plane[y0, x0] * (1 - fx) * (1 - fy) + plane[y0, x1] * fx * (1 - fy) +
            plane[y1, x0] * (1 - fx) * fy + plane[y1, x1] * fx * fy)


def _sync_labels_to_warp(mask, bboxes, grid, h, w):
    """用 content-warp 采样网格同步 mask/bboxes。

    grid 为 output→input 采样映射, 故图像与 mask 用同一 grid 拉样即可对齐;
    bbox 的 input 角点 p 对应 output 角点 q ≈ p − disp(p) (形变仅 0.5~5px)。
    """
    import torch
    import torch.nn.functional as F

    if grid is None:
        return mask, bboxes

    if mask is not None:
        m = torch.from_numpy(mask.astype(np.float32)).view(1, 1, h, w).to(grid.device)
        m_w = F.grid_sample(m, grid, mode='bilinear', padding_mode='zeros',
                            align_corners=True)
        mask = (m_w[0, 0].cpu().numpy() > 127).astype(np.uint8) * 255

    if bboxes is not None and len(bboxes) > 0:
        # grid 归一化坐标 → 像素位移场
        identity_x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        identity_y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        g = grid[0].cpu().numpy()
        disp_x = (g[..., 0] - identity_x) * (w - 1) / 2.0
        disp_y = (g[..., 1] - identity_y) * (h - 1) / 2.0

        new_bboxes = []
        for bbox in bboxes:
            cls, cx_n, cy_n, bw_n, bh_n = bbox
            cx_px, cy_px = cx_n * w, cy_n * h
            bw_px, bh_px = bw_n * w, bh_n * h
            corners = np.array([
                [cx_px - bw_px / 2, cy_px - bh_px / 2],
                [cx_px + bw_px / 2, cy_px - bh_px / 2],
                [cx_px + bw_px / 2, cy_px + bh_px / 2],
                [cx_px - bw_px / 2, cy_px + bh_px / 2],
            ], dtype=np.float32)
            sx = _bilinear_sample(disp_x, corners[:, 0], corners[:, 1])
            sy = _bilinear_sample(disp_y, corners[:, 0], corners[:, 1])
            warped = corners.copy()
            warped[:, 0] -= sx
            warped[:, 1] -= sy
            x_min = np.clip(warped[:, 0].min(), 0, w)
            y_min = np.clip(warped[:, 1].min(), 0, h)
            x_max = np.clip(warped[:, 0].max(), 0, w)
            y_max = np.clip(warped[:, 1].max(), 0, h)
            if x_max - x_min > 2 and y_max - y_min > 2:
                new_bboxes.append([
                    int(cls),
                    float((x_min + x_max) / 2 / w),
                    float((y_min + y_max) / 2 / h),
                    float((x_max - x_min) / w),
                    float((y_max - y_min) / h),
                ])
        bboxes = new_bboxes

    return mask, bboxes


def add_pimog_noise(image, mask=None, bboxes=None, rng=None):
    """拍照模拟 — 复用 physical_moire.py 的屏-摄物理模拟 (screen_capture 全链路)。

    覆盖: 标定屏幕/相机单应投影、倒格子摩尔纹(主阶+弱次阶)、光学 PSF、
    Bayer/CFA 重建、传感器噪声、环境光照/白平衡/色调、残差投影形变。
    形变只有 0.5~5px, 经 content-warp 网格同步到 mask/bboxes。
    """
    import torch

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]

    sim = _get_moire_sim()
    device = sim.core._device_anchor.device

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = (torch.from_numpy(rgb)
              .permute(2, 0, 1).unsqueeze(0)
              .to(device=device, dtype=torch.float32)
              .div_(255.0))

    generator = None
    if rng is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(rng.randint(0, 2 ** 31 - 1)))

    with torch.inference_mode():
        out = sim(tensor, generator=generator)
        grid = sim.last_grid

    out_bgr = (out[0].permute(1, 2, 0).clamp(0, 1)
               .mul(255).round().to(torch.uint8).cpu().numpy())
    out_bgr = cv2.cvtColor(out_bgr, cv2.COLOR_RGB2BGR)

    mask, bboxes = _sync_labels_to_warp(mask, bboxes, grid, h, w)
    return out_bgr, mask, bboxes


def apply_pair_noise(image, mask=None, bboxes=None, rng=None):
    """Pair噪声: 从[identity, wechat, tile_crop, pimog]随机选2种组合。

    - wechat : wechat_worst_case_compressor.py 真 JPEG 压缩, 非几何
    - pimog  : physical_moire.py 屏-摄拍照模拟, 残差形变同步 mask/bbox
    - tile_crop: 3x3 平铺+旋转+裁剪, 同步 mask/bbox (并复制 9 份 bbox)
    """
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
            image, mask, bboxes = add_tile_rotate_crop_noise(image, mask, bboxes, rng=rng)
        elif noise_type == 'pimog':
            image, mask, bboxes = add_pimog_noise(image, mask, bboxes, rng=rng)

    return image, mask, bboxes, [str(first), str(second)]


# ───────────────────── alpha融合 ─────────────────────

def alpha_blend_watermark(carrier_bgr, watermark_bgra, alpha, channel_mode='b'):
    """模拟 HLSL shader 的 alpha 融合。

    契约 (对应 overlay.cpp 旧 RGB(A) 路径 + overlay_texture.cpp 的
    SrcBlend=ONE / DestBlend=INV_SRC_ALPHA):

        out = α_eff * template + (1 - α_eff) * carrier
        α_eff = α * dynamicMask
        dynamicMask = saturate( max_c |template_c/255 - 1| * 2 )

    模板只含 0 / 255:
      - 白色(中性)像素 (255,255,255) → dynamicMask = 0 → **完全不改**
      - 黄色(信号)像素 (0,255,255)   → dynamicMask = 1 → α_eff = α
        max|out - carrier| = α × 255 = 0.032 × 255 = 8.16

    channel_mode='b' / 'yw': 黄/白 RGB 编入 (模板 0/255)
    """
    wm_bgr = watermark_bgra[:, :, :3].astype(np.float32)

    # 旧 RGB(A) 路径的 dynamicMask: 偏离白色的幅度, saturate(dev*2)
    # 0/255 模板下 dev ∈ {0, 1} → mask ∈ {0, 1}
    dev = np.max(np.abs(wm_bgr - 255.0), axis=2) / 255.0
    dynamic_mask = np.minimum(1.0, dev * 2.0).astype(np.float32)
    a_eff = (alpha * dynamic_mask)[:, :, None]

    carrier = carrier_bgr.astype(np.float32)
    result = carrier * (1.0 - a_eff) + wm_bgr * a_eff
    # 必须四舍五入再落盘: astype(uint8) 是截断, 会把 246.84 收成 246,
    # 让 |Δ| 从 8 变成 9, 突破 0.032×255=8.16 的上界。
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


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


# ───────────────────── 画布拼装 (无缩放) ─────────────────────

# 候选网格 (cols, rows), cell = (1920//cols, 1080//rows), 需整除
_CANVAS_GRIDS = (
    (1, 1), (2, 1), (2, 2), (3, 2), (3, 3),
    (4, 2), (4, 3), (4, 4), (6, 3), (6, 4), (8, 4),
)


def _fits(shape_hw, cell_h, cell_w):
    h, w = shape_hw[:2]
    return w >= cell_w and h >= cell_h


def pick_canvas_grid(src_shapes, target_w=1920, target_h=1080, min_coverage=0.85):
    """按源图尺寸选网格: 取**格子最大**(缝最少)且 ≥min_coverage 源图能 1:1 裁出的。"""
    shapes = list(src_shapes)
    for cols, rows in _CANVAS_GRIDS:
        if target_w % cols or target_h % rows:
            continue
        cell_w, cell_h = target_w // cols, target_h // rows
        cov = np.mean([_fits(s, cell_h, cell_w) for s in shapes]) if shapes else 0.0
        if cov >= min_coverage:
            return cols, rows
    cols, rows = _CANVAS_GRIDS[-1]
    return cols, rows


def crop_tile_1x1(img, cell_h, cell_w, rng):
    """1:1 裁出 cell_h x cell_w — **绝不缩放**。

    源图不够大时裁到能给的最大区域, 缺边用边缘复制补齐
    (复制不插值, 不会把纹理低通成糊的)。
    """
    h, w = img.shape[:2]
    if h >= cell_h and w >= cell_w:
        y0 = int(rng.randint(0, h - cell_h + 1))
        x0 = int(rng.randint(0, w - cell_w + 1))
        return img[y0:y0 + cell_h, x0:x0 + cell_w].copy()
    ch, cw = min(h, cell_h), min(w, cell_w)
    y0 = int(rng.randint(0, h - ch + 1)) if h > ch else 0
    x0 = int(rng.randint(0, w - cw + 1)) if w > cw else 0
    patch = img[y0:y0 + ch, x0:x0 + cw]
    if ch < cell_h or cw < cell_w:
        patch = cv2.copyMakeBorder(patch, 0, cell_h - ch, 0, cell_w - cw,
                                   borderType=cv2.BORDER_REPLICATE)
    return patch.copy()


def build_canvas(src_imgs, rng, target_w=1920, target_h=1080, grid=None):
    """无缩放把源图拼成 target_w x target_h 画布。

    resize 会把 640x480 放大 3 倍, 纹理被低通成糊的, 和真实截屏不符;
    这里一律保持原生像素密度, 每格一张源图的 1:1 裁剪:

      * grid=(1, 1) 且源图够大 (bcgd):
            单幅 1:1 随机裁剪 — 连贯场景, 无重采样, 最贴近真实截屏
      * 其它 (coco / document):
            原生分辨率网格平铺, 每格独立 1:1 裁剪

    每个数据集各自拼自己的画布, 不跨集混拼。
    """
    if grid is None:
        shapes = [im.shape for im in src_imgs]
        grid = pick_canvas_grid(shapes, target_w, target_h)
    cols, rows = grid
    cell_w = target_w // cols
    cell_h = target_h // rows

    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    k = 0
    for r in range(rows):
        for c in range(cols):
            src = src_imgs[k % len(src_imgs)]
            canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = \
                crop_tile_1x1(src, cell_h, cell_w, rng)
            k += 1

    if canvas.shape[0] == target_h and canvas.shape[1] == target_w:
        return canvas
    return crop_tile_1x1(canvas, target_h, target_w, rng)


# ───────────────────── 数据集生成 ─────────────────────

def generate_one_sample(wm_id, fix_fg_matrix, locator_pattern, alpha, rng,
                        carrier_img=None, apply_noise=True, channel_mode='b'):
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
        # 黄/白模式必须保持模板二值 0/255 — INTER_AREA 会在格子边界产生
        # 中间值, 让 max|Δ| 偏离 0.032×255; 用 NEAREST + 阈值锁死二值。
        out = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        out[:, :, :3] = np.where(out[:, :, :3] >= 128, 255, 0).astype(np.uint8)
        return out

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
    # B通道: 黄色单元格 B=0, 条纹外恢复 B=255 (白色不受影响)
    wm_full[:, :, 0][~keep] = 255

    # alpha融合 — 载体已是画布尺寸时禁止 resize (会抹掉高频纹理)
    if carrier_img is not None:
        if carrier_img.shape[:2] == (SCREEN_H, SCREEN_W):
            carrier = carrier_img
        else:
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
    parser.add_argument("--alpha", type=float, default=0.032)
    parser.add_argument("--carrier_dir", type=str, default=None,
                        help="载体图像目录 (默认纯白)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--no_noise", action="store_true", help="不加噪声")
    parser.add_argument("--channel_mode", type=str, default="b",
                        choices=["b", "yw"],
                        help="嵌入域: b/yw=黄白RGB(模板0/255)")
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
