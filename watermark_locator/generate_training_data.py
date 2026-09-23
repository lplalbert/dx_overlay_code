"""
合成训练数据生成器

复用 rs_gen_Syn_template_nums_dual.py 的模板生成逻辑，
但将最后一位（原CW0/填充）替换为唯一的定位图案（index=16）。

输出：PNG图像 + JSON标注文件

用法:
    python generate_training_data.py --num_samples 1000 --output_dir training_data
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

# 添加父目录到路径以导入原始生成器
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import rs_encode, checkerboard_locator_positions, LOCATOR_CODEWORD_INDEX

# ───────────────────── 模板生成（复用原始逻辑）─────────────────────

WATERMARK_GRID_SIZE = 8
WATERMARK_GRID_CELLS = WATERMARK_GRID_SIZE * WATERMARK_GRID_SIZE
CHANNEL_ENCODING_ALPHA = 254
MESSAGE_ROW, MESSAGE_COL = 2, 2


def gen_gaussian_tp(blockRow: int) -> np.ndarray:
    """高斯圆型模板。"""
    template = np.zeros((blockRow, blockRow), dtype=np.float32)
    CenterX = blockRow // 2
    y, x = np.ogrid[:blockRow, :blockRow]
    dist = np.sqrt((y - CenterX)**2 + (x - CenterX)**2)
    Radius = 1.0 - dist / CenterX
    mask1 = (Radius > 0) & (Radius <= 0.3)
    mask2 = Radius > 0.3
    template[mask1] = np.round(125 * np.sqrt(Radius[mask1]) * 1.25)
    template[mask2] = 125
    return template.astype(np.uint8)


def gen_rect_tp(blockRow: int) -> np.ndarray:
    """矩形模板。"""
    return np.ones((blockRow, blockRow), dtype=np.uint8) * 125


def gen_block_single_uv(
    k: int, kv: int,
    blockRow: int = 32,
    ratio_u: int = 10,
    ratio_v: int = 8,
    v_tp_fn=gen_gaussian_tp,
) -> np.ndarray:
    """
    生成单个码字块的BGRA纹理。
    k: Cr通道方向 (1=亮, 0=暗)
    kv: Cb通道方向 (1=亮, 0=暗)
    """
    y = np.full((blockRow, blockRow), 128, dtype=np.int16)
    cr = np.full((blockRow, blockRow), 128, dtype=np.int16)
    cb = np.full((blockRow, blockRow), 128, dtype=np.int16)

    # Cr（动态）用矩形模板
    if ratio_u != 0:
        cr_delta = gen_rect_tp(blockRow).astype(np.int16) * ratio_u // 10
        cr += cr_delta if k == 1 else -cr_delta

    # Cb（静态）用高斯/指定模板
    if ratio_v != 0:
        cb_delta = v_tp_fn(blockRow).astype(np.int16) * ratio_v // 10
        cb += cb_delta if kv == 1 else -cb_delta

    y = np.clip(y, 0, 255).astype(np.uint8)
    cr = np.clip(cr, 0, 255).astype(np.uint8)
    cb = np.clip(cb, 0, 255).astype(np.uint8)
    alpha = np.full((blockRow, blockRow), CHANNEL_ENCODING_ALPHA, dtype=np.uint8)
    return cv2.merge((cb, cr, y, alpha))


def generate_fix_fg_matrix_with_locator() -> np.ndarray:
    """生成包含定位图案的17×64矩阵。"""
    # 原始16行
    original = np.array([
        [1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1],
        [-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
        [-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1],
        [-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1],
        [-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1],
        [-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
        [1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1],
        [-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1],
        [1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1],
        [1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1],
        [-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1],
        [-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
        [1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1],
        [1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
        [1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1],
        [-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1],
    ], dtype=np.int8)

    # 第17行：定位图案
    # 尝试从文件加载，否则使用预设的已验证图案
    pattern_path = os.path.join(os.path.dirname(__file__), 'locator_pattern.npy')
    if os.path.exists(pattern_path):
        locator = np.load(pattern_path).reshape(1, 64).astype(np.int8)
    else:
        # 由 generate_locator_pattern.py --seed 2024 生成（最小Hamming距离=24）
        locator = np.array([[
            1,  1,  1, -1, -1,  1,  1, -1,
           -1, -1, -1, -1, -1,  1, -1, -1,
            1, -1, -1,  1, -1, -1,  1,  1,
            1, -1, -1, -1, -1,  1,  1, -1,
            1,  1,  1,  1, -1, -1,  1, -1,
            1, -1, -1, -1,  1,  1,  1,  1,
            1,  1,  1, -1, -1,  1,  1,  1,
           -1,  1, -1, -1,  1, -1,  1, -1
        ]], dtype=np.int8)

    return np.vstack([original, locator])


# 全局变量：扩展后的17×64矩阵
FIX_FG_MATRIX_EXTENDED = generate_fix_fg_matrix_with_locator()


def generate_template_for_codeword(
    codeword_index: int,
    single_block_size: int = 64,
    ratio_u: int = 10,
    ratio_v: int = 8,
    type_val: int = 0,
    inverse: bool = False,
) -> np.ndarray:
    """
    为指定码字索引生成 512×512 的 BGRA 模板图。

    与原始 rs_gen_Syn_template_nums_dual.py 完全一致的逻辑：
    fix_fg_matrix[codeword_index] 是 64 个 +1/-1 值（8×8 网格），
    每个值决定对应 64×64 子块使用 template_1 还是 template_0。

    Args:
        codeword_index: 0~16（16为定位图案）

    Returns:
        (512, 512, 4) BGRA 图像
    """
    pattern = FIX_FG_MATRIX_EXTENDED[codeword_index]  # (64,) +1/-1

    # 根据 type_val 和 inverse 决定 template_1/template_0 的 Cr/Cb 极性
    # 与原始代码 gen_wm_blocks_uv 完全一致
    if type_val == 0:
        if not inverse:
            template_1 = gen_block_single_uv(k=1, kv=1, blockRow=single_block_size,
                                              ratio_u=ratio_u, ratio_v=ratio_v)
            template_0 = gen_block_single_uv(k=0, kv=0, blockRow=single_block_size,
                                              ratio_u=ratio_u, ratio_v=ratio_v)
        else:
            template_1 = gen_block_single_uv(k=0, kv=1, blockRow=single_block_size,
                                              ratio_u=ratio_u, ratio_v=ratio_v)
            template_0 = gen_block_single_uv(k=1, kv=0, blockRow=single_block_size,
                                              ratio_u=ratio_u, ratio_v=ratio_v)
    else:
        if not inverse:
            template_1 = gen_block_single_uv(k=1, kv=0, blockRow=single_block_size,
                                              ratio_u=ratio_u, ratio_v=ratio_v)
            template_0 = gen_block_single_uv(k=0, kv=1, blockRow=single_block_size,
                                              ratio_u=ratio_u, ratio_v=ratio_v)
        else:
            template_1 = gen_block_single_uv(k=0, kv=0, blockRow=single_block_size,
                                              ratio_u=ratio_u, ratio_v=ratio_v)
            template_0 = gen_block_single_uv(k=1, kv=1, blockRow=single_block_size,
                                              ratio_u=ratio_u, ratio_v=ratio_v)

    # 用 fix_fg_matrix 的 +1/-1 图案铺排 8×8 个子块 → 512×512
    canvas = np.empty((WATERMARK_GRID_SIZE * single_block_size,
                       WATERMARK_GRID_SIZE * single_block_size, 4), dtype=np.uint8)
    for idx, val in enumerate(pattern):
        row_idx, col_idx = divmod(idx, WATERMARK_GRID_SIZE)
        r0 = row_idx * single_block_size
        c0 = col_idx * single_block_size
        canvas[r0:r0 + single_block_size, c0:c0 + single_block_size, :] = (
            template_1 if val == 1 else template_0
        )
    return canvas


def generate_full_template_templates(
    single_block_size: int = 64,
    ratio_u: int = 10,
    ratio_v: int = 8,
    type_val: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """生成17个码字模板（正向+反向）。"""
    templates = []
    templates_inv = []
    for i in range(17):  # 0~16
        t = generate_template_for_codeword(
            i, single_block_size, ratio_u, ratio_v, type_val, inverse=False
        )
        templates.append(t)
        t_inv = generate_template_for_codeword(
            i, single_block_size, ratio_u, ratio_v, type_val, inverse=True
        )
        templates_inv.append(t_inv)
    return templates, templates_inv


def generate_watermark_image(
    watermark_id: int,
    screen_w: int = 1920,
    screen_h: int = 1080,
    block_rows: int = 4,
    block_cols: int = 6,
    single_block_size: int = 64,
    ratio_u: int = 10,
    ratio_v: int = 8,
    type_val: int = 0,
    inverse: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    生成完整的水印图像（单张，用于训练）。

    Returns:
        (image_bgra, annotation_dict)
    """
    # RS编码得到16个码字（最后一位是定位图案索引16）
    wm_seq = rs_encode(watermark_id)

    # 生成17个码字模板
    templates, templates_inv = generate_full_template_templates(
        single_block_size, ratio_u, ratio_v, type_val
    )

    # 计算块尺寸
    block_w = screen_w // block_cols
    block_h = screen_h // block_rows
    msg_w = block_w // MESSAGE_COL
    msg_h = block_h // MESSAGE_ROW

    # 创建画布
    canvas = np.full((screen_h, screen_w, 3), 128, dtype=np.uint8)

    # 选择正向或反向模板
    active_templates = templates_inv if inverse else templates

    # 4个message_block（每个包含4个码字，2×2排列）
    msg_block_templates = []
    for blk_idx in range(4):
        row_start = blk_idx * 4
        codeword_row = wm_seq[row_start:row_start + 4]  # 4个码字
        a, b, c, d = codeword_row[0], codeword_row[1], codeword_row[2], codeword_row[3]
        reordered = [a, c, b, d]  # gen_message_block 的重排逻辑

        block_img = np.full((block_h, block_w, 3), 128, dtype=np.uint8)
        for si in range(MESSAGE_ROW):
            for sj in range(MESSAGE_COL):
                idx = si * MESSAGE_COL + sj
                cw = reordered[idx]
                # active_templates[cw] 是 512×512 的 BGRA 图像
                tpl = active_templates[cw]
                # resize 到子块大小 (msg_w × msg_h)
                tpl_resized = cv2.resize(tpl, (msg_w, msg_h), interpolation=cv2.INTER_AREA)
                # BGRA → BGR（取前3通道）
                block_img[
                    si * msg_h:(si + 1) * msg_h,
                    sj * msg_w:(sj + 1) * msg_w,
                ] = tpl_resized[:, :, :3]

        msg_block_templates.append(block_img)

    # 铺满画布
    for i in range(block_rows):
        for j in range(block_cols):
            k = (j + (i % 2) * 2) % 4
            y0 = i * block_h
            x0 = j * block_w
            canvas[y0:y0 + block_h, x0:x0 + block_w] = msg_block_templates[k]

    # 计算定位块位置
    loc_positions = checkerboard_locator_positions(block_rows, block_cols)
    loc_abs = []
    for col, row in loc_positions:
        # 定位块在message_block的(0,1)子位置
        cx = col * block_w + 1 * msg_w + msg_w / 2
        cy = row * block_h + 0 * msg_h + msg_h / 2
        loc_abs.append([float(cx), float(cy)])

    # 码字标签（24个块位置，每个一个码字索引）
    codeword_labels = []
    for i in range(block_rows):
        for j in range(block_cols):
            k = (j + (i % 2) * 2) % 4
            row_start = k * 4
            a, b, c, d = wm_seq[row_start], wm_seq[row_start+1], wm_seq[row_start+2], wm_seq[row_start+3]
            reordered = [a, c, b, d]
            # 每个块(2×2)包含4个码字，但我们为每个块只取第一个码字作为简化的标签
            # 完整版本应该标注所有子位置
            codeword_labels.append(int(reordered[0]))

    annotation = {
        'watermark_id': watermark_id,
        'screen_w': screen_w,
        'screen_h': screen_h,
        'block_rows': block_rows,
        'block_cols': block_cols,
        'block_w': block_w,
        'block_h': block_h,
        'msg_block_size_w': msg_w,
        'msg_block_size_h': msg_h,
        'locator_abs_positions': loc_abs,
        'codeword_labels': codeword_labels,
        'wm_seq': [int(x) for x in wm_seq],
        'inverse': inverse,
        'type_val': type_val,
    }

    return canvas, annotation


# ───────────────────── 退化处理 ─────────────────────

def apply_degradation(image: np.ndarray, severity: str = 'medium') -> np.ndarray:
    """对图像应用随机退化（模拟真实截屏）。"""
    img = image.copy()

    if severity == 'light':
        noise_std, blur_k, jpeg_q = 1.0, 0, 90
    elif severity == 'medium':
        noise_std, blur_k, jpeg_q = 3.0, 3, 75
    else:  # heavy
        noise_std, blur_k, jpeg_q = 5.0, 5, 50

    # 高斯噪声
    if np.random.random() < 0.7:
        noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 模糊
    if blur_k > 0 and np.random.random() < 0.5:
        k = blur_k if blur_k % 2 == 1 else blur_k + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    # JPEG压缩
    if np.random.random() < 0.6:
        quality = np.random.randint(jpeg_q, jpeg_q + 15)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    # 亮度扰动
    if np.random.random() < 0.5:
        delta = np.random.uniform(-10, 10)
        img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    # 对比度扰动
    if np.random.random() < 0.5:
        factor = np.random.uniform(0.9, 1.1)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return img


# ───────────────────── 主程序 ─────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="training_data",
                        help="Output directory")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--block_rows", type=int, default=4)
    parser.add_argument("--block_cols", type=int, default=6)
    parser.add_argument("--ratio_u", type=int, default=10)
    parser.add_argument("--ratio_v", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degradation", type=str, default='medium',
                        choices=['none', 'light', 'medium', 'heavy'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    print(f"=== Generating {args.num_samples} training samples ===")
    print(f"Output: {args.output_dir}")
    print(f"Resolution: {args.screen_width}×{args.screen_height}")
    print(f"Degradation: {args.degradation}")

    for i in range(args.num_samples):
        # 随机watermark_id
        wm_id = rng.randint(0, 1000000)
        # 随机是否用反向模板
        inverse = rng.random() < 0.5

        # 生成
        img, anno = generate_watermark_image(
            watermark_id=wm_id,
            screen_w=args.screen_width,
            screen_h=args.screen_height,
            block_rows=args.block_rows,
            block_cols=args.block_cols,
            ratio_u=args.ratio_u,
            ratio_v=args.ratio_v,
            inverse=inverse,
        )

        # 退化
        if args.degradation != 'none':
            img = apply_degradation(img, args.degradation)

        # 保存
        base_name = f"sample_{i:06d}_{wm_id}"
        img_path = os.path.join(args.output_dir, f"{base_name}.png")
        json_path = os.path.join(args.output_dir, f"{base_name}.json")

        cv2.imwrite(img_path, img)
        with open(json_path, 'w') as f:
            json.dump(anno, f, indent=2)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{args.num_samples}] {base_name}")

    print(f"\nDone! Generated {args.num_samples} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
