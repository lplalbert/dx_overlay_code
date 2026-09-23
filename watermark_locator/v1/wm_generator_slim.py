#!/usr/bin/env python3
"""
Legacy watermark generator retained for development comparisons only.
Production releases use watermark_generator.cpp and do not bundle Python.
Generates watermark template images from a numeric ID.
Uses galois library for Reed-Solomon encoding.
No matplotlib/GUI dependencies.
"""

import numpy as np
import cv2
import os
import argparse
import sys

# Import galois library (required)
import galois

# Initialize GF(2^4) and RS(15,5) encoder
_GF = galois.GF(2**4)
_RS = galois.ReedSolomon(n=15, k=5, field=_GF)
print("Initialized galois RS(15,5) encoder")
WATERMARK_PAYLOAD_HEX_DIGITS = 5
MAX_WATERMARK_ID = 16 ** WATERMARK_PAYLOAD_HEX_DIGITS - 1
WATERMARK_GRID_SIZE = 8
WATERMARK_GRID_CELLS = WATERMARK_GRID_SIZE * WATERMARK_GRID_SIZE
CHANNEL_ENCODING_ALPHA = 254


def gen_gaussian_tp(blockRow):
    """Generate Gaussian circular pattern for watermark unit."""
    template = np.zeros((blockRow, blockRow), dtype=np.uint8)
    CenterX = blockRow // 2
    for m in range(blockRow):
        for n in range(blockRow):
            Radius = 1.0 - np.sqrt((m - CenterX) ** 2 + (n - CenterX) ** 2) / CenterX
            if Radius <= 0:
                template[m, n] = 0
            elif Radius <= 0.3:
                template[m, n] = np.round(125 * np.sqrt(Radius) * 1.25)
            else:
                template[m, n] = 125
    return template


def gen_rect_tp(blockRow):
    """Restore the original full-cell rectangular matrix."""
    return np.ones((blockRow, blockRow), dtype=np.uint8) * 125


def gen_gaussian_tp_v2(blockRow, flat_ratio=0.7):
    """Generate a Gaussian pattern with a wider full-strength center."""
    template = np.zeros((blockRow, blockRow), dtype=np.float32)
    center = blockRow // 2
    y, x = np.ogrid[:blockRow, :blockRow]
    radius = 1.0 - np.sqrt((y - center) ** 2 + (x - center) ** 2) / center
    flat = radius > (1.0 - flat_ratio)
    gradient = (radius > 0) & (~flat)
    template[flat] = 125
    if (1.0 - flat_ratio) > 0:
        normalized = radius[gradient] / (1.0 - flat_ratio)
        template[gradient] = np.round(125 * np.sqrt(normalized) * 1.25)
    return np.clip(template, 0, 125).astype(np.uint8)


def gen_soft_rect_tp(blockRow, border=2):
    """Generate a rectangular pattern with softened borders."""
    template = np.ones((blockRow, blockRow), dtype=np.float32) * 125
    mask = np.ones((blockRow, blockRow), dtype=np.float32)
    for index in range(border):
        value = (index + 1) / (border + 1)
        mask[index, :] *= value
        mask[-1 - index, :] *= value
        mask[:, index] *= value
        mask[:, -1 - index] *= value
    return (template * mask).astype(np.uint8)


def gen_block_single_uv_decouple(
        k=1, kv=1, blockRow=32, ratio_u=10, ratio_v=8,
        v_tp_fn=gen_gaussian_tp):
    """Return BGRA data whose logical PNG RGBA channels are Y/Cr/Cb/marker."""
    y = np.full((blockRow, blockRow), 128, dtype=np.int16)
    cr = np.full((blockRow, blockRow), 128, dtype=np.int16)
    cb = np.full((blockRow, blockRow), 128, dtype=np.int16)
    v_pattern = v_tp_fn(blockRow)
    gen_rect = gen_rect_tp(blockRow)

    # OpenCV YCrCb ordering means legacy ratio_u controls dynamic Cr.
    if ratio_u != 0:
        cr_delta = gen_rect.astype(np.int16) * ratio_u // 10
        if k == 1:
            cr += cr_delta
        else:
            cr -= cr_delta

    # Legacy ratio_v controls static Cb.
    if ratio_v != 0:
        cb_delta = v_pattern.astype(np.int16) * ratio_v // 10
        if kv == 1:
            cb += cb_delta
        else:
            cb -= cb_delta

    y = np.clip(y, 0, 255).astype(np.uint8)
    cr = np.clip(cr, 0, 255).astype(np.uint8)
    cb = np.clip(cb, 0, 255).astype(np.uint8)
    alpha = np.full((blockRow, blockRow), CHANNEL_ENCODING_ALPHA, dtype=np.uint8)
    return cv2.merge((cb, cr, y, alpha))


def resize_template_image(image, width, height):
    """Resize channel-data templates without converting encoded channels."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def new_channel_canvas(height, width, channels):
    if channels != 4:
        raise ValueError("channel-data templates must use four BGRA channels")
    canvas = np.empty((height, width, channels), dtype=np.uint8)
    canvas[:, :, :3] = 128
    canvas[:, :, 3] = CHANNEL_ENCODING_ALPHA
    return canvas


# Fixed foreground matrix for 16 watermark patterns
FIX_FG_MATRIX = np.array([
    [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
    [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
    [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
    [-1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
    [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
    [-1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
    [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
    [-1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
    [-1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
    [-1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1]
])


def gen_wm_blocks_uv(message_seq, type_val=1, single_block_size=64,
                     ratio_u=10, ratio_v=8, inverse=False,
                     v_tp_fn=gen_gaussian_tp):
    """Generate a fixed 8x8 watermark block with lossless Y/Cr/Cb data."""
    seq = np.asarray(message_seq).reshape(-1)
    if seq.size != WATERMARK_GRID_CELLS:
        raise ValueError(
            f"8x8 watermark templates require exactly {WATERMARK_GRID_CELLS} cells, "
            f"got {seq.size}"
        )

    if type_val == 0:
        if not inverse:
            template_1 = gen_block_single_uv_decouple(k=1, kv=1, ratio_u=ratio_u, ratio_v=ratio_v, blockRow=single_block_size, v_tp_fn=v_tp_fn)
            template_0 = gen_block_single_uv_decouple(k=0, kv=0, ratio_u=ratio_u, ratio_v=ratio_v, blockRow=single_block_size, v_tp_fn=v_tp_fn)
        else:
            template_1 = gen_block_single_uv_decouple(k=0, kv=1, ratio_u=ratio_u, ratio_v=ratio_v, blockRow=single_block_size, v_tp_fn=v_tp_fn)
            template_0 = gen_block_single_uv_decouple(k=1, kv=0, ratio_u=ratio_u, ratio_v=ratio_v, blockRow=single_block_size, v_tp_fn=v_tp_fn)
    else:
        if not inverse:
            template_1 = gen_block_single_uv_decouple(k=1, kv=0, ratio_u=ratio_u, ratio_v=ratio_v, blockRow=single_block_size, v_tp_fn=v_tp_fn)
            template_0 = gen_block_single_uv_decouple(k=0, kv=1, ratio_u=ratio_u, ratio_v=ratio_v, blockRow=single_block_size, v_tp_fn=v_tp_fn)
        else:
            template_1 = gen_block_single_uv_decouple(k=0, kv=0, ratio_u=ratio_u, ratio_v=ratio_v, blockRow=single_block_size, v_tp_fn=v_tp_fn)
            template_0 = gen_block_single_uv_decouple(k=1, kv=1, ratio_u=ratio_u, ratio_v=ratio_v, blockRow=single_block_size, v_tp_fn=v_tp_fn)

    img_blocks = np.empty((WATERMARK_GRID_SIZE * single_block_size,
                           WATERMARK_GRID_SIZE * single_block_size, 4), dtype=np.uint8)

    for j, bit in enumerate(seq):
        row_idx, col_idx = divmod(j, WATERMARK_GRID_SIZE)
        r0, r1 = row_idx * single_block_size, (row_idx + 1) * single_block_size
        c0, c1 = col_idx * single_block_size, (col_idx + 1) * single_block_size
        img_blocks[r0:r1, c0:c1, :] = template_1 if bit == 1 else template_0

    return img_blocks


class WM_Template_Generator:
    """Watermark template generator with cached templates."""

    def __init__(self, ratio_u=10, ratio_v=8, type_val=1, v_tp_fn=gen_gaussian_tp):
        self.ratio_u = ratio_u
        self.ratio_v = ratio_v
        self.type_val = type_val
        self.v_tp_fn = v_tp_fn
        self.wm_template_imgs = [None] * 16
        self.wm_template_imgs_inverse = [None] * 16
        self._gen_16_wm_templates(inverse=False)
        self._gen_16_wm_templates(inverse=True)

    def _gen_16_wm_templates(self, inverse=False):
        for i in range(16):
            wm_template = gen_wm_blocks_uv(
                FIX_FG_MATRIX[i],
                single_block_size=64,
                ratio_u=self.ratio_u,
                ratio_v=self.ratio_v,
                type_val=self.type_val,
                inverse=inverse,
                v_tp_fn=self.v_tp_fn,
            )
            if not inverse:
                self.wm_template_imgs[i] = wm_template
            else:
                self.wm_template_imgs_inverse[i] = wm_template

    def __call__(self, nums, inverse=False):
        if not inverse:
            return self.wm_template_imgs[nums]
        else:
            return self.wm_template_imgs_inverse[nums]


def encode(data):
    """RS(15,5) encode data using galois library."""
    return _RS.encode(data)


def nums2_16(nums):
    """Convert number to base-16 representation."""
    if nums < 0:
        raise ValueError("watermark_id must be non-negative")
    if nums > MAX_WATERMARK_ID:
        raise ValueError(
            f"watermark_id={nums} exceeds the current RS(15,5) payload range "
            f"0..{MAX_WATERMARK_ID} (0x{MAX_WATERMARK_ID:05X}); "
            "use a smaller ID or redesign the encoder/decoder payload size"
        )
    ans = [0] * 5
    i = 0
    while nums > 0:
        ans[i] = nums % 16
        nums = nums // 16
        i += 1
    return ans


def get_wm_seq(nums):
    """Get watermark sequence from number."""
    data = nums2_16(nums)
    rscode_data = encode(data[::-1])
    rscode_data = [int(e) for e in rscode_data]
    wm_seq = rscode_data[::-1] + [0]
    return wm_seq


MESSAGE_ROW, MESSAGE_COL = 2, 2


def gen_message_block(messages, wm_images, wm_images_inverse, block_height, block_width, message_height, message_width):
    """Generate message block from watermark images."""
    channels = wm_images[0].shape[2]
    message_block = new_channel_canvas(block_height, block_width, channels)
    message_block_inverse = new_channel_canvas(block_height, block_width, channels)
    a, b, c, d = messages[0], messages[1], messages[2], messages[3]
    messages = [a, c, b, d]

    for i in range(MESSAGE_ROW):
        for j in range(MESSAGE_COL):
            index = i * MESSAGE_COL + j
            index = index % len(messages)
            img_type = messages[index]
            wm_image = resize_template_image(wm_images[img_type], message_width, message_height)
            message_block[
                i * message_height : (i + 1) * message_height,
                j * message_width : (j + 1) * message_width,
            ] = wm_image

            wm_image_inverse = resize_template_image(
                wm_images_inverse[img_type], message_width, message_height)
            message_block_inverse[
                i * message_height : (i + 1) * message_height,
                j * message_width : (j + 1) * message_width,
            ] = wm_image_inverse

    return message_block, message_block_inverse


def gen_watermark_imgs(all_messages, block_rows, block_cols, block_height, block_width, message_height, message_width, wm_images, wm_images_inverse):
    """Generate full watermark images."""
    channels = wm_images[0].shape[2]
    wm_blocks = [None] * 4
    wm_blocks_inverse = [None] * 4

    for i in range(len(all_messages)):
        wm_msg_block, wm_msg_block_inverse = gen_message_block(
            all_messages[i],
            wm_images,
            wm_images_inverse,
            block_height,
            block_width,
            message_height,
            message_width,
        )
        wm_blocks[i] = wm_msg_block
        wm_blocks_inverse[i] = wm_msg_block_inverse

    wm_blank = new_channel_canvas(block_rows * block_height, block_width * block_cols, channels)
    wm_blank_inverse = new_channel_canvas(block_rows * block_height, block_width * block_cols, channels)

    for i in range(block_rows):
        for j in range(block_cols):
            k = j
            if i % 2:
                k += 2
            k = k % 4
            wm_blank[
                i * block_height : (i + 1) * block_height,
                j * block_width : (j + 1) * block_width,
            ] = wm_blocks[k]
            wm_blank_inverse[
                i * block_height : (i + 1) * block_height,
                j * block_width : (j + 1) * block_width,
            ] = wm_blocks_inverse[k]

    return wm_blank, wm_blank_inverse


def main():
    parser = argparse.ArgumentParser(description="Generate watermark template images")
    parser.add_argument("--nums", type=int, default=123456, help="Watermark ID number")
    parser.add_argument("--block_rows", "-row", type=int, default=4, help="Block rows")
    parser.add_argument("--block_cols", "-col", type=int, default=6, help="Block cols")
    parser.add_argument("--screen_width", type=int, default=1920, help="Screen width")
    parser.add_argument("--screen_height", type=int, default=1080, help="Screen height")
    parser.add_argument("--save_dir", type=str, default="wm_imgs", help="Output directory")
    # Watermark pattern parameters
    parser.add_argument("--dynamic_ratio_cr", "--ratio_u", dest="ratio_u", type=int, default=10, help="Dynamic Cr amplitude (legacy: ratio_u)")
    parser.add_argument("--static_ratio_cb", "--ratio_v", dest="ratio_v", type=int, default=8, help="Static Cb amplitude (legacy: ratio_v)")
    parser.add_argument("--type_val", type=int, default=0, help="Pattern type (0 or 1, default: 0)")
    parser.add_argument(
        "--pattern",
        choices=["gaussian", "gaussian_v2", "soft_rect", "rect"], # 高斯平滑，高斯平滑v2，软矩形，矩形 
        default="rect",
        help="Static Cb sub-block pattern; dynamic Cr stays rectangular",
    )

    args = parser.parse_args()

    # 打印所有参数（调试用）
    print(f"=== Watermark Generator ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Arguments: {args}")
    print(f"Generating watermark for ID: {args.nums}")
    print(f"Pattern params: ratio_u={args.ratio_u}, ratio_v={args.ratio_v}, type_val={args.type_val}")
    print(f"Save directory: {args.save_dir}")

    # Get watermark sequence
    wm_seq = get_wm_seq(args.nums)
    message_info = np.array(wm_seq).reshape(4, 4)

    # Initialize generator with user-specified parameters
    generator = WM_Template_Generator(
        ratio_u=args.ratio_u,
        ratio_v=args.ratio_v,
        type_val=args.type_val,
        v_tp_fn={
            "gaussian": gen_gaussian_tp,
            "gaussian_v2": gen_gaussian_tp_v2,
            "soft_rect": gen_soft_rect_tp,
            "rect": gen_rect_tp,
        }[args.pattern],
    )
    wm_images = generator.wm_template_imgs
    wm_images_inverse = generator.wm_template_imgs_inverse

    # Calculate dimensions using the legacy pipeline.
    block_width = args.screen_width // args.block_cols
    block_height = args.screen_height // args.block_rows
    message_width = block_width // MESSAGE_COL
    message_height = block_height // MESSAGE_ROW

    # Generate watermark images
    wm_blank, wm_blank_inverse = gen_watermark_imgs(
        message_info,
        args.block_rows,
        args.block_cols,
        block_height,
        block_width,
        message_height,
        message_width,
        wm_images,
        wm_images_inverse,
    )

    # Save images
    os.makedirs(args.save_dir, exist_ok=True)

    save_path = os.path.join(args.save_dir, f"wm_template_{args.nums}.png")
    wm_resized = resize_template_image(wm_blank, args.screen_width, args.screen_height)
    cv2.imwrite(save_path, wm_resized)

    save_path_inverse = os.path.join(args.save_dir, f"wm_template_{args.nums}_inverse.png")
    wm_resized_inverse = resize_template_image(wm_blank_inverse, args.screen_width, args.screen_height)
    cv2.imwrite(save_path_inverse, wm_resized_inverse)

    print(f"Saved channel-data template: {save_path}")
    print(f"Saved channel-data inverse template: {save_path_inverse}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        # 写入错误日志到多个位置
        log_paths = [
            "wm_generator_error.log",  # 当前目录
            os.path.join(os.path.expanduser("~"), "wm_generator_error.log"),  # 用户目录
        ]
        for log_path in log_paths:
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(error_msg)
                print(f"Error log saved to: {log_path}")
                break
            except Exception as log_err:
                print(f"Failed to write log to {log_path}: {log_err}")

        # 保持窗口打开以便查看错误
        if sys.stdin.isatty():
            try:
                input("Press Enter to exit...")
            except EOFError:
                pass
        sys.exit(1)
