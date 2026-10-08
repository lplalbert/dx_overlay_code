#!/usr/bin/env python3
"""v1 版：生成原生 DX Overlay 黄/白 RGB 屏幕模板（交错格排布）。

这是 ``../..//generate_yellow_white_template.py`` 的 **v1 冻结副本**，
与 v2 的差异只有一处：**块级 message 排布**。

    v1 交错格:  message = (block_col + (2 if block_row % 2 else 0)) % 4
        行0:  M1 M2 M3 M4 M1 M2
        行1:  M3 M4 M1 M2 M3 M4
        行2:  M1 M2 M3 M4 M1 M2
        行3:  M3 M4 M1 M2 M3 M4

    v2 交替行:  message = (block_col % 2) + 2 * (block_row % 2)
        行0:  M1 M2 M1 M2 M1 M2
        行1:  M3 M4 M3 M4 M3 M4
        行2:  M1 M2 M1 M2 M1 M2
        行3:  M3 M4 M3 M4 M3 M4

**默认行为与顶层生成器逐字节一致**（末位填充码字 0，不含回字形），
以保证 v1 现有功能不受影响。回字形定位图案是 v1 训练数据集
(``dataset/generate_dataset.py`` 里 ``wm_seq[-1] = LOCATOR_CODEWORD_INDEX``)
的做法，与本脚本的原生契约（末位字面 0）存在遗留矛盾；本副本只**提供
``--fill hui`` 作为可选项**复现训练集那套，**默认仍是 0**，不改动 v1 行为。

其余契约与顶层生成器完全一致：

* RS(15,5) over GF(16), payload 是五位十六进制 watermark ID;
* 反转后的 RS codeword 加一个末位 0 放进既有的 8 x 12 = 96 个屏幕位置;
* 每个 codeword 是原生生成器里的 8 x 8 格矩阵;
* 正/逆 PNG 是不透明 RGBA 黄/白图;
* 条纹使用与 EXE 相同的全局屏幕坐标。

Examples:

    python generate_yellow_white_template.py --id 123456 \\
        --width 2560 --height 1440 --output yellow_templates \\
        --angle 45 --period 4 --stripe-width 2 --alternate

    python generate_yellow_white_template.py --id-start 100 --id-end 110 \\
        --width 1920 --height 1080 --output batch_templates

    # 复现 v1 训练集那套 (末位画回字形而非码字 0)
    python generate_yellow_white_template.py --id 123456 --fill hui --output v1_train
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


MAX_WATERMARK_ID = (1 << 20) - 1
GRID_SIZE = 8
GRID_CELLS = GRID_SIZE * GRID_SIZE
BLOCK_ROWS = 4
BLOCK_COLS = 6
MESSAGE_ORDER = (0, 2, 1, 3)

# v1 交错格排布：message 只依赖 (block_col, block_row 的奇偶)。
LAYOUT_NAME = "v1-staggered"


def block_message_index(block_row: int, block_col: int) -> int:
    """返回块 (block_row, block_col) 的 message 下标，取值 0..3。"""
    return (block_col + (2 if block_row % 2 else 0)) % 4


# One bit is one row-major cell of the corresponding 8 x 8 codeword.  This
# table is copied from watermark_generator.cpp; do not replace it with a new
# pseudo-random pattern, because detector compatibility depends on it.
CODEWORD_CELL_MASKS: Tuple[int, ...] = (
    0x6699996699666699,
    0x5555AAAA5555AAAA,
    0xCCCC33333333CCCC,
    0x9669699696696996,
    0xC33C3CC33CC3C33C,
    0x9999666699996666,
    0xAA5555AA55AAAA55,
    0x5A5AA5A5A5A55A5A,
    0x6969969696966969,
    0xCC33CC33CC33CC33,
    0x5AA5A55A5AA5A55A,
    0xC3C33C3CC3C33C3C,
    0xA55AA55A5AA55AA5,
    0xC3C3C3C3C3C3C3C3,
    0x3CC33CC33CC33CC3,
    0x3C3CC3C3C3C33C3C,
)

# 回字形 (同心方环) 定位图案，与 generate_locator_pattern.build_hui_pattern()
# 同构，写成与 CODEWORD_CELL_MASKS 一致的 64-bit 掩码:
#
#     ########        ring 0 最外框  信号 28 格
#     #......#        ring 1 间隙    中性 20 格
#     #.####.#        ring 2 内框    信号 12 格
#     #.#..#.#        ring 3 核心    中性  4 格 (2x2)
#     #.#..#.#        ─────────────────────────
#     #.####.#        信号 40 / 中性 24
#     #......#
#     ########
#
# 约定同上: bit 1 = 白/中性, bit 0 = 黄/信号。仅供 --fill hui 使用。
HUI_CELL_MASK = 0x007E425A5A427E00

# 序列里末位 marker 的下标，也是 v1 训练集把定位图案塞进去的那一格。
LOCATOR_SLOT = 15
# v1 训练集用的"符号值"：超过 16 个码字，指向扩展矩阵末行（回字形）。
LOCATOR_SYMBOL = 16

# RS(15,5), GF(16), primitive polynomial x^4 + x + 1, generator 2,
# first consecutive root 1.  This is the systematic generator used by the
# native encoder (the first coefficient is implicit 1).
RS_GENERATOR: Tuple[int, ...] = (1, 4, 8, 10, 12, 9, 4, 2, 12, 2, 7)


def gf_multiply(left: int, right: int) -> int:
    """Multiply two GF(16) symbols using polynomial 0x13."""

    result = 0
    left &= 0xF
    right &= 0xF
    while right:
        if right & 1:
            result ^= left
        right >>= 1
        carry = bool(left & 0x8)
        left = (left << 1) & 0xF
        if carry:
            left ^= 0x13
        left &= 0xF
    return result


def encode_watermark_sequence(watermark_id: int, fill: str = "0") -> List[int]:
    """Return the native 16-symbol sequence for a numeric watermark ID.

    ``fill`` 决定末位 marker 画什么:

    * ``"0"``   (默认) — 末位是字面符号 0，画 ``CODEWORD_CELL_MASKS[0]``。
                      这是原生生成器 / EXE 的契约，也与 v1 顶层脚本一致。
    * ``"hui"``       — 末位改成 ``LOCATOR_SYMBOL``，画回字形。
                      复现 v1 训练数据集 (``wm_seq[-1] = LOCATOR_CODEWORD_INDEX``)。
    """

    if fill not in ("0", "hui"):
        raise ValueError(f'fill must be "0" or "hui", got {fill!r}')
    if not 0 <= watermark_id <= MAX_WATERMARK_ID:
        raise ValueError(
            f"watermark ID must be in 0..{MAX_WATERMARK_ID} "
            f"(0x{MAX_WATERMARK_ID:05X}), got {watermark_id}"
        )

    # Native code stores the five payload nibbles most-significant first,
    # then performs the systematic RS division in place.
    encoded = [(watermark_id >> (4 * (4 - i))) & 0xF for i in range(5)] + [0] * 10
    payload = encoded[:5]
    for i in range(5):
        coefficient = encoded[i]
        if coefficient == 0:
            continue
        for j in range(1, len(RS_GENERATOR)):
            encoded[i + j] ^= gf_multiply(RS_GENERATOR[j], coefficient)

    # The payload is restored after parity calculation.  The screen layout
    # consumes the reversed 15-symbol codeword followed by a zero marker.
    encoded[:5] = payload
    sequence = list(reversed(encoded)) + [0]
    if fill == "hui":
        sequence[LOCATOR_SLOT] = LOCATOR_SYMBOL
    return sequence


def round_ties_to_even(numerator: int, denominator: int) -> int:
    """Round a non-negative rational number exactly as the C++ helper does."""

    quotient, remainder = divmod(numerator, denominator)
    if remainder * 2 > denominator or (
        remainder * 2 == denominator and (quotient & 1)
    ):
        quotient += 1
    return quotient


def rounded_boundary(extent: int, index: int, count: int) -> int:
    """Return round-to-even(extent * index / count)."""

    return round_ties_to_even(extent * index, count)


def _stripe_mask(
    width: int,
    height: int,
    angle: float,
    period: int,
    stripe_width: int,
) -> np.ndarray:
    """Return a boolean mask using the native global stripe coordinates.

    The C++ option stores angle as ``float`` before doing double-precision
    trigonometry.  Quantizing through ``np.float32`` here keeps arbitrary-angle
    output aligned with the executable; the four cardinal/diagonal angles use
    exact integer paths just like the native code.
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


def build_rgb_templates(
    sequence: Sequence[int],
    width: int,
    height: int,
    block_rows: int = BLOCK_ROWS,
    block_cols: int = BLOCK_COLS,
    polarity: int = 0,
    alternate: bool = False,
    pattern: str = "diagonal",
    angle: float = 45.0,
    period: int = 4,
    stripe_width: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build native-compatible positive/inverse opaque RGBA arrays.

    ``sequence`` 允许出现 ``LOCATOR_SYMBOL`` (=16)，那一格画回字形；
    其余符号必须是 0..15。这与 v1 训练集的扩展 17x64 矩阵一致。
    """

    if len(sequence) != 16 or any(
        not (0 <= int(v) < 16 or int(v) == LOCATOR_SYMBOL) for v in sequence
    ):
        raise ValueError(
            f"sequence must contain exactly 16 symbols in 0..15 or {LOCATOR_SYMBOL}"
        )
    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError("block_rows and block_cols must be positive")
    if width // block_cols < 2 or height // block_rows < 2:
        raise ValueError(
            "screen dimensions are too small for the requested 2x2 message layout"
        )
    if polarity not in (0, 1):
        raise ValueError("polarity must be in 0..1")
    if pattern not in ("uniform", "diagonal"):
        raise ValueError("pattern must be uniform or diagonal")
    if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
        raise ValueError("angle must be finite and in 0..180 degrees")
    if period < 1 or period > 4096:
        raise ValueError("period must be in 1..4096")
    if stripe_width < 0 or stripe_width > period:
        raise ValueError("stripe_width must be in 0..period")

    rows = block_rows * 2 * GRID_SIZE
    cols = block_cols * 2 * GRID_SIZE
    positive = np.full((height, width, 4), 255, dtype=np.uint8)
    inverse = np.full((height, width, 4), 255, dtype=np.uint8)
    keep = np.ones((height, width), dtype=bool)
    if pattern == "diagonal":
        keep = _stripe_mask(width, height, angle, period, stripe_width)

    for row in range(rows):
        top = rounded_boundary(height, row, rows)
        bottom = rounded_boundary(height, row + 1, rows)
        if top == bottom:
            continue
        block_row = row // (2 * GRID_SIZE)
        for col in range(cols):
            left = rounded_boundary(width, col, cols)
            right = rounded_boundary(width, col + 1, cols)
            if left == right:
                continue
            block_col = col // (2 * GRID_SIZE)
            message = block_message_index(block_row, block_col)
            position = (row // GRID_SIZE % 2) * 2 + (col // GRID_SIZE % 2)
            symbol = int(sequence[message * 4 + MESSAGE_ORDER[position]])
            cell = (row % GRID_SIZE) * GRID_SIZE + (col % GRID_SIZE)
            if symbol == LOCATOR_SYMBOL:
                bit = (HUI_CELL_MASK >> cell) & 1
            else:
                bit = (CODEWORD_CELL_MASKS[symbol] >> cell) & 1

            # Native polarity: at typeValue=0, a zero matrix bit is yellow;
            # at typeValue=1, a one matrix bit is yellow.
            yellow = (not bit) if polarity == 0 else bool(bit)
            positive_blue = 0 if yellow else 255
            inverse_blue = 255 - positive_blue if alternate else positive_blue

            if positive_blue == 0:
                region = keep[top:bottom, left:right]
                positive[top:bottom, left:right, 2][region] = 0
            if inverse_blue == 0:
                region = keep[top:bottom, left:right]
                inverse[top:bottom, left:right, 2][region] = 0

    return positive, inverse


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a RGBA array with Pillow using a lossless PNG encoder."""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("expected an H x W x 4 uint8 RGBA array")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGBA").save(path, format="PNG", optimize=False)


def _parse_int(text: str) -> int:
    try:
        return int(text, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {text}") from exc


def _parse_nonnegative_int(text: str) -> int:
    value = _parse_int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"value must be non-negative")
    return value


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate opaque yellow/white DX Overlay screen templates (v1 staggered layout)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--id", type=_parse_nonnegative_int, help="one watermark ID")
    source.add_argument(
        "--id-start", type=_parse_nonnegative_int, help="first ID in an inclusive range"
    )
    source.add_argument(
        "--train-codeword",
        type=_parse_int,
        metavar="0..15",
        help="repeat one training codeword at all 96 positions",
    )
    parser.add_argument(
        "--id-end",
        type=_parse_nonnegative_int,
        help="inclusive final ID for --id-start (defaults to --id-start)",
    )
    parser.add_argument("--width", type=_parse_int, default=1920, help="screen width")
    parser.add_argument("--height", type=_parse_int, default=1080, help="screen height")
    parser.add_argument("--block-rows", type=_parse_int, default=BLOCK_ROWS)
    parser.add_argument("--block-cols", type=_parse_int, default=BLOCK_COLS)
    parser.add_argument("--polarity", type=_parse_int, choices=(0, 1), default=0)
    parser.add_argument(
        "--pattern",
        choices=("uniform", "diagonal"),
        default="diagonal",
        help="uniform yellow cells or global diagonal stripes (default: diagonal)",
    )
    parser.add_argument("--angle", type=float, default=45.0, help="stripe angle in degrees")
    parser.add_argument("--period", type=_parse_int, default=4, help="stripe period in pixels")
    parser.add_argument(
        "--stripe-width",
        type=_parse_int,
        default=4,
        help="yellow stripe width in coordinate units",
    )
    parser.add_argument(
        "--alternate",
        action="store_true",
        help="swap yellow/white code cells in the inverse PNG (A/B mode)",
    )
    parser.add_argument(
        "--fill",
        choices=("0", "hui"),
        default="0",
        help="locator-slot fill: '0' = codeword 0 (native, default), "
        "'hui' = 回字形 (v1 training set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yellow_white_templates"),
        help="output directory (default: yellow_white_templates)",
    )
    parser.add_argument(
        "--print-sequence",
        action="store_true",
        help="print the 16-symbol RS/training sequence for each generated item",
    )
    return parser.parse_args(argv)


def _items(args: argparse.Namespace) -> Iterable[Tuple[str, List[int]]]:
    if args.train_codeword is not None:
        if args.id_end is not None or not 0 <= args.train_codeword <= 15:
            raise ValueError("training codeword must be 0..15 and cannot use --id-end")
        yield str(args.train_codeword), [args.train_codeword] * 16
        return

    if args.id is not None:
        if args.id_end is not None:
            raise ValueError("--id-end is only valid with --id-start")
        yield str(args.id), encode_watermark_sequence(args.id, fill=args.fill)
        return

    assert args.id_start is not None
    end = args.id_start if args.id_end is None else args.id_end
    if end < args.id_start:
        raise ValueError("--id-end must be greater than or equal to --id-start")
    if end > MAX_WATERMARK_ID:
        raise ValueError(f"--id-end must be <= {MAX_WATERMARK_ID}")
    for watermark_id in range(args.id_start, end + 1):
        yield str(watermark_id), encode_watermark_sequence(watermark_id, fill=args.fill)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.width <= 0 or args.height <= 0:
            raise ValueError("width and height must be positive")
        if args.id_start is not None and args.id_start > MAX_WATERMARK_ID:
            raise ValueError(f"--id-start must be <= {MAX_WATERMARK_ID}")

        for label, sequence in _items(args):
            if args.print_sequence:
                print(f"{label}: " + " ".join(f"{value:X}" for value in sequence))
            positive, inverse = build_rgb_templates(
                sequence,
                args.width,
                args.height,
                args.block_rows,
                args.block_cols,
                args.polarity,
                args.alternate,
                args.pattern,
                args.angle,
                args.period,
                args.stripe_width,
            )
            stem = f"wm_template_{label}"
            positive_path = args.output / f"{stem}.png"
            inverse_path = args.output / f"{stem}_inverse.png"
            write_png(positive_path, positive)
            write_png(inverse_path, inverse)
            print(f"wrote {positive_path}")
            print(f"wrote {inverse_path}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
