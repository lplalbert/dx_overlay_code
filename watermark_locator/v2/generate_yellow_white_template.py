#!/usr/bin/env python3
"""v2 版：生成 DX Overlay 黄/白 RGB 屏幕模板（交替行排布）。

与 v1 的差异（只有两处）:

1. **块级 message 排布** —— v2 用交替行，v1 用交错格::

       v2 交替行  message = (block_col % 2) + 2 * (block_row % 2)
           行0:  M1 M2 M1 M2 M1 M2
           行1:  M3 M4 M3 M4 M3 M4
           行2:  M1 M2 M1 M2 M1 M2
           行3:  M3 M4 M3 M4 M3 M4

       v1 交错格  message = (block_col + (2 if block_row % 2 else 0)) % 4
           行0:  M1 M2 M3 M4 M1 M2
           行1:  M3 M4 M1 M2 M3 M4
           行2:  M1 M2 M3 M4 M1 M2
           行3:  M3 M4 M1 M2 M3 M4

   为什么改：在小块级（8x12）做"固定间隔 → 相同样式"的 lag 匹配率，

       lag (小块)     v1      v2
         (4,0)     1.000   1.000     540 px
         (0,4)     0.000   1.000     640 px
         (4,4)     0.000   1.000     对角

   v2 的 (4,0)/(0,4)/(4,4) 三个都是精确不动点 → **矩形格，最小正周期唯一**。
   v1 只有 (4,0) 命中 → 交错格存在**半周期歧义**，检验结果依赖起始行。
   这是 v2 检测器"固定间隔呈现相同样式 ⇒ 找到间隔"这一先验的前提。

2. **末位填充** —— v2 **不再使用回字形**。末位 marker 就是字面符号 0，
   画 ``CODEWORD_CELL_MASKS[0]``。96 个小块全部是码字，检测目标因此
   是"小块本身"（1 类），不需要任何特殊定位图案。

   这也与原生生成器的契约一致（``encode_watermark_sequence`` 末位字面 0），
   v1 那套"末位画回字形"的训练集做法不在 v2 沿用。

其余契约与原生生成器完全一致:

* RS(15,5) over GF(16), payload 是五位十六进制 watermark ID;
* 反转后的 RS codeword 加一个末位 0 放进 8 x 12 = 96 个屏幕位置;
* 每个 codeword 是 8 x 8 格矩阵;
* 正/逆 PNG 是不透明 RGBA 黄/白图;
* 条纹使用与 EXE 相同的全局屏幕坐标（45 度、周期 4、带宽 2）。

几何层级::

    屏幕 1920x1080
      └─ 块  320x270    4 行 x 6 列 = 24 块
          └─ 小块 160x135   8 行 x 12 列 = 96 小块   ← 原子重复单元 / 观测尺寸
              └─ 格  20x16.875   8x8 = 64 格        ← 一个 RS 符号

Examples:

    python generate_yellow_white_template.py --id 123456 \\
        --width 1920 --height 1080 --output v2_templates

    python generate_yellow_white_template.py --id-start 100 --id-end 110 \\
        --width 1920 --height 1080 --output batch_templates --print-sequence

    # 打印块级排布 / 小块级样式格 (自检)
    python generate_yellow_white_template.py --id 1 --print-layout
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

# v2 交替行排布。块级 message 只依赖 (block_col % 2, block_row % 2)。
LAYOUT_NAME = "v2-alternating-rows"

# 小块（原子重复单元）的模板像素尺寸。
SUB_BLOCK_W = 160
SUB_BLOCK_H = 135

# 末位 marker 就是字面符号 0；v2 不使用回字形，因此不存在超出 0..15 的符号。
LOCATOR_SLOT = 15


def block_message_index(block_row: int, block_col: int) -> int:
    """返回块 (block_row, block_col) 的 message 下标，取值 0..3。

    v2 交替行::

        行0:  M1 M2 M1 M2 M1 M2
        行1:  M3 M4 M3 M4 M3 M4
        行2:  M1 M2 M1 M2 M1 M2
        行3:  M3 M4 M3 M4 M3 M4
    """
    return (block_col % 2) + 2 * (block_row % 2)


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


def encode_watermark_sequence(watermark_id: int) -> List[int]:
    """Return the native 16-symbol sequence for a numeric watermark ID."""

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
    return list(reversed(encoded)) + [0]


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


def sub_block_symbol_grid(
    sequence: Sequence[int],
    block_rows: int = BLOCK_ROWS,
    block_cols: int = BLOCK_COLS,
) -> np.ndarray:
    """返回小块级（block_rows*2 x block_cols*2）的符号索引格。

    这就是"样式格"：检测器的周期先验（固定间隔呈现相同样式）检验的
    就是这个格。v2 下它在 lag (4,0)/(0,4)/(4,4) 上是精确不动点。
    """
    grid = np.empty((block_rows * 2, block_cols * 2), dtype=np.int16)
    for row in range(block_rows * 2):
        for col in range(block_cols * 2):
            block_row, block_col = row // 2, col // 2
            message = block_message_index(block_row, block_col)
            position = (row % 2) * 2 + (col % 2)
            grid[row, col] = int(sequence[message * 4 + MESSAGE_ORDER[position]])
    return grid


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

    v2 不接受回字形符号：``sequence`` 的 16 个符号必须都在 0..15。
    """

    if len(sequence) != 16 or any(not 0 <= int(v) < 16 for v in sequence):
        raise ValueError("sequence must contain exactly 16 GF(16) symbols (0..15)")
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


def _print_layout(sequence: Sequence[int], block_rows: int, block_cols: int) -> None:
    """打印块级排布与小块级样式格，用于自检 v2 的周期结构。"""

    print(f"layout: {LAYOUT_NAME}")
    print(f"block message grid ({block_rows} x {block_cols}), "
          f"M1..M4 = sequence[0..15] in groups of 4:")
    for br in range(block_rows):
        cells = []
        for bc in range(block_cols):
            m = block_message_index(br, bc)
            cells.append(f"M{m + 1}")
        print("   " + " ".join(cells))

    grid = sub_block_symbol_grid(sequence, block_rows, block_cols)
    print(f"\nsub-block symbol grid ({grid.shape[0]} x {grid.shape[1]}), "
          f"tile = {SUB_BLOCK_W}x{SUB_BLOCK_H} template px:")
    for r in range(grid.shape[0]):
        print("   " + " ".join(f"{int(v):2X}" for v in grid[r]))

    def lag_match(dr: int, dc: int) -> float:
        a = grid[: grid.shape[0] - dr or None, : grid.shape[1] - dc or None]
        b = grid[dr:, dc:]
        return float((a == b).mean()) if a.size else 0.0

    print("\nperiodicity test (lag match rate on the sub-block symbol grid):")
    print(f"   {'lag':>8} {'match':>7}  {'meaning':<28}")
    for dr, dc, meaning in (
        (0, 4, "right 4 sub = 640 px"),
        (4, 0, "down 4 sub = 540 px"),
        (4, 4, "diagonal = super-cell period"),
        (2, 2, "half-period diagonal"),
        (0, 2, "half-period horizontal"),
        (2, 0, "half-period vertical"),
    ):
        print(f"   {f'({dr},{dc})':>8} {lag_match(dr, dc):>7.3f}  {meaning:<28}")
    print("   expect (4,0)/(0,4)/(4,4) = 1.000  -> rectangular lattice, "
          "unique minimal period")


def _parse_int(text: str) -> int:
    try:
        return int(text, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {text}") from exc


def _parse_nonnegative_int(text: str) -> int:
    value = _parse_int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate opaque yellow/white DX Overlay screen templates "
        "(v2 alternating-row layout, no 回字形)."
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
    parser.add_argument(
        "--print-layout",
        action="store_true",
        help="print the block message grid, sub-block symbol grid and the "
        "periodicity test, then exit",
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
        yield str(args.id), encode_watermark_sequence(args.id)
        return

    assert args.id_start is not None
    end = args.id_start if args.id_end is None else args.id_end
    if end < args.id_start:
        raise ValueError("--id-end must be greater than or equal to --id-start")
    if end > MAX_WATERMARK_ID:
        raise ValueError(f"--id-end must be <= {MAX_WATERMARK_ID}")
    for watermark_id in range(args.id_start, end + 1):
        yield str(watermark_id), encode_watermark_sequence(watermark_id)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.width <= 0 or args.height <= 0:
            raise ValueError("width and height must be positive")
        if args.id_start is not None and args.id_start > MAX_WATERMARK_ID:
            raise ValueError(f"--id-start must be <= {MAX_WATERMARK_ID}")

        items = list(_items(args))
        if args.print_layout:
            # 排布只依赖 sequence 的 16 个符号槽位，取第一项即可。
            _print_layout(items[0][1], args.block_rows, args.block_cols)
            return 0

        for label, sequence in items:
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
