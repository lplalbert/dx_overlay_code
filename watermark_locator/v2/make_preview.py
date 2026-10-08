#!/usr/bin/env python3
"""把 v2 模板按数据集管线的 alpha 契约混到载体上，产出一张加水印图。

混合公式与 ``watermark_locator/dataset/generate_dataset.py`` 的
``alpha_blend_watermark`` **逐位一致**（这里直接 import，不复制实现）::

    out    = α_eff · template + (1 − α_eff) · carrier
    α_eff  = α · saturate( max_c |template_c/255 − 1| · 2 )

0/255 模板下 dynamicMask ∈ {0, 1}：

* 白色(中性)格 (255,255,255) → mask=0 → **完全不改**
* 黄色(信号)格 (0,255,255)   → mask=1 → 三通道都朝模板值靠，
  max|Δ| = α × 255 = 0.032 × 255 = 8.16

**载体禁止 resize**：必须恰好 1920×1080，否则直接报错
（resize 会把纹理低通成糊的，混出来的图不像真实截屏）。

Examples:

    python make_preview.py \\
        --template sample/wm_template_123456.png \\
        --carrier  /data1/lpl/datasets/test/Snipaste_2026-09-24_08-21-10.png \\
        --output   sample/watermarked_123456.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 复用数据集管线的混合实现，保证逐位一致。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
from generate_dataset import alpha_blend_watermark  # noqa: E402

ALPHA = 0.032
SCREEN_W, SCREEN_H = 1920, 1080
MAX_DELTA = ALPHA * 255  # 8.16


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Blend a v2 template onto a native carrier.")
    p.add_argument("--template", type=Path, required=True, help="v2 模板 PNG (RGBA)")
    p.add_argument("--carrier", type=Path, required=True, help="载体图，必须是 1920x1080")
    p.add_argument("--output", type=Path, required=True, help="加水印输出 PNG")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--residual", type=Path, default=None,
                   help="可选：|Δ| 放大 20 倍的残差图，便于目检")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # 模板: 读成 BGR 三通道 (丢 alpha)。生成器写的是 RGBA，
    # 通道序 R,G,B,A，黄格 = RGB(255,255,0) → BGR(0,255,255)，正好是契约里的黄。
    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if template is None:
        print(f"error: cannot read template {args.template}", file=sys.stderr)
        return 2
    if template.shape[:2] != (SCREEN_H, SCREEN_W):
        print(f"error: template must be {SCREEN_W}x{SCREEN_H}, got "
              f"{template.shape[1]}x{template.shape[0]}", file=sys.stderr)
        return 2

    carrier = cv2.imread(str(args.carrier), cv2.IMREAD_COLOR)
    if carrier is None:
        print(f"error: cannot read carrier {args.carrier}", file=sys.stderr)
        return 2
    if carrier.shape[:2] != (SCREEN_H, SCREEN_W):
        # 载体绝不 resize —— 见 memory: 载体画布禁止 resize
        print(f"error: carrier must be exactly {SCREEN_W}x{SCREEN_H} (no resize), got "
              f"{carrier.shape[1]}x{carrier.shape[0]}", file=sys.stderr)
        return 2

    blended = alpha_blend_watermark(carrier, template, args.alpha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), blended):
        print(f"error: cannot write {args.output}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}")

    # ── 契约校验 ──
    diff = np.abs(blended.astype(np.int16) - carrier.astype(np.int16))
    dev = np.max(np.abs(template.astype(np.float32) - 255.0), axis=2) / 255.0
    signal = dev > 0.5          # 黄(信号)格
    neutral = ~signal           # 白(中性)格

    print(f"\nalpha = {args.alpha}   max|Δ| bound = {MAX_DELTA:.2f}")
    print(f"  template : {args.template.name}")
    print(f"  carrier  : {args.carrier.name}")
    print(f"  signal (yellow) cells : {signal.sum():8d} px  ({signal.mean()*100:.2f}%)")
    print(f"  neutral (white) cells : {neutral.sum():8d} px  ({neutral.mean()*100:.2f}%)")

    ok = True
    max_all = int(diff.max())
    print(f"  max|Δ| overall        : {max_all}  (<= {int(MAX_DELTA)})")
    if max_all > int(MAX_DELTA):
        ok = False
        print("    [FAIL] exceeds bound")

    if neutral.any():
        max_n = int(diff[neutral].max())
        print(f"  max|Δ| on white cells : {max_n}  (must be 0)")
        if max_n != 0:
            ok = False
            print("    [FAIL] white cells must be untouched")

    for c, name in enumerate("BGR"):
        vals, cnts = np.unique(diff[:, :, c][signal], return_counts=True)
        top = ", ".join(f"{int(v)}:{int(n)}" for v, n in zip(vals[-4:], cnts[-4:]))
        print(f"  Δ{name} on signal cells : max={int(diff[:, :, c][signal].max())}  [{top}]")

    # 被改的像素应当只出现在条纹 keep 掩码内（45°, period 4, width 2 → ~25%）
    changed = diff.max(axis=2) > 0
    yy, xx = np.indices(changed.shape)
    phase = np.fmod((xx + yy).astype(np.float64), 4.0)
    stripe = phase < 2
    outside = changed & ~stripe
    print(f"  changed px outside 45°/4/2 stripe : {int(outside.sum())}  (must be 0)")
    if outside.sum():
        ok = False
        print("    [FAIL] stripe keep-mask leaked")

    if args.residual is not None:
        amp = np.clip(diff.max(axis=2).astype(np.float32) * 20.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(args.residual), cv2.applyColorMap(amp, cv2.COLORMAP_INFERNO))
        print(f"wrote {args.residual}  (|Δ| x20)")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
