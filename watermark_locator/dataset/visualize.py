"""
全流程可视化: 全局斜条纹 → 码字模板 → 条纹调制 → alpha融合 → 噪声(标签同步)
"""
import cv2
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from generate_dataset import (
    stripe_mask, gen_diagonal_stripe_tp, gen_wm_block, gen_block_single_uv,
    gen_rect_tp, alpha_blend_watermark, generate_one_sample, apply_pair_noise,
    add_tile_rotate_crop_noise, add_pimog_noise, add_wechat_noise,
    get_locator_abs_rect, get_locator_positions,
    BLOCK_H, BLOCK_W, MSG_H, MSG_W, BLOCK_ROWS, BLOCK_COLS,
    SCREEN_W, SCREEN_H,
)
from generate_locator_pattern import FIX_FG_MATRIX

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis')
os.makedirs(OUT, exist_ok=True)


def make_row(items, img_h=256, bg=255):
    resized = []
    for name, img in items:
        h, w = img.shape[:2]
        scale = img_h / h
        resized.append((name, cv2.resize(img, (int(w * scale), img_h), interpolation=cv2.INTER_NEAREST)))
    pad, title_h = 8, 24
    total_w = sum(x[1].shape[1] for x in resized) + pad * (len(resized) + 1)
    row = np.full((img_h + title_h + pad, total_w, 3), bg, dtype=np.uint8)
    x = pad
    for name, img in resized:
        h, w = img.shape[:2]
        row[title_h: title_h + h, x: x + w] = img
        cv2.putText(row, name, (x, title_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(row, (x - 1, title_h - 1), (x + w, title_h + h), (80, 80, 80), 1)
        x += w + pad
    return row


def stack_rows(rows):
    pad = 12
    total_w = max(r.shape[1] for r in rows)
    total_h = sum(r.shape[0] for r in rows) + pad * (len(rows) - 1)
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    y = 0
    for r in rows:
        canvas[y: y + r.shape[0], 0: r.shape[1]] = r
        y += r.shape[0] + pad
    return canvas


def draw_labels(img, mask, bboxes, mask_color=(0, 0, 220), bbox_color=(0, 255, 0)):
    vis = img.copy()
    if mask is not None:
        m = mask > 127
        vis[m] = (vis[m].astype(np.float32) * 0.35 + np.array(mask_color) * 0.65).astype(np.uint8)
    if bboxes:
        for bbox in bboxes:
            cls, cx, cy, w, h = bbox
            x1 = int((cx - w / 2) * SCREEN_W)
            y1 = int((cy - h / 2) * SCREEN_H)
            x2 = int((cx + w / 2) * SCREEN_W)
            y2 = int((cy + h / 2) * SCREEN_H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), bbox_color, 2)
    return vis


def bgra_to_bgr(bgra):
    cb = bgra[:, :, 0].astype(np.float32)
    cr = bgra[:, :, 1].astype(np.float32)
    y  = bgra[:, :, 2].astype(np.float32)
    r = np.clip(y + 1.402 * (cr - 128), 0, 255)
    g = np.clip(y - 0.714136 * (cr - 128) - 0.344136 * (cb - 128), 0, 255)
    b = np.clip(y + 1.772 * (cb - 128), 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])


# ═══ Row 1: 全局斜条纹掩码 ═══
keep_global = stripe_mask(SCREEN_W, SCREEN_H, angle=45.0, period=4, stripe_width=2)
keep_vis = (keep_global * 255).astype(np.uint8)
keep_bgr = cv2.cvtColor(keep_vis, cv2.COLOR_GRAY2BGR)
keep_small = cv2.resize(keep_bgr, (480, 270), interpolation=cv2.INTER_NEAREST)

# 8x8 放大
keep_8x8 = stripe_mask(8, 8, angle=45.0, period=4, stripe_width=2)
keep_8x8_vis = cv2.cvtColor((keep_8x8 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
keep_8x8_vis = cv2.resize(keep_8x8_vis, (128, 128), interpolation=cv2.INTER_NEAREST)

row1 = make_row([
    ('global stripe mask (45deg p=4 w=2)', keep_small),
    ('stripe 8x8 detail', keep_8x8_vis),
    (f'coverage: {keep_global.sum()/keep_global.size*100:.0f}%', np.full((270, 100, 3), 255, dtype=np.uint8)),
], img_h=270)

# ═══ Row 2: 码字模板 (均匀填充, 无条纹) ═══
t0 = bgra_to_bgr(gen_block_single_uv(0, 0, 64, v_tp_fn=gen_rect_tp))
t1 = bgra_to_bgr(gen_block_single_uv(1, 1, 64, v_tp_fn=gen_rect_tp))

locator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'locator_pattern.npy')
locator_pattern = np.load(locator_path) if os.path.exists(locator_path) else FIX_FG_MATRIX[0].copy()
cw5_block = bgra_to_bgr(gen_wm_block(FIX_FG_MATRIX[5], block_size=32, v_tp_fn=gen_rect_tp))
locator_block = bgra_to_bgr(gen_wm_block(locator_pattern, block_size=32, v_tp_fn=gen_rect_tp))

row2 = make_row([
    ('t0 (bit=0, uniform)', t0),
    ('t1 (bit=1, uniform)', t1),
    ('CW5 block (uniform)', cw5_block),
    ('Locator block (uniform)', locator_block),
], img_h=200)

# ═══ Row 3: 条纹调制后的效果 ═══
# 构建完整的 wm_full 看条纹调制效果
rng0 = np.random.RandomState(42)
blended, bboxes, mask, noise_types = generate_one_sample(
    12345, FIX_FG_MATRIX, locator_pattern, 0.016, rng0, apply_noise=False
)

ycrcb = cv2.cvtColor(blended, cv2.COLOR_BGR2YCrCb)
cb_enh = ((ycrcb[:, :, 2].astype(np.float32) - 128) * 15 + 128).clip(0, 255).astype(np.uint8)
cb_vis = cv2.cvtColor(cb_enh, cv2.COLOR_GRAY2BGR)

row3 = make_row([
    ('carrier (white)', np.full_like(blended, 255)),
    ('watermarked (a=0.016)', blended),
    ('Cb (15x, global stripe)', cb_vis),
], img_h=270)

# ═══ Row 4: 各噪声 + 标签同步 ═══
noise_demos = []
noise_demos.append(('identity', blended.copy(), mask.copy(), [list(b) for b in bboxes]))
noise_demos.append(('wechat', add_wechat_noise(blended.copy()), mask.copy(), [list(b) for b in bboxes]))

tc_img, tc_mask, tc_bboxes = add_tile_rotate_crop_noise(blended.copy(), mask.copy(), [list(b) for b in bboxes])
noise_demos.append(('tile_crop', tc_img, tc_mask, tc_bboxes))

pg_img, pg_mask, pg_bboxes = add_pimog_noise(blended.copy(), mask.copy(), [list(b) for b in bboxes])
noise_demos.append(('pimog', pg_img, pg_mask, pg_bboxes))

pn_img, pn_mask, pn_bboxes, pn_types = apply_pair_noise(
    blended.copy(), mask.copy(), [list(b) for b in bboxes], np.random.RandomState(99))
noise_demos.append((f'pair:{"+".join(pn_types)}', pn_img, pn_mask, pn_bboxes))

row4_items = [(name, draw_labels(img, msk, bbs)) for name, img, msk, bbs in noise_demos]
row4 = make_row(row4_items, img_h=270)

# ═══ Row 5: tile_crop 标签对比 ═══
before_vis = draw_labels(blended, mask, bboxes)
after_vis = draw_labels(tc_img, tc_mask, tc_bboxes)

def bbox_text(bbs, title):
    img = np.full((270, 480, 3), 255, dtype=np.uint8)
    cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for i, b in enumerate(bbs[:6]):
        txt = f'{int(b[0])}  {b[1]:.4f}  {b[2]:.4f}  {b[3]:.4f}  {b[4]:.4f}'
        cv2.putText(img, txt, (10, 48 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1, cv2.LINE_AA)
    return img

row5 = make_row([
    ('BEFORE noise', before_vis),
    ('AFTER tile_crop', after_vis),
    ('bbox before', bbox_text(bboxes, f'Before ({len(bboxes)} boxes)')),
    ('bbox after', bbox_text(tc_bboxes, f'After tile_crop ({len(tc_bboxes)} boxes)')),
], img_h=270)

# ═══ 组装 ═══
canvas = stack_rows([row1, row2, row3, row4, row5])
out_path = os.path.join(OUT, 'full_pipeline.png')
cv2.imwrite(out_path, canvas)
print(f'Saved: {out_path}')

print(f'\n=== 标签同步验证 ===')
for name, img, msk, bbs in noise_demos:
    print(f'{name}: {len(bbs)} boxes, mask pixels={np.count_nonzero(msk)}')
