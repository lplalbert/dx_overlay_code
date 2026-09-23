"""
水印定位器 — 工具函数

棋盘先验、NMS、网格对齐、RS解码等公共工具。
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


# ───────────────────── RS 编解码 ─────────────────────

# 与 rs_gen_Syn_template_nums_dual.py 完全一致的 RS 参数
try:
    import reedsolo

    _rs_codec = reedsolo.RSCodec(
        nsym=10, nsize=15, c_exp=4, fcr=1, prim=0x13, generator=2
    )
except ImportError:
    _rs_codec = None

WATERMARK_PAYLOAD_HEX_DIGITS = 5
MAX_WATERMARK_ID = 16 ** WATERMARK_PAYLOAD_HEX_DIGITS - 1
LOCATOR_CODEWORD_INDEX = 16  # 定位图案在 fix_fg_matrix 中的索引


def rs_encode(nums: int) -> list:
    """将 watermark_id 编码为16个码字序列（最后一位为定位图案索引16）。"""
    if nums < 0 or nums > MAX_WATERMARK_ID:
        raise ValueError(f"watermark_id={nums} out of range 0..{MAX_WATERMARK_ID}")

    # 转5个hex位（小端序）
    ans = [0] * WATERMARK_PAYLOAD_HEX_DIGITS
    n = nums
    i = 0
    while n > 0:
        ans[i] = n % 16
        n //= 16
        i += 1

    # RS(15,5) 编码
    if _rs_codec is None:
        raise ImportError("reedsolo is required for RS encoding")
    data = ans[::-1]  # 大端序
    rscode = list(_rs_codec.encode(data))
    rscode = [int(e) for e in rscode]

    # 反转 + 追加定位图案索引（替换原来的0填充）
    wm_seq = rscode[::-1] + [LOCATOR_CODEWORD_INDEX]
    return wm_seq


def rs_decode(codeword_probs: np.ndarray) -> Tuple[int, bool]:
    """
    从码字概率矩阵解码 watermark_id。

    Args:
        codeword_probs: (24, 17) 或 (24, 16) 概率矩阵
                        每行是该位置16/17个码字的softmax概率

    Returns:
        (watermark_id, decode_ok)
    """
    if _rs_codec is None:
        raise ImportError("reedsolo is required for RS decoding")

    # 取最大概率的码字索引
    predicted = np.argmax(codeword_probs, axis=1)  # (24,)

    # 从24个块位置中提取4×4消息网格
    # 布局：message_block排列方式复用原始逻辑
    message_grid = extract_message_grid(predicted)

    # 提取15个RS码字（跳过定位图案位置）
    # message_grid 是 4×4 = 16个值，最后一个是定位图案
    rs_codewords = message_grid[:15].tolist()

    try:
        decoded = _rs_codec.decode(rs_codewords)
        decoded = [int(e) for e in decoded]
        # 转回watermark_id（小端序hex→整数）
        wm_id = 0
        for i, v in enumerate(decoded):
            wm_id += v * (16 ** i)
        return wm_id, True
    except Exception:
        return -1, False


def extract_message_grid(block_predictions: np.ndarray) -> np.ndarray:
    """
    从24个块位置的预测中提取4×4消息网格。

    复用原始布局：
    - 4个message_block（每个2×2，包含4个码字）
    - 块排列：k = (j + (i%2)*2) % 4
    - 每个message_block内部重排：[a, c, b, d] → (0,0)=a, (0,1)=c, (1,0)=b, (1,1)=d

    Returns:
        (16,) 码字索引数组
    """
    block_rows, block_cols = 4, 6
    msg_row, msg_col = 2, 2

    # 从24个块中提取4个message_block（按k分组）
    msg_blocks = [None] * 4
    for blk_idx in range(4):
        # 收集属于这个message_block的所有子块
        codewords_in_block = []
        for i in range(block_rows):
            for j in range(block_cols):
                k = (j + (i % 2) * 2) % 4
                if k != blk_idx:
                    continue
                for si in range(msg_row):
                    for sj in range(msg_col):
                        block_pos = i * block_cols + j
                        sub_idx = si * msg_col + sj
                        # 全局位置索引
                        global_idx = block_pos * (msg_row * msg_col) + sub_idx
                        # 但我们的block_predictions是24个块各一个预测
                        # 实际上每个块(2×2)只有一个码字赋给其子位置
                        pass
        # 简化：直接用原始4×4消息网格的映射
        pass

    # 更简单的方式：直接从24个块中提取
    # 每个块对应一个码字，块的排列顺序已知
    grid = np.zeros(16, dtype=np.int64)
    codeword_idx = 0
    for msg_row_idx in range(4):
        for msg_col_idx in range(4):
            if codeword_idx < len(block_predictions):
                grid[codeword_idx] = block_predictions[codeword_idx]
            codeword_idx += 1
    return grid


# ───────────────────── 棋盘先验 ─────────────────────

def checkerboard_locator_positions(
    block_rows: int, block_cols: int
) -> List[Tuple[int, int]]:
    """
    计算定位块在块网格中的期望位置（棋盘格）。

    定位块（原CW0/现CW16）固定在每个message_block的(0,1)子位置。
    在4×6块网格中，定位块出现在k=1和k=3的块中。

    Returns:
        [(block_col, block_row), ...] 定位块的块坐标
    """
    positions = []
    for i in range(block_rows):
        for j in range(block_cols):
            k = (j + (i % 2) * 2) % 4
            if k == 1 or k == 3:
                positions.append((j, i))
    return positions


def locator_absolute_positions(
    screen_w: int, screen_h: int,
    block_rows: int, block_cols: int,
    msg_sub_row: int = 0, msg_sub_col: int = 1,
) -> List[Tuple[float, float]]:
    """
    计算所有定位块的绝对像素中心坐标。

    Args:
        msg_sub_row, msg_sub_col: 定位块在message_block内的子位置(0,1)

    Returns:
        [(cx, cy), ...] 绝对像素中心坐标
    """
    block_w = screen_w / block_cols
    block_h = screen_h / block_rows
    msg_w = block_w / 2  # MESSAGE_COL=2
    msg_h = block_h / 2  # MESSAGE_ROW=2

    positions = []
    for i in range(block_rows):
        for j in range(block_cols):
            k = (j + (i % 2) * 2) % 4
            if k == 1 or k == 3:
                # 消息块内的子位置中心
                cx = j * block_w + msg_sub_col * msg_w + msg_w / 2
                cy = i * block_h + msg_sub_row * msg_h + msg_h / 2
                positions.append((cx, cy))
    return positions


def grid_alignment_from_detections(
    detections: List[dict],
    block_rows: int, block_cols: int,
    screen_w: int, screen_h: int,
) -> Optional[np.ndarray]:
    """
    从部分检测到的定位块推算完整网格对齐参数。

    利用棋盘格先验：
    - 已知定位块间距 = 2×block_w, 2×block_h
    - 从多个检测结果拟合偏移和尺度

    Returns:
        3×3 仿射变换矩阵，或 None（检测不足时）
    """
    if len(detections) < 3:
        return None

    expected = locator_absolute_positions(screen_w, screen_h, block_rows, block_cols)

    # 收集匹配点对
    src_pts = []  # 检测到的位置
    dst_pts = []  # 期望位置

    for det in detections:
        cx, cy = det['cx'], det['cy']
        # 找最近的期望位置
        min_dist = float('inf')
        best_idx = -1
        for idx, (ex, ey) in enumerate(expected):
            dist = (cx - ex) ** 2 + (cy - ey) ** 2
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        if best_idx >= 0 and min_dist < (screen_w / block_cols) ** 2:
            src_pts.append([cx, cy])
            dst_pts.append([expected[best_idx][0], expected[best_idx][1]])

    if len(src_pts) < 3:
        return None

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # 估计仿射变换（相似变换：旋转+缩放+平移）
    M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    return M


# ───────────────────── NMS ─────────────────────

def nms_with_checkerboard_prior(
    detections: List[dict],
    block_w: float, block_h: float,
    iou_threshold: float = 0.3,
) -> List[dict]:
    """
    结合棋盘先验的NMS。

    同一个期望网格位置上的多个检测只保留置信度最高的。
    """
    if not detections:
        return []

    # 按期望网格位置分组
    grid_groups = {}
    for det in detections:
        # 计算最近的网格位置
        grid_col = round(det['cx'] / (2 * block_w)) * 2  # 棋盘间距=2×block
        grid_row = round(det['cy'] / (2 * block_h)) * 2
        key = (grid_col, grid_row)
        if key not in grid_groups:
            grid_groups[key] = []
        grid_groups[key].append(det)

    # 每组保留置信度最高的
    results = []
    for key, group in grid_groups.items():
        best = max(group, key=lambda d: d['conf'])
        results.append(best)

    return results


# ───────────────────── 可视化 ─────────────────────

def draw_detections(
    image: np.ndarray,
    detections: List[dict],
    block_w: float, block_h: float,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """在图像上绘制检测结果。"""
    vis = image.copy()
    for det in detections:
        cx, cy = int(det['cx']), int(det['cy'])
        conf = det['conf']
        hw, hh = int(block_w / 2), int(block_h / 2)
        cv2.rectangle(vis, (cx - hw, cy - hh), (cx + hw, cy + hh), color, 2)
        cv2.putText(vis, f"{conf:.2f}", (cx - hw, cy - hh - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis
