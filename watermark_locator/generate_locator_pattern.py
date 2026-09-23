"""
生成唯一定位图案（Locator Pattern）

定位图案需满足：
1. 与 fix_fg_matrix 的所有16行 Hamming 距离 > 20
2. 具有独特空间结构（非对称，便于CNN区分）
3. 输出为 numpy 数组，可追加到 fix_fg_matrix

用法:
    python generate_locator_pattern.py [--output locator_pattern.npy]
"""

import numpy as np
import os
import argparse

# 原始16个码字模板
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


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """计算两个+1/-1向量的Hamming距离（不同位数）。"""
    return int(np.sum(a != b))


def min_hamming_to_matrix(pattern: np.ndarray, matrix: np.ndarray) -> int:
    """计算pattern与matrix所有行的最小Hamming距离。"""
    return min(hamming_distance(pattern, matrix[i]) for i in range(len(matrix)))


def count_transitions(pattern: np.ndarray) -> int:
    """计算8×8网格中相邻单元格值不同的次数（水平+垂直）。"""
    m = pattern.reshape(8, 8)
    h = sum(1 for i in range(8) for j in range(7) if m[i, j] != m[i, j+1])
    v = sum(1 for i in range(7) for j in range(8) if m[i, j] != m[i+1, j])
    return h + v


def has_isolated_points(pattern: np.ndarray) -> bool:
    """
    检测8×8网格中是否存在孤立点。

    孤立点定义：某个单元格的4个上下左右邻居全都是相反值。
    这种点在JPEG压缩后极易被抹平，导致模式信息丢失。

    边界上的点只检查存在的邻居（2~3个）。
    """
    m = pattern.reshape(8, 8)
    for i in range(8):
        for j in range(8):
            val = m[i, j]
            neighbors = []
            if i > 0: neighbors.append(m[i-1, j])
            if i < 7: neighbors.append(m[i+1, j])
            if j > 0: neighbors.append(m[i, j-1])
            if j < 7: neighbors.append(m[i, j+1])
            # 所有邻居都与当前值相反 → 孤立点
            if all(n != val for n in neighbors):
                return True
    return False


def min_distance_after_shift(pattern: np.ndarray, matrix: np.ndarray) -> int:
    """
    计算图案在所有循环平移下，与矩阵中每行的最小Hamming距离。

    这确保定位图案即使经过平移后也不会与任何码字混淆。
    平移包括水平和垂直方向的循环移位。

    Args:
        pattern: (64,) 定位图案
        matrix: (16, 64) 码字矩阵

    Returns:
        所有平移组合下的最小Hamming距离
    """
    m = pattern.reshape(8, 8)
    min_dist = 64

    for row_shift in range(8):
        for col_shift in range(8):
            # 循环平移
            shifted = np.roll(np.roll(m, row_shift, axis=0), col_shift, axis=1)
            shifted_flat = shifted.reshape(64)

            # 与矩阵中每行比较
            for i in range(len(matrix)):
                dist = int(np.sum(shifted_flat != matrix[i]))
                if dist < min_dist:
                    min_dist = dist
                    if min_dist == 0:
                        return 0  # 完全匹配，提前退出

    return min_dist


def generate_locator_pattern(
    min_distance: int = 22,
    max_transitions: int = 32,
    max_attempts: int = 200000,
    seed: int = 2024,
) -> np.ndarray:
    """
    生成满足约束的定位图案。

    策略：
    1. 用伪随机搜索生成候选
    2. 确保与所有16个码字的Hamming距离 > min_distance
    3. 额外约束：不具有左右对称（增加CNN可区分性）
    4. 额外约束：过渡次数 <= max_transitions（控制高频能量，抗压缩）
    5. 额外约束：无孤立点（单像素被相反值包围，压缩后必被抹平）
    6. 额外约束：循环平移后与码字的最小距离 > shift_distance（防混淆）

    Args:
        min_distance: 与所有码字的最小Hamming距离下限
        max_transitions: 最大允许的相邻过渡次数（越低频率越低）
        max_attempts: 最大搜索次数
        seed: 随机种子

    Returns:
        (64,) int8 数组，值为+1或-1
    """
    rng = np.random.RandomState(seed)

    best_pattern = None
    best_min_dist = 0
    shift_distance = max(12, min_distance - 8)  # 平移后的最小距离

    for attempt in range(max_attempts):
        # 生成随机候选（+1/-1均匀分布）
        candidate = rng.choice([-1, 1], size=64).astype(np.int8)

        # 约束1：平衡性（32个+1, 32个-1）
        plus_count = np.sum(candidate == 1)
        if plus_count != 32:
            continue

        # 约束2：无孤立点（压缩鲁棒性）
        if has_isolated_points(candidate):
            continue

        # 约束3：过渡次数（控制高频能量）
        trans = count_transitions(candidate)
        if trans > max_transitions:
            continue

        # 约束4：与所有码字的距离
        dist = min_hamming_to_matrix(candidate, FIX_FG_MATRIX)
        if dist < min_distance:
            continue

        # 约束5：非左右对称
        m = candidate.reshape(8, 8)
        if np.all(m == m[:, ::-1]):
            continue

        # 约束6：循环平移后与码字不相似
        shift_dist = min_distance_after_shift(candidate, FIX_FG_MATRIX)
        if shift_dist < shift_distance:
            continue

        # 找到满足所有约束的候选
        if dist > best_min_dist:
            best_min_dist = dist
            best_pattern = candidate.copy()
            print(f"  Attempt {attempt+1}: min_dist={dist}, shift_min={shift_dist}, transitions={trans}")

            # 如果距离已经足够好，提前停止
            if dist >= min_distance + 4:
                break

    if best_pattern is None:
        raise RuntimeError(
            f"Failed to generate pattern after {max_attempts} attempts. "
            f"Try reducing min_distance (currently {min_distance}) or "
            f"increasing max_transitions (currently {max_transitions})."
        )

    return best_pattern


def validate_pattern(pattern: np.ndarray) -> bool:
    """验证定位图案满足所有约束。"""
    m = pattern.reshape(8, 8)

    # 平衡性
    if np.sum(pattern == 1) != 32:
        print("FAIL: pattern is not balanced (not 32 +1 / 32 -1)")
        return False

    # 过渡次数
    trans = count_transitions(pattern)
    print(f"  Transition count: {trans} (limit: {32})")
    if trans > 32:
        print("WARN: high transition count (may affect compression robustness)")

    # 与每个码字的距离
    all_pass = True
    for i in range(16):
        dist = hamming_distance(pattern, FIX_FG_MATRIX[i])
        status = "OK" if dist >= 20 else "FAIL"
        if dist < 20:
            all_pass = False
        print(f"  CW{i:2d}: Hamming dist = {dist:2d} [{status}]")

    # 非对称性
    lr_sym = np.all(m == m[:, ::-1])
    tb_sym = np.all(m == m[::-1, :])
    print(f"  Left-right symmetric: {lr_sym}")
    print(f"  Top-bottom symmetric: {tb_sym}")

    return all_pass


def save_pattern(pattern: np.ndarray, output_path: str):
    """保存定位图案。"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, pattern)
    print(f"Saved locator pattern to {output_path}")

    # 同时保存可读的文本版本
    txt_path = output_path.replace('.npy', '.txt')
    np.savetxt(txt_path, pattern.reshape(8, 8), fmt='%+d', delimiter=' ')
    print(f"Saved readable version to {txt_path}")


def get_extended_fix_fg_matrix(locator_pattern: np.ndarray) -> np.ndarray:
    """返回追加了定位图案的17×64矩阵。"""
    return np.vstack([FIX_FG_MATRIX, locator_pattern.reshape(1, 64)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate unique locator pattern")
    parser.add_argument("--output", type=str, default="locator_pattern.npy",
                        help="Output .npy file path")
    parser.add_argument("--min-distance", type=int, default=22,
                        help="Minimum Hamming distance to all codewords")
    parser.add_argument("--max-transitions", type=int, default=32,
                        help="Max transitions in 8x8 grid (lower = more low-freq, more robust)")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=== Generating Locator Pattern ===")
    print(f"  min_distance={args.min_distance}, max_transitions={args.max_transitions}")
    pattern = generate_locator_pattern(
        min_distance=args.min_distance,
        max_transitions=args.max_transitions,
        seed=args.seed,
    )

    print(f"\nGenerated pattern (8×8):")
    print(pattern.reshape(8, 8))

    print(f"\n=== Validation ===")
    ok = validate_pattern(pattern)

    if ok:
        save_pattern(pattern, args.output)

        # 也保存扩展矩阵
        ext_matrix = get_extended_fix_fg_matrix(pattern)
        ext_path = args.output.replace('.npy', '_extended_matrix.npy')
        np.save(ext_path, ext_matrix)
        print(f"Saved extended 17×64 matrix to {ext_path}")
    else:
        print("\nWARNING: Pattern does not meet all constraints!")
