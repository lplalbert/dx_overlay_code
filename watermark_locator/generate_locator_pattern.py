"""
生成唯一定位图案（Locator Pattern）— 回字形

定位图案采用**回字形**（同心方环 / nested square rings）:
    外环(信号) + 间隙环(中性) + 内环(信号) + 核心(中性)

        Y Y Y Y Y Y Y Y
        Y W W W W W W Y
        Y W Y Y Y Y W Y
        Y W Y W W Y W Y
        Y W Y W W Y W Y
        Y W Y Y Y Y W Y
        Y W W W W W W Y
        Y Y Y Y Y Y Y Y

    Y = -1  黄色(信号), 条纹调制
    W = +1  白色(中性), 始终不变

回字形有强角点 + 独特自相关峰, 比伪随机图案更容易定位。
注意: 真"回"不是 32/32 平衡的 (信号 40 / 中性 24), 故不再约束平衡性。

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


def build_hui_pattern() -> np.ndarray:
    """
    构造确定性的回字形定位图案 (8x8, +1/-1)。

    四层同心方环, 交替 信号/中性/信号/中性:
      ring 0 (最外框)   = -1  信号   28 格
      ring 1 (间隙)     = +1  中性   20 格
      ring 2 (内框)     = -1  信号   12 格
      ring 3 (核心 2x2) = +1  中性    4 格

    即汉字"回"的字形: 外"囗"套内"口", 内口是空心的。
    信号 40 / 中性 24 — 不再要求 32/32 平衡 (真回本身就不平衡)。

    Returns:
        (64,) int8 数组，值为 +1(白/中性) 或 -1(黄/信号)
    """
    m = np.ones((8, 8), dtype=np.int8)          # 默认中性 +1
    # ring 0: 最外框
    m[0, :] = -1
    m[7, :] = -1
    m[:, 0] = -1
    m[:, 7] = -1
    # ring 2: 内框 (2..5 的边界)
    m[2, 2:6] = -1
    m[5, 2:6] = -1
    m[2:6, 2] = -1
    m[2:6, 5] = -1
    # ring 1 (1..6 除去 ring2) 与 ring 3 (核心 2x2) 保持 +1
    return m.reshape(64).astype(np.int8)


def generate_locator_pattern(
    min_distance: int = 16,
    max_transitions: int = 64,
    max_attempts: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """
    生成回字形定位图案 (确定性, 不再伪随机搜索)。

    回字形有强角点 + 独特自相关峰, 便于 CNN/匹配定位。
    与 16 个码字的 Hamming 距离由 validate_pattern 报告;
    回字形本身是中心对称的 (这是定位子的特性, 不是缺陷)。

    Args:
        min_distance / max_transitions / max_attempts / seed:
            兼容旧接口, 现已不参与搜索 (保留避免调用方报错)。

    Returns:
        (64,) int8 数组，值为 +1 或 -1
    """
    del min_distance, max_transitions, max_attempts, seed  # 兼容旧签名
    pattern = build_hui_pattern()
    print("  Built 回字形 (nested square rings) locator, deterministic")
    return pattern


def validate_pattern(pattern: np.ndarray) -> bool:
    """验证定位图案 (回字形结构 + 与码字的可区分性)。"""
    m = pattern.reshape(8, 8)

    # 取值合法性
    if not np.all(np.isin(pattern, (-1, 1))):
        print("FAIL: pattern must be +1 / -1 only")
        return False

    n_sig = int(np.sum(pattern == -1))
    n_neu = int(np.sum(pattern == 1))
    print(f"  Signal(-1) / Neutral(+1): {n_sig} / {n_neu}  (回字形期望 40 / 24)")
    if (n_sig, n_neu) != (40, 24):
        print("WARN: not the 回字形 40/24 split")

    # 回字形结构: 四层同心方环
    expect = build_hui_pattern()
    is_hui = bool(np.array_equal(pattern, expect))
    print(f"  Is 回字形 (nested rings): {is_hui}")

    # 过渡次数 / 孤立点 (报告用; 回字形结构低频, 不作硬约束)
    trans = count_transitions(pattern)
    print(f"  Transition count: {trans}")
    print(f"  Isolated points: {has_isolated_points(pattern)}")

    # 与每个码字的距离 (回字形与伪随机码字应有足够距离)
    all_pass = is_hui
    dists = []
    for i in range(16):
        dist = hamming_distance(pattern, FIX_FG_MATRIX[i])
        dists.append(dist)
        status = "OK" if dist >= 16 else "FAIL"
        if dist < 16:
            all_pass = False
        print(f"  CW{i:2d}: Hamming dist = {dist:2d} [{status}]")
    print(f"  min Hamming to codewords: {min(dists)}")

    shift_dist = min_distance_after_shift(pattern, FIX_FG_MATRIX)
    print(f"  min Hamming after shift:  {shift_dist}")

    # 对称性 (回字形是中心对称的 — 定位子特性)
    lr_sym = bool(np.all(m == m[:, ::-1]))
    tb_sym = bool(np.all(m == m[::-1, :]))
    print(f"  Left-right symmetric: {lr_sym}  (回字形预期 True)")
    print(f"  Top-bottom symmetric: {tb_sym}  (回字形预期 True)")

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
    parser = argparse.ArgumentParser(description="Generate 回字形 locator pattern")
    parser.add_argument("--output", type=str, default="locator_pattern.npy",
                        help="Output .npy file path")
    args = parser.parse_args()

    print("=== Generating Locator Pattern (回字形) ===")
    pattern = generate_locator_pattern()

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
