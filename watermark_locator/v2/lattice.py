#!/usr/bin/env python3
"""v2 格点几何 + 码字符号解码 + RS(15,5) 解码。

检测网络只负责"找出 96 个码字块"。**精密测量全部在这里做**，纯确定性算法，
不学习、不改网络 —— 见 DESIGN.md §5（框宽当不了尺子，间距要从框中心的格点读）。

三层：

1. ``fit_lattice`` / ``ransac_lattice``
       框中心 → (s, theta, tx, ty)。s 给出 pitch = (160·s, 135·s)。
       一维 CRLB: σ_a = σ_c·√12/(M·√N)；N=96, M=12, σ_c=1px → 0.02%。
2. ``decode_codeword``
       一个码字的 8×8 格 → 对 16 张已知掩码做匹配滤波 → 符号 ID。
3. ``decode_watermark``
       96 个符号 → 按序列槽位投票 → 16 符号 → RS(15,5) → watermark ID。

掩码约定与 ``generate_yellow_white_template.CODEWORD_CELL_MASKS`` 一致：
bit=0 → 黄(信号, B=0)，bit=1 → 白(中性, B=255)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── 模板几何 (s=1, 即 1920×1080 屏幕) ──────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080
BLOCK_ROWS, BLOCK_COLS = 4, 6
CODEWORD_W, CODEWORD_H = 160, 135          # 码字 = 1/4 个 M 块，最小检测单元
CELL_COLS, CELL_ROWS = 8, 8                # 一个码字 8×8 格
SUB_ROWS, SUB_COLS = BLOCK_ROWS * 2, BLOCK_COLS * 2   # 8×12 = 96 个码字

MESSAGE_ORDER = (0, 2, 1, 3)
LOCATOR_SLOT = 15

# ── 解码拒识门 ────────────────────────────────────────────────────────
# 一个槽位有 6 份空间冗余副本 (96 码字 / 16 槽)。相干累加的胜者可能被少数
# 高分副本拉走，所以除了累加分数的 top1-top2 分差，还要数"几份副本自己也
# 认这个符号"。dd 在 240 样本上按硬投票的 slot_ok 扫操作点：<=2 全是解错，
# >=3 解错归零且 clean 锚点零损失。这里取它的等价物。
MIN_AGREE = 3
# 定位槽 (slot 15) 的证据量下限 —— 不能只要求 n_obs > 0，一个噪声格就满足。
# 每槽正常有 6 份副本，取半数。
MARKER_MIN_OBS = 3

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


# ════════════════════════════════════════════════════════════════════
# 1. 格点拟合
# ════════════════════════════════════════════════════════════════════

@dataclass
class LatticeFit:
    """格点拟合结果。pitch 是 s=1 时 (160,135) 的 s 倍。"""
    s: float
    theta_deg: float
    tx: float
    ty: float
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    rms: float = float('nan')

    @property
    def pitch(self) -> Tuple[float, float]:
        return (CODEWORD_W * self.s, CODEWORD_H * self.s)

    @property
    def symbol_pitch(self) -> Tuple[float, float]:
        """同序列槽位（逐像素相同）的 6 份拷贝之间的间距 = 4 × pitch。"""
        return (4 * CODEWORD_W * self.s, 4 * CODEWORD_H * self.s)

    def nearest(self, cx: float, cy: float) -> Tuple[int, int, float, float]:
        """(cx,cy) 最近的格点 → (col, row, 残差x, 残差y)。"""
        th = math.radians(self.theta_deg)
        c, sn = math.cos(th), math.sin(th)
        dx, dy = cx - self.tx, cy - self.ty
        u = (c * dx + sn * dy) / (CODEWORD_W * self.s)      # 列 (浮点)
        v = (-sn * dx + c * dy) / (CODEWORD_H * self.s)     # 行 (浮点)
        j, k = int(round(u)), int(round(v))
        rx = dx - (c * j * CODEWORD_W * self.s - sn * k * CODEWORD_H * self.s)
        ry = dy - (sn * j * CODEWORD_W * self.s + c * k * CODEWORD_H * self.s)
        return j, k, rx, ry

    def site_xy(self, j: float, k: float) -> Tuple[float, float]:
        """格点 (col,row) 的像素坐标（:meth:`nearest` 的逆）。"""
        th = math.radians(self.theta_deg)
        c, sn = math.cos(th), math.sin(th)
        x = self.tx + c * j * CODEWORD_W * self.s - sn * k * CODEWORD_H * self.s
        y = self.ty + sn * j * CODEWORD_W * self.s + c * k * CODEWORD_H * self.s
        return x, y

    def cell_corners(self, j: float, k: float) -> np.ndarray:
        """格点 (col,row) 对应码字区域的四角 (4,2)，左上起顺时针。

        **格点是码字的中心**，不是角点 —— ``fit_lattice`` 拟合的是框中心，
        所以码字区域 = 中心 ± (CODEWORD_W·s/2, CODEWORD_H·s/2)，再转 theta。
        """
        cx, cy = self.site_xy(j, k)
        hw = CODEWORD_W * self.s / 2.0
        hh = CODEWORD_H * self.s / 2.0
        th = math.radians(self.theta_deg)
        c, sn = math.cos(th), math.sin(th)
        out = np.empty((4, 2), dtype=np.float64)
        for i, (dx, dy) in enumerate(((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))):
            out[i, 0] = cx + c * dx - sn * dy
            out[i, 1] = cy + sn * dx + c * dy
        return out

    def cell_rect(self, j: float, k: float, pad: float = 0.0) -> Tuple[int, int, int, int]:
        """格点 (col,row) 码字区域的**轴对齐外接框** → 整数 (x0,y0,x1,y1)。

        theta=0 时正好是码字的像素矩形；有倾角时是它的外接框，会略大。
        """
        c = self.cell_corners(j, k)
        x0 = int(math.floor(c[:, 0].min() - pad))
        y0 = int(math.floor(c[:, 1].min() - pad))
        x1 = int(math.ceil(c[:, 0].max() + pad))
        y1 = int(math.ceil(c[:, 1].max() + pad))
        return x0, y0, x1, y1

    def cell_window(self, j: float, k: float) -> Tuple[int, int, int, int]:
        """码字的**固定尺寸**像素窗 (x0,y0,x1,y1)，尺寸 = round(160s)×round(135s)。

        与 :meth:`cell_rect` 的区别：这里尺寸是整数定值，不随浮点原点抖动，
        保证切出来的 patch 恒等于一个码字的像素矩形（解码按比例切 8×8 格，
        patch 多一个像素格边界就歪了）。
        """
        cw = int(round(CODEWORD_W * self.s))
        ch = int(round(CODEWORD_H * self.s))
        cx, cy = self.site_xy(j, k)
        x0 = int(round(cx - cw / 2.0))
        y0 = int(round(cy - ch / 2.0))
        return x0, y0, x0 + cw, y0 + ch


def _fit_axis(t: np.ndarray, u: np.ndarray) -> Tuple[float, float]:
    """最小二乘 u = a·t + b，返回 (a, b)。"""
    tc = t - t.mean()
    uc = u - u.mean()
    denom = float((tc * tc).sum())
    if denom <= 0:
        return 0.0, float(u.mean())
    a = float((tc * uc).sum() / denom)
    return a, float(u.mean() - a * t.mean())


def _mode_gap(d: np.ndarray, rel: float = 0.02) -> float:
    """差值的**众数** = 最大的一簇（簇内容差 ≤ rel·值）。

    为什么不用中位数：误报点落进两条真列之间会把一条真列间距劈成两个碎片，
    碎片彼此不同、各出现一次，真列间距却重复 (列数-1) 次。中位数把这些碎片
    一起数进去就被带走 —— 实测 12 列 + 8 误报时中位数给 209.8px（真值 216，
    偏 3%），**216 一次都没进间距候选表**，拟合只剩半间距可选（真点全落在
    隔线上，内点一分不少），这就是 s_true=1.35 那次 25% 误差的成因。
    众数只认"反复出现的那个间距"，碎片摊成底噪，动不了它。
    """
    v = np.sort(np.asarray(d, dtype=np.float64))
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 0.0
    if v.size == 1:
        return float(v[0])
    best_n, best_s, cur_n, cur_s = 0, 0.0, 1, float(v[0])
    for a, b in zip(v[:-1], v[1:]):
        if b - a <= rel * max(a, 1.0):
            cur_n += 1
            cur_s += float(b)
        else:
            if cur_n > best_n:
                best_n, best_s = cur_n, cur_s
            cur_n, cur_s = 1, float(b)
    if cur_n > best_n:
        best_n, best_s = cur_n, cur_s
    return best_s / best_n


def _cluster_pitch(coords: np.ndarray, thr: float) -> float:
    """按 thr 切簇，返回簇心间距的**众数**。切不出 ≥2 簇则返回 0。"""
    v = np.sort(coords)
    if v.size < 2:
        return 0.0
    d = np.diff(v)
    cuts = np.where(d > thr)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [v.size]])
    centers = np.array([v[a:b].mean() for a, b in zip(starts, ends)])
    if centers.size < 2:
        return 0.0
    return _mode_gap(np.diff(centers))


def _estimate_pitch(coords: np.ndarray, hint: Optional[float] = None) -> float:
    """稳健估一维间距：**先聚类，再用簇心间距**。

    两个坑都不能踩：

    1. 直接取 ``np.diff(sorted).median()`` —— 簇内 1–2px 的小差值会淹没列间距；
    2. 直接取"大于阈值的相邻差"的中位数 —— 那是**簇边缘到边缘**的距离，
       等于间距减去簇内散布，会系统性偏小（实测 216 → 211.9，偏 4px）。

    簇心间距才是无偏的。
    """
    v = np.sort(np.asarray(coords, dtype=np.float64))
    if v.size < 2:
        return float(hint) if hint else float(CODEWORD_W)
    d = np.diff(v)
    if d.size == 0:
        return float(hint) if hint else float(CODEWORD_W)
    if hint and hint > 0:
        thr = 0.3 * float(hint)          # 簇内散布 ≈ 8px ≪ 30% hint
    else:
        thr = max(3.0, 0.5 * float(np.percentile(d, 90)))
    p = _cluster_pitch(v, thr)
    return p if p > 0 else (float(hint) if hint else float(v[-1] - v[0]) or float(CODEWORD_W))


def _phase_origin(coords: np.ndarray, pitch: float, tol: float = 3.0) -> float:
    """格点原点：**按内点数扫相位**，取内点最多的那个。

    以前用圆均值（``coord mod pitch`` 的角度平均）估计原点，那不是稳健估计：
    20 个同相位的真点被 30 个均匀误报一搅，合向量幅值只剩 0.4，相位能偏 7px
    > tol=3 —— ``_fit_once`` 起步就全盘外点、``inl.sum() < 3`` 直接早退，
    拟合死在 0 内点上，连真间距候选都来不及精修（实测 20 真 + 30 误报时
    四个候选**全部** 0 内点）。

    这里改成一维确定性相位搜索：候选相位 = 各点自身的相位（把该点对齐到
    格点），逐个数内点取最大。这正是我们要的那个量，且对误报免疫——
    30 个均匀点摊到相位上是噪声，真点在单一相位上堆出一个尖峰。
    N=50 时 2500 次比较，可以忽略。
    """
    if pitch <= 0:
        return 0.0
    r = np.mod(coords, pitch)
    if r.size == 0:
        return 0.0
    half = pitch / 2.0
    best_t, best_n = float(r[0]), -1
    for t in r:
        d = np.abs(np.mod(r - t + half, pitch) - half)
        n = int((d <= tol).sum())
        if n > best_n:
            best_n, best_t = n, float(t)
    # 用最佳相位下的内点做圆均值精修，把起点再往真值上推一把
    d = np.abs(np.mod(r - best_t + half, pitch) - half)
    sel = r[d <= max(tol, 1e-9)]
    if sel.size:
        ang = 2.0 * np.pi * sel / pitch
        m = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
        best_t = (m % (2.0 * np.pi)) * pitch / (2.0 * np.pi)
    return best_t


def _inlier_mask(pts: np.ndarray, px: float, py: float,
                 tx: float, ty: float, tol: float) -> np.ndarray:
    """(px,py,tx,ty) 下的内点掩码，与 :func:`_fit_once` 的归格标同一套公式。"""
    if not (px > 0 and py > 0):
        return np.zeros(pts.shape[0], dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    j = np.round((x - tx) / px)
    k = np.round((y - ty) / py)
    rx = x - (tx + j * px)
    ry = y - (ty + k * py)
    return (np.abs(rx) <= tol) & (np.abs(ry) <= tol)


def _fit_once(pts: np.ndarray, px: float, py: float,
              tx: float, ty: float, tol: float):
    """一次"归格标 → 判内点 → 回归"。返回**与掩码配套**的 (px, py, tx, ty, inlier_mask)。

    两个必须防的退化：

    1. **掩码与参数不配套**。以前把"精修前"那一轮的 ``inl`` 跟"精修后"的
       ``(px,py,tx,ty)`` 一起返回，退化解下两者差出十万八千里 —— 掩码报 90 内点、
       ``rms`` 却 57px，评分照着这个假掩码把退化解排到第一。现在精修完**重算**掩码。
    2. **px → 0 的自激**。内点全落在同一条格线上时 ``_fit_axis`` 分母为 0、回出
       px=0；下一轮 ``max(px, 1e-9)`` 让每一点都不偏不倚"落在格线上"（残差恒 ≈ 0），
       全体变内点，再回归出一个莫名其妙的间距，自激到收敛。精修步必须拒绝非正间距
       （以及 < 2·tol 的间距 —— 那种格点密到每点都在线上，共识无从谈起）。
    """
    x, y = pts[:, 0], pts[:, 1]
    inl = _inlier_mask(pts, px, py, tx, ty, tol)
    if int(inl.sum()) < 3:
        return px, py, tx, ty, inl
    j = np.round((x - tx) / px)
    k = np.round((y - ty) / py)
    px_new, tx_new = _fit_axis(j[inl], x[inl])
    py_new, ty_new = _fit_axis(k[inl], y[inl])
    if not (px_new > 2.0 * tol and py_new > 2.0 * tol):
        return px, py, tx, ty, inl
    inl_new = _inlier_mask(pts, px_new, py_new, tx_new, ty_new, tol)
    return px_new, py_new, tx_new, ty_new, inl_new


def _pitch_candidates(coords: np.ndarray, hint: Optional[float]) -> List[float]:
    """一维间距候选。

    **必须先定间距再定原点**：hint 带 5% 误差时，相位在 12 个周期上累积漂移
    0.6 个 pitch，直接拿 hint 去归格标会全盘皆输。所以用"簇心间距"（与原点无关），
    hint 只作兜底。

    **hint 绝不能成为唯一入口**：hint 错（误报框更多、尺度被小框拖走）时，
    只按 hint 生成候选会把真间距整个锁死在候选表外面，拟合只剩退化解可选
    —— 这正是 20 真 + 30 误报那次"间距 32.8px、内点 1"的成因。所以不管有没有
    hint，都把无 hint 的阈值扫掠并进来，靠内点数裁决谁对。
    """
    out: List[float] = []
    v = np.sort(np.asarray(coords, dtype=np.float64))
    if hint and hint > 0:
        p = _cluster_pitch(v, 0.3 * float(hint))
        if p > 0:
            out.append(p)
        out.append(float(hint))
    d = np.diff(v) if v.size > 1 else np.array([1.0])
    base = max(3.0, 0.5 * float(np.percentile(d, 90)))
    if hint and hint > 0:
        base = min(base, float(hint))      # 别扫到 hint 的好几倍以外去
    for k in (0.5, 1.0, 2.0, 4.0):
        p = _cluster_pitch(v, base * k)
        if p > 0 and all(abs(p - q) / p > 0.05 for q in out):
            out.append(p)
    # 去重、保序
    uniq: List[float] = []
    for p in out:
        if p > 0 and all(abs(p - q) / q > 0.02 for q in uniq):
            uniq.append(float(p))
    return uniq or [float(CODEWORD_W)]


def _line_occupancy(idx: np.ndarray) -> float:
    """指标号的格线占用率 = 用了几条线 / 跨度内有几条线 ∈ (0, 1]。

    = 1 表示点把跨度内的格线都占满了（间距对）；= 1/m 表示点只占每第 m 条线
    —— 拟合出的间距是真值的 **1/m**，多出来的 (m-1)/m 条线一个点都没解释。

    这是拆 1/m 倍间距的量，而且**经得起误报污染**：1/3 倍间距再落进 1 个
    误报补到中间线上时，最小步长从 3 塌成 1（完全失效），占用率却只从
    0.33 爬到 0.37 —— 多一条线分母是 35，动不了。实测 36 真 + 20 误报那次
    就是被这**一个**误报骗过的：内点 37 vs 36、最小步长 1×1，而占用率是
    0.37×1.0 对 1.0×1.0，一眼分开。

    为什么是逐轴各算一次再相乘，而不是二维"内点数/包围盒格位数"：二维那版
    在稀疏观测下真解只有 20/96 = 0.21，方向是**反的**（半间距反而把包围盒缩
    小到 6×4 拿 0.83）。逐轴的占用率与"检出了多少个码字"无关，稀疏观测下
    真解照样 ≈ 1。
    """
    u = np.unique(idx)
    if u.size == 0:
        return 0.0
    span = int(u.max() - u.min()) + 1
    return float(u.size) / float(max(span, 1))


_SCALE_LO = 0.35        # 拟合间距相对 hint 的下界
_SCALE_HI = 2.8         # 上界


def _scale_ok(p: float, h: Optional[float]) -> bool:
    """拟合间距 ``p`` 是否在 hint ``h`` 的可信量程内。``h is None`` 时不设限。"""
    if not h or h <= 0:
        return True
    return (_SCALE_LO * h) <= p <= (_SCALE_HI * h)


def _fit_axis_aligned(pts: np.ndarray, hx: Optional[float], hy: Optional[float],
                      tol: float, rounds: int):
    """theta=0 的格点拟合主体，返回 (px, py, tx, ty, inl, score)。"""
    x, y = pts[:, 0], pts[:, 1]
    best = None
    best_score = -1e18
    best_any = None
    best_any_score = -1e18
    for cpx in _pitch_candidates(x, hx):
        for cpy in _pitch_candidates(y, hy):
            if cpx <= 0 or cpy <= 0:
                continue
            cx, cy = cpx, cpy
            ctx = _phase_origin(x, cx, tol)
            cty = _phase_origin(y, cy, tol)
            for _ in range(rounds):
                cx, cy, ctx, cty, inl = _fit_once(pts, cx, cy, ctx, cty, tol)
            # 评分 = 内点数 × 格线占用率。
            #
            # 内点数必须主导：稀疏观测（96 个码字只检出 20 个）下真解照样赢，
            # 误报堆不出共识。但内点数单独不够 —— 1/m 倍间距下真点**照样落在
            # 格线上**，内点数一分不少甚至更多（误报还会补到中间线上白送一个），
            # 由格线占用率拆开（见 _line_occupancy）：真解占用率 ≈ 1，1/m 倍
            # 间距 ≈ 1/m，乘上去才是"解释掉了多少真格线"。
            cnt = int(inl.sum())
            ji = np.round((pts[inl, 0] - ctx) / max(cx, 1e-9))
            ki = np.round((pts[inl, 1] - cty) / max(cy, 1e-9))
            occ = _line_occupancy(ji) * _line_occupancy(ki)
            score = cnt * occ
            if score > best_any_score:
                best_any_score, best_any = score, (cx, cy, ctx, cty, inl, score)
            # **量程门槛**（不是罚分）：框宽/框高是尺度的测量值，不是可有可无的
            # 提示。转角搜索里有一个致命混淆 —— 160×135 的格点整体转 φ 角后，
            # 它的 x 投影落在间距 135·sin φ 的细格线上，**每一个点都严丝合缝**，
            # 内点数比真解还多（实测 φ≈9.7° 时细间距 22.5px、89 内点，压过真解
            # 的 78）。这是旋转格点的投影混淆，光靠数据分辨不了，必须用尺度先验。
            # 而 ±6° 的搜索范围内这个混淆间距最大只有 135·sin6° ≈ 14px，
            # 远在 0.35·hint 之下 —— 量程一卡，整族混淆一起出局。
            # （罚分形式不行：罚轻了拦不住 89 vs 78，罚重了又会在 hint 偏小
            # 时把真解一起拦掉。门槛只要求"在同一个量级"，对 hint 的精度不敏感。）
            if not _scale_ok(cx, hx) or not _scale_ok(cy, hy):
                continue
            if score > best_score:
                best_score = score
                best = (cx, cy, ctx, cty, inl, score)
    return best if best is not None else best_any


def fit_lattice(centers: np.ndarray,
                pitch_hint: Optional[Tuple[float, float]] = None,
                tol: float = 3.0, rounds: int = 5,
                reject_outliers: bool = True,
                theta_span: float = 0.0, theta_step: float = 0.5) -> LatticeFit:
    """格点最小二乘：centers = (N,2) 的框中心。

    确定性迭代，不做随机 RANSAC —— 两采样定出的间距可能是真间距的整数倍，
    在规则格上不可靠。这里用**残差内点判定**做共识：真码字互相印证，
    孤立的图标误报落不到格点上，被剔掉。这就是 DESIGN.md §7 的格点共识。

    ``pitch_hint`` = (px0, py0) 粗间距初值，一般传框宽/框高中位数。
    框宽当不了精密尺子（§5），当初值正好 —— 会自动被迭代精修掉。

    ``theta_span > 0`` 时在 ±theta_span 度上做**确定性网格搜索**（不做随机采样）：
    tile_crop 的 ±5° 旋转、屏摄的残余单应都会让格点整体转一个小角，
    theta=0 假设下 5° 在 1920px 上漂 167px ≈ 1 个 pitch，会全盘皆输。
    每个候选角度先把点绕质心转回 0°，再跑轴对齐拟合，按同一评分择优。
    """
    pts = np.asarray(centers, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("need at least 2 centers")
    n = pts.shape[0]

    hx = float(pitch_hint[0]) if pitch_hint else None
    hy = float(pitch_hint[1]) if pitch_hint else None

    if theta_span and theta_span > 0:
        nstep = int(round(theta_span / max(theta_step, 1e-6)))
        thetas = [i * theta_step for i in range(-nstep, nstep + 1)]
    else:
        thetas = [0.0]

    cx0, cy0 = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    best = None
    best_score = -1e18
    best_theta = 0.0
    for th in thetas:
        if th == 0.0:
            p = pts
        else:
            r = math.radians(th)
            c, sn = math.cos(r), math.sin(r)
            dx, dy = pts[:, 0] - cx0, pts[:, 1] - cy0
            # 转回轴对齐: 逆时针转 -theta
            p = np.stack([cx0 + c * dx + sn * dy, cy0 - sn * dx + c * dy], axis=1)
        res = _fit_axis_aligned(p, hx, hy, tol, rounds)
        if res is None:
            continue
        _, _, _, _, _, score = res
        if score > best_score:
            best_score = score
            best = res
            best_theta = float(th)

    px, py, tx, ty, inl, _ = best
    if not reject_outliers:
        inl = np.ones(n, dtype=bool)

    # 把轴对齐解转回原坐标系: 原点也要跟着转
    if best_theta != 0.0:
        r = math.radians(best_theta)
        c, sn = math.cos(r), math.sin(r)
        # 轴对齐系里原点是 (tx,ty)；转回 = 绕同一质心顺时针转 best_theta
        dtx, dty = tx - cx0, ty - cy0
        tx = cx0 + c * dtx - sn * dty
        ty = cy0 + sn * dtx + c * dty

    s = ((px / CODEWORD_W) + (py / CODEWORD_H)) / 2.0
    fit = LatticeFit(s=float(s), theta_deg=float(best_theta), tx=float(tx), ty=float(ty),
                     inliers=np.asarray(inl, dtype=bool))
    resid = []
    for (cx_, cy_) in pts:
        _, _, rx, ry = fit.nearest(float(cx_), float(cy_))
        resid.append(math.hypot(rx, ry))
    fit.rms = float(np.sqrt(np.mean(np.square(resid)))) if resid else float('nan')
    return fit


def ransac_lattice(centers: np.ndarray, tol: float = 3.0,
                   iters: int = 400, min_inliers: int = 8,
                   pitch_hint: Optional[Tuple[float, float]] = None,
                   rng: Optional[np.random.Generator] = None,
                   theta_span: float = 6.0) -> LatticeFit:
    """兼容入口：现在是确定性格点共识拟合（见 fit_lattice）。

    ``iters`` / ``rng`` 保留只为不破坏旧调用签名，不再使用随机采样。
    """
    del iters, rng, min_inliers
    return fit_lattice(centers, pitch_hint=pitch_hint, tol=tol, theta_span=theta_span)


def read_interval(fit: LatticeFit) -> dict:
    """从格点拟合读出"固定间距"。"""
    pw, ph = fit.pitch
    sw, sh = fit.symbol_pitch
    return {
        's': fit.s,
        'interval_px': [pw, ph],
        'symbol_interval_px': [sw, sh],
        'theta_deg': fit.theta_deg,
        'origin': [fit.tx, fit.ty],
        'n_inliers': int(fit.inliers.sum()) if fit.inliers.size else 0,
        'rms_px': fit.rms,
    }


# ════════════════════════════════════════════════════════════════════
# 2. 码字符号匹配滤波
# ════════════════════════════════════════════════════════════════════

def _mask_to_sign(symbol: int) -> np.ndarray:
    """8×8 符号模板 → ±1：bit=0(黄,信号) → +1，bit=1(白,中性) → -1。

    与 ``generate_yellow_white_template.build_rgb_templates`` 逐位一致：
    ``cell = row * 8 + col``，``bit = (mask >> cell) & 1`` —— **bit0 是左上角**。
    """
    m = CODEWORD_CELL_MASKS[symbol]
    bits = np.array([(m >> i) & 1 for i in range(64)], dtype=np.float64)
    return (1.0 - 2.0 * bits).reshape(CELL_ROWS, CELL_COLS)


_SIGN_TEMPLATES = np.stack([_mask_to_sign(s) for s in range(16)])   # (16,8,8)
# 信号是**单极**的：黄格写 B=0 → 该像素变暗；白格 alpha_eff=0 → 一个像素都不改。
# 所以参考掩码是 0/1（黄=1），不是 ±1。±1 在"排序"意义上等价（T_s 与 sign_s 仿射），
# 但一旦做归一化相关就不等价了 —— ‖T_s‖ 随黄格数变化，‖sign_s‖ 恒定。
_YELLOW_TEMPLATES = (_SIGN_TEMPLATES > 0).astype(np.float64)        # (16,8,8)


def codeword_cell_means(patch_bgr: np.ndarray) -> np.ndarray:
    """取一个码字区域的 8×8 格均值（B 通道，信号只写在 B）。

    patch_bgr: (H,W,3) 或 (H,W)；必须是**原生像素密度**的码字区域。
    格边界用 round-to-even，与模板生成器的 rounded_boundary 一致。
    """
    if patch_bgr.ndim == 3:
        plane = patch_bgr[:, :, 0].astype(np.float64)     # B 通道
    else:
        plane = patch_bgr.astype(np.float64)
    h, w = plane.shape
    out = np.zeros((CELL_ROWS, CELL_COLS), dtype=np.float64)
    for r in range(CELL_ROWS):
        top = int(round(h * r / CELL_ROWS))
        bot = int(round(h * (r + 1) / CELL_ROWS))
        for c in range(CELL_COLS):
            left = int(round(w * c / CELL_COLS))
            right = int(round(w * (c + 1) / CELL_COLS))
            out[r, c] = plane[top:bot, left:right].mean() if bot > top and right > left else np.nan
    return out


STRIPE_PERIOD = 4
STRIPE_DUTY = 2          # phase < 2 写入 → 占比 25%


def stripe_phase_of(x: float, y: float, offset: int = 0) -> int:
    """图像像素 (x,y) 处的条纹相位，offset = (dx+dy) % 4 是取景偏移。

    模板渲染用 ``coordinate = xx + yy``（45°），相位只依赖 x+y 这**一个**
    自由度，所以全局未知量是 1 个整数 φ ∈ {0,1,2,3}，不是两个。
    """
    return int(x + y + offset) % STRIPE_PERIOD


def _stripe_keep(h: int, w: int, phase: int) -> np.ndarray:
    """patch 局部坐标里的条纹掩码。``phase`` = stripe_phase_of(x0, y0, offset)。"""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.fmod(xx + yy + float(phase), float(STRIPE_PERIOD)) < float(STRIPE_DUTY)


def codeword_stripe_contrast(patch_bgr: np.ndarray, stripe_phase: int = 0) -> np.ndarray:
    """一个码字 → 8×8 格"黄格证据" h（越大越像该格是黄的）。

    模板的真实写法（``build_rgb_templates``）是：
      白格 → B=255 全像素，alpha_eff=0，**一个像素都不改**；
      黄格 → 只有条纹像素（25%）写 B=0，alpha_eff=0.032 → 该像素变暗 3.2%·自身值；
             同格的非条纹像素（75%）也**一个像素都不改**。

    所以同一格内的非条纹像素就是**无调制的载体真值**，比任何高通/减平面都更准
    —— 载体纹理在 20×17 px 的格内被完全采样掉。对比度定义为

        h = mean(非条纹) - mean(条纹)

    黄格 h ≈ +0.032·C > 0，白格 h ≈ 0。与 16 张 ±1 掩码相关就是匹配滤波。
    （推导：噪声模型 d = α(T−C) 下，相关 T_s 与相关 sign_s 排序完全相同，
    因为 T_s = 127.5·(1−sign_s) 是仿射关系 —— 所以 ±1 掩码已经是最优形式。）
    """
    plane = patch_bgr[:, :, 0].astype(np.float64) if patch_bgr.ndim == 3 \
        else patch_bgr.astype(np.float64)
    h, w = plane.shape
    keep = _stripe_keep(h, w, stripe_phase)
    out = np.zeros((CELL_ROWS, CELL_COLS), dtype=np.float64)
    for r in range(CELL_ROWS):
        top = int(round(h * r / CELL_ROWS))
        bot = int(round(h * (r + 1) / CELL_ROWS))
        for c in range(CELL_COLS):
            left = int(round(w * c / CELL_COLS))
            right = int(round(w * (c + 1) / CELL_COLS))
            if bot <= top or right <= left:
                continue
            m = keep[top:bot, left:right]
            p = plane[top:bot, left:right]
            n_on = int(m.sum())
            n_off = m.size - n_on
            if n_on == 0 or n_off == 0:
                continue
            out[r, c] = (p[~m].mean() - p[m].mean()) * n_on
    return out


def _cell_edges(n: int) -> List[int]:
    return [int(round(n * k / CELL_ROWS)) for k in range(CELL_ROWS + 1)]


def codeword_scores(patch_bgr: np.ndarray, stripe_phase: int = 0) -> np.ndarray:
    """一个码字 → 16 个匹配滤波分（越大越像该符号）。

    **平面是 G−B，不是 B。** 信号契约（``alpha_blend_watermark``）：黄格
    (0,255,255) 的 ``dynamicMask=1``，Δ = 0.032·(T−C)；白格 mask=0，完全不变。
    于是

        Δ_B   = −0.032·C_B                  → 载体依赖，**暗载体上归零**
        Δ(G−B) = +0.032·(255−C_G+C_B)       → 中性灰恒 **8.16**

    实测（拍前 9 张，只换平面、其余不动）：B 平面 **0/9**，G−B **9/9**。
    B 平面在拍前能过只是信号太强盖住了建模错误。

    **相关是零均值归一化的（NCC），不是裸相关。** 裸相关会被载体在 4px 条纹频率上的
    纹理能量灌满 —— 实测随机噪声的分差中位 11509，比真水印实拍（459~2976）还大。
    NCC 把这层能量归一化掉。

    参考是 ``(条纹 & 黄格)`` 的 **0/1 单极**掩码（见 ``_YELLOW_TEMPLATES``）。
    解析化：设格 i 的 ``n_i = keep 上像素数``、``S_i = o 在 keep 上的和``，符号 s
    的黄格集 Y_s，则
        n = Σ_{i∈Y} n_i ,  S = Σ_{i∈Y} S_i
        Σ(o−o̅)(ref−r̅) = S − o̅·n ,  Σ(ref−r̅)² = n − n²/(h·w)
    16 个符号各 O(64)，不再逐符号建参考图。
    """
    if patch_bgr.ndim == 3:
        o = (patch_bgr[:, :, 1].astype(np.float64)
             - patch_bgr[:, :, 0].astype(np.float64))
    else:
        o = patch_bgr.astype(np.float64)
    h, w = o.shape
    keep = _stripe_keep(h, w, stripe_phase)
    ys, xs = _cell_edges(h), _cell_edges(w)
    n_i = np.zeros(CELL_ROWS * CELL_COLS, dtype=np.float64)
    s_i = np.zeros_like(n_i)
    for r in range(CELL_ROWS):
        for c in range(CELL_COLS):
            y0, y1 = ys[r], ys[r + 1]
            x0, x1 = xs[c], xs[c + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            m = keep[y0:y1, x0:x1]
            p = o[y0:y1, x0:x1]
            n_i[r * CELL_COLS + c] = float(m.sum())
            s_i[r * CELL_COLS + c] = float(p[m].sum()) if m.any() else 0.0
    o_mean = float(o.mean())
    o_ss = float(((o - o_mean) ** 2).sum())
    hw = float(h * w)
    out = np.full(16, -1.0, dtype=np.float64)
    for s in range(16):
        y = _YELLOW_TEMPLATES[s].reshape(-1)
        n = float((y * n_i).sum())
        if n <= 1.0 or n >= hw - 1.0:
            continue
        S = float((y * s_i).sum())
        num = S - o_mean * n
        den = o_ss * (n - n * n / hw)
        out[s] = num / np.sqrt(den) if den > 0 else -1.0
    return out


def codeword_evidence(patch_bgr: np.ndarray, stripe_phase: int = 0) -> float:
    """相位判据：G−B 条纹对比的正部之和。与符号无关，真相位下最大。

    **相位判据和符号打分是两回事**：相位只问"条纹对没对齐"，用格内
    ``mean(非条纹) − mean(条纹)`` 的正部就是最直接的量，不必再套 NCC。
    实测把这里换成 NCC 正部之和会伤拍前（9/9 → 8/9）—— NCC 的逐符号参考范数
    把额外方差带进相位搜索。

    平面取 **G−B** 而不是 B：条纹门是靠把 B 写成 255 关掉 ``dynamicMask`` 的，
    但留下来的黄格信号在 G−B 上是常数 8.16，在 B 上是 −0.032·C_B（暗载体归零）。

    错相位下条纹/非条纹对调，黄格的对比会变号（≈ −⅓·α·C），正部塌到 0。
    """
    if patch_bgr.ndim == 3:
        plane = (patch_bgr[:, :, 1].astype(np.float64)
                 - patch_bgr[:, :, 0].astype(np.float64))
    else:
        plane = patch_bgr.astype(np.float64)
    h, w = plane.shape
    keep = _stripe_keep(h, w, stripe_phase)
    ys, xs = _cell_edges(h), _cell_edges(w)
    total = 0.0
    for r in range(CELL_ROWS):
        for c in range(CELL_COLS):
            y0, y1, x0, x1 = ys[r], ys[r + 1], xs[c], xs[c + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            m = keep[y0:y1, x0:x1]
            p = plane[y0:y1, x0:x1]
            if not m.any() or m.all():
                continue
            total += max(0.0, float(p[~m].mean() - p[m].mean()))
    return float(total)


def decode_codeword(patch_bgr: np.ndarray,
                    stripe_phase: int = 0) -> Tuple[int, float, float]:
    """一个码字 → (符号 ID, 最高分, 次高分)。分数差越大越可靠。

    这是确定性检测，不需要训练 —— 见 DESIGN.md §4.4。
    ``stripe_phase`` 见 :func:`stripe_phase_of`；不确定就交给
    :func:`decode_patches` 去搜 4 个相位。
    """
    scores = codeword_scores(patch_bgr, stripe_phase)
    order = np.argsort(scores)
    best = int(order[-1])
    return best, float(scores[best]), float(scores[order[-2]])


def decode_symbols(patches: Sequence[np.ndarray]) -> List[Tuple[int, float]]:
    """一批码字区域 → [(符号, 置信度)]。"""
    out = []
    for p in patches:
        sym, sc, _ = decode_codeword(p)
        out.append((sym, sc))
    return out


# ════════════════════════════════════════════════════════════════════
# 3. 96 符号 → 16 槽位投票 → RS(15,5) 解码
# ════════════════════════════════════════════════════════════════════

def subblock_symbol_grid(sequence: Sequence[int]) -> np.ndarray:
    """16 槽序列 → 8×12 的码字符号格（与 v2 排布一致）。"""
    seq = list(sequence)
    if len(seq) != 16:
        raise ValueError('sequence must have 16 symbols')
    grid = np.zeros((SUB_ROWS, SUB_COLS), dtype=np.int64)
    for r in range(SUB_ROWS):
        for c in range(SUB_COLS):
            br, bc = r // 2, c // 2
            m = (bc % 2) + 2 * (br % 2)          # v2 交替行
            pos = (r % 2) * 2 + (c % 2)
            grid[r, c] = int(seq[m * 4 + MESSAGE_ORDER[pos]])
    return grid


def slot_of(sub_row: int, sub_col: int) -> int:
    """码字位置 (sub_row, sub_col) 对应的 16 槽下标。"""
    br, bc = sub_row // 2, sub_col // 2
    m = (bc % 2) + 2 * (br % 2)
    pos = (sub_row % 2) * 2 + (sub_col % 2)
    return m * 4 + MESSAGE_ORDER[pos]


def _vote(assignments: Sequence[Tuple[int, int, float]]) -> Tuple[List[int], List[int], float, List[float]]:
    """按槽位多数投票。assignments = [(slot, symbol, score), ...]。

    返回 (seq16, n_obs16, 总分, gap16)。空槽位填 -1 并在 n_obs 里记 0，
    **不擅自填 0** —— 空槽不是"观测到符号 0"。

    ``gap16`` = 头名票数 − 次名票数；**平票 gap=0 表示零证据**，
    与"观测到符号 0"是两回事，调用方据此拒识。
    """
    votes: List[List[Tuple[int, float]]] = [[] for _ in range(16)]
    for sl, sym, sc in assignments:
        if 0 <= int(sl) < 16:
            votes[int(sl)].append((int(sym), float(sc)))
    seq: List[int] = []
    n_obs: List[int] = []
    gaps: List[float] = []
    total = 0.0
    for v in votes:
        n_obs.append(len(v))
        if not v:
            seq.append(-1)
            gaps.append(0.0)
            continue
        syms = np.array([s for s, _ in v])
        cnt = np.bincount(np.clip(syms, 0, 15), minlength=16)
        win = int(np.argmax(cnt))
        order = np.argsort(-cnt)
        seq.append(win)
        gaps.append(float(cnt[order[0]] - cnt[order[1]]) if len(order) > 1
                    else float(cnt[order[0]]))
        # 只累获胜那一方的匹配置信度 —— 错票不该给分
        total += float(sum(sc for s, sc in v if s == win))
    return seq, n_obs, total, gaps


def vote_sequence(symbols: Sequence[int]) -> List[int]:
    """96 个码字符号 → 16 槽位多数投票（行优先填满 8×12）。

    每个槽位在全屏有 6 份**逐像素相同**的拷贝（lag 640×540），
    所以投票就是把相干累积用到极致。
    """
    assignments = []
    idx = 0
    for r in range(SUB_ROWS):
        for c in range(SUB_COLS):
            if idx >= len(symbols):
                break
            assignments.append((slot_of(r, c), int(symbols[idx]), 1.0))
            idx += 1
    seq, _, _, _ = _vote(assignments)
    return [0 if s < 0 else s for s in seq]


def align_and_decode(sites: Sequence[Tuple[int, int, int, float]]) -> dict:
    """格位+符号 → 16 槽序列 → watermark ID，自动消解模板原点。

    ``sites`` = ``[(col, row, symbol, score), ...]``，(col,row) 来自
    ``LatticeFit.nearest``。

    **格点只定出相对原点**：``slot_of`` 在 row/col 上周期都是 4，所以
    模板的绝对原点有 4×4 = 16 种可能。消歧靠两件事：

    1. **槽 15 是固定标记** —— ``encode_watermark_sequence`` 的第 16 个符号
       恒为 0，不管 ID 是多少；
    2. **RS(15,5)** —— 原点错了会把符号搅到别的槽里，解出非法码字。

    对 16 个偏移各做一次投票 + RS，按 (标记位, RS 合法, 纠错数少, 置信度高) 择优。
    """
    best = None
    best_key = None
    for dr in range(4):
        for dc in range(4):
            assignments = [(slot_of(k + dr, j + dc), sym, sc) for (j, k, sym, sc) in sites]
            seq, n_obs, total, gaps = _vote(assignments)
            n_empty = sum(1 for n in n_obs if n == 0)
            # 零证据槽 = 空槽 / 平票 / 胜者票数不足。全零序列是合法 RS 码字且
            # seq[15]==0 天然过 marker_ok，零证据帧会在排序键上拿满分、解成
            # id=0 —— 幻觉 ID。RS(15,5) 预算 2e+f <= 10，无证据槽**超过 10 就
            # 整帧放弃**，不送 RS，**绝不截断到 10 再硬解**（全平图会凑成
            # f=10 + 5 个 0 = 合法全零码字，幻觉通道重开）。
            bare = [bool(n_obs[t] == 0 or gaps[t] <= 0 or gaps[t] < MIN_AGREE)
                    for t in range(16)]
            n_bare = sum(bare)
            if n_bare > 10:
                continue
            # 空槽位不是"观测到符号 0"：送进 RS 前才填 0，无证据槽按**擦除**送
            seq_clean = [0 if s < 0 else s for s in seq]
            payload, nfix = sequence_to_payload(seq_clean, known16=[not b for b in bare])
            marker_ok = (n_obs[15] >= MARKER_MIN_OBS and seq[15] == 0)
            # 排序键**先看 RS 合法性**，marker_ok 只能做次要佐证，绝不能压过它。
            # 反过来会灾难：正解常因定位槽被擦而 marker_ok=0，垃圾偏移却碰巧在
            # 槽 15 读出 0 —— 于是 nfix=-1 的垃圾把 nfix=0 的正解挤掉。
            # 实测 wm153364_c02（实拍前 PNG）：正解 (0,0) nfix=0，被 (2,0)/(3,2)
            # 两个 nfix=-1 的偏移压到第 3，整帧解废。RS 合法是**跨 15 槽的联合**
            # 证据，marker 只是单槽 1 位测试，量级差着数量级，必须次之。
            # 剩下两项在两个都合法时才起作用：nfix 少的更可信，marker 相同时
            # 再用零证据槽数。
            # ``total``（赢分之和）对 16 组的**标签置换不变** —— 16 个偏移给的是
            # 同一个 6 路分组、只是换标签，每组赢分之和恒为常数（实测 16 个偏移
            # 全是 1143573.0），裁决不了原点。留作最后一档只为定序稳定，别指望它。
            key = (
                1 if payload is not None else 0,
                -(nfix if payload is not None else 99),
                1 if marker_ok else 0,
                -n_bare,
                -n_empty,
                total,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    'shift': (dr, dc),
                    'sequence': seq_clean,
                    'n_obs': n_obs,
                    'id': payload,
                    'nfix': nfix,
                    'score': total,
                    'n_empty_slots': n_empty,
                    'n_bare_slots': n_bare,
                }
    if best is None:
        return {'shift': None, 'sequence': [-1] * 16, 'n_obs': [0] * 16,
                'id': None, 'nfix': -1, 'score': 0.0,
                'n_empty_slots': 16, 'n_bare_slots': 16, 'refused': 'zero_evidence'}
    return best


def align_and_decode_scores(score_sites: Sequence[Tuple[int, int, np.ndarray]]) -> dict:
    """16 维分数向量 → 16 槽序列 → ID，消解模板原点。

    ``score_sites`` = ``[(col, row, scores16), ...]``。

    与 :func:`align_and_decode` 的区别只在"怎么把 6 份拷贝合成一个槽"：
    这里把 6 份的 16 维分数**直接相加**（相干累积，噪声独立 → SNR × √6），
    不做硬投票。硬投票只用到 argmax，把 51:49 和 99:1 当成一样的，
    在逐码字正确率只有 55% 时会把信号丢光。
    """
    best = None
    best_key = None
    for dr in range(4):
        for dc in range(4):
            acc = np.zeros((16, 16), dtype=np.float64)
            n_obs = np.zeros(16, dtype=np.int64)
            # 逐副本独立 argmax 的直方图 —— 相干累加的胜者可能被少数高分副本
            # 拉走，用它数"几份拷贝自己也认这个符号"，等价于硬投票的 slot_ok。
            cnt = np.zeros((16, 16), dtype=np.int64)
            for j, k, sc in score_sites:
                sl = slot_of(int(k) + dr, int(j) + dc)
                if 0 <= sl < 16:
                    s = np.asarray(sc, dtype=np.float64)
                    acc[sl] += s
                    n_obs[sl] += 1
                    cnt[sl, int(np.argmax(s))] += 1
            seq: List[int] = []
            total = 0.0
            n_bare = 0            # 零证据槽数：空槽 / 平票 / 副本支持不足
            bare = np.zeros(16, dtype=bool)   # 送 RS 时按擦除处理, 不要填 0
            n_agree = np.zeros(16, dtype=np.int64)
            for t in range(16):
                if n_obs[t] == 0:
                    seq.append(-1)          # 空槽不是"观测到符号 0"
                    n_bare += 1
                    bare[t] = True
                    continue
                order = np.argsort(-acc[t])
                win = int(order[0])
                gap = float(acc[t, order[0]] - acc[t, order[1]]) if len(order) > 1 \
                    else float(acc[t, order[0]])
                seq.append(win)
                total += float(acc[t, win])
                n_agree[t] = int(cnt[t, win])
                # 拒识条件：分差非正（平票）或 独立副本支持不足。
                # 后者是 dd 的 k=3 门在相干累加下的等价物 —— 240 样本实测
                # slot_ok<=2 全部是解错, >=3 让解错归零, clean 锚点零损失。
                if gap <= 0.0 or n_agree[t] < MIN_AGREE:
                    n_bare += 1
                    bare[t] = True
            n_empty = int((n_obs == 0).sum())
            # 零证据拒识：全零序列是**合法** RS 码字，且 seq[15]==0 天然过
            # marker_ok，所以「整帧塌成 0」会在排序键上拿满分、解成 id=0 ——
            # 那是幻觉 ID，比解不出危险。RS(15,5) 的预算是 2e+f <= 10，
            # 无证据槽**超过 10 就整帧放弃**，不送 RS。**绝不截断到 10 再硬解**
            # —— 全平图会凑成 f=10 + 5 个 0 = 合法全零码字，幻觉通道重开。
            if n_bare > 10:
                continue
            seq_clean = [0 if s < 0 else s for s in seq]
            payload, nfix = sequence_to_payload(seq_clean, known16=~bare)
            # 注意这里**不用** MIN_AGREE：marker_ok 只做 16 个模板原点的消歧，
            # 加严它会改偏移排序、把正解挤掉（实测 153364 就是这么丢的）。
            # MIN_AGREE 只用于判该槽是否算零证据。
            marker_ok = (n_obs[15] >= MARKER_MIN_OBS and seq[15] == 0)
            # 排序键**先看 RS 合法性**，marker_ok 只能做次要佐证，绝不能压过它。
            # 反过来会灾难：正解常因定位槽被擦而 marker_ok=0，垃圾偏移却碰巧在
            # 槽 15 读出 0 —— 于是 nfix=-1 的垃圾把 nfix=0 的正解挤掉。
            # 实测 wm153364_c02（实拍前 PNG）：正解 (0,0) nfix=0，被 (2,0)/(3,2)
            # 两个 nfix=-1 的偏移压到第 3，整帧解废。RS 合法是**跨 15 槽的联合**
            # 证据，marker 只是单槽 1 位测试，量级差着数量级，必须次之。
            # 剩下两项在两个都合法时才起作用：nfix 少的更可信，marker 相同时
            # 再用零证据槽数。
            # ``total``（赢分之和）对 16 组的**标签置换不变** —— 16 个偏移给的是
            # 同一个 6 路分组、只是换标签，每组赢分之和恒为常数（实测 16 个偏移
            # 全是 1143573.0），裁决不了原点。留作最后一档只为定序稳定，别指望它。
            key = (
                1 if payload is not None else 0,
                -(nfix if payload is not None else 99),
                1 if marker_ok else 0,
                -n_bare,
                -n_empty,
                total,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    'shift': (dr, dc),
                    'sequence': seq_clean,
                    'n_obs': [int(v) for v in n_obs],
                    'n_agree': [int(v) for v in n_agree],
                    'id': payload,
                    'nfix': nfix,
                    'score': total,
                    'n_empty_slots': n_empty,
                    'n_bare_slots': n_bare,
                }
    if best is None:
        # 16 个偏移全部零证据 —— 整帧没有任何符号信息，明确拒绝而不是吐 id=0
        return {'shift': None, 'sequence': [-1] * 16, 'n_obs': [0] * 16,
                'n_agree': [0] * 16,
                'id': None, 'nfix': -1, 'score': 0.0,
                'n_empty_slots': 16, 'n_bare_slots': 16, 'refused': 'zero_evidence'}
    return best


def decode_patches(patches: Sequence[np.ndarray],
                   boxes: Sequence[Sequence[int]],
                   stripe_offset: Optional[int] = None) -> dict:
    """主入口：码字图像 + 格位 → watermark ID。

    ``patches[i]`` 对应 ``boxes[i] = (col, row, x0, y0[, x1, y1])``，
    ``(x0, y0)`` 是该码字在**原图**里的像素原点，用来算条纹相位
    （见 :func:`stripe_phase_of`）。``col/row`` 来自 :meth:`LatticeFit.nearest`。

    ``stripe_offset`` = (dx+dy) % 4，取景偏移已知就填上；``None`` 则在
    0..3 里搜。相位判据用 :func:`codeword_evidence`（与符号无关的正证据），
    不用符号分数 —— 错相位下黄格对比度会变号，正部直接塌到 0，判据很硬。
    """
    cands = (0, 1, 2, 3) if stripe_offset is None else (int(stripe_offset) % 4,)
    best = None
    best_key = None
    for phi in cands:
        energy = 0.0
        score_sites = []
        syms = []
        for patch, box in zip(patches, boxes):
            j, k, x0, y0 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            lp = stripe_phase_of(x0, y0, phi)
            energy += codeword_evidence(patch, lp)
            sc = codeword_scores(patch, lp)
            score_sites.append((j, k, sc))
            syms.append(int(np.argmax(sc)))
        out = align_and_decode_scores(score_sites)
        # 择优：RS 解出 ID 是最强证据，其次纠错数少/空槽少，
        # 都打平时才用证据能量挑相位。能量只在相位之间比较，量纲不进 score。
        key = (
            1 if out['id'] is not None else 0,
            -(out['nfix'] if out['id'] is not None else 99),
            -out['n_empty_slots'],
            energy,
            out['score'],
        )
        if best_key is None or key > best_key:
            best_key = key
            best = dict(out)
            best['stripe_phase'] = int(phi)
            best['evidence'] = float(energy)
            best['symbols'] = syms
    return best


# ── GF(16) 与 RS(15,5) ────────────────────────────────────────────
# primitive polynomial x^4 + x + 1 = 0x13，本原元 2；与编码器一致。

_RS_GENERATOR: Tuple[int, ...] = (1, 4, 8, 10, 12, 9, 4, 2, 12, 2, 7)


def _gf_mul(a: int, b: int) -> int:
    res = 0
    a &= 0xF
    b &= 0xF
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        carry = bool(a & 0x8)
        a = (a << 1) & 0xF
        if carry:
            a ^= 0x13
        a &= 0xF
    return res


def _gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    for inv in range(1, 16):
        if _gf_mul(b, inv) == 1:
            return _gf_mul(a, inv)
    raise ZeroDivisionError


def _gf_pow(a: int, e: int) -> int:
    r = 1
    while e > 0:
        if e & 1:
            r = _gf_mul(r, a)
        a = _gf_mul(a, a)
        e >>= 1
    return r


def _poly_eval(p: Sequence[int], x: int) -> int:
    """Horner，**p[0] 是最高次项系数**（码字 r 的存储顺序）。"""
    y = 0
    for c in p:
        y = _gf_mul(y, x) ^ (c & 0xF)
    return y


def _poly_eval_low(p: Sequence[int], x: int) -> int:
    """Horner，**p[0] 是常数项**（Λ / Ω 的 BM 存储顺序）。"""
    y = 0
    for c in reversed(p):
        y = _gf_mul(y, x) ^ (c & 0xF)
    return y


def rs_encode(payload5: Sequence[int]) -> List[int]:
    """5 个 payload nibble (MSB 在前) → 15 符号系统码字。与编码器一致。"""
    if len(payload5) != 5:
        raise ValueError('payload must be 5 nibbles')
    enc = [int(v) & 0xF for v in payload5] + [0] * 10
    pay = enc[:5]
    for i in range(5):
        coef = enc[i]
        if coef == 0:
            continue
        for j in range(1, len(_RS_GENERATOR)):
            enc[i + j] ^= _gf_mul(_RS_GENERATOR[j], coef)
    enc[:5] = pay
    return enc


def rs_decode(codeword15: Sequence[int]) -> Tuple[Optional[List[int]], int]:
    """RS(15,5) 解码 → (payload5 或 None, 纠正的符号数)。

    可纠 t=5 个符号错误。payload5 = codeword[0:5]（MSB 在前）。
    """
    r = [int(v) & 0xF for v in codeword15]
    if len(r) != 15:
        raise ValueError('codeword must have 15 symbols')

    # 生成多项式 g(x) = ∏(x - α^i), i=1..10
    g = [1]
    for i in range(1, 11):
        ng = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            ng[j] ^= _gf_mul(c, _gf_pow(2, i))
            ng[j + 1] ^= c
        g = ng
    # g[0] 是最高次系数，归一
    lead = g[0]
    g = [_gf_div(c, lead) for c in g]

    syn = [_poly_eval(r, _gf_pow(2, i)) for i in range(1, 11)]
    if all(s == 0 for s in syn):
        return list(r[:5]), 0

    # Berlekamp-Massey
    C = [1] + [0] * 10
    B = [1] + [0] * 10
    L, m, b = 0, 1, 1
    for n in range(10):
        d = syn[n]
        for i in range(1, L + 1):
            d ^= _gf_mul(C[i], syn[n - i])
        if d == 0:
            m += 1
            continue
        coef = _gf_div(d, b)
        T = C[:]
        for i in range(0, 11 - m):
            C[i + m] ^= _gf_mul(coef, B[i])
        if 2 * L <= n:
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            m += 1

    if L > 5 or L == 0:
        return None, -1

    # Chien search：找 Λ(x) 的根（Λ 的系数是低次在前，用 _poly_eval_low）。
    # r 的下标 pos ↔ 多项式次数 14-pos；Λ 的根在 X_i^{-1} = α^(pos+1)，
    # 所以 Λ(α^i)=0 ⟺ pos = (i-1) mod 15。
    err_pos = []
    for i in range(15):
        if _poly_eval_low(C, _gf_pow(2, i)) == 0:
            err_pos.append((i - 1) % 15)
    if len(err_pos) != L:
        return None, -1

    # Forney：Ω(x) = [S(x)·Λ(x)] mod x^10
    S = syn
    omega = [0] * 10
    for i in range(10):
        acc = S[i]
        for j in range(1, L + 1):
            if i - j >= 0:
                acc ^= _gf_mul(C[j], S[i - j])
        omega[i] = acc

    # Λ'(x) 形式导数（GF(2^m) 只留奇次项）
    def _lam_prime(x):
        val = 0
        for i in range(1, L + 1, 2):
            val ^= _gf_mul(C[i], _gf_pow(x, i - 1))
        return val

    corrected = r[:]
    for pos in err_pos:
        # 错误位置多项式根在 X_i^{-1}；X_i = α^(14-pos)，故 X_i^{-1} = α^(15-(14-pos))
        xi_inv = _gf_pow(2, (15 - (14 - pos)) % 15)
        num = _poly_eval_low(omega, xi_inv)      # Ω 也是低次在前
        den = _lam_prime(xi_inv)
        if den == 0:
            return None, -1
        # Forney, GF(2^m) 上负号消失, fcr=1 ⇒ e_i = Ω(X_i^{-1}) / Λ'(X_i^{-1})
        corrected[pos] ^= _gf_div(num, den)

    chk = [_poly_eval(corrected, _gf_pow(2, i)) for i in range(1, 11)]
    if any(v != 0 for v in chk):
        return None, -1
    return list(corrected[:5]), L


# 系统码的 5×15 生成矩阵：G[k] = rs_encode(第 k 个单位向量)。RS 是**线性**的，
# 所以 enc(p) = Σ_k p_k·G[k]，擦除解码就是解 GF(16) 上的 5 元一次方程组。
_RS_GEN_MATRIX = np.array(
    [rs_encode([1 if i == k else 0 for i in range(5)]) for k in range(5)],
    dtype=np.int64)                                       # (5, 15)


def rs_decode_erased(codeword15: Sequence[int],
                     known: Sequence[bool]) -> Optional[List[int]]:
    """擦除解码：``known[pos]=True`` 的位置**假定无错**，其余是擦除。

    RS(15,5) 的唯一可解条件是 ``2e + f <= 10``。走这条路时 ``e = 0``，
    所以只要可信位置 ``>= 5`` 就唯一确定码字；多余方程就是**合法性校验**
    —— 有一个对不上就整个拒绝，不会吐出半对的 ID。

    为什么需要它：把不可信槽填 0 再走纠错路，等于把"不知道"伪装成"符号 0"，
    预算全被这些假符号吃光。实测拍后 rect 有 6 张 ``n_bare = 8~10``，
    即 5~7 个槽有证据 —— 填 0 时 RS 接不住，按擦除解则正好在预算内。
    """
    r = [int(v) & 0xF for v in codeword15]
    if len(r) != 15:
        raise ValueError('codeword must have 15 symbols')
    idx = [i for i in range(15) if bool(known[i])]
    # **可信位置少于 8 个就直接拒绝。** 5 个未知数 k 个方程，冗余 k−5 条，
    # 一个随机错误 payload 骗过校验的概率是 16^{-(k-5)}。一帧要试 16 个原点
    # 偏移，要让整帧假接受率 < 1% 就得 16·16^{-(k-5)} < 0.01 → **k ≥ 8**。
    # 实测教训：门开到 k=5（零冗余，恒有解）时，n_bare=10 的图解出
    # 871421 / 835979 / 908765 一串假 ID，连拍前 wm153364 都被错原点解成
    # 31919 —— 排序键把"RS 合法"排第一，假解就压过真解了。
    if len(idx) < 8:
        return None

    def _solve(sub_idx):
        """GF(16) 高斯消元解 G[:,sub_idx] p = r[sub_idx]；不一致/秩亏返回 None。"""
        A = np.array([_RS_GEN_MATRIX[:, pos] for pos in sub_idx], dtype=np.int64)
        b = np.array([r[pos] for pos in sub_idx], dtype=np.int64)
        M = np.concatenate([A, b.reshape(-1, 1)], axis=1)
        m, n = M.shape[0], 5
        row = 0
        pivots = [-1] * n
        for col in range(n):
            sel = None
            for i in range(row, m):
                if M[i, col]:
                    sel = i
                    break
            if sel is None:
                continue
            if sel != row:
                M[[row, sel]] = M[[sel, row]]
            inv = _gf_div(1, int(M[row, col]))
            M[row] = [_gf_mul(int(v), inv) for v in M[row]]
            for i in range(m):
                if i != row and M[i, col]:
                    f = int(M[i, col])
                    M[i] = [int(M[i, j]) ^ _gf_mul(f, int(M[row, j]))
                            for j in range(n + 1)]
            pivots[col] = row
            row += 1
            if row == m:
                break
        for i in range(row, m):
            if M[i, n]:
                return None                      # 不一致 → 可信位置里有错
        if any(p < 0 for p in pivots):
            return None
        p = [0] * 5
        for col in range(n):
            p[col] = int(M[pivots[col], n])
        enc = rs_encode(p)
        if any(enc[pos] != r[pos] for pos in sub_idx):
            return None
        return p

    p = _solve(idx)
    if p is None:
        return None
    # **留一稳定性**：任意去掉一个可信位置，剩下的解必须一样。
    # 这是代数校验之外的独立证据 —— 有一个可信位置读错时，包含它的那些子集
    # 会给出另一个 payload，立刻暴露。全对时所有子集一致，自然通过。
    for drop in range(len(idx)):
        q = _solve(idx[:drop] + idx[drop + 1:])
        if q is None or q != p:
            return None
    return p


def sequence_to_payload(seq16: Sequence[int],
                        known16: Optional[Sequence[bool]] = None
                        ) -> Tuple[Optional[int], int]:
    """16 槽序列（屏幕顺序 = reversed(codeword) + [0]）→ watermark ID。

    ``known16``（可选）标出哪些槽**有证据**。给了就走双路：

    1. 纠错路 —— ``rs_decode``，可纠 ``2e <= 10`` 即 5 个错，不区分擦除；
    2. 擦除路 —— ``rs_decode_erased``，把没证据的槽当擦除，``2e + f <= 10``。

    先纠错后擦除：纠错路能吃掉"有证据但读错"的情况，擦除路专治"证据不够
    被门挡掉"的槽。两条都失败才返回 None。

    返回 (ID 或 None, 纠正符号数)。走擦除路成功时返回 ``nfix = 0``
    （没纠正任何**错**，只是没拿假符号去填）。
    """
    s = [int(v) & 0xF for v in seq16]
    if len(s) != 16:
        raise ValueError('need 16 symbols')
    codeword = list(reversed(s[:15]))
    payload, nfix = rs_decode(codeword)
    if payload is None and known16 is not None:
        k = [bool(v) for v in known16]
        if len(k) == 16:
            payload = rs_decode_erased(codeword, list(reversed(k[:15])))
            nfix = 0 if payload is not None else -1
    if payload is None:
        return None, -1
    wid = 0
    for v in payload:
        wid = (wid << 4) | (v & 0xF)
    return wid, nfix


# ════════════════════════════════════════════════════════════════════
# 自检
# ════════════════════════════════════════════════════════════════════

def _selftest() -> int:
    print('— 掩码性质 —')
    yellow = [64 - bin(m).count('1') for m in CODEWORD_CELL_MASKS]
    print('  每掩码黄格数 (应恒 32):', set(yellow))
    ds = set()
    for i in range(16):
        for j in range(i + 1, 16):
            ds.add(bin(CODEWORD_CELL_MASKS[i] ^ CODEWORD_CELL_MASKS[j]).count('1'))
    print('  两两 Hamming (应恒 32):', ds)

    print('— 格点拟合 (含 8 个孤立误报) —')
    rng = np.random.default_rng(0)
    for s_true in (0.55, 1.0, 1.35, 2.1):
        pts = np.array([[(j + .5) * CODEWORD_W * s_true, (k + .5) * CODEWORD_H * s_true]
                        for k in range(8) for j in range(12)], dtype=np.float64)
        pts += rng.normal(0, 1.5, pts.shape)
        bad = rng.uniform(0, [SCREEN_W * s_true, SCREEN_H * s_true], size=(8, 2))
        all_pts = np.vstack([pts, bad])
        # 框宽/框高是**初值**，不是精密尺子 —— 带 5% 偏差也要能收敛
        hint = (CODEWORD_W * s_true * 1.05, CODEWORD_H * s_true * 0.95)
        fit = ransac_lattice(all_pts, tol=3.0, pitch_hint=hint)
        err = abs(fit.s - s_true) / s_true * 100
        ninl = int(fit.inliers.sum())
        # 内点允许漏掉几个：tol=3px 对 σ=1.5px 噪声，单点约 9% 概率超限
        # 关键是 8 个误报全被剔除，且间距误差远小于 0.5% 指标
        bad_kept = int(fit.inliers[96:].sum())
        ok = err < 0.5 and bad_kept == 0 and ninl >= 80
        print(f'  s_true={s_true:.2f}  s_fit={fit.s:.5f}  相对误差={err:.3f}%  '
              f'内点 {ninl}/104 (真 96, 误报残留 {bad_kept})  rms={fit.rms:.2f}px  '
              f'{"OK" if ok else "FAIL"}')
        last = fit
    print('  interval:', read_interval(last))

    print('— 无 hint (纯间距启发式) —')
    fit = ransac_lattice(all_pts, tol=3.0)
    err = abs(fit.s - 2.1) / 2.1 * 100
    print(f'  s_fit={fit.s:.5f}  相对误差={err:.3f}%  内点 {int(fit.inliers.sum())}  '
          f'{"OK" if err < 0.5 else "FAIL"}')

    print('— 小角度旋转搜索 (tile_crop / 屏摄残余单应) —')
    for th_true in (-4.2, 0.0, 3.6):
        s_true = 1.0
        pts = np.array([[(j + .5) * CODEWORD_W * s_true, (k + .5) * CODEWORD_H * s_true]
                        for k in range(8) for j in range(12)], dtype=np.float64)
        pts += rng.normal(0, 1.2, pts.shape)
        r = math.radians(th_true)
        c_, s_ = math.cos(r), math.sin(r)
        cx0, cy0 = pts[:, 0].mean(), pts[:, 1].mean()
        dx, dy = pts[:, 0] - cx0, pts[:, 1] - cy0
        rot = np.stack([cx0 + c_ * dx - s_ * dy, cy0 + s_ * dx + c_ * dy], axis=1)
        fit = fit_lattice(rot, pitch_hint=(CODEWORD_W * s_true, CODEWORD_H * s_true),
                          tol=3.0, theta_span=6.0, theta_step=0.5)
        err = abs(fit.s - s_true) / s_true * 100
        dth = abs(fit.theta_deg - th_true)
        ok = err < 0.5 and dth < 0.6
        print(f'  theta_true={th_true:+.1f}°  theta_fit={fit.theta_deg:+.2f}°  '
              f's_err={err:.3f}%  内点 {int(fit.inliers.sum())}/96  '
              f'{"OK" if ok else "FAIL"}')

    print('— 原点消歧 (格点只定相对原点, 16 种偏移) —')
    print('  注: id=0x00000 → 全零序列在任何偏移下都不变, 原点**原理上不可定**,'
          ' 只要求 ID 正确。')
    test_ids = [0x12345, 0xABCDE, 0xFFFFF, 0x00000] + \
               [int(v) for v in rng.integers(0, 1 << 20, size=8)]
    n_fail = 0
    for wid in test_ids:
        cw = rs_encode([(wid >> (4 * (4 - i))) & 0xF for i in range(5)])
        seq16 = list(reversed(cw)) + [0]        # encode_watermark_sequence 同款
        grid = subblock_symbol_grid(seq16)
        degenerate = len(set(seq16)) == 1       # 序列与原点无关
        for dr, dc in ((0, 0), (1, 2), (3, 3), (2, 1)):
            sites = []
            for k in range(8):
                for j in range(12):
                    # 拟合原点比真原点偏 (dc, dr) 个格 → nearest 返回 (j-dc, k-dr)
                    sites.append((j - dc, k - dr, int(grid[k, j]), 1.0))
            out = align_and_decode(sites)
            ok = out['id'] == wid and out['nfix'] == 0
            if not degenerate:
                ok = ok and out['shift'] == (dr, dc)
            if not ok:
                n_fail += 1
                got = out['id']
                print(f'  FAIL id=0x{wid:05X} 偏移(dr={dr},dc={dc}) → '
                      f'shift={out["shift"]} id={got} nfix={out["nfix"]}')
    print(f'  {len(test_ids)} 个 ID × 4 个偏移: '
          f'{"全部 OK" if n_fail == 0 else f"{n_fail} 个 FAIL"}')

    print('— RS 编解码往返 (0..5 个随机错) —')
    for wid in (0x00000, 0x12345, 0xABCDE, 0xFFFFF):
        payload = [(wid >> (4 * (4 - i))) & 0xF for i in range(5)]
        cw = rs_encode(payload)
        line = [f'  id=0x{wid:05X}']
        for nerr in range(6):
            bad_cw = cw[:]
            for p in range(nerr):
                bad_cw[p * 3 % 15] ^= (p + 1) & 0xF or 1
            out, nf = rs_decode(bad_cw)
            line.append(f'{nerr}错:{"OK" if out == payload else "FAIL"}')
        print('  '.join(line))
        # 6 个**互不相同**的位置出错应当纠不了 (t=5)
        bad_cw = cw[:]
        for p in (0, 2, 4, 6, 8, 10):
            bad_cw[p] ^= 0x7
        out, _ = rs_decode(bad_cw)
        print(f'    6 错应失败: {"OK(返回None或错值)" if out != payload else "异常成功"}')

    print('— 16 槽序列 → ID —')
    for wid in (0x12345, 0xABCDE):
        payload = [(wid >> (4 * (4 - i))) & 0xF for i in range(5)]
        cw = rs_encode(payload)
        seq16 = list(reversed(cw)) + [0]          # 屏幕顺序
        got, nfix = sequence_to_payload(seq16)
        print(f'  id=0x{wid:05X} → 0x{got:05X} (nfix={nfix}) '
              f'{"OK" if got == wid else "FAIL"}')

    print('— 端到端：模板 → 96 码字 → 相干累积 → ID —')
    from PIL import Image
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    tp = os.path.join(here, 'sample', 'wm_template_123456.png')
    if not os.path.exists(tp):
        print('  (跳过，未找到 sample/wm_template_123456.png)')
        return 0
    img = np.array(Image.open(tp).convert('RGB'))[:, :, ::-1]   # RGB → BGR
    wid = 123456
    payload = [(wid >> (4 * (4 - i))) & 0xF for i in range(5)]
    seq16 = list(reversed(rs_encode(payload))) + [0]
    grid = subblock_symbol_grid(seq16)

    patches, boxes, symbols = [], [], []
    ok = 0
    for r in range(SUB_ROWS):
        for c in range(SUB_COLS):
            y0, x0 = r * CODEWORD_H, c * CODEWORD_W
            patch = img[y0:y0 + CODEWORD_H, x0:x0 + CODEWORD_W]
            patches.append(patch)
            boxes.append((c, r, x0, y0))
            got = decode_codeword(patch, stripe_phase_of(x0, y0))[0]
            symbols.append(got)
            ok += (got == grid[r, c])
    print(f'  逐码字匹配滤波: {ok}/96 正确 (期望 96)')
    voted = vote_sequence(symbols)
    print(f'  16 槽硬投票:    {"OK" if voted == list(seq16) else "FAIL"}')
    got_id, nfix = sequence_to_payload(voted)
    print(f'  RS 解码:        0x{wid:05X} → '
          f'{("0x%05X" % got_id) if got_id is not None else "None"} (nfix={nfix})  '
          f'{"OK" if got_id == wid else "FAIL"}')

    print('— decode_patches：未知条纹相位 + 相干累积 —')
    bad = 0
    for off in range(4):
        # 人为制造 (dx+dy)%4 = off 的取景偏移：把每个 patch 的像素原点平移。
        # 解码器看到的相位是 (x0+off+y0+phi)，真值是 (x0+y0) → phi 应选 -off。
        boxes2 = [(j, k, x0 + off, y0) for (j, k, x0, y0) in boxes]
        out = decode_patches(patches, boxes2, stripe_offset=None)
        got = out['id']
        good = (got == wid and out['stripe_phase'] == ((-off) % 4))
        bad += (not good)
        print(f'  offset={off} → 相位{out["stripe_phase"]}(期望{(-off) % 4}) '
              f'ID=0x{got:05X} (nfix={out["nfix"]})  {"OK" if good else "FAIL"}')
    print(f'  未知相位搜索: {"OK" if bad == 0 else f"{bad} 例 FAIL"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(_selftest())
