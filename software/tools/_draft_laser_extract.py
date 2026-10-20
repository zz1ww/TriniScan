"""草稿：细脊 + 连续性约束的激光线提取（不修改项目代码）。

思路（针对黑白相机 + 灰度激光线）：
1. 逐列找所有局部峰（比左右邻域都亮）
2. 对每个峰算"半高宽"：峰值降到半高时的左右跨度
   - 激光线：2~6 px（细）
   - 白格 / 纸板边 / 手：几十 px（宽）
   - 只保留窄峰
3. 峰值本身必须够亮（相对该列中位数）
4. 连续性：逐列按上一点的 y 做窗口搜索，剔除跳变
"""
from __future__ import annotations

import numpy as np


def _col_peaks(col: np.ndarray, min_val: float, max_width: float):
    """返回该列所有满足条件的局部峰 (y, peak_val, width)。"""
    n = col.shape[0]
    out = []
    if n < 5:
        return out
    # 局部极大（>= 左右邻域）
    c = col
    is_max = np.zeros(n, bool)
    is_max[1:-1] = (c[1:-1] > c[:-2]) & (c[1:-1] >= c[2:])
    ys = np.flatnonzero(is_max)
    for y in ys:
        pk = float(c[y])
        if pk < min_val:
            continue
        half = pk * 0.5
        # 向左找半高
        x = y
        while x > 0 and c[x - 1] > half and (y - x) < max_width * 3:
            x -= 1
        left = x
        x = y
        while x < n - 1 and c[x + 1] > half and (x - y) < max_width * 3:
            x += 1
        right = x
        w = right - left + 1
        out.append((int(y), pk, float(w)))
    return out


def extract_laser_thin(gray: np.ndarray,
                       min_peak: float = 200.0,
                       max_width: float = 6.0,
                       bg_percentile: float = 60.0,
                       bg_margin: float = 60.0,
                       max_jump: float = 12.0) -> np.ndarray:
    """提取细激光线，返回 (N,2) 亚像素 (u, v)。

    Parameters
    ----------
    gray : uint8 / float 灰度图
    min_peak : 峰值绝对下限（灰度）
    max_width : 允许的最大半高宽（像素）—— 用于排除宽白格
    bg_percentile : 用于估计背景的分位数
    bg_margin : 峰值至少超过背景多少
    max_jump : 相邻列 y 允许的最大跳变（连续性约束）
    """
    g = gray.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    H, W = g.shape

    cand = np.full((W, 2), np.nan, np.float32)  # (u, v)
    cand_val = np.zeros(W, np.float32)

    for u in range(W):
        col = g[:, u]
        bg = float(np.percentile(col, bg_percentile))
        thr = max(min_peak, bg + bg_margin)
        peaks = _col_peaks(col, thr, max_width)
        if not peaks:
            continue
        # 只留窄的
        narrow = [p for p in peaks if p[2] <= max_width]
        if not narrow:
            continue
        # 取最亮
        y, pk, w = max(narrow, key=lambda t: t[1])
        # 亚像素：抛物线拟合
        if 0 < y < H - 1:
            y0, y1, y2 = col[y - 1], col[y], col[y + 1]
            den = (y0 - 2 * y1 + y2)
            dy = 0.5 * (y0 - y2) / den if abs(den) > 1e-6 else 0.0
            dy = float(np.clip(dy, -1, 1))
        else:
            dy = 0.0
        cand[u] = (u, y + dy)
        cand_val[u] = pk

    # --- 连续性：从最亮列开始双向跟踪 ---
    valid = ~np.isnan(cand[:, 0])
    if valid.sum() == 0:
        return np.zeros((0, 2))

    keep = np.zeros(W, bool)
    # 先做"跳变"过滤：与左右已保留点的中值比较
    us = np.flatnonzero(valid)
    # 迭代：以最亮列为种子，向两侧扩展
    seed = int(us[np.argmax(cand_val[us])])
    keep[seed] = True

    for step in (-1, 1):
        last_y = float(cand[seed, 1])
        empty = 0
        u = seed + step
        while 0 <= u < W:
            if not valid[u]:
                empty += 1
                if empty > 30:      # 连续 30 列无点 → 认为线断了，停止
                    break
                u += step
                continue
            y = float(cand[u, 1])
            if abs(y - last_y) <= max_jump:
                keep[u] = True
                last_y = y
                empty = 0
            else:
                empty += 1
                if empty > 30:
                    break
            u += step

    pts = cand[keep]
    if len(pts) == 0:
        return np.zeros((0, 2))
    return pts.astype(np.float64)


if __name__ == "__main__":
    import sys
    import cv2

    for f in sys.argv[1:]:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        p = extract_laser_thin(g)
        print(f, len(p))
