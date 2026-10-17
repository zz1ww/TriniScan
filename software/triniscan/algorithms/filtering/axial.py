"""轴向裁剪：剔除物体上下方的背景/支架孤立环带。

动机
----
转台扫描时激光常同时打到**物体后方的背景/支架**，在点云里表现为
若干条与物体分离的环带（沿轴高度不同、半径也不同）。这类点用 RANSAC
平面拟合往往抓错目标（会误判物体的竖直面），而"沿轴高度分布"最能
把它们和物体分开。

三种手段（可组合）：

1. ``auto_trim``：自动取"沿轴高度分布的**主体连通区间**"；
2. ``height_min_m`` / ``height_max_m``：硬门限；
3. ``radius_max_m``：切掉离轴过远的散点。

``auto_trim`` 曾经会删掉整个物体
--------------------------------
旧判据是"某箱计数 > ``density_ratio`` × 峰值"。立方体绕轴扫描时，
**顶面**的 4000+ 点全挤进一两个箱，而**侧面**沿高度均匀稀疏（每箱
约 150 点）——阈值被顶面抬高到 413，侧面全部低于阈值，直方图只剩
顶面一箱，于是返回 ``[49.7, 50.0] mm``，把侧面整个删掉。

这就是"少一面"的真正机制。现行判据见 :func:`dominant_band`：密度
条件 + 间隙容差 + 区间填充率，三者共同判定，"一密一疏"的分布也能
正确圈出完整高度范围。

安全阀
------
即使判定出错，本插件也不会让物体"消失"：如果区间内点数少于
``min_keep_ratio`` × 总点数，视为判定不可信，**放弃裁剪**并在报告
里记录 ``auto_trim_skipped``。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...contracts import Cloud, FILTERS, ParamSpec, RunContext
from .base import FilterPlugin

__all__ = ["AxialTrimFilter", "dominant_band"]


# ---------------------------------------------------------------------------
def dominant_band(along: np.ndarray,
                  bins: int = 120,
                  density_ratio: float = 0.10,
                  max_gap_bins: int = 8,
                  min_fill_ratio: float = 0.02) -> Optional[tuple]:
    """在"沿轴坐标"直方图中找出物体主体的**连通覆盖区间**。

    为什么不能用"峰值比例"当阈值
    ----------------------------
    旧实现的判据是"某箱计数 > ``density_ratio`` × 峰值"。它有一个致命
    假设：**物体各高度的点密度大致均匀**。

    线结构光转台扫描里这个假设经常不成立。一个立方体绕 Z 轴扫描时：

    - 侧面（4 个立面）的点沿高度**均匀稀疏**分布，每箱约 150 点；
    - 顶面（水平面）的点几乎**全部落在 1~2 个箱里**，可达 4000+ 点。

    于是峰值为 4130，阈值 413，所有侧面箱都低于阈值 → 直方图只剩顶面
    那一箱 → 返回区间 ``[49.7, 50.0] mm`` → **整个物体被删掉**。

    这正是"少一面 / 体积偏小"的真正机制，且旧实现完全静默。

    现行判据（两个条件）
    -------------------
    1. **密度条件**：``hist[i] > peak × density_ratio`` ——仍用于识别
       "真正稀疏到可视为空"的箱（背景噪声、孤立环带）；
    2. **间隙条件**：两个密集箱之间如果相隔超过 ``max_gap_bins`` 个空箱，
       就认为已跨出物体。

    并把两者的连通区间**合并**起来：只要区间内的"填充率"
    （非空箱占比）不低于 ``min_fill_ratio``，就视为同一个物体——
    这样"顶面超密 + 侧面平均"的分布也能被正确圈成一个区间。

    Parameters
    ----------
    along : (N,) 各点沿轴坐标
    bins : 直方图分箱数
    density_ratio : 密度判据（相对峰值）。越大越保守（越容易截断）
    max_gap_bins : 允许跨越的空箱数。应大于"物体内部正常稀疏带"的
        宽度，小于"物体与背景之间"的间隔
    min_fill_ratio : 区间内非空箱的最小占比，低于此值说明中间是真空洞

    Returns
    -------
    (lo, hi) 或 None（无法判定）

    Notes
    -----
    返回的区间是**最长**的合格连通段，而不是"点数最多的段"。
    """
    along = np.asarray(along, dtype=np.float64).ravel()
    if along.size < 20 or np.ptp(along) < 1e-9:
        return None
    hist, edges = np.histogram(along, bins=bins)
    peak = int(hist.max())
    if peak <= 0:
        return None

    # --- 判据 1：用**中位数**而非峰值定"空箱"阈值 -------------------
    # 峰值会被"一个超密的面"独占（立方体顶面可占全部点的 20%），
    # 用它定阈值会把所有"相对稀疏"的箱判成空。中位数对单个尖峰
    # 免疫：一半箱子比它密就算"有数据"。
    nonzero = hist[hist > 0]
    if nonzero.size == 0:
        return None
    median = float(np.median(nonzero))
    thr = max(1.0, median * density_ratio * 10.0)   # density_ratio≈0.1 → thr≈median
    dense = hist > thr

    # 若判据过严导致几乎无密集箱，退化为"非空即密集"
    if dense.sum() < max(2, bins // 20):
        dense = hist > 0

    # --- 判据 2：按间隙把密集箱聚簇 -------------------------------
    idx = np.flatnonzero(dense)
    groups: list[list[int]] = [[int(idx[0])]]
    for cur in idx[1:]:
        if cur - groups[-1][-1] - 1 > max_gap_bins:
            groups.append([int(cur)])
        else:
            groups[-1].append(int(cur))

    # --- 判据 3：取跨度最长、填充率合格的簇 -----------------------
    best = None
    for group in groups:
        lo_i, hi_i = group[0], group[-1]
        span = hi_i - lo_i + 1
        if span <= 0:
            continue
        filled = int((hist[lo_i:hi_i + 1] > 0).sum())
        if filled / span < min_fill_ratio:
            continue
        points = int(hist[lo_i:hi_i + 1].sum())
        score = (span, points)      # 先比覆盖高度，再比点数
        if best is None or score > best[0]:
            best = (score, lo_i, hi_i)

    if best is None:
        return None
    _, lo_i, hi_i = best
    return float(edges[lo_i]), float(edges[hi_i + 1])


# ---------------------------------------------------------------------------
@FILTERS.register("axial")
class AxialTrimFilter(FilterPlugin):
    """按沿轴高度 / 半径裁剪。"""

    name = "axial"
    label = "轴向裁剪"
    # 直方图基于当前点云分布 → 与顺序耦合
    depends_on_current = True
    params = (
        ParamSpec("reconstruction.axial.auto_trim", bool, False,
                  label="自动主体区间",
                  help="用沿轴高度直方图自动确定主体的上下边界", short="auto_trim"),
        ParamSpec("reconstruction.axial.bins", int, 120,
                  low=20, high=500,
                  label="直方图箱数", help="自动主体区间的分辨率", short="bins"),
        ParamSpec("reconstruction.axial.density_ratio", float, 0.10,
                  low=0.01, high=0.9,
                  label="密度门槛比例",
                  help="低于 峰值×该比例 的箱视为空，会把主体截断", short="density_ratio"),
        ParamSpec("reconstruction.axial.max_gap_bins", int, 8,
                  low=1, high=40, short="max_gap_bins",
                  label="允许跨越的空箱数",
                  help="两个密集箱相隔超过此值即认为离开物体；"
                       "应大于物体内部正常稀疏带宽度"),
        ParamSpec("reconstruction.axial.min_fill_ratio", float, 0.02,
                  low=0.0, high=0.5, short="min_fill_ratio",
                  label="区间最小填充率",
                  help="区间内非空箱占比低于此值视为中间是空洞"),
        ParamSpec("reconstruction.axial.min_keep_ratio", float, 0.10,
                  low=0.0, high=1.0, short="min_keep_ratio",
                  label="裁剪安全阀",
                  help="自动区间保留点数低于 总点数×该比例 时放弃裁剪"
                       "（防止误判把物体删光）"),
        ParamSpec("reconstruction.axial.height_min_m", float, None,
                  unit="m", label="高度下限", short="height_min_m"),
        ParamSpec("reconstruction.axial.height_max_m", float, None,
                  unit="m", label="高度上限", short="height_max_m"),
        ParamSpec("reconstruction.axial.radius_max_m", float, None,
                  unit="m", label="最大半径", short="radius_max_m"),
    )

    def compute_mask(self, cloud: Cloud, ctx: RunContext
                     ) -> tuple[np.ndarray, dict]:
        params = self.read_params(ctx.cfg)
        n = len(cloud)
        if n == 0:
            return np.zeros(0, dtype=bool), {"reason": "空点云",
                                             "applied": False}
        if ctx.axis_point is None and ctx.calib is None:
            return np.ones(n, dtype=bool), {"reason": "无转轴，跳过",
                                            "applied": False}

        point, axis = _axis(ctx)
        rel = cloud.points - point
        along = rel @ axis
        perp = rel - np.outer(along, axis)
        radius = np.linalg.norm(perp, axis=1)

        h_min = params.get("height_min_m")
        h_max = params.get("height_max_m")
        r_max = params.get("radius_max_m")
        auto = bool(params["auto_trim"])

        keep = np.ones(n, dtype=bool)
        reasons: list[str] = []
        band = None

        # 与旧实现一致：auto 仅在三个门限**全为空**时生效
        extra: dict = {"auto_trim_skipped": ""}
        if auto and h_min is None and h_max is None and r_max is None:
            band = dominant_band(
                along,
                bins=int(params["bins"]),
                density_ratio=float(params["density_ratio"]),
                max_gap_bins=int(params["max_gap_bins"]),
                min_fill_ratio=float(params["min_fill_ratio"]),
            )
            if band is not None:
                lo_b, hi_b = band
                m = (along >= lo_b) & (along <= hi_b)
                keep_ratio = m.sum() / max(n, 1)
                min_keep = float(params["min_keep_ratio"])
                if m.sum() < 20:
                    extra["auto_trim_skipped"] = (
                        f"主体区间仅 {int(m.sum())} 点，判定不可信")
                elif keep_ratio < min_keep:
                    # 安全阀：判据可能把"一个面"当成了整个物体
                    extra["auto_trim_skipped"] = (
                        f"主体区间仅覆盖 {keep_ratio * 100:.1f}% 的点"
                        f"（低于安全阀 {min_keep * 100:.0f}%），已放弃裁剪")
                else:
                    keep &= m
                    extra["dominant_band_m"] = [lo_b, hi_b]
                    reasons.append(f"自动主体区间 [{lo_b * 1e3:.0f},"
                                   f"{hi_b * 1e3:.0f}] mm")

        if h_min is not None:
            keep &= along >= float(h_min)
            reasons.append(f"h>={float(h_min) * 1e3:.0f}mm")
        if h_max is not None:
            keep &= along <= float(h_max)
            reasons.append(f"h<={float(h_max) * 1e3:.0f}mm")
        if r_max is not None:
            keep &= radius <= float(r_max)
            reasons.append(f"r<={float(r_max) * 1e3:.0f}mm")

        removed = int((~keep).sum())
        if removed == 0:
            return np.ones(n, dtype=bool), {
                "reason": "无越界点", "applied": False}
        return keep, {
            "applied": True,
            "removed": removed,
            "dominant_band_m": band,
            "along_min_m": float(along.min()),
            "along_max_m": float(along.max()),
            "reason": "，" .join(reasons),
        }


def _axis(ctx: RunContext):
    if ctx.axis_point is not None and ctx.axis_dir is not None:
        p = np.asarray(ctx.axis_point, dtype=np.float64).reshape(3)
        d = np.asarray(ctx.axis_dir, dtype=np.float64).reshape(3)
    else:
        p = ctx.calib.axis_point
        d = ctx.calib.axis_dir
    nrm = np.linalg.norm(d)
    return p, (d / nrm if nrm > 1e-12 else d)
