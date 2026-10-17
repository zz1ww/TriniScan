"""转盘（载物台）表面点去除。

为什么需要
----------
物体放在转盘上扫描时，激光会同时打在**转盘表面**，形成大量属于
"载物台"而非"目标物体"的点。这些点若不去除：

- 会被 Poisson 当成真实表面，重建出多余的"底盘"；
- 会让封闭化/体积积分把底盘也算进去，体积**系统性偏大**。

去除原理（几何拟合，无需人工框选）
----------------------------------
转盘俯视近似**平面**，且其法向与**转轴方向**一致（都在竖直方向）。
因此：

1. 用 RANSAC 在点云中找出**主平面**；
2. 校验该平面法向是否与转轴方向近似平行（夹角 < 容差）——
   满足才认为是转盘，避免误删物体的真实平面；
3. 剔除落在该平面上/下方（沿轴距离 < 容差）的点；
4. 可选：仅剔除距转轴半径 < ``radius`` 的点（转盘边缘外不误伤）。

本模块为纯几何、可配置、可单测，并返回可解释的去除信息。

两种方法
--------
1. :func:`remove_turntable_surface`（``method="ransac"``）：RANSAC 拟合
   主平面 + 法向与轴平行校验。**对平面占比高的场景有效**，但物体的
   大面积竖直面会干扰平面选择。
2. :func:`remove_turntable_axis_height`（``method="axis_height"`）：
   直接以**轴标定给出的轴向基准** ``h_table`` 切割——因为转轴已经标定，
   转盘面的高度是标定体系的固有量，无需从点云反推。**更稳、更可控**。
   Cyclop/Horus 的坐标系就是把 Z=0 定义在转台面上（见
   ``Jesus89/3DScanScience`` 第 6 节），本方法即该约定的工程化实现。

:func:`estimate_table_height` 可在未显式给出 ``h_table`` 时，从点云的
``(r, h)`` 极坐标分布中自动估计（找"轴向最低且径向延伸最宽"的密集层）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common.logging_utils import get_logger
from ..common.robust import robust_plane_fit

log = get_logger(__name__)

__all__ = ["TurntableRemovalResult", "remove_turntable_surface",
           "estimate_table_height", "remove_turntable_axis_height"]


@dataclass
class TurntableRemovalResult:
    """转盘去除结果。"""

    points: np.ndarray                  # (M, 3) 保留下来的点
    removed_mask: np.ndarray            # (N,) bool，True 表示被剔除
    plane: Optional[np.ndarray]         # 拟合到的转盘平面（无则 None）
    num_input: int = 0
    num_removed: int = 0
    applied: bool = False               # 是否真的执行了去除
    reason: str = ""                    # 未执行/跳过原因

    @property
    def removed_ratio(self) -> float:
        if self.num_input <= 0:
            return 0.0
        return self.num_removed / float(self.num_input)


def remove_turntable_surface(
    points: np.ndarray,
    axis_dir: Optional[np.ndarray] = None,
    ransac_dist: float = 0.001,
    min_inlier_ratio: float = 0.15,
    align_tol_deg: float = 15.0,
    margin: float = 0.0005,
    axis_point: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    band_below: float = 0.0,
    keep_upper_only: bool = False,
    outlier_mm: Optional[float] = None,
    seed: int = 0,
) -> TurntableRemovalResult:
    """从点云中剔除转盘表面点。

    Parameters
    ----------
    points : (N, 3) 世界坐标系点云
    axis_dir : (3,) 转轴方向（用于校验平面朝向）；None 时跳过朝向校验
    ransac_dist : 平面内点距离阈值 (m)
    min_inlier_ratio : 平面内点占比低于此值则不认为是转盘（避免误删）
    align_tol_deg : 平面法向与转轴方向的允许夹角（度）
    margin : 平面法向容差带厚度 (m)：|距离| < margin 视为"在盘面上"
    axis_point : 轴上一点（配合 radius 做半径裁剪）
    radius : 仅剔除距转轴半径小于此值的点（m）；None 表示不限制
    band_below : 额外剔除盘面**下方**该厚度内的点（m）
    keep_upper_only : 是否直接保留平面以上（含 margin）的点（更强）
    outlier_mm : 稳健拟合离群阈值（mm）；None 自适应
    seed : RANSAC 随机种子

    Returns
    -------
    TurntableRemovalResult
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n < 10:
        return TurntableRemovalResult(
            points=points, removed_mask=np.zeros(n, dtype=bool),
            plane=None, num_input=n, applied=False,
            reason="点数过少，跳过转盘去除")

    margin = max(float(margin), 0.0)
    band_below = max(float(band_below), 0.0)

    # 1. RANSAC 找主平面（比 IRLS 更能抵御大比例盘面点）
    fit = robust_plane_fit(
        points, method="ransac", outlier_threshold=ransac_dist,
        max_iter=2000, seed=seed,
    )
    plane = fit.plane
    inlier_ratio = fit.num_inliers / float(n)

    # 2. 校验：内点占比需足够，且平面法向须与转轴方向平行
    if inlier_ratio < min_inlier_ratio:
        return TurntableRemovalResult(
            points=points, removed_mask=np.zeros(n, dtype=bool),
            plane=plane, num_input=n, applied=False,
            reason=(f"未找到足够大的主平面（内点占比 "
                    f"{inlier_ratio * 100:.0f}% < "
                    f"{min_inlier_ratio * 100:.0f}%），跳过"))

    if axis_dir is not None:
        axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
        na = np.linalg.norm(axis)
        if na > 1e-12:
            axis = axis / na
            cos_ang = abs(float(plane[:3] @ axis))
            cos_ang = min(max(cos_ang, 0.0), 1.0)
            angle = float(np.degrees(np.arccos(cos_ang)))
            if angle > align_tol_deg:
                return TurntableRemovalResult(
                    points=points, removed_mask=np.zeros(n, dtype=bool),
                    plane=plane, num_input=n, applied=False,
                    reason=(f"主平面法向与转轴夹角 {angle:.1f}° > "
                            f"{align_tol_deg:.1f}°，判定为物体表面，跳过"))

    # 3. 计算沿法向的带符号距离
    signed = points @ plane[:3] + plane[3]

    # 4. 半径裁剪（可选）
    radial_mask = None
    if radius is not None and axis_point is not None:
        rel = points - np.asarray(axis_point, dtype=np.float64).reshape(3)
        if axis_dir is not None:
            axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
            na = np.linalg.norm(axis)
            axis = axis / na if na > 1e-12 else axis
            rel = rel - np.outer(rel @ axis, axis)
        radial_mask = np.linalg.norm(rel, axis=1) <= radius

    if keep_upper_only:
        candidate = signed >= margin
        if radial_mask is not None:
            # 半径内：保留上方；半径外：全部保留
            candidate = candidate | (~radial_mask)
        removed_mask = ~candidate
    else:
        # 盘面附近（±margin）及下方 band_below 内剔除
        near_plane = np.abs(signed) <= margin
        below = (signed < -margin) & (signed >= -(margin + band_below))
        near_plane = near_plane | (below if band_below > 0 else False)
        if radial_mask is not None:
            near_plane = near_plane & radial_mask
        removed_mask = near_plane

    kept = points[~removed_mask]
    result = TurntableRemovalResult(
        points=kept, removed_mask=removed_mask, plane=plane,
        num_input=n, num_removed=int(removed_mask.sum()), applied=True,
        reason=(f"剔除转盘面点 {int(removed_mask.sum())} "
                f"({removed_mask.sum() / n * 100:.1f}%)"),
    )
    log.info("转盘去除: %s（主平面内点占比 %.0f%%）",
             result.reason, inlier_ratio * 100)
    return result


# ---------------------------------------------------------------------------
# 方法 2：轴基准高度法（Cyclop 坐标约定）
# ---------------------------------------------------------------------------
def _axis_frame(points: np.ndarray, axis_point: np.ndarray,
                axis_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把点云投影到轴坐标系，返回 ``(h, r)``。

    ``h`` 为沿轴坐标（相对 ``axis_point``），``r`` 为到轴的径向距离。
    """
    axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    nrm = np.linalg.norm(axis)
    axis = axis / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
    origin = np.asarray(axis_point, dtype=np.float64).reshape(3)
    rel = points - origin
    h = rel @ axis
    perp = rel - np.outer(h, axis)
    r = np.linalg.norm(perp, axis=1)
    return h, r


def estimate_table_height(points: np.ndarray,
                          axis_point: np.ndarray,
                          axis_dir: np.ndarray,
                          *,
                          bins: int = 100,
                          width_ratio: float = 0.5,
                          min_count: int = 20) -> Optional[float]:
    """从点云自动估计**转盘面**的轴向高度 ``h_table``。

    判据
    ----
    转盘面在点云中的特征：**位于物体下方（h 最小侧）**，且**径向延伸
    最宽**（盘面半径大于物体截面）。因此：

    1. 按 ``h`` 分箱，统计每箱的点数；
    2. 只考虑 h 较小的那部分（``width_ratio`` 比例，默认下半）；
    3. 在点数达到 ``min_count`` 的候选中，取 **h 最小** 的一箱作为盘面
       （转盘面在物体下方，是最底的密集层）。

    Parameters
    ----------
    points : (N, 3) 世界系点云
    axis_point, axis_dir : 轴
    bins : 轴向分箱数
    width_ratio : 只在 h 分布的下侧该比例内搜索（转盘在下方）
    min_count : 视为"密集层"的最少点数

    Returns
    -------
    float 或 None : 估计的 ``h_table``（与轴同单位，通常 m）
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < max(min_count * 2, 50):
        return None
    h, _ = _axis_frame(points, axis_point, axis_dir)
    if h.size == 0 or np.ptp(h) < 1e-9:
        return None

    lo_all, hi_all = float(h.min()), float(h.max())
    span = hi_all - lo_all
    # 只在下侧 width_ratio 区间内搜索（转盘位于物体下方）
    hi_search = lo_all + span * float(np.clip(width_ratio, 0.05, 1.0))
    sel = h[(h >= lo_all) & (h <= hi_search)]
    if sel.size < min_count:
        sel = h
    edges = np.linspace(float(sel.min()), float(sel.max()),
                        max(bins, 4) + 1)
    counts, edges = np.histogram(sel, bins=edges)
    valid = np.nonzero(counts >= int(min_count))[0]
    if valid.size == 0:
        # 放宽到至少取最大箱
        valid = np.array([int(np.argmax(counts))])
    k = int(valid[0])                       # h 最小的密集箱
    h_table = float(0.5 * (edges[k] + edges[k + 1]))
    log.info("自动估计转盘高度 h_table=%.2f mm（密集箱点数 %d）",
             h_table * 1e3, int(counts[k]))
    return h_table


def remove_turntable_axis_height(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    h_table: float,
    *,
    margin: float = 0.0005,
    band_below: float = 0.002,
    radius: Optional[float] = None,
    clamp_to_cloud_min: bool = True,
) -> TurntableRemovalResult:
    """按**轴基准高度** ``h_table`` 剔除转盘面点（Cyclop 坐标约定）。

    与 RANSAC 平面法相比，本方法**不依赖点云中的平面结构**：只要转轴
    已标定，转盘面的高度就是标定体系的固有量，直接把 ``h <= h_table +
    margin`` 的点剔除即可。这对"物体有大面积竖直面"（会干扰 RANSAC
    选面）的场景尤其稳健。

    Parameters
    ----------
    points : (N, 3) 世界系点云
    axis_point, axis_dir : 轴
    h_table : 转盘面沿轴高度（同轴坐标系单位）
    margin : 盘面**上方**仍视为盘面的厚度 (m)
    band_below : 盘面**下方**剔除的厚度 (m)
    radius : 仅剔除距轴半径 <= 此值的点（m）；None 不限。
        用于避免误删转盘半径之外的物体/背景。
    clamp_to_cloud_min : 若 ``h_table`` 低于点云最小 h，说明标定基准
        与本次点云不吻合，此时把它抬到点云最小 h（安全兜底），
        并在结果 reason 中说明。

    Returns
    -------
    TurntableRemovalResult
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n < 10:
        return TurntableRemovalResult(
            points=points, removed_mask=np.zeros(n, dtype=bool),
            plane=None, num_input=n, applied=False,
            reason="点数过少，跳过转盘去除")

    h, r = _axis_frame(points, axis_point, axis_dir)
    h_tab = float(h_table)
    note = ""
    if clamp_to_cloud_min:
        h_min = float(h.min())
        if h_tab < h_min:
            note = (f"（h_table {h_tab*1e3:.1f}mm 低于点云最低 "
                    f"{h_min*1e3:.1f}mm，已抬升至点云最低）")
            h_tab = h_min

    margin = max(float(margin), 0.0)
    band_below = max(float(band_below), 0.0)
    # 剔除：h <= h_table + margin（盘面及其下方 band_below 内）
    cut_hi = h_tab + margin
    below = h_tab - margin - band_below
    remove = (h <= cut_hi) & (h >= below)

    if radius is not None and float(radius) > 0:
        remove &= (r <= float(radius))

    removed = int(remove.sum())
    if removed == 0:
        return TurntableRemovalResult(
            points=points, removed_mask=np.zeros(n, dtype=bool),
            plane=None, num_input=n, applied=False,
            reason=f"轴基准 {h_tab*1e3:.1f}mm 处无点可剔除{note}")

    kept = points[~remove]
    result = TurntableRemovalResult(
        points=kept, removed_mask=remove, plane=None,
        num_input=n, num_removed=removed, applied=True,
        reason=(f"轴基准法 h_table={h_tab*1e3:.1f}mm：剔除 {removed} 点"
                f"（{removed / n * 100:.1f}%）{note}"),
    )
    log.info("转盘去除[轴基准]: %s", result.reason)
    return result
