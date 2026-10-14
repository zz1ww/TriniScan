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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common.logging_utils import get_logger
from ..common.robust import robust_plane_fit

log = get_logger(__name__)

__all__ = ["TurntableRemovalResult", "remove_turntable_surface"]


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
