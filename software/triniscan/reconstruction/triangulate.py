"""三角测量：像素 + 相机内参 + 激光平面 → 相机系三维点。

原理
----
线结构光三角测量中，每个激光条纹中心像素对应一条从相机光心出发的射线；
该射线与激光光平面在空间中的**唯一交点**即为被测表面上的三维点::

    射线:  P(t) = O + t * D        (O=光心, D=单位方向)
    平面:  n · P + d = 0
    ⇒      t = -(n·O + d) / (n·D)
    ⇒      P = O + t*D

实现要点
--------
- 全程向量化，支持一次处理整条中心线（数千点）。
- 可选先对像素去畸变，保证精度。
- 平面法向符号约定：相机在平面"外侧"，即 ``plane[3] < 0``，
  使所有有效交点位于射线正方向（t > 0）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.geometry import normalize_plane
from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["triangulate", "pixel_to_rays", "undistort_points"]


def undistort_points(pixels: np.ndarray, K: np.ndarray,
                     dist: np.ndarray) -> np.ndarray:
    """对像素坐标去畸变（保持在内参 K 定义的成像模型下）。"""
    import cv2
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pixels, K, dist, P=K)
    return out.reshape(-1, 2)


def pixel_to_rays(pixels: np.ndarray, K: np.ndarray,
                  dist: Optional[np.ndarray] = None) -> np.ndarray:
    """像素坐标 → 相机系单位方向向量（向量化）。

    Parameters
    ----------
    pixels : (N, 2) 像素 (u, v)
    K : (3, 3) 相机内参
    dist : 畸变系数（可选）

    Returns
    -------
    directions : (N, 3) 单位方向
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    pts = undistort_points(pixels, K, dist) if dist is not None else pixels

    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    dirs = homogeneous @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return dirs / norms


def triangulate(centerline: np.ndarray, K: np.ndarray,
                plane: np.ndarray,
                dist: Optional[np.ndarray] = None,
                return_mask: bool = False):
    """把中心线像素还原为相机坐标系三维点（向量化）。

    Parameters
    ----------
    centerline : (N, 2) 亚像素中心点 (u, v)
    K : (3, 3) 相机内参
    plane : (4,) 激光平面 [a, b, c, d]
    dist : 畸变系数（可选，建议提供）
    return_mask : 是否同时返回有效掩码

    Returns
    -------
    points : (M, 3) 相机系三维点
    mask : (N,) bool（当 return_mask=True）
    """
    pixels = np.asarray(centerline, dtype=np.float64).reshape(-1, 2)
    if pixels.size == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return (empty, np.zeros(0, dtype=bool)) if return_mask else empty

    plane = normalize_plane(plane)
    dirs = pixel_to_rays(pixels, K, dist)

    denom = dirs @ plane[:3]                 # (N,)
    numer = plane[3]                         # 相机原点代入: n·0 + d = d

    valid = np.abs(denom) > 1e-9
    t = np.zeros(len(denom))
    t[valid] = -numer / denom[valid]
    valid &= t > 0

    points = np.empty((0, 3), dtype=np.float64)
    if np.any(valid):
        points = dirs[valid] * t[valid, None]

    if return_mask:
        return points, valid
    return points
