"""几何工具库：射线、平面、旋转、拟合。

坐标系约定
----------
- 相机坐标系：右手系，X 向右，Y 向下，Z 沿光轴向前（OpenCV 约定）。
- 平面表示：``plane = [a, b, c, d]``，满足 ``a*x + b*y + c*z + d = 0``，
  法向量 ``n = [a, b, c]`` 归一化为单位向量。
- 三维点 / 向量：NumPy ``(3,)`` 数组。
- 点集：NumPy ``(N, 3)`` 数组。

本模块为纯函数库，不持有状态，便于单元测试与复用。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "normalize",
    "normalize_plane",
    "plane_from_point_normal",
    "plane_from_three_points",
    "fit_plane_svd",
    "point_plane_signed_distance",
    "intersect_ray_plane",
    "intersect_rays_plane_batch",
    "rodrigues_matrix",
    "transform_about_axis",
    "rigid_transform",
    "apply_transform",
    "fit_circle_3d",
    "decompose_rotation",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 向量 / 平面
# ---------------------------------------------------------------------------
def normalize(vec: np.ndarray) -> np.ndarray:
    """归一化向量（零向量抛错）。"""
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < _EPS:
        raise ValueError("无法归一化零向量")
    return vec / norm


def normalize_plane(plane: np.ndarray) -> np.ndarray:
    """归一化平面，使法向量为单位向量。"""
    plane = np.asarray(plane, dtype=np.float64).reshape(4)
    n = plane[:3]
    norm = np.linalg.norm(n)
    if norm < _EPS:
        raise ValueError("平面法向量为零")
    return plane / norm


def plane_from_point_normal(point: np.ndarray,
                            normal: np.ndarray) -> np.ndarray:
    """由平面上一点与法向量构造平面 ``[a, b, c, d]``。"""
    n = normalize(normal)
    d = -float(n @ np.asarray(point, dtype=np.float64))
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def plane_from_three_points(p0: np.ndarray, p1: np.ndarray,
                            p2: np.ndarray) -> np.ndarray:
    """由不共线三点构造平面。"""
    p0, p1, p2 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2))
    normal = np.cross(p1 - p0, p2 - p0)
    if np.linalg.norm(normal) < _EPS:
        raise ValueError("三点共线，无法确定平面")
    return plane_from_point_normal(p0, normal)


def fit_plane_svd(points: np.ndarray) -> np.ndarray:
    """最小二乘拟合平面（SVD 法）。

    Parameters
    ----------
    points : (N, 3), N >= 3

    Returns
    -------
    plane : (4,) 单位法向 [a, b, c, d]
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        raise ValueError("拟合平面至少需要 3 个点")
    centroid = points.mean(axis=0)
    centered = points - centroid
    # 最小奇异值对应的右奇异向量即为法向
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    d = -float(normal @ centroid)
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


def point_plane_signed_distance(points: np.ndarray,
                                plane: np.ndarray) -> np.ndarray:
    """点到平面的带符号距离（沿单位法向）。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    plane = normalize_plane(plane)
    return points @ plane[:3] + plane[3]


# ---------------------------------------------------------------------------
# 射线-平面求交
# ---------------------------------------------------------------------------
def intersect_ray_plane(origin: np.ndarray, direction: np.ndarray,
                        plane: np.ndarray,
                        require_forward: bool = True) -> Optional[float]:
    """射线 ``origin + t * direction`` 与平面求交。

    Parameters
    ----------
    origin : (3,) 射线起点
    direction : (3,) 射线方向（无需单位化，函数内部会归一化）
    plane : (4,) 平面
    require_forward : 仅返回 t > 0 的解

    Returns
    -------
    t : 交点参数；平行或（require_forward 时）在背后则返回 None
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = normalize(direction)
    plane = normalize_plane(plane)

    denom = float(plane[:3] @ direction)
    if abs(denom) < 1e-9:
        return None  # 射线与平面平行
    t = -float(plane[:3] @ origin + plane[3]) / denom
    if require_forward and t <= 0:
        return None
    return t


def intersect_rays_plane_batch(origins: np.ndarray, directions: np.ndarray,
                               plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """批量射线-平面求交（向量化）。

    Parameters
    ----------
    origins : (N, 3)
    directions : (N, 3)（未归一化亦可）
    plane : (4,)

    Returns
    -------
    points : (M, 3) 有效交点
    valid : (N,) bool 掩码
    """
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    plane = normalize_plane(plane)

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < _EPS] = 1.0
    dirs = directions / norms

    denom = dirs @ plane[:3]                       # (N,)
    numer = origins @ plane[:3] + plane[3]         # (N,)

    valid = np.abs(denom) > 1e-9
    t = np.zeros(len(denom), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t[valid] = -numer[valid] / denom[valid]
    valid &= t > 0
    t = t[valid]

    points = origins[valid] + t[:, None] * dirs[valid]
    return points, valid


# ---------------------------------------------------------------------------
# 旋转 / 刚体变换
# ---------------------------------------------------------------------------
def rodrigues_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """罗德里格斯公式：绕单位轴旋转的 3x3 旋转矩阵。"""
    k = normalize(axis)
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle_rad) * K + \
        (1.0 - np.cos(angle_rad)) * (K @ K)


def rigid_transform(rotation: np.ndarray,
                    translation: np.ndarray) -> np.ndarray:
    """由旋转与平移构造 4x4 齐次变换矩阵。"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def transform_about_axis(axis_point: np.ndarray, axis_dir: np.ndarray,
                         angle_rad: float) -> np.ndarray:
    """绕"过 axis_point、方向 axis_dir"的轴旋转 angle_rad 的 4x4 变换。

    变换顺序：平移到原点 → 旋转 → 平移回去。
    """
    axis_point = np.asarray(axis_point, dtype=np.float64).reshape(3)
    R = rodrigues_matrix(axis_dir, angle_rad)
    t = axis_point - R @ axis_point
    return rigid_transform(R, t)


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """对 (N, 3) 点集应用 4x4 变换。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points.copy()
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    return (transform @ homogeneous.T).T[:, :3]


def decompose_rotation(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """把旋转矩阵分解为 ``(旋转轴, 旋转角)``。"""
    R = np.asarray(rotation, dtype=np.float64)
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if abs(angle) < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(angle - np.pi) < 1e-6:
        # 180° 的奇异情形：由对角元素恢复轴
        diag = np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None)
        axis = np.sqrt(diag)
        # 修正符号
        axis[1] = np.copysign(axis[1], R[0, 1])
        axis[2] = np.copysign(axis[2], R[0, 2])
        return normalize(axis), angle
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return normalize(axis), angle


# ---------------------------------------------------------------------------
# 圆拟合
# ---------------------------------------------------------------------------
def fit_circle_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """拟合三维点所在空间圆。

    Parameters
    ----------
    points : (N, 3), N >= 3

    Returns
    -------
    center : (3,) 圆心
    normal : (3,) 圆面单位法向（即旋转轴方向）
    radius : float 半径
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        raise ValueError("拟合圆至少需要 3 个点")

    centroid = points.mean(axis=0)
    centered = points - centroid
    # 圆所在平面：由前两个主成分张成，法向为最小奇异向量
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    u, v = vh[0], vh[1]

    uv = centered @ np.column_stack([u, v])          # 投影到平面内 2D
    A = np.column_stack([2.0 * uv[:, 0], 2.0 * uv[:, 1],
                         np.ones(len(uv))])
    b = (uv ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    radius = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    center = centroid + cx * u + cy * v
    return center, normal, radius
