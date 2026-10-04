"""三角测量：像素点 + 相机内参 + 激光平面 → 相机系三维点。"""
from __future__ import annotations

import numpy as np


def _ray_from_pixel(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """像素 → 相机系单位方向向量。"""
    Kinv = np.linalg.inv(K)
    p = Kinv @ np.array([u, v, 1.0])
    return p / np.linalg.norm(p)


def _intersect_ray_plane(o: np.ndarray, d: np.ndarray,
                         plane: np.ndarray) -> float | None:
    """射线与平面交点参数 t（返回 None 表示平行/反向）。"""
    a, b, c, dd = plane
    denom = a * d[0] + b * d[1] + c * d[2]
    if abs(denom) < 1e-9:
        return None
    t = -(a * o[0] + b * o[1] + c * o[2] + dd) / denom
    if t <= 0:
        return None
    return float(t)


def triangulate(centerline: np.ndarray, K: np.ndarray,
                plane: np.ndarray,
                dist: np.ndarray | None = None) -> np.ndarray:
    """把中心线像素点还原为相机坐标系三维点。

    Parameters
    ----------
    centerline : (N, 2) 亚像素中心点 (u, v)
    K : (3, 3) 相机内参
    plane : (4,) 激光平面 [a, b, c, d]
    dist : 畸变系数（可选，先去畸变）

    Returns
    -------
    points : (M, 3) 相机系三维点
    """
    pts = np.asarray(centerline, dtype=np.float64)
    if dist is not None and len(pts) > 0:
        pts = cv2_undistort_points(pts, K, dist)

    origin = np.zeros(3)
    out = []
    for (u, v) in pts:
        d = _ray_from_pixel(u, v, K)
        t = _intersect_ray_plane(origin, d, plane)
        if t is None:
            continue
        out.append(origin + t * d)
    return np.array(out, dtype=np.float64)


def cv2_undistort_points(pts: np.ndarray, K: np.ndarray,
                         dist: np.ndarray) -> np.ndarray:
    import cv2
    p = pts.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(p, K, dist, P=K)
    return und.reshape(-1, 2)


if __name__ == "__main__":
    # 自测：构造虚拟数据
    K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], float)
    plane = np.array([0.0, 0.7, 0.7, -300.0])  # 一个平面
    cl = np.array([[900, 500], [960, 540], [1020, 580]], float)
    pts = triangulate(cl, K, plane)
    print("三维点:\n", pts)
