"""转台旋转轴标定。

通过在每个转台角度提取标记物特征点，拟合旋转圆，
得到旋转轴在相机坐标系下的位置 point 与方向 direction。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def fit_circle_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """拟合三维点所在圆，返回 (圆心, 法向, 半径)。"""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    # 建立平面内正交基
    u = vh[0]
    v = vh[1]
    uv = centered @ np.column_stack([u, v])  # (N,2)

    # 二维圆拟合 (代数法)
    A = np.column_stack([2 * uv[:, 0], 2 * uv[:, 1],
                         np.ones(len(uv))])
    b = (uv ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    center = centroid + cx * u + cy * v
    return center, normal, r


def calibrate_axis(
    feature_points: np.ndarray,
    out_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """由绕轴旋转的一组特征点标定旋转轴。

    Parameters
    ----------
    feature_points : (N,3)
        每个转台角度下，标记物特征点在相机系的三维坐标
        （需用三角测量得到）。

    Returns
    -------
    (point, direction) : 轴上一点与单位方向向量
    """
    if len(feature_points) < 3:
        raise RuntimeError("转轴标定点数不足（<3）")
    center, normal, r = fit_circle_3d(np.asarray(feature_points))
    print(f"转轴拟合: center={center}, dir={normal}, r={r:.4f}")

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)),
                    exist_ok=True)
        np.savez(out_path, point=center, direction=normal)
        print(f"已保存: {out_path}")
    return center, normal


def rotation_matrix_about_axis(axis_dir: np.ndarray,
                               angle_rad: float) -> np.ndarray:
    """罗德里格斯公式：绕单位轴方向的旋转矩阵 3x3。"""
    k = axis_dir / np.linalg.norm(axis_dir)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + \
        (1 - np.cos(angle_rad)) * (K @ K)


def transform_about_axis(point: np.ndarray, axis_dir: np.ndarray,
                         angle_rad: float) -> np.ndarray:
    """绕"过 point、方向 axis_dir"的轴旋转 angle_rad 的 4x4 变换。"""
    R = rotation_matrix_about_axis(axis_dir, angle_rad)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = point - R @ point
    return T


def load_axis_calib(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["point"], data["direction"]


if __name__ == "__main__":
    print("转轴标定需先获得各角度下特征点的三维坐标，见 docs/04_标定方案.md")
