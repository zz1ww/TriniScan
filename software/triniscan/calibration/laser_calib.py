"""激光平面标定。

求激光平面在相机坐标系下的方程:
    a*X + b*Y + c*Z + d = 0

方法（棋盘格多姿态法）:
    1. 让激光线投射到已知位姿的棋盘格平面上；
    2. 求棋盘格平面在相机系下的方程（由 solvePnP 得外参）；
    3. 提取激光中心线像素，与棋盘格平面求交得到相机系三维点；
    4. 多张图收集射线与不同平面的交点（或直接用平面内的点），
       最小二乘拟合激光平面。

简化实现：若已有相机内参，可由像素反投影射线，求射线与
棋盘格平面的交点，作为激光平面上的三维点。
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import cv2
import numpy as np


def _plane_from_chessboard(rvec, tvec) -> np.ndarray:
    """由棋盘格外参得到棋盘格平面在相机系的方程 [a,b,c,d]。"""
    R, _ = cv2.Rodrigues(rvec)
    # 棋盘格平面在自身系为 Z=0, 法向 (0,0,1)
    n = R @ np.array([[0.0], [0.0], [1.0]])
    n = n.flatten()
    d = -float(n @ tvec.flatten())
    return np.array([n[0], n[1], n[2], d])


def _ray_from_pixel(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """像素 → 相机系单位方向向量。"""
    Kinv = np.linalg.inv(K)
    p = Kinv @ np.array([u, v, 1.0])
    return p / np.linalg.norm(p)


def _intersect_ray_plane(o: np.ndarray, d: np.ndarray,
                         plane: np.ndarray) -> Optional[np.ndarray]:
    """射线 o + t*d 与平面 ax+by+cz+d=0 求交。"""
    a, b, c, dd = plane
    denom = a * d[0] + b * d[1] + c * d[2]
    if abs(denom) < 1e-9:
        return None
    t = -(a * o[0] + b * o[1] + c * o[2] + dd) / denom
    if t <= 0:
        return None
    return o + t * d


def calibrate_laser_plane(
    pairs: list[dict],
    K: np.ndarray,
    pattern_cols: int = 9,
    pattern_rows: int = 6,
    square_size_m: float = 0.020,
    out_path: Optional[str] = None,
) -> np.ndarray:
    """标定激光平面。

    Parameters
    ----------
    pairs : list of dict
        每项形如 {"image": path, "centerline": np.ndarray[[u,v],...]}
        或仅 {"image": path}（内部会尝试提取中心线）。
        图像需同时包含棋盘格与激光线，或提供已求好的 centerline。
    K : (3,3) 相机内参

    Returns
    -------
    plane : (4,) [a, b, c, d]
    """
    objp = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows] \
        .T.reshape(-1, 2)
    objp *= square_size_m
    pattern_size = (pattern_cols, pattern_rows)

    all_points = []
    for item in pairs:
        img = cv2.imread(item["image"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ok:
            print(f"[跳过] 棋盘格未找到: {item['image']}")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        okp, rvec, tvec = cv2.solvePnP(objp, corners, K, None)
        if not okp:
            continue
        plane = _plane_from_chessboard(rvec, tvec)

        centerline = item.get("centerline")
        if centerline is None:
            print(f"[警告] 缺少 centerline: {item['image']}")
            continue

        for (u, v) in centerline:
            d = _ray_from_pixel(u, v, K)
            o = np.zeros(3)
            pt = _intersect_ray_plane(o, d, plane)
            if pt is not None:
                all_points.append(pt)

    if len(all_points) < 3:
        raise RuntimeError("激光平面标定点不足（<3）")

    pts = np.array(all_points)
    plane = _fit_plane(pts)
    print(f"激光平面拟合: n=({plane[0]:.4f},{plane[1]:.4f},"
          f"{plane[2]:.4f}), d={plane[3]:.4f}, 点数={len(pts)}")

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)),
                    exist_ok=True)
        np.savez(out_path, plane=plane)
        print(f"已保存: {out_path}")
    return plane


def _fit_plane(pts: np.ndarray) -> np.ndarray:
    """最小二乘拟合平面 [a,b,c,d]，单位法向。"""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    d = -float(normal @ centroid)
    # 约定 d 使平面可正可负，统一符号（法向朝相机/朝下）
    return np.array([normal[0], normal[1], normal[2], d])


def load_laser_calib(path: str) -> np.ndarray:
    return np.load(path)["plane"]


if __name__ == "__main__":
    from triniscan.core.config import Config

    cfg = Config.load()
    K = load_camera_calib(cfg.get("calibration.camera_file"))["K"] \
        if os.path.exists(cfg.get("calibration.camera_file")) else None
    print("激光平面标定需准备 图像+中心线 数据对，见 docs/04_标定方案.md")
