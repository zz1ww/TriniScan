"""激光平面标定。

目标
----
求激光光平面在**相机坐标系**下的方程::

    a*x + b*y + c*z + d = 0

这是把「激光中心线像素 + 相机射线」还原为三维点的关键，其精度直接决定
整个系统的深度精度，因此本项目把它列为**最关键标定**。

标定原理（棋盘格多姿态法）
--------------------------
1. 把标定棋盘格置于激光光路中，使激光线投射在棋盘格平面上；
2. 相机同时拍到**棋盘格**与**激光线**；
3. 由棋盘格角点 + 相机内参 ``solvePnP`` 求得棋盘格平面在相机系的位姿，
   从而得到该平面方程；
4. 提取激光中心线像素，反投影为射线，与棋盘格平面求交，
   得到一组位于**激光平面上**的三维点；
5. 更换棋盘格位姿重复若干次（**至少 2 个不同平面姿态**，建议 >= 4），
   汇集所有三维点，最小二乘（SVD）拟合出激光平面。

为什么需要多个不同姿态
----------------------
若只用一个平面姿态，所有采样点几乎共面，无法约束激光平面的法向。
多个不同倾角的平面才能充分约束。

质量评估
--------
- **拟合残差**：所有点到拟合平面的 RMS 距离，越小越好；
- **视角分布**：各姿态贡献的点数应均衡；
- 建议残差 < 0.5 mm，否则检查棋盘格检测、中心线提取或内参。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from ..common.geometry import (
    fit_plane_svd,
    intersect_ray_plane,
    normalize,
    normalize_plane,
    plane_from_point_normal,
    point_plane_signed_distance,
)
from ..common.io_utils import read_image, save_calibration
from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "LaserCalibResult",
    "calibrate_laser_plane",
    "laser_plane_from_chessboard",
    "pixel_rays",
    "load_laser_calib",
    "save_laser_calib",
]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class LaserCalibResult:
    """激光平面标定结果。"""

    plane: np.ndarray                        # (4,) [a, b, c, d] 单位法向
    residual_rms: float                      # 拟合残差 RMS (m)
    num_points: int                          # 参与拟合的点数
    num_views: int                           # 有效视角数
    view_point_counts: list[int] = field(default_factory=list)

    @property
    def normal(self) -> np.ndarray:
        return self.plane[:3].copy()

    @property
    def offset(self) -> float:
        return float(self.plane[3])


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def laser_plane_from_chessboard(rvec: np.ndarray,
                                tvec: np.ndarray) -> np.ndarray:
    """由棋盘格外参求棋盘格平面在相机系的方程。

    棋盘格自身坐标系为 Z=0 平面，法向为 ``(0,0,1)``。
    外参 ``(R, t)`` 把棋盘格点变到相机系：``X_cam = R X_board + t``。
    因此平面法向为 ``R @ (0,0,1)``，过点 ``t``。
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    normal_board = np.array([0.0, 0.0, 1.0])
    normal_cam = R @ normal_board
    point_cam = np.asarray(tvec, dtype=np.float64).reshape(3)
    return plane_from_point_normal(point_cam, normal_cam)


def pixel_rays(pixels: np.ndarray, K: np.ndarray,
               dist: Optional[np.ndarray] = None) -> np.ndarray:
    """像素坐标 → 相机系单位方向向量（可选先去畸变）。

    Parameters
    ----------
    pixels : (N, 2) 像素 (u, v)
    K : (3, 3) 相机内参
    dist : 畸变系数（可选）

    Returns
    -------
    directions : (N, 3) 单位方向（相机系）
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    pts = pixels
    if dist is not None:
        pts = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), K, dist, P=K
        ).reshape(-1, 2)

    homogeneous = np.column_stack(
        [pts, np.ones(len(pts), dtype=np.float64)]
    )
    dirs = homogeneous @ np.linalg.inv(K).T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return dirs / norms


# ---------------------------------------------------------------------------
# 主标定流程
# ---------------------------------------------------------------------------
def calibrate_laser_plane(
    pairs: Sequence[dict],
    K: np.ndarray,
    dist: Optional[np.ndarray] = None,
    pattern_cols: int = 9,
    pattern_rows: int = 6,
    square_size_m: float = 0.020,
    min_points_per_view: int = 20,
    out_path: Optional[str] = None,
) -> LaserCalibResult:
    """用棋盘格多姿态法标定激光平面。

    Parameters
    ----------
    pairs : 序列，每项为 dict::

            {
                "image": <str 图像路径> 或 "gray": <np.ndarray 灰度图>,
                "centerline": <(N,2) 中心线像素，可选>,
                "roi": <(x,y,w,h) 可选>,
                "threshold": <float 可选>,
            }

        若未提供 ``centerline``，则用默认参数（或 roi/threshold）自动提取。
    K : (3, 3) 相机内参
    dist : 畸变系数（可选，强烈建议提供）
    pattern_cols, pattern_rows : 棋盘格内角点数
    square_size_m : 方格实际尺寸（米）
    min_points_per_view : 单个视角最少有效点数，不足则该视角剔除
    out_path : 若非空，保存标定结果

    Returns
    -------
    LaserCalibResult
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    objp = _chessboard_object_points(pattern_cols, pattern_rows, square_size_m)
    pattern_size = (pattern_cols, pattern_rows)

    all_points: list[np.ndarray] = []
    view_counts: list[int] = []
    valid_views = 0

    for idx, item in enumerate(pairs):
        gray = _load_gray(item)
        if gray is None:
            log.warning("视角 %d: 无法读取图像，跳过", idx)
            continue

        plane_board = _solve_chessboard_plane(
            gray, objp, pattern_size, K, dist
        )
        if plane_board is None:
            log.warning("视角 %d: 棋盘格检测失败，跳过", idx)
            continue

        pixels = _get_centerline(item, gray)
        if pixels is None or len(pixels) < min_points_per_view:
            log.warning("视角 %d: 中心线点不足 (%d)，跳过",
                        idx, 0 if pixels is None else len(pixels))
            continue

        pts = _intersect_pixels_with_plane(pixels, K, dist, plane_board)
        if len(pts) < min_points_per_view:
            log.warning("视角 %d: 与棋盘格平面有效交点不足 (%d)，跳过",
                        idx, len(pts))
            continue

        all_points.append(pts)
        view_counts.append(len(pts))
        valid_views += 1
        log.debug("视角 %d: 贡献 %d 点", idx, len(pts))

    if valid_views < 2:
        raise RuntimeError(
            f"有效视角不足（{valid_views} < 2）。激光平面标定至少需要 "
            f"2 个不同棋盘格姿态，建议 >= 4。"
        )

    points = np.vstack(all_points)
    if len(points) < 3:
        raise RuntimeError(f"标定点不足（{len(points)} < 3）")

    plane = fit_plane_svd(points)
    plane = _orient_plane_toward_camera(plane)

    residual = float(np.sqrt(np.mean(
        point_plane_signed_distance(points, plane) ** 2
    )))

    result = LaserCalibResult(
        plane=plane,
        residual_rms=residual,
        num_points=len(points),
        num_views=valid_views,
        view_point_counts=view_counts,
    )

    log.info(
        "激光平面标定完成: n=(%.4f, %.4f, %.4f), d=%.4f, "
        "RMS=%.4f mm, 点数=%d, 视角=%d",
        plane[0], plane[1], plane[2], plane[3],
        residual * 1e3, len(points), valid_views,
    )
    if residual > 5e-4:
        log.warning("激光平面拟合残差偏大 (%.3f mm)，建议检查标定质量",
                    residual * 1e3)

    if out_path:
        save_laser_calib(out_path, result)
    return result


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------
def _chessboard_object_points(cols: int, rows: int,
                              square: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square
    return objp


def _load_gray(item: dict) -> Optional[np.ndarray]:
    if "gray" in item and item["gray"] is not None:
        g = item["gray"]
        return g if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    path = item.get("image")
    if path is None:
        return None
    img = read_image(path, must_exist=False)
    if img is None:
        return None
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _solve_chessboard_plane(gray: np.ndarray, objp: np.ndarray,
                            pattern_size: tuple[int, int],
                            K: np.ndarray,
                            dist: Optional[np.ndarray]) -> Optional[np.ndarray]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return None

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    ok, rvec, tvec = cv2.solvePnP(
        objp, corners, K,
        dist if dist is not None else np.zeros(5),
    )
    if not ok:
        return None
    return laser_plane_from_chessboard(rvec, tvec)


def _get_centerline(item: dict,
                    gray: np.ndarray) -> Optional[np.ndarray]:
    if item.get("centerline") is not None:
        return np.asarray(item["centerline"], dtype=np.float64).reshape(-1, 2)

    # 延迟导入避免循环依赖
    from ..extraction.centerline import extract_centerline

    class _Cfg:
        def __init__(self, d): self._d = d
        def get(self, k, default=None): return self._d.get(k, default)

    cfg = _Cfg({
        "method": item.get("method", "steger"),
        "threshold": item.get("threshold", 60.0),
        "smooth_sigma": item.get("smooth_sigma", 1.5),
        "steger_sigma": item.get("steger_sigma", 1.5),
        "roi": item.get("roi", None),
    })
    return extract_centerline(gray, cfg)


def _intersect_pixels_with_plane(pixels: np.ndarray, K: np.ndarray,
                                 dist: Optional[np.ndarray],
                                 plane: np.ndarray) -> np.ndarray:
    dirs = pixel_rays(pixels, K, dist)
    origin = np.zeros(3)
    pts = []
    for d in dirs:
        t = intersect_ray_plane(origin, d, plane)
        if t is not None:
            pts.append(origin + t * d)
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def _orient_plane_toward_camera(plane: np.ndarray) -> np.ndarray:
    """统一平面法向符号：使相机原点处到平面的带符号距离为负
    （即法向背离相机、指向场景），保证后续三角测量的 t>0 约定一致。"""
    plane = normalize_plane(plane)
    # 相机原点 (0,0,0) 代入: d = plane[3]
    if plane[3] > 0:
        plane = -plane
    return plane


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------
def save_laser_calib(path: str, result: LaserCalibResult) -> None:
    save_calibration(
        path,
        arrays={
            "plane": result.plane,
            "residual_rms": np.array(result.residual_rms),
            "num_points": np.array(result.num_points),
            "num_views": np.array(result.num_views),
            "view_point_counts": np.array(result.view_point_counts,
                                          dtype=np.int64),
        },
        meta={"type": "laser_plane"},
    )


def load_laser_calib(path: str) -> LaserCalibResult:
    from ..common.io_utils import load_calibration
    arrays, _ = load_calibration(path)
    return LaserCalibResult(
        plane=arrays["plane"],
        residual_rms=float(arrays.get("residual_rms", np.nan)),
        num_points=int(arrays.get("num_points", 0)),
        num_views=int(arrays.get("num_views", 0)),
        view_point_counts=list(
            np.asarray(arrays.get("view_point_counts", [])).tolist()
        ),
    )
