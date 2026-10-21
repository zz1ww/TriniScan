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

from ...common.geometry import (
    intersect_ray_plane,
    normalize_plane,
    plane_from_point_normal,
    point_plane_signed_distance,
)
from ...common.io_utils import read_image, save_calibration
from ...common.logging_utils import get_logger
from ...common.robust import robust_plane_fit

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
    residual_rms: float                      # 稳健拟合残差 RMS (m)
    num_points: int                          # 参与拟合的点数
    num_views: int                           # 有效视角数
    view_point_counts: list[int] = field(default_factory=list)
    # --- 鲁棒拟合附加信息（向后兼容，旧文件加载时取默认值）---
    per_view_rms: list[float] = field(default_factory=list)   # 每视角残差(m)
    rejected_views: list[int] = field(default_factory=list)   # 被剔除视角
    num_inliers: int = 0                     # 稳健内点数
    outlier_ratio: float = 0.0               # 离群点比例
    method: str = "svd"                      # 使用的拟合方法

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
    robust: bool = True,
    robust_method: str = "irls",
    view_reject_mm: Optional[float] = 1.0,
    outlier_mm: Optional[float] = 0.5,
    min_views: int = 2,
    clip_margin_px: float = 0.0,
    channel: str = "gray",
) -> LaserCalibResult:
    """用棋盘格多姿态法标定激光平面（含稳健估计与输入校验）。

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
    robust : 是否用稳健拟合（IRLS/RANSAC）替代普通最小二乘
    robust_method : ``"irls"`` 或 ``"ransac"``（``robust=False`` 时忽略）
    view_reject_mm : 单视角残差中位数超过此值则剔除该视角（mm；None 关闭）
    outlier_mm : 稳健拟合的离群距离阈值（mm；None 由尺度自适应）
    min_views : 可接受的最少有效视角数
    clip_margin_px : 把激光中心线裁剪到棋盘格角点凸包内时允许的外扩余量
        （像素）。默认 0 表示严格限制在凸包内；激光线若穿过棋盘格边缘
        可适当放大（如 10~30），以保留贴边处的有效点。
    channel : 中心线提取通道。

        - ``"red_laser"``（彩色相机推荐）：``R - max(G, B)``，压制棋盘格
          灰白背景、木框与环境光；
        - ``"gray"``（黑白相机）：普通灰度。
        - 若图像本身是单通道，``preprocess`` 会**自动降级**为 gray，
          因此彩色配置用于黑白相机也不会报错。可被每个 ``pairs`` 元素里
          的 ``channel`` 键覆盖。

    Returns
    -------
    LaserCalibResult
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    default_channel = ("gray" if channel is None else str(channel)) or "gray"
    _validate_inputs(pattern_cols, pattern_rows, square_size_m,
                     min_points_per_view, min_views)
    objp = _chessboard_object_points(pattern_cols, pattern_rows, square_size_m)
    pattern_size = (pattern_cols, pattern_rows)

    per_view_points: list[np.ndarray] = []
    per_view_index: list[int] = []
    view_counts: list[int] = []

    for idx, item in enumerate(pairs):
        gray = _load_gray(item)
        if gray is None:
            log.warning("视角 %d: 无法读取图像，跳过", idx)
            continue

        board = _solve_chessboard_plane(
            gray, objp, pattern_size, K, dist
        )
        if board is None:
            log.warning("视角 %d: 棋盘格检测失败，跳过", idx)
            continue
        plane_board, board_corners = board

        pixels = _get_centerline(item, _load_image(item), default_channel)
        if pixels is None or len(pixels) < min_points_per_view:
            log.warning("视角 %d: 中心线点不足 (%d)，跳过",
                        idx, 0 if pixels is None else len(pixels))
            continue

        # 裁剪到棋盘格范围内：丢弃落在棋盘格平面之外（打到木框/手/
        # 背景）的激光线像素——它们是残差偏大的主因。
        n_before = len(pixels)
        pixels = _clip_centerline_to_board(
            pixels, board_corners, margin_px=clip_margin_px)
        n_after = len(pixels)
        if n_after < min_points_per_view:
            log.warning("视角 %d: 裁剪到棋盘格内后中心线点不足 "
                        "(%d -> %d)，跳过", idx, n_before, n_after)
            continue
        log.debug("视角 %d: 中心线 %d -> %d 点（棋盘格内）",
                  idx, n_before, n_after)

        pts = _intersect_pixels_with_plane(pixels, K, dist, plane_board)
        if len(pts) < min_points_per_view:
            log.warning("视角 %d: 与棋盘格平面有效交点不足 (%d)，跳过",
                        idx, len(pts))
            continue

        per_view_points.append(pts)
        per_view_index.append(idx)
        view_counts.append(len(pts))
        log.debug("视角 %d: 贡献 %d 点", idx, len(pts))

    if len(per_view_points) < min_views:
        raise RuntimeError(
            f"有效视角不足（{len(per_view_points)} < {min_views}）。"
            f"激光平面标定至少需要 {min_views} 个不同棋盘格姿态，"
            f"建议 >= 4。"
        )

    # --- 一阶：全局稳健拟合，得到初版平面 ---
    initial_points = np.vstack(per_view_points)
    if len(initial_points) < 3:
        raise RuntimeError(f"标定点不足（{len(initial_points)} < 3）")

    reject_thr = (view_reject_mm * 1e-3
                  if view_reject_mm is not None else None)

    # --- 二：按单视角残差剔除劣质视角（用初版平面评估）---
    plane0 = _fit_plane(initial_points, robust, robust_method, outlier_mm)
    kept_points: list[np.ndarray] = []
    kept_index: list[int] = []
    kept_counts: list[int] = []
    per_view_rms_all: list[float] = []
    rejected: list[int] = []

    for pts, idx, cnt in zip(per_view_points, per_view_index, view_counts):
        rms = float(np.sqrt(np.mean(
            point_plane_signed_distance(pts, plane0) ** 2)))
        per_view_rms_all.append(rms)
        if reject_thr is not None and rms > reject_thr:
            rejected.append(idx)
            log.warning("视角 %d: 残差 %.3f mm 超限（> %.3f mm），剔除",
                        idx, rms * 1e3, view_reject_mm)
            continue
        kept_points.append(pts)
        kept_index.append(idx)
        kept_counts.append(cnt)

    if len(kept_points) < min_views:
        log.warning("剔除劣质视角后有效视角不足（%d < %d），"
                    "回退为使用全部视角", len(kept_points), min_views)
        kept_points, kept_index, kept_counts = (
            per_view_points, per_view_index, view_counts)
        per_view_rms_all = [float(np.sqrt(np.mean(
            point_plane_signed_distance(p, plane0) ** 2)))
            for p in kept_points]
        rejected = []

    # --- 三：最终拟合（稳健）---
    points = np.vstack(kept_points)
    fit = robust_plane_fit(
        points,
        method=(robust_method if robust else "svd"),
        outlier_threshold=(outlier_mm * 1e-3
                           if outlier_mm is not None else None),
    )
    plane = _orient_plane_toward_camera(fit.plane)

    # 最终残差：优先报告稳健内点 RMS，同时给出全体 RMS
    residual = float(fit.residual_rms)
    residual_all = float(np.sqrt(np.mean(
        point_plane_signed_distance(points, plane) ** 2)))

    result = LaserCalibResult(
        plane=plane,
        residual_rms=residual,
        num_points=len(points),
        num_views=len(kept_points),
        view_point_counts=kept_counts,
        per_view_rms=per_view_rms_all,
        rejected_views=rejected,
        num_inliers=int(fit.num_inliers),
        outlier_ratio=float(fit.outlier_ratio),
        method=fit.method,
    )

    log.info(
        "激光平面标定完成: n=(%.4f, %.4f, %.4f), d=%.4f, "
        "RMS=%.4f mm（全体 %.4f mm）, 内点 %d/%d, 方法=%s, "
        "视角=%d（剔除 %d）, 点数=%d",
        plane[0], plane[1], plane[2], plane[3],
        residual * 1e3, residual_all * 1e3,
        result.num_inliers, result.num_points, result.method,
        result.num_views, len(rejected), len(points),
    )
    if residual > 0.5e-3:
        log.warning("激光平面拟合残差偏大 (%.3f mm)，建议检查标定质量",
                    residual * 1e3)

    if out_path:
        save_laser_calib(out_path, result)
    return result


def _validate_inputs(cols: int, rows: int, square_size_m: float,
                     min_points_per_view: int, min_views: int) -> None:
    """对形参与靶标参数做合理性校验（尽早暴露"16 mm 事故"类问题）。"""
    if cols < 3 or rows < 3:
        raise ValueError(f"棋盘格内角点数过小: {cols}x{rows}（应 >= 3x3）")
    if square_size_m <= 0:
        raise ValueError(f"方格尺寸必须为正: {square_size_m}")
    if not (1e-4 <= square_size_m <= 0.2):
        log.warning("方格尺寸 %.4f m 超出常见范围（0.1~200 mm），"
                    "请确认单位与实测一致", square_size_m)
    if min_points_per_view < 3:
        raise ValueError("min_points_per_view 应 >= 3")
    if min_views < 2:
        raise ValueError("min_views 应 >= 2")


def _fit_plane(points: np.ndarray, robust: bool, robust_method: str,
               outlier_mm: Optional[float]) -> np.ndarray:
    """便捷：按开关拟合并返回单位法向平面。"""
    fit = robust_plane_fit(
        points,
        method=(robust_method if robust else "svd"),
        outlier_threshold=(outlier_mm * 1e-3
                           if outlier_mm is not None else None),
    )
    return _orient_plane_toward_camera(fit.plane)


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------
def _chessboard_object_points(cols: int, rows: int,
                              square: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square
    return objp


def _pick_image(item: dict) -> Optional[np.ndarray]:
    """取出**原始图像**（尽量保留 BGR 彩色）。

    支持的键（按优先级）：

    - ``"bgr"`` / ``"image_arr"`` : 原始彩色/灰度数组（推荐，彩色相机用）；
    - ``"image"`` : 图像路径（按 IMREAD_COLOR 读入，保留颜色）；
    - ``"gray"`` : 灰度数组（兼容旧调用；此时无法恢复颜色）。
    """
    for key in ("bgr", "image_arr", "gray"):
        if key in item and item[key] is not None:
            return item[key]
    path = item.get("image")
    if path is None:
        return None
    return read_image(path, flags=cv2.IMREAD_COLOR, must_exist=False)


def _load_gray(item: dict) -> Optional[np.ndarray]:
    """取得**灰度图**（供棋盘格检测用）。

    彩色输入按 BGR→GRAY 转换；灰度输入原样返回。
    """
    img = _pick_image(item)
    if img is None:
        return None
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _load_image(item: dict) -> Optional[np.ndarray]:
    """读取**原始图像**（尽可能保留 BGR 颜色），供颜色通道分离的中心线提取。

    棋盘格检测仍需灰度图（另由 :func:`_load_gray` 提供）。

    Notes
    -----
    - 传入彩色图 → 原样返回（3 通道），``red_laser`` 等颜色差分可用；
    - 传入灰度图 → 转成 3 通道 BGR（各通道相同），此时颜色差分等价于
      灰度，``preprocess`` 会把 ``red_laser`` **自动降级为 gray**，
      黑白相机走这条路径。
    """
    img = _pick_image(item)
    if img is None:
        return None
    if img.ndim == 2:                       # 灰度 → 伪 BGR（单色相机）
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _solve_chessboard_plane(gray: np.ndarray, objp: np.ndarray,
                            pattern_size: tuple[int, int],
                            K: np.ndarray,
                            dist: Optional[np.ndarray]
                            ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """检测棋盘格并求解其平面位姿。

    Returns
    -------
    (plane, corners) 或 None
        ``plane`` 为 (4,) 平面方程；``corners`` 为 (N,2) 亚像素角点像素
        （用于把激光中心线裁剪到棋盘格范围内）。
    """
    # 复用相机标定里的角点检测器：优先 SB（sector-based）检测器，
    # 它对光照不均、轻微模糊、低对比度远优于经典 findChessboardCorners。
    # 本项目实测：13 张激光标定图里经典检测器只检出 8 张，SB 检出 13 张；
    # 漏检的那几张恰恰是残差最大的视角，被丢弃后剩余视角残差反而变大
    # （表现为"剔除劣质视角后有效视角不足"）。
    from .camera_calib import _detect_corners

    corners = _detect_corners(gray, pattern_size, use_sb=True)
    if corners is None:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        objp, corners, K,
        dist if dist is not None else np.zeros(5),
    )
    if not ok:
        return None
    plane = laser_plane_from_chessboard(rvec, tvec)
    return plane, corners.reshape(-1, 2).astype(np.float64)


def _clip_centerline_to_board(pixels: np.ndarray, corners: np.ndarray,
                              margin_px: float = 0.0
                              ) -> np.ndarray:
    """把中心线像素裁剪到棋盘格角点凸包内。

    激光线常贯穿整幅图像（延伸到木框、手、背景等），这些位于棋盘格
    **物理平面之外**的像素若参与反投影求交，会得到严重错误的三维点，
    是激光平面标定残差偏大的主因。这里用角点凸包（可选外扩
    ``margin_px`` 像素）只保留落在棋盘格范围内的中心线点。

    Parameters
    ----------
    pixels : (N,2) 中心线像素 (x, y)
    corners : (M,2) 棋盘格角点像素
    margin_px : 凸包外扩余量（像素）；0 表示严格限制在凸包内。

    Returns
    -------
    (K,2) 位于棋盘格内的中心线点（可能为空）。
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pixels.size == 0 or corners.size == 0:
        return pixels.reshape(-1, 2)

    hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)

    if margin_px and margin_px > 0:
        # 以角点中心为基准，把凸包顶点沿径向外扩 margin_px 像素
        center = hull.mean(axis=0)
        vec = hull - center
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        hull = (center + vec * (1.0 + margin_px / norm)).astype(np.float32)

    inside = np.array([
        cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0
        for x, y in pixels
    ], dtype=bool)
    return pixels[inside]


def _get_centerline(item: dict,
                    image: np.ndarray,
                    default_channel: str = "red_laser"
                    ) -> Optional[np.ndarray]:
    """从标定图提取激光中心线。

    Parameters
    ----------
    item : 视角描述 dict；可含 ``centerline``（直接给定）、``method``、
        ``threshold``、``smooth_sigma``、``steger_sigma``、``roi``、``channel``。
    image : 原始图像（保留 BGR 时颜色差分可用）。
    default_channel : 未显式指定 ``channel`` 时使用的提取通道。

        - 彩色相机（默认）用 ``"red_laser"``（``R - max(G,B)``），可压制
          棋盘格灰白背景与木框等环境光干扰；
        - 黑白相机请设为 ``"gray"``。若图像本身是单通道，``preprocess``
          会**自动降级**为 gray，因此即便这里给了 ``red_laser`` 也不会
          报错，只是等价于 gray。
    """
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
        # 通道随相机类型而定：彩色相机用颜色差分（压制棋盘格/背景），
        # 黑白相机在 preprocess 内自动降级为 gray（见 centerline.preprocess）。
        "channel": item.get("channel", default_channel),
    })
    return extract_centerline(image, cfg)


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
            "per_view_rms": np.array(result.per_view_rms, dtype=np.float64),
            "rejected_views": np.array(result.rejected_views, dtype=np.int64),
            "num_inliers": np.array(result.num_inliers),
            "outlier_ratio": np.array(result.outlier_ratio),
        },
        meta={"type": "laser_plane", "method": result.method},
    )


def load_laser_calib(path: str) -> LaserCalibResult:
    from ...common.io_utils import load_calibration
    arrays, meta = load_calibration(path)
    return LaserCalibResult(
        plane=arrays["plane"],
        residual_rms=float(arrays.get("residual_rms", np.nan)),
        num_points=int(arrays.get("num_points", 0)),
        num_views=int(arrays.get("num_views", 0)),
        view_point_counts=list(
            np.asarray(arrays.get("view_point_counts", [])).tolist()
        ),
        per_view_rms=list(
            np.asarray(arrays.get("per_view_rms", [])).tolist()
        ),
        rejected_views=[int(v) for v in
                        np.asarray(arrays.get("rejected_views", [])).tolist()],
        num_inliers=int(arrays.get("num_inliers", 0)),
        outlier_ratio=float(arrays.get("outlier_ratio", 0.0)),
        method=str(meta.get("method", "svd")),
    )


# ----------------------------------------------------------------------
# 双图差分（无激光 / 有激光）
# ----------------------------------------------------------------------
@dataclass
class DiffPairResult:
    """一对（无激光, 有激光）图像的差分结果。

    Attributes
    ----------
    ok : 是否可用于标定
    reason : ``ok=False`` 时的原因（中文，可直接显示给用户）
    diff : 差分灰度图（``uint8``，``lit - dark`` 截负），``ok=False`` 时为 None
    changed_ratio : 变化像素占比
    bright_ratio : 差分后亮区（超过阈值）像素占比
    """

    ok: bool
    reason: str = ""
    diff: Optional[np.ndarray] = None
    changed_ratio: float = 0.0
    bright_ratio: float = 0.0


def to_gray_image(image: np.ndarray) -> np.ndarray:
    """把任意输入图统一成 ``uint8`` 单通道灰度。"""
    if image is None:
        raise ValueError("图像为空")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def diff_laser_images(dark: np.ndarray, lit: np.ndarray, *,
                      threshold: float = 60.0,
                      brightness_drift_max: float = 0.02,
                      same_ratio_max: float = 0.001,
                      board_shift_px: float = 1.5,
                      board_shift_ratio_max: float = 0.01,
                      bright_ratio_max: float = 0.05,
                      pattern_cols: int = 11,
                      pattern_rows: int = 8,
                      use_sb: bool = True) -> DiffPairResult:
    """对一对「关激光 / 开激光」图像做差分，并校验配对是否可信。

    差分的**前提**是「除激光以外画面完全没变」。只做 ``lit - dark`` 而不
    校验，会在下面几种情况下静默产生错误标定：

    - 两张图其实是同一次拍摄（用户重复按了快门）；
    - 拍两张之间手碰了棋盘格；
    - 开关激光时环境光/曝光也变了。

    因此这里除了计算差分，还逐条校验，任何一条不满足即返回
    ``ok=False`` 并给出原因。

    Parameters
    ----------
    dark, lit : 无激光图、有激光图
    threshold : 差分阈值，``diff > threshold`` 视为激光
    brightness_drift_max : 两帧**平均亮度**相对差上限。真配对只改激光，
        平均亮度几乎不变；自动曝光重新收敛会造成明显漂移
    same_ratio_max : 变化像素占比低于此值判为「两张几乎相同」
    board_shift_px : 棋盘格角点平均位移超过此值判为「板动了」
    board_shift_ratio_max : 角点位移/格宽 超过此比例也判为「板动了」
    bright_ratio_max : 差分亮区占比超过此值判为「光源整体变了」
    pattern_cols, pattern_rows : 棋盘内角点，用于检测板是否移动
    use_sb : 是否优先用 ``findChessboardCornersSB``

    Returns
    -------
    DiffPairResult
    """
    if dark is None or lit is None:
        return DiffPairResult(False, "图像为空")

    d = to_gray_image(dark)
    l = to_gray_image(lit)
    if d.shape != l.shape:
        return DiffPairResult(
            False,
            f"尺寸不一致（{d.shape[1]}x{d.shape[0]} vs "
            f"{l.shape[1]}x{l.shape[0]}），请确认分辨率相同")

    h, w = d.shape
    total = float(h * w)

    # --- 1. 差分 ---
    diff = np.clip(l.astype(np.int16) - d.astype(np.int16),
                   0, 255).astype(np.uint8)

    changed_ratio = float(np.count_nonzero(diff) / total)
    bright_ratio = float(np.count_nonzero(diff > threshold) / total)

    # --- 2. 全局亮度漂移：自动曝光的指纹 ---
    # 真配对只改激光，两帧平均亮度应几乎一致；若相机自动曝光在两次
    # 快门之间重新收敛，两帧整体亮度会系统性偏移，差分图变成一片
    # 弥散的"半透明残影"，看起来像低对比度原图而不是一条细线。
    mean_dark = float(d.mean())
    drift = abs(float(l.mean()) - mean_dark)
    if mean_dark > 1e-6 and drift / mean_dark > brightness_drift_max:
        return DiffPairResult(
            False,
            f"两帧整体亮度差 {drift:.1f}（{drift / mean_dark * 100:.1f}%，"
            f"阈值 {brightness_drift_max * 100:.0f}%），说明相机自动曝光"
            "在两次快门之间发生了变化。差分只剩弥散残影，无法定位激光；"
            "请关闭自动曝光、锁定参数后再拍",
            diff=diff, changed_ratio=changed_ratio,
            bright_ratio=bright_ratio)

    if changed_ratio < same_ratio_max:
        return DiffPairResult(
            False,
            f"两张图几乎相同（仅 {changed_ratio * 100:.3f}% 像素不同），"
            "疑似重复拍摄：拍第 2 张前请确认激光已开关",
            diff=diff, changed_ratio=changed_ratio,
            bright_ratio=bright_ratio)

    if bright_ratio > bright_ratio_max:
        return DiffPairResult(
            False,
            f"差分亮区占 {bright_ratio * 100:.1f}% "
            f"(> {bright_ratio_max * 100:.0f}%)，远超一条激光线的面积。"
            "常见原因：拍两张之间移动了棋盘格/手，或开关激光时环境光、"
            "曝光也变了。请锁定曝光并保持画面静止后重拍",
            diff=diff, changed_ratio=changed_ratio,
            bright_ratio=bright_ratio)

    # --- 3. 棋盘格是否在两帧间移动（检测得到才校验）---
    # 此处**不**因检测失败而否决：板离得远/激光太弱时检测不到属正常，
    # 而后续 calibrate_laser_plane 本身还会再判一次棋盘。把"检不到"
    # 当成致命错误会让标定在可用数据上也无法进行。
    shift, _reason = _board_shift_px(
        d, l, (pattern_cols, pattern_rows), use_sb)
    if shift is not None and shift > board_shift_px:
        return DiffPairResult(
            False,
            f"棋盘格在两帧之间移动了 {shift:.1f}px（阈值 "
            f"{board_shift_px:.1f}px），差分会把板边缘误判为激光；"
            "请拍两张时保持棋盘格不动",
            diff=diff, changed_ratio=changed_ratio,
            bright_ratio=bright_ratio)

    return DiffPairResult(
        True, "", diff=diff, changed_ratio=changed_ratio,
        bright_ratio=bright_ratio)


def _board_shift_px(dark: np.ndarray, lit: np.ndarray, pattern, use_sb: bool):
    """返回两张图中棋盘角点的平均位移（像素）；检测不到返回 (None, 原因)。"""
    def _corners(img):
        if use_sb and hasattr(cv2, "findChessboardCornersSB"):
            flags = (cv2.CALIB_CB_EXHAUSTIVE
                     + cv2.CALIB_CB_ACCURACY)
            ok, c = cv2.findChessboardCornersSB(img, pattern, flags=flags)
            if ok:
                return c.reshape(-1, 2)
        ok, c = cv2.findChessboardCorners(
            img, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
        return c.reshape(-1, 2).astype(np.float64)

    c_dark = _corners(dark)
    c_lit = _corners(lit)
    if c_dark is None or c_lit is None:
        which = "无激光图" if c_dark is None else "有激光图"
        return None, f"{which}检测不到棋盘格，无法校验配对（请检查角点设置）"
    if c_dark.shape != c_lit.shape:
        return None, "两张图检测到的角点数不一致"
    return float(np.linalg.norm(c_dark - c_lit, axis=1).mean()), ""


def extract_centerline_from_diff(diff: np.ndarray, *,
                                 threshold: float = 60.0,
                                 min_px: int = 20,
                                 max_halfwidth: float = 7.0
                                 ) -> np.ndarray:
    """从差分图提取激光中心线，返回逐列中心点 ``(N, 2)``。

    差分图里只剩激光（以及少量残余噪声），因此可以比常规提取更粗暴，
    但仍需避开两个坑：

    **不要只取最大连通域**：激光穿过黑白格时两侧亮度差很大，线会在
    黑白交界处断开（实测一张图可碎成 30 多段），取最大域会丢掉大半。

    **不要用「竖直」判据**：激光斜穿棋盘时，同一段碎片的 x 跨度很大，
    ``h > 3w`` 会把它们误杀。正确判据是「细」——用距离变换求每个
    连通域的最大内切半径，激光线通常 < 6 像素，而白纸/手/木框等
    块状物远大于此。该判据与方向无关。

    Parameters
    ----------
    diff : 差分图（``uint8``）
    threshold : 亮度阈值
    min_px : 连通域最少像素数
    max_halfwidth : 最大内切半径阈值（像素），超过视为块状物丢弃

    Returns
    -------
    (N, 2) float64 数组，每行 ``(x, y_subpixel)``；无线时返回空数组。
    """
    if diff is None or diff.size == 0:
        return np.zeros((0, 2))

    mask = (diff > threshold).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_px:
            continue
        sub = (lbl[y:y + h, x:x + w] == i).astype(np.uint8)
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        if dist.max() > max_halfwidth:
            continue
        keep[y:y + h, x:x + w] |= (sub > 0)

    ys, xs = np.where(keep)
    if xs.size == 0:
        return np.zeros((0, 2))

    # 逐列按灰度加权求亚像素 y
    cols: dict = {}
    for x, y in zip(xs, ys):
        cols.setdefault(int(x), []).append(int(y))

    pts = []
    for x in sorted(cols):
        yy = np.array(cols[x], dtype=np.int64)
        w = diff[yy, x].astype(np.float64)
        s = w.sum()
        if s <= 0:
            continue
        pts.append((float(x), float((yy * w).sum() / s)))
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))
