"""转轴**现场**标定：放一个小球在转台上，转动即可标定。

与 :mod:`triniscan.calibration.axis_calib` 的关系
-------------------------------------------------
- ``axis_calib.calibrate_axis(points)`` 吃**已经算好的** 3D 特征点，
  保持原样、不做任何修改（离线/自定义流程仍可用）；
- 本模块负责把它**前面那段缺失的流程**补上：
  现场从图像里测出这些 3D 特征点，再调用 ``calibrate_axis``。

原理
----
转台上偏心放一个已知半径的**黑色哑光小球**，转台每转一个角度拍一张：

1. 在图像中找到球上的**同一物理点**（两种方法，见下）；
2. 用相机内参 + 激光平面做**三角测量**得该点的 3D 坐标；
3. 因小球随转台刚性旋转，这些 3D 点落在**同一个空间圆**上；
4. 拟合该圆 → 圆心即轴上一点，圆法向即轴方向（复用 ``fit_circle_3d``）。

两种取点方法
------------
- ``"peak"``   **最亮极点法**：激光打在黑球上形成极亮斑点，取亮度加权
  质心。需**关闭环境光**，背景尽量暗。简单、快。
- ``"sphere"`` **球心法**：在**亮背景**下检测黑色球的轮廓圆，结合已知
  半径反求**球心**三维坐标。球心是完美固定点，精度更高。

注意
----
- 轴的位置误差会随测量半径放大，应尽量精确；
- 采样角度宜覆盖较大范围（如每 30° 共 12 点）以稳定拟合。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "SpotNotFound",
    "detect_spot_peak",
    "detect_sphere_center",
    "sphere_center_from_pixel",
    "AxisLiveCollector",
    "RegionOfInterest",
]


class SpotNotFound(RuntimeError):
    """图像中未找到符合要求的标记点。"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
@dataclass
class RegionOfInterest:
    """像素搜索窗口（u 越小越左，v 越小越上）。"""

    u: int
    v: int
    half: int = 60

    def slice(self, shape: tuple[int, int]) -> tuple[slice, slice]:
        h, w = shape[:2]
        v0 = max(int(self.v) - self.half, 0)
        v1 = min(int(self.v) + self.half + 1, h)
        u0 = max(int(self.u) - self.half, 0)
        u1 = min(int(self.u) + self.half + 1, w)
        return slice(v0, v1), slice(u0, u1)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """转灰度 float32；彩色按 BGR 处理。"""
    if image.ndim == 2:
        gray = image
    else:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.asarray(gray, dtype=np.float32)


def _window(image: np.ndarray, roi: Optional[RegionOfInterest],
            shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """按 ROI 截取，返回 (子图, 行偏移, 列偏移)。"""
    if roi is None:
        return image, 0, 0
    sy, sx = roi.slice(shape)
    return image[sy, sx], sy.start, sx.start


# ---------------------------------------------------------------------------
# 方法一：最亮极点法
# ---------------------------------------------------------------------------
def detect_spot_peak(image: np.ndarray,
                     threshold: float = 60.0,
                     roi: Optional[RegionOfInterest] = None,
                     win_px: int = 25) -> np.ndarray:
    """在（暗背景的）图像中定位激光亮点，返回亚像素 ``(u, v)``。

    思路：激光打在哑光**黑**球上形成全场最亮的斑点。先做背景抑制，
    取显著亮点，再在其邻域内做**亮度平方加权质心**得到亚像素中心。

    Parameters
    ----------
    image : 灰度或 BGR 图像
    threshold : 亮点相对背景的灰度阈值
    roi : 可选搜索窗口（应先粗定位球的大致位置）
    win_px : 精定位窗口半径（像素）

    Raises
    ------
    SpotNotFound : 未找到满足条件的亮点
    """
    gray = _to_gray(image)
    sub, off_y, off_x = _window(gray, roi, gray.shape)
    if sub.size == 0:
        raise SpotNotFound("搜索窗口为空")

    # 背景估计：亮点是小面积高亮，用中位数作背景
    bg = float(np.median(sub))
    sig = sub - bg
    sig[sub < threshold] = 0.0
    if not np.any(sig > 0):
        raise SpotNotFound(
            f"未见亮点（阈值 {threshold:.0f}，背景 {bg:.0f}）")

    # 粗定位：最亮像素
    peak_idx = int(np.argmax(sig))
    py, px = np.unravel_index(peak_idx, sub.shape)

    # 精定位：峰值邻域内做平方加权质心
    h, w = sub.shape
    y0 = max(py - win_px, 0)
    y1 = min(py + win_px + 1, h)
    x0 = max(px - win_px, 0)
    x1 = min(px + win_px + 1, w)
    patch = sig[y0:y1, x0:x1].astype(np.float64)

    # 只保留较亮的像素，抑制漫反射噪声
    cut = 0.5 * patch.max()
    wgt = np.where(patch >= cut, patch ** 2, 0.0)
    total = wgt.sum()
    if total <= 1e-12:
        raise SpotNotFound("亮点能量过弱")

    yy, xx = np.mgrid[0:wgt.shape[0], 0:wgt.shape[1]]
    cy = float((yy * wgt).sum() / total) + y0 + off_y
    cx = float((xx * wgt).sum() / total) + x0 + off_x
    return np.array([cx, cy], dtype=np.float64)


# ---------------------------------------------------------------------------
# 方法二：球心法
# ---------------------------------------------------------------------------
def detect_sphere_center(image: np.ndarray,
                         min_radius_px: float = 8.0,
                         max_radius_px: float = 240.0,
                         invert: bool = True) -> tuple[np.ndarray, float]:
    """检测（亮背景下的）黑色球，返回 ``(圆心像素(u,v), 半径px)``。

    Parameters
    ----------
    image : 灰度或 BGR 图像
    min_radius_px, max_radius_px : 半径搜索范围（像素）
    invert : True 表示"暗球亮背景"（黑球 → 取暗区域）

    Raises
    ------
    SpotNotFound : 未检测到符合半径范围的圆
    """
    import cv2

    gray = _to_gray(image).astype(np.uint8)
    blur = cv2.medianBlur(gray, 5)

    # Hough 的 param2 是累加器阈值：越大越严格。用逐级放宽的方式，
    # 先试高质量圆，找不到再降低要求，兼顾稳健与精度。
    circles = None
    for param2 in (30.0, 25.0, 20.0, 15.0):
        found = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(blur.shape[0], blur.shape[1]),
            param1=120, param2=param2,
            minRadius=int(min_radius_px), maxRadius=int(max_radius_px),
        )
        if found is not None and len(found) > 0:
            circles = found
            break

    if circles is None or len(circles) == 0:
        raise SpotNotFound(
            f"未检测到球轮廓（半径范围 "
            f"{min_radius_px:.0f}~{max_radius_px:.0f}px）")

    # 若指定"暗球"，选圆内部平均灰度最低的那个
    cand = circles[0]
    if invert:
        scores = []
        for (cx, cy, r) in cand:
            rr = max(int(r * 0.6), 2)
            y0, y1 = max(int(cy) - rr, 0), min(int(cy) + rr + 1,
                                               blur.shape[0])
            x0, x1 = max(int(cx) - rr, 0), min(int(cx) + rr + 1,
                                               blur.shape[1])
            scores.append(float(blur[y0:y1, x0:x1].mean())
                          if y1 > y0 and x1 > x0 else 1e9)
        best = int(np.argmin(scores))
    else:
        best = int(np.argmax([c[2] for c in cand]))

    cx, cy, r = cand[best]
    return np.array([float(cx), float(cy)], dtype=np.float64), float(r)


def sphere_center_from_pixel(pixel: np.ndarray, radius_px: float,
                             sphere_radius_m: float,
                             K: np.ndarray, dist=None) -> np.ndarray:
    """由像素圆心 + 球半径，反求**球心**三维坐标（相机系）。

    原理：针孔模型下，球的像素半径 ``r`` 与其距相机的深度 ``Z`` 近似
    满足 ``r ≈ f · R / Z``（``f`` 为等效焦距，``R`` 为球实际半径）。
    故 ``Z ≈ f · R / r``，球心 = 视线方向 ``d`` × ``Z``。

    Parameters
    ----------
    pixel : (2,) 圆心像素 (u, v)
    radius_px : 球的像素半径
    sphere_radius_m : 球的实际半径 (m)
    K : (3,3) 相机内参
    dist : 畸变系数（可选）

    Returns
    -------
    center : (3,) 球心相机系坐标 (m)
    """
    from ..calibration.laser_calib import pixel_rays

    if radius_px <= 1e-6:
        raise SpotNotFound("球半径（像素）过小，无法反求球心")

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    direction = pixel_rays(
        np.asarray(pixel, dtype=np.float64).reshape(1, 2), K, dist)[0]
    f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))

    # 精确关系：球在视线方向的总角半径为 a，sin(a) = R/|C|；
    # 像素半径 r ≈ f·tan(a)（小孔模型）。故
    #     r/f = R / sqrt(|C|² - R²)  =>  |C| = sqrt((f·R/r)² + R²)
    # 相比 r = f·R/|C| 的粗略近似，这里消除了 R 与 |C| 可比时的
    # 系统偏差（乒乓球在 ~300mm 处约 0.6mm 的定标偏差）。
    t = f * float(sphere_radius_m) / float(radius_px)
    depth = float(np.sqrt(t * t + float(sphere_radius_m) ** 2))
    return direction * depth


# ---------------------------------------------------------------------------
# 采集器：把上述原语串成"现场标定"链路
# ---------------------------------------------------------------------------
class AxisLiveCollector:
    """转轴现场标定采集器。

    逐角度喂入图像，内部完成「找点 → 三角测量 → 收集 3D 点」，
    最后调用**原有的** :func:`~triniscan.calibration.calibrate_axis`。

    Parameters
    ----------
    K : (3,3) 相机内参
    plane : (4,) 激光平面 [a,b,c,d]
    sphere_radius_m : 标记球半径 (m)
    method : "peak" 或 "sphere"
    dist : 畸变系数（可选）
    """

    def __init__(self, K: np.ndarray, plane: np.ndarray,
                 sphere_radius_m: float = 0.020,
                 method: str = "peak",
                 dist=None) -> None:
        if method not in ("peak", "sphere"):
            raise ValueError(
                f"未知方法 '{method}'，可选 'peak' / 'sphere'")
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.plane = np.asarray(plane, dtype=np.float64).reshape(4)
        self.dist = None if dist is None else np.asarray(dist,
                                                         dtype=np.float64)
        self.sphere_radius_m = float(sphere_radius_m)
        self.method = method

        self.points: list[np.ndarray] = []      # 已收集的 3D 点
        self.pixels: list[np.ndarray] = []      # 对应像素
        self.angles: list[float] = []           # 对应转台角度
        self.errors: list[str] = []             # 每个角度的失败原因

    # ------------------------------------------------------------------
    def add_image(self, image: np.ndarray,
                  angle_deg: float = 0.0,
                  roi: Optional[RegionOfInterest] = None) -> np.ndarray:
        """处理一个角度的图像，成功则收录该角度 3D 点并返回之。

        Raises
        ------
        SpotNotFound : 该角度未找到标记点（调用方可跳过继续）
        """
        pixel = self.find_pixel(image, roi)
        point = self.pixel_to_point(pixel)
        self.points.append(point)
        self.pixels.append(pixel)
        self.angles.append(float(angle_deg))
        return point

    def find_pixel(self, image: np.ndarray,
                   roi: Optional[RegionOfInterest] = None) -> np.ndarray:
        """按当前方法找出标记点像素坐标。"""
        cfg = getattr(self, "_algo_cfg", {})
        if self.method == "peak":
            return detect_spot_peak(
                image,
                threshold=cfg.get("spot_threshold", 60.0),
                roi=roi,
                win_px=cfg.get("spot_win_px", 25),
            )
        center, radius = detect_sphere_center(
            image,
            min_radius_px=cfg.get("min_radius_px", 8.0),
            max_radius_px=cfg.get("max_radius_px", 240.0),
        )
        self._last_radius_px = radius
        return center

    def pixel_to_point(self, pixel: np.ndarray) -> np.ndarray:
        """像素 → 3D 点。``peak`` 取球面交点；``sphere`` 取球心。"""
        pixel = np.asarray(pixel, dtype=np.float64).reshape(2)
        if self.method == "peak":
            # 球面被照亮处即激光平面上的点：与平面求交即可
            from ..reconstruction.triangulate import triangulate

            pts, mask = triangulate(pixel.reshape(1, 2), self.K,
                                    self.plane, self.dist,
                                    return_mask=True)
            if not mask[0]:
                raise SpotNotFound("该像素与激光平面无有效交点")
            return pts[0]

        # sphere：由像素圆心与球半径反求球心
        r_px = getattr(self, "_last_radius_px", 0.0)
        if r_px <= 1e-6:
            raise SpotNotFound("球心法缺少像素半径")

        return sphere_center_from_pixel(
            pixel, r_px, self.sphere_radius_m, self.K, self.dist)

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def stack(self) -> np.ndarray:
        """已收集点组成的 (N,3) 数组。"""
        return (np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
                if self.points else np.empty((0, 3)))

    def save_points(self, path: str) -> None:
        """把现场测得的 3D 点存为 .npy（可复查 / 用原命令重标）。"""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        np.save(path, self.stack)
        log.info("已保存现场转轴点 (%d 点): %s", self.count, path)

    def calibrate(self, out_path: Optional[str] = None,
                  save_points_to: Optional[str] = None):
        """用收集到的点完成转轴标定（调用原有 ``calibrate_axis``）。"""
        from .axis_calib import calibrate_axis

        if save_points_to:
            self.save_points(save_points_to)
        return calibrate_axis(self.stack, out_path=out_path)

    def add_angle(self, image: np.ndarray, angle_deg: float,
                  roi: Optional[RegionOfInterest] = None) -> np.ndarray:
        """同 :meth:`add_image`，但把失败原因记入 ``errors`` 而不抛出。

        便于自动采集时"跳过错过的角度、继续下一个"。
        """
        try:
            return self.add_image(image, angle_deg=angle_deg, roi=roi)
        except SpotNotFound as exc:
            self.errors.append(f"{angle_deg:.0f}°: {exc}")
            return None

    def summary(self) -> str:
        """采集结果的人类可读摘要。"""
        text = f"{self.count} 个有效视角"
        if self.errors:
            text += f"，{len(self.errors)} 个失败"
        return text
