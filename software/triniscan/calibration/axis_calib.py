"""转台旋转轴标定。

目标
----
确定转台旋转轴在**相机坐标系**下的位置 ``point`` 与方向 ``direction``，
用于把各视角点云变换到统一坐标系（多视角配准的基础）。

原理
----
转台上放一标记物，转台每转一个角度拍一张，提取标记物特征点的三维坐标
（由激光三角测量得到）。这些点分布在以旋转轴为轴的圆上，
拟合圆即得轴的位置与方向。

注意
----
- 轴的位置误差会随测量半径放大，应尽量精确；
- 采样角度宜覆盖较大范围（如每 30° 共 12 点）以稳定拟合；
- 拟合后应检查残差，过大说明特征点提取或角度记录有误。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..common.geometry import fit_circle_3d as _fit_circle_3d
from ..common.geometry import transform_about_axis as _transform_axis
from ..common.io_utils import save_calibration
from ..common.logging_utils import get_logger
from ..common.robust import robust_circle_3d_fit

log = get_logger(__name__)

__all__ = [
    "AxisCalibResult",
    "calibrate_axis",
    "save_axis_calib",
    "load_axis_calib",
    "transform_about_axis",
]


@dataclass
class AxisCalibResult:
    """转轴标定结果。"""

    point: np.ndarray           # (3,) 轴上一点（相机系）
    direction: np.ndarray       # (3,) 单位方向向量（相机系）
    radius: float = 0.0         # 拟合圆半径
    residual_rms: float = 0.0   # 拟合残差 RMS (m)
    num_points: int = 0
    num_inliers: int = 0        # 稳健内点数
    outlier_ratio: float = 0.0  # 离群点比例
    method: str = "svd"         # 拟合方法

    def transform(self, angle_deg: float) -> np.ndarray:
        """返回绕该轴转 ``angle_deg`` 的 4x4 变换。"""
        return _transform_axis(self.point, self.direction,
                               np.deg2rad(angle_deg))


def calibrate_axis(
    feature_points: np.ndarray,
    out_path: Optional[str] = None,
    robust: bool = True,
    method: str = "ransac",
    angles_deg: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
) -> AxisCalibResult:
    """由绕轴旋转的一组特征点标定旋转轴。

    Parameters
    ----------
    feature_points : (N, 3), N >= 3
        各转角下标记物特征点在相机系的三维坐标。
    out_path : 非空则保存。
    robust : 是否使用稳健拟合（RANSAC）以抵抗跳点。
    method : ``"ransac"`` 或 ``"lsq"``（``robust=False`` 时忽略）。
    angles_deg : 各点对应的转台角度（可选）。用于**固定转轴方向的符号**，
        使 `direction` 与旋转正方向一致，避免后续配准方向反了。
    threshold : RANSAC 径向残差阈值（m）；None 时自适应。

    Returns
    -------
    AxisCalibResult
    """
    points = np.asarray(feature_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise RuntimeError(f"转轴标定点数不足（{len(points)} < 3）")

    if robust:
        fit = robust_circle_3d_fit(points, method=method, threshold=threshold)
        center, normal, radius = fit.center, fit.normal, fit.radius
        residual = float(fit.residual_rms)
        num_inliers = int(fit.num_inliers)
        outlier_ratio = float(fit.outlier_ratio)
        used_method = fit.method
    else:
        center, normal, radius = _fit_circle_3d(points)
        normal = normal / np.linalg.norm(normal)
        residual = _circle_residual(points, center, normal, radius)
        num_inliers = len(points)
        outlier_ratio = 0.0
        used_method = "lsq"

    normal = normal / np.linalg.norm(normal)

    # 用角度单调性确定转轴方向的符号（右手定则），避免配准方向反向。
    if angles_deg is not None:
        normal = _orient_axis_by_angles(
            points, center, normal, np.asarray(angles_deg, dtype=np.float64))

    result = AxisCalibResult(
        point=center,
        direction=normal,
        radius=float(radius),
        residual_rms=float(residual),
        num_points=len(points),
        num_inliers=num_inliers,
        outlier_ratio=outlier_ratio,
        method=used_method,
    )

    log.info(
        "转轴标定完成: point=(%.4f, %.4f, %.4f), "
        "dir=(%.4f, %.4f, %.4f), r=%.4f m, RMS=%.4f mm "
        "(内点 %d/%d, 方法=%s)",
        *center, *normal, radius, residual * 1e3,
        num_inliers, len(points), used_method,
    )

    if out_path:
        save_axis_calib(out_path, result)
    return result


def _orient_axis_by_angles(points: np.ndarray, center: np.ndarray,
                           normal: np.ndarray,
                           angles_deg: np.ndarray) -> np.ndarray:
    """用角度序列的旋转方向（右手定则）统一转轴方向符号。

    对相邻两点，计算它们在圆平面内的**有符号夹角**；若其符号与角度
    增量一致，则当前 ``normal`` 方向正确，否则取反。
    """
    if len(points) != len(angles_deg) or len(points) < 2:
        return normal

    rel = points - center
    axial = rel @ normal
    radial = rel - np.outer(axial, normal)
    # 构造平面内正交基
    u = radial[0]
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        return normal
    u = u / nu
    v = np.cross(normal, u)

    xy = np.column_stack([radial @ u, radial @ v])   # (N, 2)
    theta = np.arctan2(xy[:, 1], xy[:, 0])           # 平面内极角
    # 解开相位跳变
    theta_unwrapped = np.unwrap(theta)

    dtheta = np.diff(theta_unwrapped)
    dphi = np.diff(np.deg2rad(angles_deg))
    # 只取有意义（非零）的角度增量
    valid = np.abs(dphi) > 1e-9
    if not np.any(valid):
        return normal
    agreement = float(np.sum(np.sign(dtheta[valid]) * np.sign(dphi[valid])))
    if agreement < 0:
        log.debug("转轴方向与角度增量不一致，已翻转符号")
        return -normal
    return normal


def _circle_residual(points: np.ndarray, center: np.ndarray,
                     normal: np.ndarray, radius: float) -> float:
    """点到拟合圆的径向距离 RMS。"""
    rel = points - center
    # 去掉沿轴分量
    axial = rel @ normal
    radial_vec = rel - np.outer(axial, normal)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    return float(np.sqrt(np.mean((radial_dist - radius) ** 2)))


def transform_about_axis(axis_point: np.ndarray, axis_dir: np.ndarray,
                         angle_rad: float) -> np.ndarray:
    """代理到 common.geometry，保持旧接口兼容。"""
    return _transform_axis(axis_point, axis_dir, angle_rad)


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------
def save_axis_calib(path: str, result: AxisCalibResult) -> None:
    save_calibration(
        path,
        arrays={
            "point": result.point,
            "direction": result.direction,
            "radius": np.array(result.radius),
            "residual_rms": np.array(result.residual_rms),
            "num_points": np.array(result.num_points),
            "num_inliers": np.array(result.num_inliers),
            "outlier_ratio": np.array(result.outlier_ratio),
        },
        meta={"type": "rotation_axis", "method": result.method},
    )


def load_axis_calib(path: str) -> AxisCalibResult:
    from ..common.io_utils import load_calibration
    arrays, meta = load_calibration(path)
    return AxisCalibResult(
        point=arrays["point"],
        direction=arrays["direction"],
        radius=float(arrays.get("radius", 0.0)),
        residual_rms=float(arrays.get("residual_rms", 0.0)),
        num_points=int(arrays.get("num_points", 0)),
        num_inliers=int(arrays.get("num_inliers", 0)),
        outlier_ratio=float(arrays.get("outlier_ratio", 0.0)),
        method=str(meta.get("method", "svd")),
    )
