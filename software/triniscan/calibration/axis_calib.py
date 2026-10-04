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
from typing import Optional

import numpy as np

from ..common.geometry import fit_circle_3d as _fit_circle_3d
from ..common.geometry import transform_about_axis as _transform_axis
from ..common.io_utils import save_calibration
from ..common.logging_utils import get_logger

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

    def transform(self, angle_deg: float) -> np.ndarray:
        """返回绕该轴转 ``angle_deg`` 的 4x4 变换。"""
        return _transform_axis(self.point, self.direction,
                               np.deg2rad(angle_deg))


def calibrate_axis(
    feature_points: np.ndarray,
    out_path: Optional[str] = None,
) -> AxisCalibResult:
    """由绕轴旋转的一组特征点标定旋转轴。

    Parameters
    ----------
    feature_points : (N, 3), N >= 3
        各转角下标记物特征点在相机系的三维坐标。
    out_path : 非空则保存。

    Returns
    -------
    AxisCalibResult
    """
    points = np.asarray(feature_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise RuntimeError(f"转轴标定点数不足（{len(points)} < 3）")

    center, normal, radius = _fit_circle_3d(points)
    normal = normal / np.linalg.norm(normal)

    # 残差：各点到拟合圆的径向偏差
    residual = _circle_residual(points, center, normal, radius)

    result = AxisCalibResult(
        point=center,
        direction=normal,
        radius=float(radius),
        residual_rms=float(residual),
        num_points=len(points),
    )

    log.info(
        "转轴标定完成: point=(%.4f, %.4f, %.4f), "
        "dir=(%.4f, %.4f, %.4f), r=%.4f m, RMS=%.4f mm",
        *center, *normal, radius, residual * 1e3,
    )

    if out_path:
        save_axis_calib(out_path, result)
    return result


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
        },
        meta={"type": "rotation_axis"},
    )


def load_axis_calib(path: str) -> AxisCalibResult:
    from ..common.io_utils import load_calibration
    arrays, _ = load_calibration(path)
    return AxisCalibResult(
        point=arrays["point"],
        direction=arrays["direction"],
        radius=float(arrays.get("radius", 0.0)),
        residual_rms=float(arrays.get("residual_rms", 0.0)),
        num_points=int(arrays.get("num_points", 0)),
    )
