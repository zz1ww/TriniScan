"""标定加载与质量门限。

从重构前的 ``core/pipeline.py`` 抽出，**逻辑逐行保持不变**，只是换了
位置并补充了"降级路径"的记录（假轴兜底会进报告而不是只打日志）。

转轴有两种来源（``calibration.axis_mode``）：

- ``"z_axis"``：不标定，直接假设转轴为相机系 Z 轴。成熟开源转台
  扫描器的通行做法——转台轴本就物理竖直，只要物体在转台中心、
  相机大致水平即可，省去易出错的球心标定。
- ``"calibrated"``：从 ``calibration.axis_file`` 载入实测轴。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..algorithms.calibration import (AxisCalibResult, format_reports,
                                      load_axis_calib, load_camera_calib,
                                      load_laser_calib, reports_to_dict,
                                      validate_calibrations)
from ..common.logging_utils import get_logger
from ..contracts import CalibrationBundle, StageReport

log = get_logger(__name__)

__all__ = ["load_calibrations", "quality_thresholds", "synthetic_z_axis"]


def load_calibrations(cfg) -> CalibrationBundle:
    """载入三类标定，并**立即做质量校验**。

    ``calibration.strict=true`` 时不合格将中止（由
    :func:`validate_calibrations` 抛异常）。
    """
    camera = load_camera_calib(cfg.resolve("calibration.camera_file"))
    laser = load_laser_calib(cfg.resolve("calibration.laser_file"))

    axis_mode = str(cfg.get("calibration.axis_mode", "z_axis"))
    fallbacks: list[str] = []
    if axis_mode == "z_axis":
        axis = synthetic_z_axis(cfg)
        axis_for_quality = None          # 合成轴无需质量校验
    else:
        axis = load_axis_calib(cfg.resolve("calibration.axis_file"))
        axis_for_quality = axis

    reports = validate_calibrations(
        camera=camera, laser=laser, axis=axis_for_quality,
        thresholds=quality_thresholds(cfg),
        strict=bool(cfg.get("calibration.strict", False)),
    )
    log.info("\n%s", format_reports(reports))

    if axis_mode == "z_axis":
        log.info("转轴: z_axis 模式（不标定）— dir=%s, point=%s",
                 np.round(axis.direction, 4).tolist(),
                 np.round(axis.point, 4).tolist())
    else:
        log.info("转轴: calibrated 模式 — RMS=%.3f mm",
                 axis.residual_rms * 1e3)
    log.info("标定已载入: 相机 RMS=%.3f px, 激光平面 RMS=%.3f mm",
             camera.rms, laser.residual_rms * 1e3)

    bundle = CalibrationBundle(
        camera=camera, laser=laser, axis=axis,
        reports=reports_to_dict(reports), axis_mode=axis_mode)
    bundle._fallbacks = fallbacks        # type: ignore[attr-defined]
    return bundle


# ---------------------------------------------------------------------------
def synthetic_z_axis(cfg) -> AxisCalibResult:
    """构造"相机系 Z 轴"的合成本征轴（不标定模式）。

    对应重构前的 ``ScanPipeline._synthetic_z_axis``。注意这是**兜底
    路径**：若点云的旋转中心与相机 Z 轴不重合，拼接会散开。报告里
    会以 ``axis_mode=z_axis`` 显示，便于区分。
    """
    point = np.asarray(cfg.get("calibration.z_axis_point", [0.0, 0.0, 0.0]),
                       dtype=np.float64).reshape(3)
    direction = np.asarray(
        cfg.get("calibration.z_axis_dir", [0.0, 0.0, 1.0]),
        dtype=np.float64).reshape(3)
    nrm = np.linalg.norm(direction)
    if nrm < 1e-12:
        raise ValueError("calibration.z_axis_dir 不能为零向量")
    return AxisCalibResult(
        point=point, direction=direction / nrm,
        radius=0.0, residual_rms=0.0,
        num_points=0, num_inliers=0, outlier_ratio=0.0,
        method="z_axis",
    )


def quality_thresholds(cfg) -> dict:
    """从配置拼装阈值覆盖（缺省则用 quality 模块默认值）。"""
    def _pair(prefix: str) -> Optional[dict]:
        good = cfg.get(f"quality.{prefix}_good", None)
        warn = cfg.get(f"quality.{prefix}_warn", None)
        if good is None and warn is None:
            return None
        d = {}
        if good is not None:
            d["good"] = good
        if warn is not None:
            d["warn"] = warn
        return d

    overrides: dict = {}
    for name, key in (("camera", "camera_rms"),
                      ("laser", "laser_rms"),
                      ("axis", "axis_rms")):
        pair = _pair(key)
        if pair:
            overrides[name] = pair
    return overrides
