"""流水线标定校验集成测试（不依赖硬件）。

重点验证 P0 修复：**坏标定不会静默通过**——
``ScanPipeline.load_calibrations`` 加载后即校验，严格模式直接中止。
"""
import numpy as np
import pytest

from triniscan.core.config import Config
from triniscan.calibration import (
    save_camera_calib,
    save_laser_calib,
    save_axis_calib,
    CameraCalibResult,
    LaserCalibResult,
    AxisCalibResult,
)


def _write_calibs(tmp_path, laser_rms_mm, camera_rms, axis_rms_mm):
    cam = CameraCalibResult(
        K=np.array([[1200.0, 0, 960], [0, 1200, 540], [0, 0, 1]]),
        dist=np.zeros(5), image_size=(1920, 1080),
        rms=camera_rms, num_images=15, per_view_errors=[camera_rms] * 15,
    )
    laser = LaserCalibResult(
        plane=np.array([0.0, 0.0, 1.0, -0.3]),
        residual_rms=laser_rms_mm * 1e-3, num_points=800, num_views=6,
        view_point_counts=[130] * 6,
    )
    axis = AxisCalibResult(
        point=np.zeros(3), direction=np.array([0.0, 0.0, 1.0]),
        radius=0.05, residual_rms=axis_rms_mm * 1e-3, num_points=12,
    )
    cpath = str(tmp_path / "cam.npz")
    lpath = str(tmp_path / "laser.npz")
    apath = str(tmp_path / "axis.npz")
    save_camera_calib(cpath, cam)
    save_laser_calib(lpath, laser)
    save_axis_calib(apath, axis)
    return cpath, lpath, apath


def _cfg(tmp_path, paths, strict=False):
    cam, laser, axis = paths
    return Config({
        "calibration": {
            "camera_file": cam, "laser_file": laser, "axis_file": axis,
            "strict": strict,
        },
        "quality": {},
    }, root=str(tmp_path))


def test_good_calibrations_pass(tmp_path):
    from triniscan.core.pipeline import ScanPipeline
    from triniscan.calibration.quality import reports_to_dict

    paths = _write_calibs(tmp_path, laser_rms_mm=0.2,
                          camera_rms=0.2, axis_rms_mm=0.05)
    pipe = ScanPipeline(_cfg(tmp_path, paths))
    pipe.load_calibrations()
    levels = [r["level"] for r in reports_to_dict(pipe._calib_reports)]
    assert all(l == "ok" for l in levels)


def test_bad_laser_flagged(tmp_path):
    from triniscan.core.pipeline import ScanPipeline

    paths = _write_calibs(tmp_path, laser_rms_mm=16.0,
                          camera_rms=0.2, axis_rms_mm=0.05)
    pipe = ScanPipeline(_cfg(tmp_path, paths))
    pipe.load_calibrations()  # 非严格：仅记录，不抛
    reports = pipe._calib_reports
    laser_report = [r for r in reports if "激光" in r.name][0]
    assert laser_report.level.value == "fail"


def test_strict_mode_aborts_on_bad_laser(tmp_path):
    from triniscan.core.pipeline import ScanPipeline

    paths = _write_calibs(tmp_path, laser_rms_mm=16.0,
                          camera_rms=0.2, axis_rms_mm=0.05)
    pipe = ScanPipeline(_cfg(tmp_path, paths, strict=True))
    with pytest.raises(RuntimeError):
        pipe.load_calibrations()
