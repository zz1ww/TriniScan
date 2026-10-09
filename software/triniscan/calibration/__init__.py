"""标定模块：相机内参、激光平面、转轴。

用法::

    from triniscan.calibration import calibrate_camera, calibrate_laser_plane
"""
from .camera_calib import (
    CameraCalibResult,
    calibrate_camera,
    calibrate_camera_from_files,
    load_camera_calib,
    save_camera_calib,
)
from .laser_calib import (
    LaserCalibResult,
    calibrate_laser_plane,
    load_laser_calib,
    save_laser_calib,
)
from .axis_calib import (
    AxisCalibResult,
    calibrate_axis,
    load_axis_calib,
    save_axis_calib,
    transform_about_axis,
)

__all__ = [
    "CameraCalibResult", "calibrate_camera",
    "calibrate_camera_from_files",
    "load_camera_calib", "save_camera_calib",
    "LaserCalibResult", "calibrate_laser_plane",
    "load_laser_calib", "save_laser_calib",
    "AxisCalibResult", "calibrate_axis",
    "load_axis_calib", "save_axis_calib",
    "transform_about_axis",
]
