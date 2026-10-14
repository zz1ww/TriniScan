"""激光平面稳健标定测试（合成数据）。"""
import numpy as np
import pytest

from triniscan.calibration.laser_calib import (
    LaserCalibResult,
    load_laser_calib,
    save_laser_calib,
)
from triniscan.calibration.axis_calib import (
    AxisCalibResult,
    load_axis_calib,
    save_axis_calib,
)


def test_laser_result_extra_fields_default():
    r = LaserCalibResult(plane=np.array([0, 0, 1.0, -0.3]),
                         residual_rms=1e-4, num_points=100, num_views=4)
    assert r.per_view_rms == []
    assert r.rejected_views == []
    assert r.method == "svd"


def test_laser_save_load_roundtrip(tmp_path):
    r = LaserCalibResult(
        plane=np.array([0.0, 0.0, 1.0, -0.3]),
        residual_rms=2e-4, num_points=800, num_views=6,
        view_point_counts=[100] * 6, per_view_rms=[1e-4] * 6,
        rejected_views=[2], num_inliers=780, outlier_ratio=0.025,
        method="irls",
    )
    path = str(tmp_path / "laser.npz")
    save_laser_calib(path, r)
    back = load_laser_calib(path)
    assert np.allclose(back.plane, r.plane)
    assert back.method == "irls"
    assert back.rejected_views == [2]
    assert pytest.approx(back.outlier_ratio, rel=1e-6) == 0.025


def test_axis_save_load_roundtrip(tmp_path):
    r = AxisCalibResult(
        point=np.array([0.0, 0.0, 0.03]),
        direction=np.array([0.0, 0.0, 1.0]),
        radius=0.05, residual_rms=5e-5, num_points=12,
        num_inliers=11, outlier_ratio=1 / 12, method="ransac",
    )
    path = str(tmp_path / "axis.npz")
    save_axis_calib(path, r)
    back = load_axis_calib(path)
    assert np.allclose(back.point, r.point)
    assert back.method == "ransac"
    assert back.num_inliers == 11


def test_axis_direction_sign_from_angles():
    """给角度序列时，转轴方向应与旋转正方向一致（右手定则）。"""
    from triniscan.calibration.axis_calib import calibrate_axis

    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    r = 0.05
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta),
                           np.full_like(theta, 0.03)])
    angles = np.degrees(theta)
    res = calibrate_axis(pts, angles_deg=angles)
    assert abs(res.direction[2]) > 0.99
    # 按 +z 旋转时，点随角度逆时针 → 方向应为 +z
    assert res.direction[2] > 0
