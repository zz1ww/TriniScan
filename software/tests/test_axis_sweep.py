"""转轴**自动扫描**（sweep_axis_calibration）的测试。

用注入的假相机 / 假转台驱动完整流程，无需真实硬件。
"""
import math
import os

import cv2
import numpy as np
import pytest

from triniscan.calibration import (
    CameraCalibResult,
    LaserCalibResult,
    load_axis_calib,
    save_camera_calib,
    save_laser_calib,
)
from triniscan.ui.capture_session import AxisSweepResult, \
    sweep_axis_calibration

K = np.array([[800.0, 0.0, 320.0],
              [0.0, 800.0, 240.0],
              [0.0, 0.0, 1.0]])
AXIS_RADIUS = 0.060
DEPTH = 0.30
PLANE = np.array([0.0, 0.0, 1.0, -DEPTH])
SPHERE_R = 0.020


class _FakeConfig:
    """最小配置替身。"""

    def __init__(self, root, *, camera_file="", laser_file="",
                 axis_file="", method="peak", views=12):
        self.root = root
        self.raw = {"camera": {}, "turntable": {}}
        self._d = {
            "calibration.camera_file": camera_file,
            "calibration.laser_file": laser_file,
            "calibration.axis_file": axis_file,
            "axis_calib.sphere_radius_m": SPHERE_R,
            "axis_calib.method": method,
            "axis_calib.num_views": views,
            "axis_calib.spot_threshold": 60.0,
            "axis_calib.spot_win_px": 25,
            "axis_calib.min_radius_px": 8.0,
            "axis_calib.max_radius_px": 240.0,
            "axis_calib.archive_dir": "arch",
        }

    def get(self, key, default=None):
        return self._d.get(key, default)

    def set(self, key, value):
        self._d[key] = value


def _write_calibs(tmp_path):
    cam = tmp_path / "cam.npz"
    laser = tmp_path / "laser.npz"
    save_camera_calib(str(cam), CameraCalibResult(
        K=K, dist=np.zeros(5), image_size=(640, 480), rms=0.1,
        num_images=10))
    save_laser_calib(str(laser), LaserCalibResult(
        plane=PLANE, residual_rms=1e-5, num_points=200, num_views=5,
        view_point_counts=[40] * 5))
    return str(cam), str(laser)


class _FakeCamera:
    """按转台当前角度合成一张亮点图。"""

    def __init__(self, table):
        self._t = table

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def grab(self):
        th = math.radians(self._t.angle)
        c = np.array([AXIS_RADIUS * math.cos(th),
                      AXIS_RADIUS * math.sin(th), DEPTH])
        px = K @ c
        u, v = px[0] / px[2], px[1] / px[2]
        img = np.zeros((480, 640), np.uint8)
        cv2.circle(img, (int(round(u)), int(round(v))), 40, 8, -1)
        cv2.circle(img, (int(round(u)), int(round(v))), 4, 255, -1)
        return cv2.GaussianBlur(img, (5, 5), 0)


class _FakeTurntable:
    def __init__(self):
        self.angle = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def rotate_deg(self, deg):
        self.angle = (self.angle + deg) % 360.0
        return True


# ---------------------------------------------------------------------------
def test_sweep_happy_path(tmp_path):
    cam, laser = _write_calibs(tmp_path)
    axis_out = str(tmp_path / "axis.npz")
    cfg = _FakeConfig(str(tmp_path), camera_file=cam, laser_file=laser,
                      axis_file=axis_out)
    table = _FakeTurntable()
    progress = []

    result = sweep_axis_calibration(
        cfg, on_progress=lambda i, n, m: progress.append((i, n, m)),
        make_camera=lambda: _FakeCamera(table),
        make_turntable=lambda: table)

    assert isinstance(result, AxisSweepResult)
    assert result.ok, result.message
    assert result.num_views == 12
    assert result.num_failed == 0
    # 轴方向应接近 ±z
    assert abs(abs(result.direction[2]) - 1.0) < 0.05
    assert abs(result.radius_m - AXIS_RADIUS) < 0.01
    # 产物落盘
    assert os.path.exists(axis_out)
    assert result.points_path and os.path.exists(result.points_path)
    # 存档的 .npy 可被原有加载器读取
    loaded = load_axis_calib(axis_out)
    assert loaded.num_points == 12
    # 进度回调按 1..12 递增
    assert [p[0] for p in progress] == list(range(1, 13))


def test_sweep_missing_camera_file(tmp_path):
    _, laser = _write_calibs(tmp_path)
    cfg = _FakeConfig(str(tmp_path), camera_file="", laser_file=laser)
    result = sweep_axis_calibration(cfg)
    assert not result.ok
    assert "相机标定" in result.message


def test_sweep_missing_laser_file(tmp_path):
    cam, _ = _write_calibs(tmp_path)
    cfg = _FakeConfig(str(tmp_path), camera_file=cam, laser_file="")
    result = sweep_axis_calibration(cfg)
    assert not result.ok
    assert "激光平面" in result.message


def test_sweep_all_views_fail(tmp_path):
    """全程找不到标记球：应给出明确失败信息而非崩溃。"""
    cam, laser = _write_calibs(tmp_path)
    cfg = _FakeConfig(str(tmp_path), camera_file=cam, laser_file=laser)
    table = _FakeTurntable()

    class _BlankCamera(_FakeCamera):
        def grab(self):
            return np.zeros((480, 640), np.uint8)   # 全黑，无亮点

    result = sweep_axis_calibration(
        cfg, make_camera=lambda: _BlankCamera(table),
        make_turntable=lambda: table)
    assert not result.ok
    assert "有效视角不足" in result.message
    assert result.num_views == 0


def test_sweep_tolerates_some_bad_views(tmp_path):
    """部分角度失败：其余角度仍能完成标定。"""
    cam, laser = _write_calibs(tmp_path)
    cfg = _FakeConfig(str(tmp_path), camera_file=cam, laser_file=laser,
                      axis_file=str(tmp_path / "axis.npz"))
    table = _FakeTurntable()

    class _SometimesBlank(_FakeCamera):
        def grab(self):
            if abs(self._t.angle - 120.0) < 1e-6:
                return np.zeros((480, 640), np.uint8)
            return super().grab()

    result = sweep_axis_calibration(
        cfg, make_camera=lambda: _SometimesBlank(table),
        make_turntable=lambda: table)
    assert result.ok, result.message
    assert result.num_views == 11
    assert result.num_failed == 1


def test_sweep_stop_flag(tmp_path):
    """stop_flag 生效：应在中途停止并因点数不足而失败。"""
    cam, laser = _write_calibs(tmp_path)
    cfg = _FakeConfig(str(tmp_path), camera_file=cam, laser_file=laser)
    table = _FakeTurntable()
    state = {"n": 0}

    def _stop():
        state["n"] += 1
        return state["n"] > 2      # 第 3 次检查后停止

    result = sweep_axis_calibration(
        cfg, make_camera=lambda: _FakeCamera(table),
        make_turntable=lambda: table, stop_flag=_stop)
    assert not result.ok
    assert result.num_views <= 3
