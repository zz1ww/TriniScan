"""转轴**现场**标定（axis_live）的单元测试。

用合成图像验证两种取点方法，并确认全链路（找点 → 三角测量 →
拟合圆）能从图像恢复已知转轴。
"""
import math

import cv2
import numpy as np
import pytest

from triniscan.calibration import (
    AxisLiveCollector,
    SpotNotFound,
    calibrate_axis,
    detect_sphere_center,
    detect_spot_peak,
    sphere_center_from_pixel,
)

K = np.array([[800.0, 0.0, 320.0],
              [0.0, 800.0, 240.0],
              [0.0, 0.0, 1.0]])
SPHERE_R = 0.020          # 乒乓球半径
AXIS_RADIUS = 0.060       # 球心到转轴的距离
DEPTH = 0.30              # 转轴平面深度
PLANE = np.array([0.0, 0.0, 1.0, -DEPTH])


def _spot_image(center_px, radius_px, bright=True):
    """生成黑球 + 激光亮斑（暗背景）图像。"""
    img = np.zeros((480, 640), np.uint8)
    cv2.circle(img, (int(round(center_px[0])), int(round(center_px[1]))),
               int(round(radius_px)), 8, -1)
    if bright:
        cv2.circle(img, (int(round(center_px[0])), int(round(center_px[1]))),
                   4, 255, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def _sphere_image(center_px, radius_px):
    """生成黑球 + 亮背景图像。"""
    img = np.full((480, 640), 230, np.uint8)
    cv2.circle(img, (int(round(center_px[0])), int(round(center_px[1]))),
               int(round(radius_px)), 15, -1)
    return img


def _project(point):
    px = K @ np.asarray(point, dtype=np.float64)
    return px[:2] / px[2]


# ---------------------------------------------------------------------------
# 方法一：最亮极点
# ---------------------------------------------------------------------------
def test_detect_spot_peak_center():
    img = _spot_image((330, 250), 60)
    uv = detect_spot_peak(img, threshold=60.0)
    assert np.linalg.norm(uv - np.array([330.0, 250.0])) < 2.0


def test_detect_spot_peak_not_found_on_blank():
    with pytest.raises(SpotNotFound):
        detect_spot_peak(np.zeros((200, 200), np.uint8), threshold=60.0)


def test_detect_spot_peak_respects_roi():
    from triniscan.calibration import RegionOfInterest

    img = np.zeros((480, 640), np.uint8)
    cv2.circle(img, (200, 200), 5, 255, -1)     # 亮斑 A
    cv2.circle(img, (500, 400), 5, 255, -1)     # 亮斑 B
    roi = RegionOfInterest(u=500, v=400, half=60)
    uv = detect_spot_peak(img, threshold=60.0, roi=roi)
    assert np.linalg.norm(uv - np.array([500.0, 400.0])) < 3.0


# ---------------------------------------------------------------------------
# 方法二：球心
# ---------------------------------------------------------------------------
def test_detect_sphere_center():
    img = _sphere_image((330, 250), 60)
    uv, r = detect_sphere_center(img)
    assert np.linalg.norm(uv - np.array([330.0, 250.0])) < 8.0
    assert abs(r - 60.0) < 10.0


def test_detect_sphere_center_not_found():
    with pytest.raises(SpotNotFound):
        detect_sphere_center(np.zeros((480, 640), np.uint8))


def test_sphere_center_from_pixel_exact():
    """精确公式应能无偏反求球心。"""
    point = np.array([0.06, 0.0, DEPTH])
    px = _project(point)
    true_dist = float(np.linalg.norm(point))
    # 真实像素半径：f * tan(asin(R/|C|))
    r_px = 800.0 * math.tan(math.asin(SPHERE_R / true_dist))
    got = sphere_center_from_pixel(px, r_px, SPHERE_R, K)
    assert np.linalg.norm(got - point) < 2e-3      # < 2 mm


def test_sphere_center_from_pixel_rejects_tiny_radius():
    with pytest.raises(SpotNotFound):
        sphere_center_from_pixel(np.array([320.0, 240.0]), 0.0,
                                 SPHERE_R, K)


# ---------------------------------------------------------------------------
# 采集器与全链路
# ---------------------------------------------------------------------------
def test_collector_rejects_bad_method():
    with pytest.raises(ValueError):
        AxisLiveCollector(K, PLANE, method="nope")


def test_collector_peak_end_to_end():
    """全链路：12 张亮点图 → 恢复转轴。"""
    c = AxisLiveCollector(K, PLANE, sphere_radius_m=SPHERE_R,
                          method="peak")
    c._algo_cfg = {"spot_threshold": 60.0, "spot_win_px": 25}
    for deg in range(0, 360, 30):
        th = math.radians(deg)
        point = np.array([AXIS_RADIUS * math.cos(th),
                          AXIS_RADIUS * math.sin(th), DEPTH])
        px = _project(point)
        r_px = 800.0 * SPHERE_R / DEPTH
        c.add_image(_spot_image(px, r_px), angle_deg=deg)
    assert c.count == 12
    res = c.calibrate()
    # 轴方向应接近 ±z
    assert abs(abs(res.direction[2]) - 1.0) < 0.05
    assert abs(res.radius - AXIS_RADIUS) < 0.01


def test_collector_sphere_end_to_end():
    """全链路：12 张球心图 → 恢复转轴（精确公式，应很准）。"""
    c = AxisLiveCollector(K, PLANE, sphere_radius_m=SPHERE_R,
                          method="sphere")
    c._algo_cfg = {"min_radius_px": 8.0, "max_radius_px": 240.0}
    true_pts = []
    for deg in range(0, 360, 30):
        th = math.radians(deg)
        point = np.array([AXIS_RADIUS * math.cos(th),
                          AXIS_RADIUS * math.sin(th), DEPTH])
        true_pts.append(point)
        px = _project(point)
        # 用精确像素半径生成图像（避免针孔近似偏差）
        r_px = 800.0 * math.tan(math.asin(
            SPHERE_R / np.linalg.norm(point)))
        c.add_image(_sphere_image(px, r_px), angle_deg=deg)
    assert c.count == 12
    res = c.calibrate()
    assert abs(abs(res.direction[2]) - 1.0) < 0.05
    assert abs(res.radius - AXIS_RADIUS) < 0.008


def test_collector_skips_bad_view_then_calibrates():
    """个别角度找不到点：应抛异常但不影响其余角度。"""
    c = AxisLiveCollector(K, PLANE, sphere_radius_m=SPHERE_R,
                          method="peak")
    c._algo_cfg = {"spot_threshold": 60.0, "spot_win_px": 25}
    for deg in range(0, 360, 30):
        th = math.radians(deg)
        point = np.array([AXIS_RADIUS * math.cos(th),
                          AXIS_RADIUS * math.sin(th), DEPTH])
        px = _project(point)
        if deg == 90:
            with pytest.raises(SpotNotFound):
                c.add_image(np.zeros((480, 640), np.uint8), angle_deg=deg)
            continue
        c.add_image(_spot_image(px, 800.0 * SPHERE_R / DEPTH),
                    angle_deg=deg)
    assert c.count == 11
    assert c.calibrate().num_points == 11


def test_collector_save_points(tmp_path):
    c = AxisLiveCollector(K, PLANE, sphere_radius_m=SPHERE_R)
    c.points = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    out = tmp_path / "axis_points.npy"
    c.save_points(str(out))
    assert out.exists()
    arr = np.load(str(out))
    assert arr.shape == (2, 3)
    assert np.allclose(arr[1], [4.0, 5.0, 6.0])


def test_calibrate_requires_enough_points():
    c = AxisLiveCollector(K, PLANE, sphere_radius_m=SPHERE_R)
    c.points = [np.array([0.0, 0.0, 0.3])]      # 仅 1 点
    with pytest.raises(RuntimeError):
        c.calibrate()


# ---------------------------------------------------------------------------
# 回归：原有离线接口不受影响
# ---------------------------------------------------------------------------
def test_original_calibrate_axis_unchanged():
    """原有 calibrate_axis 仍按 (N,3) 点直接标定。"""
    pts = []
    for deg in range(0, 360, 30):
        th = math.radians(deg)
        pts.append([AXIS_RADIUS * math.cos(th),
                    AXIS_RADIUS * math.sin(th), DEPTH])
    res = calibrate_axis(np.asarray(pts))
    assert abs(abs(res.direction[2]) - 1.0) < 1e-9
    assert abs(res.radius - AXIS_RADIUS) < 1e-9
    assert res.residual_rms < 1e-12
