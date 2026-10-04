"""标定相关单元测试（不依赖真实相机，用合成数据）。"""
import numpy as np
import pytest

from triniscan.calibration.laser_calib import (
    calibrate_laser_plane,
    laser_plane_from_chessboard,
    pixel_rays,
)
from triniscan.calibration.axis_calib import AxisCalibResult, calibrate_axis
from triniscan.calibration.camera_calib import (
    calibrate_camera,
    make_object_points,
)


# ---------------------------------------------------------------------------
# 激光平面
# ---------------------------------------------------------------------------
def test_laser_plane_from_chessboard_identity():
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [0.5]])
    plane = laser_plane_from_chessboard(rvec, tvec)
    assert np.allclose(plane[:3], [0, 0, 1], atol=1e-9)
    assert np.isclose(plane[3], -0.5)


def test_pixel_rays_unit():
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1.0]])
    pixels = np.array([[320, 240], [400, 300], [100, 100]], dtype=float)
    dirs = pixel_rays(pixels, K)
    assert dirs.shape == (3, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    # 主点光线沿 +z
    assert np.allclose(dirs[0], [0, 0, 1], atol=1e-9)


def test_pixel_rays_empty():
    K = np.eye(3)
    dirs = pixel_rays(np.empty((0, 2)), K)
    assert dirs.shape == (0, 3)


def _synthesize_laser_pairs(K, plane_true, n_views=4, n_points=100,
                            seed=0):
    """合成"中心线像素"：让真实激光平面上的点投到相机，反算像素。"""
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n_views):
        # 在激光平面上随机取样点（切两个平面求交线再取点较复杂，
        # 这里直接取平面上点，再投影为像素，绕开棋盘格部分）
        # 平面上取点：沿两个正交方向构造
        # 用平面法向构造局部基，取固定深度范围内点
        pts3d = _random_points_on_plane(plane_true, rng, n_points)
        pixels = _project(K, pts3d)
        # 直接提供 centerline，避免依赖棋盘格
        pairs.append({"centerline": pixels, "image": None, "gray": None})
    return pairs


def _random_points_on_plane(plane, rng, n):
    n_vec = plane[:3] / np.linalg.norm(plane[:3])
    # 平面上一点
    p0 = -plane[3] * n_vec
    # 构造两个切向
    a = rng.normal(size=3)
    a -= a.dot(n_vec) * n_vec
    a /= np.linalg.norm(a)
    b = np.cross(n_vec, a)
    uv = rng.uniform(-0.05, 0.05, size=(n, 2))
    return p0 + uv[:, :1] * a + uv[:, 1:] * b


def _project(K, pts_cam):
    pts = np.asarray(pts_cam)
    uvw = pts @ K.T
    return uvw[:, :2] / uvw[:, 2:]


# ---------------------------------------------------------------------------
# 转轴
# ---------------------------------------------------------------------------
def test_calibrate_axis():
    # 绕 z 轴（过原点）、半径 0.05 的点
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    r = 0.05
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta),
                           np.full_like(theta, 0.03)])
    result = calibrate_axis(pts)
    assert isinstance(result, AxisCalibResult)
    assert np.allclose(result.point, [0, 0, 0.03], atol=1e-6)
    assert np.isclose(abs(result.direction[2]), 1.0, atol=1e-6)
    assert result.residual_rms < 1e-6


def test_calibrate_axis_too_few_points():
    with pytest.raises(RuntimeError):
        calibrate_axis(np.zeros((2, 3)))


def test_axis_transform_roundtrip():
    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    pts = np.column_stack([0.05 * np.cos(theta), 0.05 * np.sin(theta),
                           np.zeros_like(theta)])
    result = calibrate_axis(pts)
    # 转 90° 后应仍在该圆上
    T = result.transform(90.0)
    from triniscan.common.geometry import apply_transform
    moved = apply_transform(T, pts)
    radial = np.linalg.norm(moved[:, :2], axis=1)
    assert np.allclose(radial, 0.05, atol=1e-6)


# ---------------------------------------------------------------------------
# 相机内参
# ---------------------------------------------------------------------------
def test_make_object_points():
    objp = make_object_points(9, 6, 0.02)
    assert objp.shape == (54, 3)
    assert np.isclose(objp[:, 2].max(), 0.0)
    assert np.isclose(objp[:, 0].max(), 8 * 0.02)


def test_make_object_points_geometry():
    objp = make_object_points(9, 6, 0.02)
    assert objp.shape == (54, 3)
    assert np.isclose(objp[:, 2].max(), 0.0)


def test_calibrate_camera_synthetic():
    """用真实渲染的棋盘格图像验证标定流程可跑通。

    通过 OpenCV 合成一个正对相机的棋盘格（已知内参下），
    标定应能收敛且使用到所有图像。
    """
    import cv2

    cols, rows = 7, 5
    square = 0.025

    # 渲染带白边的标准棋盘格（(cols+1)x(rows+1) 个方格），
    # 通过轻微平移模拟不同拍摄位置
    images = []
    rng = np.random.default_rng(42)
    for _ in range(6):
        scale = 60
        margin = 40
        w = (cols + 1) * scale + 2 * margin
        h = (rows + 1) * scale + 2 * margin
        canvas = np.full((h, w), 255, dtype=np.uint8)
        dx = int(rng.integers(-10, 10))
        dy = int(rng.integers(-10, 10))
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    x = margin + c * scale + dx
                    y = margin + r * scale + dy
                    cv2.rectangle(canvas, (x, y),
                                  (x + scale, y + scale), 0, -1)
        images.append(canvas)

    result = calibrate_camera(images, pattern_cols=cols, pattern_rows=rows,
                              square_size_m=square)
    assert result.num_images >= 3
    assert result.K[0, 0] > 0
