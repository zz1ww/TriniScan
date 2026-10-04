"""几何工具库单元测试。"""
import numpy as np
import pytest

from triniscan.common.geometry import (
    apply_transform,
    decompose_rotation,
    fit_circle_3d,
    fit_plane_svd,
    intersect_ray_plane,
    normalize,
    normalize_plane,
    plane_from_point_normal,
    plane_from_three_points,
    point_plane_signed_distance,
    rigid_transform,
    rodrigues_matrix,
    transform_about_axis,
)


def test_normalize():
    v = normalize(np.array([3.0, 0.0, 4.0]))
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, [0.6, 0.0, 0.8])


def test_normalize_zero_raises():
    with pytest.raises(ValueError):
        normalize(np.zeros(3))


def test_plane_from_point_normal():
    plane = plane_from_point_normal(np.array([0, 0, 1.0]),
                                    np.array([0, 0, 2.0]))
    assert np.allclose(plane, [0, 0, 1, -1])


def test_plane_from_three_points():
    p = plane_from_three_points(
        np.array([1, 0, 0.0]), np.array([0, 1, 0.0]),
        np.array([0, 0, 1.0]))
    assert np.isclose(np.linalg.norm(p[:3]), 1.0)
    # 三点到平面距离为 0
    dist = point_plane_signed_distance(
        np.array([[1, 0, 0.0], [0, 1, 0.0], [0, 0, 1.0]]), p)
    assert np.allclose(dist, 0, atol=1e-9)


def test_fit_plane_svd():
    # z = 0.5 平面上的随机点
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, size=(50, 2))
    pts = np.column_stack([xy, np.full(50, 0.5)])
    plane = fit_plane_svd(pts)
    assert np.isclose(abs(plane[2]), 1.0, atol=1e-6)
    assert np.allclose(point_plane_signed_distance(pts, plane), 0, atol=1e-9)


def test_intersect_ray_plane():
    # 射线从原点沿 +z，与 z=2 平面交于 t=2
    plane2 = plane_from_point_normal(np.array([0, 0, 2.0]),
                                     np.array([0, 0, 1.0]))
    t2 = intersect_ray_plane(np.zeros(3), np.array([0, 0, 1.0]), plane2)
    assert np.isclose(t2, 2.0)


def test_intersect_ray_plane_parallel():
    # 平面法向 (0,1,0)，射线方向 (1,0,0) 与之垂直 → 平行，无交点
    plane = plane_from_point_normal(np.array([0, 1, 0.0]),
                                    np.array([0, 1, 0.0]))
    t = intersect_ray_plane(np.zeros(3), np.array([1, 0, 0.0]), plane)
    assert t is None


def test_rodrigues_90deg():
    R = rodrigues_matrix(np.array([0, 0, 1.0]), np.pi / 2)
    v = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [0, 1, 0], atol=1e-9)


def test_rodrigues_identity():
    R = rodrigues_matrix(np.array([1, 0, 0.0]), 0.0)
    assert np.allclose(R, np.eye(3))


def test_transform_about_axis():
    # 绕 z 轴（过原点）转 90°，点 (1,0,0) → (0,1,0)
    T = transform_about_axis(np.zeros(3), np.array([0, 0, 1.0]),
                             np.pi / 2)
    p = apply_transform(T, np.array([[1.0, 0, 0]]))
    assert np.allclose(p[0], [0, 1, 0], atol=1e-9)


def test_transform_about_offset_axis():
    # 轴过 (1,0,0)、方向 +z；点 (2,0,0) 转 180° → (0,0,0)
    T = transform_about_axis(np.array([1.0, 0, 0]),
                             np.array([0, 0, 1.0]), np.pi)
    p = apply_transform(T, np.array([[2.0, 0, 0]]))
    assert np.allclose(p[0], [0, 0, 0], atol=1e-9)


def test_decompose_rotation():
    axis = normalize(np.array([1.0, 2, 3]))
    angle = 0.7
    R = rodrigues_matrix(axis, angle)
    ax, ang = decompose_rotation(R)
    assert np.isclose(ang, angle, atol=1e-6)
    assert np.allclose(np.abs(ax), np.abs(axis), atol=1e-6)


def test_fit_circle_3d():
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, 30)
    r = 0.05
    circle = np.column_stack([r * np.cos(theta), r * np.sin(theta),
                              np.full(30, 0.1)])
    center, normal, radius = fit_circle_3d(circle)
    assert np.allclose(center, [0, 0, 0.1], atol=1e-6)
    assert np.isclose(radius, 0.05, atol=1e-6)
    assert np.isclose(abs(normal[2]), 1.0, atol=1e-6)


def test_rigid_transform():
    T = rigid_transform(np.eye(3), np.array([1.0, 2, 3]))
    assert np.allclose(T[:3, 3], [1, 2, 3])
    assert np.allclose(T[:3, :3], np.eye(3))
