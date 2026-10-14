"""去转盘表面点单元测试。"""
import numpy as np
import pytest

from triniscan.reconstruction.turntable_removal import (
    TurntableRemovalResult,
    remove_turntable_surface,
)


def _scene(n_grid=40, span=0.12, obj_h=0.05, obj_r=0.02, seed=0):
    """构造场景：z=0 处的大平面（转盘）+ 平面上的物体点云。"""
    rng = np.random.default_rng(seed)
    # 转盘（竖直方向为 z 轴，法向 +z）
    xy = rng.uniform(-span, span, size=(n_grid * n_grid, 2))
    disk = np.column_stack([xy, np.zeros(len(xy))])
    # 物体：半径 r 的柱体
    ang = rng.uniform(0, 2 * np.pi, size=1500)
    rad = obj_r * np.sqrt(rng.uniform(0, 1, size=1500))
    obj = np.column_stack([
        rad * np.cos(ang), rad * np.sin(ang),
        rng.uniform(0.001, obj_h, size=1500),
    ])
    pts = np.vstack([disk, obj])
    pts += rng.normal(scale=1e-5, size=pts.shape)
    return pts, len(disk)


def test_removes_disk_keeps_object():
    pts, n_disk = _scene()
    res = remove_turntable_surface(
        pts, axis_dir=np.array([0.0, 0.0, 1.0]),
        ransac_dist=0.001, margin=0.0005, keep_upper_only=True,
    )
    assert res.applied
    # 大部分转盘点被剔除
    assert res.num_removed > 0.5 * n_disk
    # 物体点（z>0.005）基本保留
    high = pts[:, 2] > 0.005
    kept_high = (~res.removed_mask[high]).sum()
    assert kept_high > 0.9 * high.sum()


def test_skips_when_no_plane():
    # 纯随机点云，没有明显平面 → 不应误删
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.05, 0.05, size=(500, 3))
    res = remove_turntable_surface(pts, axis_dir=np.array([0.0, 0.0, 1.0]),
                                   min_inlier_ratio=0.5)
    assert not res.applied
    assert res.num_removed == 0


def test_skips_when_plane_not_aligned_with_axis():
    # 平面法向与转轴垂直 → 应是物体侧壁，跳过
    rng = np.random.default_rng(2)
    xy = rng.uniform(-0.05, 0.05, size=(2000, 2))
    wall = np.column_stack([np.zeros(len(xy)), xy])  # x=0 平面，法向 = x
    res = remove_turntable_surface(
        wall, axis_dir=np.array([0.0, 0.0, 1.0]), align_tol_deg=15.0)
    assert not res.applied


def test_radius_limit():
    pts, _ = _scene(span=0.15)
    res = remove_turntable_surface(
        pts, axis_dir=np.array([0.0, 0.0, 1.0]),
        axis_point=np.zeros(3), radius=0.03,
        ransac_dist=0.001, margin=0.0005, keep_upper_only=True,
    )
    assert res.applied
    # 半径外的转盘点不应被删除
    far = np.linalg.norm(pts[:, :2], axis=1) > 0.05
    assert res.removed_mask[far].sum() == 0


def test_too_few_points():
    res = remove_turntable_surface(np.zeros((3, 3)))
    assert not res.applied
    assert isinstance(res, TurntableRemovalResult)
