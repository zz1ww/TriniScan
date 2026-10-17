"""转盘去除：轴基准高度法测试。

核心场景：**物体带大面积竖直面**（RANSAC 平面法会被带偏），
轴基准高度法应仍然稳定。
"""
import numpy as np

from triniscan.reconstruction.turntable_removal import (
    estimate_table_height,
    remove_turntable_axis_height,
    remove_turntable_surface,
)

AXIS_POINT = np.array([0.0, 0.0, 0.0])
AXIS_DIR = np.array([0.0, 0.0, 1.0])


def _disk(radius=0.03, z=0.0, n=2000, noise=0.0, seed=0):
    """转盘面点（z=0 平面圆盘）。"""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    x = r * np.cos(th)
    y = r * np.sin(th)
    zs = np.full(n, z) + (rng.normal(0, noise, n) if noise > 0 else 0.0)
    return np.column_stack([x, y, zs])


def _box(size=0.02, z0=0.001, n=1500, seed=1):
    """方块点云（含竖直面）——用于测试对 RANSAC 的干扰。"""
    rng = np.random.default_rng(seed)
    h = size
    # 顶面 + 四个竖直面
    pts = []
    per = n // 5
    for _ in range(per):   # 顶面
        pts.append([rng.uniform(-size, size), rng.uniform(-size, size),
                    z0 + h])
    for face in range(4):  # 四个竖直面
        for _ in range(per):
            a = rng.uniform(-size, size)
            b = rng.uniform(z0, z0 + h)
            if face == 0:
                pts.append([a, -size, b])
            elif face == 1:
                pts.append([a, size, b])
            elif face == 2:
                pts.append([-size, a, b])
            else:
                pts.append([size, a, b])
    return np.asarray(pts)


def test_estimate_table_height_finds_disk_below_box():
    """转盘面（z=0 圆盘）在方块（z>0）下方，应被估计到 0 附近。"""
    pts = np.vstack([_disk(radius=0.03, z=0.0), _box(z0=0.001)])
    h = estimate_table_height(pts, AXIS_POINT, AXIS_DIR)
    assert h is not None
    assert abs(h - 0.0) < 0.003


def test_axis_height_removes_disk_keeps_box():
    """轴基准法：应剔除 z≈0 的盘面点，保留方块点。"""
    disk = _disk(radius=0.03, z=0.0)
    box = _box(z0=0.001)
    pts = np.vstack([disk, box])
    res = remove_turntable_axis_height(
        pts, AXIS_POINT, AXIS_DIR, h_table=0.0, margin=0.0005,
    )
    assert res.applied
    assert res.num_removed >= len(disk) * 0.8      # 大部分盘面点被剔除
    # 方块点（z >= 0.001+margin）应保留
    kept_z = res.points[:, 2]
    assert (kept_z > 0.001).sum() > len(box) * 0.8


def test_axis_height_not_fooled_by_vertical_faces():
    """关键：物体有竖直面时，轴基准法不应把整个物体删掉。"""
    disk = _disk(radius=0.03, z=0.0)
    box = _box(z0=0.001)
    pts = np.vstack([disk, box])

    # 对照：RANSAC 可能选到方块竖直面（法向 ⊥ 轴）而被跳过
    ransac_res = remove_turntable_surface(
        pts, axis_dir=AXIS_DIR, axis_point=AXIS_POINT,
        align_tol_deg=15.0,
    )

    # 轴基准法：稳定剔除盘面
    axis_res = remove_turntable_axis_height(
        pts, AXIS_POINT, AXIS_DIR, h_table=0.0, margin=0.0005,
    )
    assert axis_res.applied
    # 保留点里方块应占多数
    kept_upper = (axis_res.points[:, 2] > 0.001).sum()
    assert kept_upper > 0.7 * len(box)


def test_axis_height_margin_protects_low_object():
    """若物体底部紧贴盘面，margin 应保护其不被误删。"""
    disk = _disk(radius=0.03, z=0.0)
    obj = _box(z0=0.0008)          # 物体底面略高于盘面
    pts = np.vstack([disk, obj])
    res = remove_turntable_axis_height(
        pts, AXIS_POINT, AXIS_DIR, h_table=0.0, margin=0.0005,
    )
    kept_z = res.points[:, 2]
    assert kept_z.size > 0
    assert (kept_z >= 0.0005).all()


def test_axis_height_radius_limit_protects_outside():
    """限定半径时应只删半径内的盘面点，半径外不动。"""
    disk = _disk(radius=0.03, z=0.0)
    far = np.array([[0.08, 0.0, 0.0], [0.0, 0.08, 0.0]])
    pts = np.vstack([disk, far])
    res = remove_turntable_axis_height(
        pts, AXIS_POINT, AXIS_DIR, h_table=0.0, margin=0.0005,
        radius=0.035,
    )
    # 半径外的两个点应保留
    r = np.linalg.norm(res.points[:, :2], axis=1)
    assert (r > 0.05).sum() == 2


def test_axis_height_clamp_below_cloud_min():
    """h_table 低于点云最低时，应抬升到点云最低（安全兜底）。"""
    pts = _box(z0=0.005)           # 所有点 z >= 0.005
    res = remove_turntable_axis_height(
        pts, AXIS_POINT, AXIS_DIR, h_table=-0.01, margin=0.0005,
    )
    # 不应删除所有点
    assert len(res.points) > 0
    assert "抬升" in res.reason


def test_axis_height_empty_input():
    res = remove_turntable_axis_height(
        np.zeros((0, 3)), AXIS_POINT, AXIS_DIR, h_table=0.0)
    assert not res.applied


def test_estimate_table_height_none_on_tiny_cloud():
    assert estimate_table_height(np.zeros((5, 3)), AXIS_POINT, AXIS_DIR) is None
