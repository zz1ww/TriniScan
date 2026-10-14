"""网格平面封底（close_mesh bottom_plane）与激光稳健标定测试。"""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from triniscan.volume import close_mesh, is_watertight


def test_watertight_cylinder_untouched():
    cyl = trimesh.creation.cylinder(radius=0.03, height=0.08, sections=64)
    res = close_mesh(cyl)
    assert res.watertight
    assert res.fallback_used == ""
    assert res.num_boundary_loops == 0


def test_open_bottom_capped_by_plane():
    """去掉柱体底面 → 有开孔 → 应被平面封底补回，且水密。"""
    cyl = trimesh.creation.cylinder(radius=0.03, height=0.08, sections=64)
    # 找到位于底部（z 最小）的面并删除，制造开口
    z = cyl.triangles[:, :, 2].mean(axis=1)
    keep = z > (z.min() + 1e-6)
    open_mesh = trimesh.Trimesh(vertices=cyl.vertices,
                                faces=cyl.faces[keep], process=False)
    assert not open_mesh.is_watertight

    res = close_mesh(open_mesh, fallback="bottom_plane",
                     axis_dir=np.array([0.0, 0.0, 1.0]))
    assert res.fallback_used.startswith("bottom_plane")
    assert res.watertight
    assert res.capped_loops >= 1


def test_is_watertight_compat():
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    assert is_watertight(box)


def test_bottom_plane_preferred_over_convex_hull():
    """平面封底不应像凸包那样把体积撑大。"""
    cyl = trimesh.creation.cylinder(radius=0.03, height=0.08, sections=64)
    z = cyl.triangles[:, :, 2].mean(axis=1)
    keep = z > (z.min() + 1e-6)
    open_mesh = trimesh.Trimesh(vertices=cyl.vertices,
                                faces=cyl.faces[keep], process=False)

    plane_res = close_mesh(open_mesh, fallback="bottom_plane",
                           axis_dir=np.array([0.0, 0.0, 1.0]))
    expected = np.pi * 0.03 ** 2 * 0.08
    assert abs(plane_res.mesh.volume - expected) < 0.1 * expected
