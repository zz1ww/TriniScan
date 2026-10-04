"""体积计算与网格封闭化测试（含标准件精度验证）。"""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from triniscan.volume import (
    close_mesh,
    compute_volume,
    compute_volume_divergence,
    compute_volume_voxel,
    volume_m3_to_cm3,
)


def test_cube_volume():
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    vol = compute_volume_divergence(box)
    assert np.isclose(vol, 1e-3, rtol=1e-6)  # 0.1³ = 1e-3 m³


def test_cylinder_volume():
    cyl = trimesh.creation.cylinder(radius=0.05, height=0.1, sections=256)
    vol = compute_volume_divergence(cyl)
    expected = np.pi * 0.05 ** 2 * 0.1
    assert np.isclose(vol, expected, rtol=1e-2)


def test_sphere_volume():
    sph = trimesh.creation.icosphere(subdivisions=5, radius=0.05)
    vol = compute_volume_divergence(sph)
    expected = 4 / 3 * np.pi * 0.05 ** 3
    assert np.isclose(vol, expected, rtol=5e-3)


def test_voxel_cross_check():
    # 体素法受边界量化影响（约 ±1 体素/面），用较细体素并放宽容差
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    vol_div = compute_volume_divergence(box)
    vol_vox = compute_volume_voxel(box, pitch=0.002)
    assert np.isclose(vol_div, vol_vox, rtol=0.15)


def test_compute_volume_result():
    box = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    result = compute_volume(box, cross_check=True)
    assert result.method == "divergence"
    assert result.watertight
    assert np.isclose(result.volume_cm3, 125.0, rtol=1e-3)  # 5cm 立方体
    assert result.cross_check_m3 is not None


def test_non_watertight_raises():
    # 一个开口的平面网格不是水密
    plane = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    plane.faces = plane.faces[:-2]  # 破坏水密
    with pytest.raises(ValueError):
        compute_volume_divergence(plane, require_watertight=True)


def test_close_mesh_watertight():
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    result = close_mesh(box)
    assert result.watertight
    assert result.fallback_used == ""


def test_volume_unit_conversion():
    assert np.isclose(volume_m3_to_cm3(1e-3), 1000.0)
