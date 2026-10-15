"""点云配准的稳健性测试（需要 open3d）。

覆盖修复：``_ensure_normals`` 在共面/共球点云上不应因 Qhull 精度错误
而崩溃；退化（点数过少）点云的 ICP 应安全返回恒等结果。
"""
from __future__ import annotations

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from triniscan.reconstruction.register import (  # noqa: E402
    _ensure_normals,
    _icp,
)


def _plane_cloud(n: int = 200, z: float = 0.16):
    """构造近似共面的点云（单视角激光线点云的典型形态）。"""
    pts = np.zeros((n, 3))
    pts[:, 0] = np.random.uniform(-0.01, 0.01, n)
    pts[:, 1] = np.random.uniform(-0.02, 0.02, n)
    pts[:, 2] = z + np.random.normal(0, 1e-6, n)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_ensure_normals_coplanar_does_not_crash():
    """共面点云会触发 Qhull 精度错误，_ensure_normals 必须容错通过。"""
    pcd = _plane_cloud()
    _ensure_normals(pcd, voxel_size=0.0005)   # 不应抛出
    assert len(pcd.normals) == len(pcd.points)


def test_icp_degenerate_returns_identity():
    """点数过少的退化点云，ICP 应安全返回恒等变换而非崩溃。"""
    pcd = _plane_cloud()
    tiny = o3d.geometry.PointCloud()
    tiny.points = o3d.utility.Vector3dVector(np.random.rand(3, 3))
    result = _icp(tiny, pcd, voxel_size=0.0005)
    assert np.allclose(result.transformation, np.eye(4))
    assert result.fitness == 0.0


def test_icp_identical_clouds_converges():
    """同一片点云自配准应得到恒等变换、fitness≈1。"""
    pcd = _plane_cloud()
    result = _icp(pcd, pcd, voxel_size=0.0005)
    assert result.fitness > 0.9
    assert np.allclose(result.transformation, np.eye(4), atol=1e-6)
