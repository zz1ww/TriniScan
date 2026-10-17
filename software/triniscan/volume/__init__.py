"""【兼容层】旧的 ``triniscan.volume`` 路径。

实现已迁移到 :mod:`triniscan.algorithms.closing` 与
:mod:`triniscan.algorithms.volumetry`。
"""
from __future__ import annotations

from ..algorithms.closing.close_mesh import (        # noqa: F401
    CloseMeshResult, close_mesh, is_watertight, mesh_from_open3d,
    mesh_to_open3d)
from ..algorithms.volumetry.volume import (          # noqa: F401
    VolumeResult, compute_volume, compute_volume_divergence,
    compute_volume_voxel, volume_m3_to_cm3)

__all__ = [
    "close_mesh", "CloseMeshResult", "is_watertight",
    "mesh_from_open3d", "mesh_to_open3d",
    "compute_volume", "VolumeResult",
    "compute_volume_divergence", "compute_volume_voxel",
    "volume_m3_to_cm3",
]
