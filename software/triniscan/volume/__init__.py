"""体积计算模块：网格封闭化 + 体积积分。"""
from .close_mesh import (
    CloseMeshResult,
    close_mesh,
    is_watertight,
    mesh_from_open3d,
    mesh_to_open3d,
)
from .volume import (
    VolumeResult,
    compute_volume,
    compute_volume_divergence,
    compute_volume_voxel,
    volume_m3_to_cm3,
)

__all__ = [
    "close_mesh", "CloseMeshResult", "is_watertight",
    "mesh_from_open3d", "mesh_to_open3d",
    "compute_volume", "VolumeResult",
    "compute_volume_divergence", "compute_volume_voxel",
    "volume_m3_to_cm3",
]
