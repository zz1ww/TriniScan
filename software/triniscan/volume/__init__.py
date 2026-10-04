"""体积计算模块：网格封闭化 + 体积积分。"""
from .close_mesh import close_mesh
from .volume import compute_volume, compute_volume_voxel

__all__ = ["close_mesh", "compute_volume", "compute_volume_voxel"]
