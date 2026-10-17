"""网格封闭化：把开放三角网格补成水密体。"""
from .close_mesh import (CloseMeshResult, close_mesh, is_watertight,  # noqa: F401
                         mesh_from_open3d, mesh_to_open3d)

__all__ = ["close_mesh", "CloseMeshResult", "is_watertight",
           "mesh_from_open3d", "mesh_to_open3d"]
