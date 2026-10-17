"""三维重建模块：三角测量、多视角配准、网格重建、去转盘。"""
from .triangulate import triangulate, pixel_to_rays, undistort_points
from .register import (
    ViewCloud,
    RegisterResult,
    register_views,
    to_open3d,
)
from .mesh import (
    MeshResult,
    prepare_normals,
    reconstruct_mesh,
    reconstruct_mesh_ball_pivoting,
)
from .turntable_removal import (
    TurntableRemovalResult,
    estimate_table_height,
    remove_turntable_axis_height,
    remove_turntable_surface,
)
from .point_filter import (
    FilterResult,
    filter_by_depth,
    filter_by_bounds,
    remove_small_clusters,
)

__all__ = [
    "triangulate", "pixel_to_rays", "undistort_points",
    "ViewCloud", "RegisterResult", "register_views", "to_open3d",
    "MeshResult", "reconstruct_mesh", "reconstruct_mesh_ball_pivoting",
    "prepare_normals",
    "TurntableRemovalResult", "remove_turntable_surface",
    "estimate_table_height", "remove_turntable_axis_height",
    "FilterResult", "filter_by_depth", "filter_by_bounds",
    "remove_small_clusters",
]
