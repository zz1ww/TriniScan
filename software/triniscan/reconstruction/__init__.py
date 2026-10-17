"""【兼容层】旧的 ``triniscan.reconstruction`` 路径。

实现已按职责拆分到：

- :mod:`triniscan.algorithms.features.triangulate`   三角测量
- :mod:`triniscan.algorithms.registration.register`  多视角配准
- :mod:`triniscan.algorithms.meshing.mesh`           表面重建
- :mod:`triniscan.algorithms.filtering.*`            点云过滤 / 去转盘

**请勿在此添加新逻辑。**
"""
from __future__ import annotations

from ..algorithms.features.triangulate import (      # noqa: F401
    pixel_to_rays, triangulate, undistort_points)
from ..algorithms.registration.register import (     # noqa: F401
    RegisterResult, ViewCloud, register_views, to_open3d)
from ..algorithms.meshing.mesh import (              # noqa: F401
    MeshResult, prepare_normals, reconstruct_mesh,
    reconstruct_mesh_ball_pivoting)
from ..algorithms.filtering.turntable_removal import (  # noqa: F401
    TurntableRemovalResult, estimate_table_height,
    remove_turntable_axis_height, remove_turntable_surface)
from ..algorithms.filtering.point_filter import (    # noqa: F401
    FilterResult, filter_by_bounds, filter_by_depth,
    remove_small_clusters)

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
