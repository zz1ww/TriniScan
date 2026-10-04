"""点云 → 三角网格重建。

方法
----
- **Poisson 重建**（默认）：平滑、抗噪、自动补洞，适合本系统。
- **Ball-Pivoting**：保边、忠实，对噪声敏感，作为备选。

Poisson 重建会生成一个"封闭的等值面"，可能包含低置信度的多余面片，
需按密度分位裁剪。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:  # pragma: no cover
    o3d = None
    _HAS_O3D = False

__all__ = ["reconstruct_mesh", "reconstruct_mesh_ball_pivoting"]


@dataclass
class MeshResult:
    """重建结果。"""

    mesh: "o3d.geometry.TriangleMesh"
    method: str
    num_vertices: int
    num_triangles: int


def _require_open3d() -> None:
    if not _HAS_O3D:
        raise ImportError("需要 open3d 进行网格重建: pip install open3d")


def _prepare_normals(pcd, search_radius: float = 0.002,
                     max_nn: int = 30) -> None:
    """估计并一致化法向（Poisson 的前提）。"""
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=max_nn
            )
        )
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    # 统一朝外：让法向尽量背离质心
    centroid = np.asarray(pcd.points).mean(axis=0)
    pcd.orient_normals_towards_camera_location(centroid)


def reconstruct_mesh(
    pcd,
    poisson_depth: int = 9,
    density_quantile: float = 0.02,
    normalize_normals: bool = True,
) -> MeshResult:
    """Poisson 重建 + 密度裁剪。

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
    poisson_depth : 八叉树深度（越大越细，也越慢/越噪），建议 8~10
    density_quantile : 裁掉密度最低的该分位面片
    normalize_normals : 是否先一致化法向

    Returns
    -------
    MeshResult
    """
    _require_open3d()
    if normalize_normals:
        _prepare_normals(pcd)

    mesh, densities = o3d.geometry.TriangleMesh \
        .create_from_point_cloud_poisson(pcd, depth=poisson_depth)

    densities = np.asarray(densities)
    if densities.size > 0 and density_quantile > 0:
        threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    mesh.compute_vertex_normals()
    result = MeshResult(
        mesh=mesh,
        method="poisson",
        num_vertices=len(mesh.vertices),
        num_triangles=len(mesh.triangles),
    )
    log.info("Poisson 重建完成: %d 顶点, %d 面片",
             result.num_vertices, result.num_triangles)
    return result


def reconstruct_mesh_ball_pivoting(
    pcd,
    radii: Optional[list[float]] = None,
) -> MeshResult:
    """Ball-Pivoting 重建（保边备选）。"""
    _require_open3d()
    if radii is None:
        radii = [0.001, 0.002, 0.004]
    _prepare_normals(pcd)

    mesh = o3d.geometry.TriangleMesh \
        .create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    mesh.compute_vertex_normals()

    result = MeshResult(
        mesh=mesh,
        method="ball_pivoting",
        num_vertices=len(mesh.vertices),
        num_triangles=len(mesh.triangles),
    )
    log.info("Ball-Pivoting 重建完成: %d 顶点, %d 面片",
             result.num_vertices, result.num_triangles)
    return result
