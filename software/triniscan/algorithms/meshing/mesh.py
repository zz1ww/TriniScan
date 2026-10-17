"""点云 → 三角网格重建。

方法
----
- **Poisson 重建**（默认）：平滑、抗噪、自动补洞，适合本系统。
- **Ball-Pivoting**：保边、忠实，对噪声敏感，作为备选。

Poisson 重建会生成一个"封闭的等值面"，可能包含低置信度的多余面片，
需按密度分位裁剪。

稳健性
------
Poisson 依赖**一致的法向**。法向一致化（``orient_normals_consistent_
tangent_plane``）内部会调用 Qhull，遇到退化/坏点云会抛异常。本模块：

- 自适应选择近邻数 ``k``（随点数、且上限保护）；
- 该步骤失败时**自动回退**到"朝向相机/质心"的定向，保证流程不中断；
- Poisson 的八叉树深度可按点云尺度自适应。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...common.logging_utils import get_logger

log = get_logger(__name__)

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:  # pragma: no cover
    o3d = None
    _HAS_O3D = False

__all__ = ["reconstruct_mesh", "reconstruct_mesh_ball_pivoting",
           "MeshResult", "prepare_normals"]


@dataclass
class MeshResult:
    """重建结果。"""

    mesh: "o3d.geometry.TriangleMesh"
    method: str
    num_vertices: int
    num_triangles: int
    depth_used: int = 0
    normals_fallback: bool = False
    num_points: int = 0
    notes: list[str] = field(default_factory=list)


def _require_open3d() -> None:
    if not _HAS_O3D:
        raise ImportError("需要 open3d 进行网格重建: pip install open3d")


def prepare_normals(pcd, search_radius: float = 0.002,
                    max_nn: int = 30,
                    prefer_tangent: bool = True) -> bool:
    """估计并一致化法向（Poisson 的前提）。

    稳健化处理：

    1. 自适应近邻数 ``k = clip(len(points)-1, 3, max_nn)``；
    2. 优先用"切平面一致化"（效果最好）；
    3. 失败（Qhull 退化）时回退到"朝向相机/质心"的定向，不抛异常。

    Returns
    -------
    bool
        是否发生了回退（True 表示使用了兜底定向）。
    """
    _require_open3d()
    points = np.asarray(pcd.points)
    n = len(points)
    if n == 0:
        return False

    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=max_nn
            )
        )

    # 自适应 k：必须 < 点数，且不小于 3
    k = int(min(max_nn, max(n - 1, 3)))
    fallback = False
    if prefer_tangent:
        try:
            pcd.orient_normals_consistent_tangent_plane(k=k)
        except Exception as exc:  # noqa: BLE001 - Qhull/退化点云
            fallback = True
            log.warning("一致化法向失败（k=%d），回退到朝向质心定向: %s",
                        k, exc)
            centroid = points.mean(axis=0)
            pcd.orient_normals_towards_camera_location(centroid)
    else:
        fallback = True
        centroid = points.mean(axis=0)
        pcd.orient_normals_towards_camera_location(centroid)
    return fallback


# 向后兼容旧名
def _prepare_normals(pcd, search_radius: float = 0.002,
                     max_nn: int = 30) -> None:
    prepare_normals(pcd, search_radius=search_radius, max_nn=max_nn)


def _suggest_depth(pcd, requested: int, adaptive: bool) -> int:
    """按点云尺度/点数建议 Poisson 深度（限制在合理区间 7~11）。"""
    if not adaptive:
        return int(requested)
    n = len(pcd.points)
    # 点越多、包围盒越大，允许越深；但避免过大导致爆内存
    if n < 20_000:
        depth = min(requested, 8)
    elif n < 200_000:
        depth = requested
    else:
        depth = max(requested, 10)
    return int(np.clip(depth, 7, 11))


def reconstruct_mesh(
    pcd,
    poisson_depth: int = 9,
    density_quantile: float = 0.02,
    normalize_normals: bool = True,
    adaptive_depth: bool = True,
    orient_max_nn: int = 30,
) -> MeshResult:
    """Poisson 重建 + 密度裁剪。

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
    poisson_depth : 八叉树深度（越大越细，也越慢/越噪），建议 8~10
    density_quantile : 裁掉密度最低的该分位面片
    normalize_normals : 是否先一致化法向
    adaptive_depth : 是否按点云尺度自适应调整深度
    orient_max_nn : 法向一致化的近邻数上限

    Returns
    -------
    MeshResult
    """
    _require_open3d()
    notes: list[str] = []
    fallback = False
    if normalize_normals:
        # 直接传入数值以控制参数；点云会被就地修改（调用方持有副本）
        fallback = prepare_normals(pcd, max_nn=orient_max_nn)
        if fallback:
            notes.append("法向一致化回退（可能影响封闭性）")

    depth = _suggest_depth(pcd, poisson_depth, adaptive_depth)
    if depth != poisson_depth:
        notes.append(f"深度自适应 {poisson_depth}→{depth}")

    try:
        mesh, densities = o3d.geometry.TriangleMesh \
            .create_from_point_cloud_poisson(pcd, depth=depth)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Poisson 重建失败（depth={depth}）: {exc}") from exc

    densities = np.asarray(densities)
    if densities.size > 0 and density_quantile > 0:
        quantile = float(np.clip(density_quantile, 0.0, 0.5))
        threshold = np.quantile(densities, quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    result = MeshResult(
        mesh=mesh,
        method="poisson",
        num_vertices=len(mesh.vertices),
        num_triangles=len(mesh.triangles),
        depth_used=depth,
        normals_fallback=fallback,
        num_points=len(pcd.points),
        notes=notes,
    )
    log.info("Poisson 重建完成: %d 顶点, %d 面片 (depth=%d%s)",
             result.num_vertices, result.num_triangles, depth,
             "，法向回退" if fallback else "")
    return result


def reconstruct_mesh_ball_pivoting(
    pcd,
    radii: Optional[list[float]] = None,
) -> MeshResult:
    """Ball-Pivoting 重建（保边备选）。"""
    _require_open3d()
    if radii is None:
        radii = [0.001, 0.002, 0.004]
    fallback = prepare_normals(pcd)

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
        normals_fallback=fallback,
        num_points=len(pcd.points),
    )
    log.info("Ball-Pivoting 重建完成: %d 顶点, %d 面片",
             result.num_vertices, result.num_triangles)
    return result
