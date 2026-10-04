"""点云 → 三角网格重建（Poisson）。"""
from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False


def reconstruct_mesh(pcd,
                     poisson_depth: int = 9,
                     density_quantile: float = 0.02,
                     remove_outlier_k: int = 20,
                     remove_outlier_std: float = 2.0):
    """点云 Poisson 重建 + 密度裁剪。

    Returns
    -------
    o3d.geometry.TriangleMesh
    """
    if not _HAS_O3D:
        raise ImportError("需要 open3d: pip install open3d")

    pcd = pcd.remove_statistical_outlier(
        nb_neighbors=remove_outlier_k,
        std_ratio=remove_outlier_std)[0]
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh \
        .create_from_point_cloud_poisson(pcd, depth=poisson_depth)

    # 按密度分位裁剪低置信度面片
    densities = np.asarray(densities)
    if len(densities) > 0:
        thresh = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < thresh)

    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__":
    print("需输入点云数据；见 docs/05_体积计算方案.md")
