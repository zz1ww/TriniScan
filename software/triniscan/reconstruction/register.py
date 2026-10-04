"""多视角点云配准拼接。"""
from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

from triniscan.calibration.axis_calib import transform_about_axis


def register_views(views: list[dict],
                   axis_point: np.ndarray,
                   axis_dir: np.ndarray,
                   voxel_size: float = 0.0005,
                   icp_refine: bool = True) -> "o3d.geometry.PointCloud":
    """把各视角点云按转轴角度变换到统一坐标系并拼接。

    Parameters
    ----------
    views : list of dict
        每项 {"angle_deg": float, "points": (N,3) np.ndarray}
    axis_point, axis_dir : 转轴标定结果
    voxel_size : 配准/下采样体素
    icp_refine : 是否做 ICP 精配准

    Returns
    -------
    o3d.geometry.PointCloud
    """
    if not _HAS_O3D:
        raise ImportError("需要 open3d: pip install open3d")

    merged = o3d.geometry.PointCloud()
    first = True
    for view in views:
        pts = np.asarray(view["points"], dtype=np.float64)
        if len(pts) == 0:
            continue
        angle = np.deg2rad(view.get("angle_deg", 0.0))
        T = transform_about_axis(axis_point, axis_dir, angle)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_w = (T @ pts_h.T).T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_w)

        if first:
            merged = pcd
            first = False
        else:
            if icp_refine:
                trans = _icp(pcd, merged, voxel_size)
                pcd = pcd.transform(trans)
            merged += pcd

    merged = merged.voxel_down_sample(voxel_size)
    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    return merged


def _icp(source: "o3d.geometry.PointCloud",
         target: "o3d.geometry.PointCloud",
         voxel_size: float) -> np.ndarray:
    """点到面/点到点 ICP 精配准，返回 4x4 变换。"""
    src = source.voxel_down_sample(voxel_size)
    tgt = target.voxel_down_sample(voxel_size)
    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30))
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, voxel_size * 2, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result.transformation
