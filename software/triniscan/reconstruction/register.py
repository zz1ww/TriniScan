"""多视角点云配准与拼接。

流程
----
1. 按转台角度，把每个视角的点云绕**旋转轴**变换到统一坐标系；
2. 用 ICP 对相邻视角做精配准（补偿转台角度误差）；
3. 合并、下采样、去离群，得到完整点云。

坐标约定
--------
所有视角点云先在**相机坐标系**下（由三角测量得到），
变换到以转轴为参考的**世界坐标系**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..common.geometry import apply_transform, transform_about_axis
from ..common.logging_utils import get_logger

log = get_logger(__name__)

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:  # pragma: no cover
    o3d = None
    _HAS_O3D = False

__all__ = ["ViewCloud", "register_views", "to_open3d"]


@dataclass
class ViewCloud:
    """单个视角的点云与角度。"""

    angle_deg: float
    points: np.ndarray                  # (N, 3) 相机系
    meta: dict = field(default_factory=dict)


def _require_open3d() -> None:
    if not _HAS_O3D:
        raise ImportError(
            "需要 open3d 进行点云配准: pip install open3d"
        )


def to_open3d(points: np.ndarray) -> "o3d.geometry.PointCloud":
    """NumPy 点集 → Open3D 点云。"""
    _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )
    return pcd


def register_views(
    views: Sequence[ViewCloud],
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    voxel_size: float = 0.0005,
    icp_refine: bool = True,
    icp_max_dist_factor: float = 2.0,
    statistical_outlier_k: int = 20,
    statistical_outlier_std: float = 2.0,
) -> "o3d.geometry.PointCloud":
    """把各视角点云变换到统一坐标系并融合。

    Parameters
    ----------
    views : ViewCloud 序列
    axis_point, axis_dir : 转轴标定结果
    voxel_size : 下采样与 ICP 的体素尺度 (m)
    icp_refine : 是否做 ICP 精配准
    icp_max_dist_factor : ICP 最大对应距离 = factor × voxel_size
    statistical_outlier_k, statistical_outlier_std : 去离群参数

    Returns
    -------
    o3d.geometry.PointCloud: 融合后的点云
    """
    _require_open3d()
    if not views:
        raise ValueError("没有可配准的视角")

    merged: Optional["o3d.geometry.PointCloud"] = None

    for view in views:
        pts = np.asarray(view.points, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            continue

        transform = transform_about_axis(
            axis_point, axis_dir, np.deg2rad(view.angle_deg)
        )
        pts_world = apply_transform(transform, pts)
        pcd = to_open3d(pts_world)

        if merged is None:
            merged = pcd
        else:
            if icp_refine:
                trans = _icp(pcd, merged, voxel_size,
                             icp_max_dist_factor)
                pcd = pcd.transform(trans)
            merged += pcd

    if merged is None:
        raise ValueError("所有视角均为空，无法配准")

    merged = merged.voxel_down_sample(voxel_size)
    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=statistical_outlier_k,
        std_ratio=statistical_outlier_std,
    )
    log.info("点云配准完成: %d 点, 体素 %.1f mm",
             len(merged.points), voxel_size * 1e3)
    return merged


def _icp(source: "o3d.geometry.PointCloud",
         target: "o3d.geometry.PointCloud",
         voxel_size: float,
         max_dist_factor: float) -> np.ndarray:
    """点到面 ICP 精配准，返回 4x4 变换。"""
    src = source.voxel_down_sample(voxel_size)
    tgt = target.voxel_down_sample(voxel_size)

    radius = voxel_size * 2
    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel_size * max_dist_factor,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration
        .TransformationEstimationPointToPlane(),
    )
    return result.transformation
