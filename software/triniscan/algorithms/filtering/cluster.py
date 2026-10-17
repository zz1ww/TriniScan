"""连通域去噪（DBSCAN 聚类）：保留最大的若干簇。

为什么它危险
------------
重构前它是 ``reconstruction.cluster_keep_top: 1``——**只留最大簇，其余
全部删除**。物体在棱边、遮挡带、反光断口处一旦断成两簇，小的一簇
（可能正是"少一面"的那一面）就会被整块删掉，且**没有任何日志**。

本插件**行为与旧实现完全一致**（默认 ``keep_top=1``），但额外输出
诊断信息，让"删了几个簇、每个多大"直接可读：

- ``clusters``：簇数
- ``cluster_sizes``：各簇点数（降序）
- ``kept_clusters``：实际保留的簇编号
- ``detached_size_ratio``：最大被删簇 / 保留簇 的比例

有了这些，若某次扫描"少一面"，报告里会直接显示
``cluster_sizes: [76400, 15200]``——第二大簇被删了 15200 点。
"""
from __future__ import annotations

import numpy as np

from ...contracts import Cloud, FILTERS, ParamSpec, RunContext
from .base import FilterPlugin
from .point_filter import remove_small_clusters

__all__ = ["ClusterFilter"]


@FILTERS.register("cluster")
class ClusterFilter(FilterPlugin):
    """欧氏聚类去噪。"""

    name = "cluster"
    label = "连通域去噪"
    depends_on_current = False          # 聚类基于点间距，纯局部判断
    params = (
        ParamSpec("reconstruction.cluster_eps_m", float, 0.002,
                  low=1e-4, high=0.05, unit="m",
                  label="邻域半径",
                  help="两点距离小于此值视为连通。太小会把物体拆散，"
                       "太大会把噪声并进物体", short="eps_m"),
        ParamSpec("reconstruction.cluster_min_points", int, 10,
                  low=2, high=500,
                  label="最少核心点", help="DBSCAN 核心点的最少邻居数", short="min_points"),
        ParamSpec("reconstruction.cluster_keep_top", int, 1,
                  low=1, high=20,
                  label="保留簇数",
                  help="只保留最大的前 k 个簇。⚠️ k=1 时其余簇（可能包含"
                       "物体的某一面）会被整块删除", short="keep_top"),
    )

    def compute_mask(self, cloud: Cloud, ctx: RunContext
                     ) -> tuple[np.ndarray, dict]:
        params = self.read_params(ctx.cfg)
        n = len(cloud)
        if n == 0:
            return np.zeros(0, dtype=bool), {"reason": "空点云",
                                             "applied": False}

        res = remove_small_clusters(
            cloud.points,
            eps=float(params["eps_m"]),
            min_points=int(params["min_points"]),
            keep_top=int(params["keep_top"]),
        )
        keep = np.asarray(res.keep_mask, dtype=bool)
        extra = {
            "applied": bool(res.applied),
            "eps_m": float(params["eps_m"]),
            "min_points": int(params["min_points"]),
            "keep_top": int(params["keep_top"]),
            "removed_ratio": round(res.removed_ratio, 6),
            "cluster_sizes": _cluster_sizes(cloud.points,
                                            float(params["eps_m"]),
                                            int(params["min_points"])),
            "noise_points": _noise_count(cloud.points,
                                         float(params["eps_m"]),
                                         int(params["min_points"])),
            "reason": res.reason or "未启用",
        }
        return keep, extra


def _noise_count(points: np.ndarray, eps: float, min_points: int) -> int:
    """噪声点数（DBSCAN 标为 -1 的点）。诊断用，失败返回 0。"""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        labels = np.asarray(pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=False))
        return int((labels == -1).sum())
    except Exception:                   # noqa: BLE001
        return 0


def _cluster_sizes(points: np.ndarray, eps: float,
                   min_points: int) -> list[int]:
    """单独跑一次聚类取各簇大小（**仅供诊断**，失败则返回空）。

    之所以重复聚类：旧实现只返回"删了多少点"，不告诉你有几个簇、各多大。
    这正是排查"少一面"最需要的信息。诊断失败不影响过滤结果。
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        labels = np.asarray(pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=False))
        uniq = [c for c in np.unique(labels) if c >= 0]
        return sorted((int((labels == c).sum()) for c in uniq), reverse=True)
    except Exception:                   # noqa: BLE001 - 诊断失败不影响主流程
        return []
