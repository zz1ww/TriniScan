"""统计离群过滤（Statistical Outlier Removal, SOR）。

算法
----
对每个点 ``p``：

1. 取最近 ``nb_neighbors`` 个邻居，算平均距离 ``d̄(p)``；
2. 对全体点求 ``d̄`` 的均值 ``μ`` 与标准差 ``σ``；
3. ``d̄(p) > μ + std_ratio·σ`` 的点判为离群。

它假设**点云密度均匀**。对线结构光转台扫描，这个假设经常不成立：
棱边、遮挡带、薄壁区域的点天然稀疏，``d̄`` 偏大，会被整片误删。

因此本插件**默认关闭**。若启用，务必先用 ``report.json`` 里的
``removed_ratio`` 检查删除比例——超过 5% 通常意味着在误删真实表面。

与旧实现的区别：旧版参数（``nb_neighbors=20``、``std_ratio=2.0``）
**硬编码在 ``register_views`` 里，不可配置**；这里全部开放。
"""
from __future__ import annotations

import numpy as np

from ...contracts import Cloud, FILTERS, ParamSpec, RunContext
from .base import FilterPlugin

__all__ = ["StatisticalOutlierFilter"]


@FILTERS.register("statistical")
class StatisticalOutlierFilter(FilterPlugin):
    """统计离群（SOR）。"""

    name = "statistical"
    label = "统计离群"
    depends_on_current = False
    params = (
        ParamSpec("reconstruction.statistical_nb_neighbors", int, 20,
                  low=5, high=200,
                  label="邻居数", help="计算平均距离用的近邻数", short="nb_neighbors"),
        ParamSpec("reconstruction.statistical_std_ratio", float, 2.0,
                  low=0.1, high=10.0,
                  label="标准差倍数",
                  help="平均距离超过 μ+kσ 判为离群。k 越小删得越狠", short="std_ratio"),
    )

    def compute_mask(self, cloud: Cloud, ctx: RunContext
                     ) -> tuple[np.ndarray, dict]:
        params = self.read_params(ctx.cfg)
        nb = int(params["nb_neighbors"])
        std = float(params["std_ratio"])
        n = len(cloud)
        if n < nb:
            return np.ones(n, dtype=bool), {
                "reason": f"点数 {n} 少于邻居数 {nb}，跳过",
                "applied": False}

        try:
            import open3d as o3d
        except Exception as exc:        # noqa: BLE001
            return np.ones(n, dtype=bool), {
                "reason": f"Open3D 不可用（{exc}），跳过",
                "applied": False}

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb,
                                                std_ratio=std)
        keep = np.zeros(n, dtype=bool)
        keep[np.asarray(ind, dtype=np.int64)] = True
        removed = int((~keep).sum())
        return keep, {
            "applied": True,
            "nb_neighbors": nb, "std_ratio": std,
            "removed_ratio": removed / n,
            "reason": (f"统计离群(邻居={nb}, k={std}): 剔除 {removed} 点"
                       f" ({removed / n * 100:.1f}%)"),
        }
