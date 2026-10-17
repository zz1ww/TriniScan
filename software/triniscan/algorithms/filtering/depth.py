"""深度过滤：按**相机系**深度剔除远近点。

这是整条链里**最早生效**的过滤（在三角测量后立即执行），也是"飞碟"
点云的主要成因治理：环境光、窗缝、反光会在远处三角出虚假点。

行为与重构前 ``reconstruction.filter_by_depth`` 完全一致：

- 只有上下限**至少一个为正**时才生效；
- 否则原样返回（``applied=False``），防止误配成"全部丢弃"。
"""
from __future__ import annotations

import numpy as np

from ...contracts import Cloud, FILTERS, ParamSpec, RunContext
from .base import FilterPlugin

__all__ = ["DepthFilter"]


@FILTERS.register("depth")
class DepthFilter(FilterPlugin):
    """相机系深度门限。"""

    name = "depth"
    label = "深度门限"
    params = (
        ParamSpec("reconstruction.depth_min_m", float, 0.0,
                  low=0.0, high=10.0, unit="m",
                  label="深度下限", help="小于此距离的点丢弃；0 表示不限", short="depth_min_m"),
        ParamSpec("reconstruction.depth_max_m", float, 0.3,
                  low=0.0, high=10.0, unit="m",
                  label="深度上限", help="大于此距离的点丢弃；0 表示不限", short="depth_max_m"),
    )

    def compute_mask(self, cloud: Cloud, ctx: RunContext
                     ) -> tuple[np.ndarray, dict]:
        params = self.read_params(ctx.cfg)
        lo = float(params.get("depth_min_m") or 0.0)
        hi = float(params.get("depth_max_m") or 0.0)
        n = len(cloud)
        if n == 0:
            return np.zeros(0, dtype=bool), {"reason": "空点云", "applied": False}
        if lo <= 0.0 and hi <= 0.0:
            return np.ones(n, dtype=bool), {
                "reason": "上下限均为 0，未启用", "applied": False}

        depth = cloud.points[:, 2]
        keep = np.ones(n, dtype=bool)
        if lo > 0.0:
            keep &= depth >= lo
        if hi > 0.0:
            keep &= depth <= hi

        removed = int((~keep).sum())
        return keep, {
            "applied": True,
            "depth_min_m": lo, "depth_max_m": hi,
            "depth_p01_m": float(np.percentile(depth, 1)) if n else 0.0,
            "depth_p99_m": float(np.percentile(depth, 99)) if n else 0.0,
            "reason": (f"深度门限 [{lo if lo > 0 else '-∞'}, "
                       f"{hi if hi > 0 else '+∞'}] m: 剔除 {removed} 点"
                       f" ({removed / n * 100:.1f}%)"),
        }
