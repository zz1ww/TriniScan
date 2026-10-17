"""物理边界过滤：按**世界系**半径 + 轴向高度剔除越界点。

与"去转盘"不同，这里剔除的是**转台半径之外 / 高度范围之外**的虚假点
（反光、环境光）。三个门限全部为空时**不启用**——避免误伤大物体。

行为与重构前 ``ScanPipeline._filter_bounds`` 一致：门限全空时直接
返回原云（``applied=False``），而不是把所有点都判为越界。
"""
from __future__ import annotations

import numpy as np

from ...contracts import Cloud, FILTERS, ParamSpec, RunContext
from .base import FilterPlugin

__all__ = ["BoundsFilter"]


@FILTERS.register("bounds")
class BoundsFilter(FilterPlugin):
    """世界系物理边界（半径 + 沿轴高度）。"""

    name = "bounds"
    label = "物理边界"
    depends_on_current = False          # 纯按点自身坐标判断
    params = (
        ParamSpec("reconstruction.bounds_radius_m", float, 0.0,
                  low=0.0, high=2.0, unit="m",
                  label="最大半径", help="距转轴超过此半径的点丢弃；0 表示不限", short="radius_m"),
        ParamSpec("reconstruction.bounds_z_min_m", float, None,
                  label="轴向下限", help="沿轴高度低于此值丢弃；空表示不限", short="z_min_m"),
        ParamSpec("reconstruction.bounds_z_max_m", float, None,
                  label="轴向上限", help="沿轴高度高于此值丢弃；空表示不限", short="z_max_m"),
    )

    def compute_mask(self, cloud: Cloud, ctx: RunContext
                     ) -> tuple[np.ndarray, dict]:
        params = self.read_params(ctx.cfg)
        r_max = float(params.get("radius_m") or 0.0)
        lo = params.get("z_min_m")
        hi = params.get("z_max_m")
        n = len(cloud)

        if r_max <= 0.0 and lo is None and hi is None:
            return np.ones(n, dtype=bool), {
                "reason": "三个门限均未设置，未启用", "applied": False}
        if n == 0:
            return np.zeros(0, dtype=bool), {"reason": "空点云",
                                             "applied": False}

        axis_point, axis = _axis(ctx)
        rel = cloud.points - axis_point
        axial = rel @ axis
        radial = rel - np.outer(axial, axis)
        r_dist = np.linalg.norm(radial, axis=1)

        keep = np.ones(n, dtype=bool)
        if r_max > 0.0:
            keep &= r_dist <= r_max
        if lo is not None:
            keep &= axial >= float(lo)
        if hi is not None:
            keep &= axial <= float(hi)

        removed = int((~keep).sum())
        parts = []
        if r_max > 0.0:
            parts.append(f"r<={r_max * 1e3:.0f}mm")
        if lo is not None or hi is not None:
            parts.append(f"h∈[{lo if lo is not None else '-∞'},"
                         f"{hi if hi is not None else '+∞'}]m")
        return keep, {
            "applied": True,
            "radius_max_m": r_max, "z_min_m": lo, "z_max_m": hi,
            "radius_p99_m": float(np.percentile(r_dist, 99)) if n else 0.0,
            "reason": (f"边界门限({', '.join(parts)}): 剔除 {removed} 点"
                       f" ({removed / n * 100:.1f}%)"),
        }


def _axis(ctx: RunContext):
    """取转轴（优先 ctx，回退到 ctx.calib）。"""
    if ctx.axis_point is not None and ctx.axis_dir is not None:
        p = np.asarray(ctx.axis_point, dtype=np.float64).reshape(3)
        d = np.asarray(ctx.axis_dir, dtype=np.float64).reshape(3)
    elif ctx.calib is not None:
        p = ctx.calib.axis_point
        d = ctx.calib.axis_dir
    else:
        raise ValueError("边界过滤需要转轴，但 ctx 中没有 axis_point/axis_dir")
    nrm = np.linalg.norm(d)
    return p, (d / nrm if nrm > 1e-12 else d)
