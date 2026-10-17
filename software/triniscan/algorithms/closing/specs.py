"""网格封闭化的参数声明。

``close_fallback="bottom_plane"`` 此前在 pipeline 与 close_mesh.py 各写
一次默认值；现在只在下面这一处。
"""
from __future__ import annotations

from triniscan.contracts import AlgorithmSpec, CLOSERS, ParamSpec

from .close_mesh import close_mesh

__all__ = ["CLOSE_SPEC", "specs"]


CLOSE_SPEC = AlgorithmSpec(
    name="default",
    label="平面封底",
    func=close_mesh,
    help="按边界环封口：优先封最靠底部的近似平面环，失败时按 fallback 降级",
    tags=("拓扑", "默认"),
    params=(
        ParamSpec("reconstruction.close_fallback", str, "bottom_plane",
                  choices=("bottom_plane", "convex_hull", "voxel"),
                  short="fallback",
                  label="兜底策略",
                  help="bottom_plane=平面封底(推荐)；convex_hull=凸包；"
                       "voxel=体素并集"),
        ParamSpec("reconstruction.allow_convex_hull", bool, True,
                  short="allow_convex_hull",
                  label="允许凸包兜底",
                  help="平面封底仍失败时是否允许凸包（会把凹陷填平）"),
        ParamSpec("reconstruction.close_cap_all_planar_loops", bool, True,
                  short="cap_all_planar_loops",
                  label="封闭所有平面环",
                  help="否时只封最底/最顶一个环"),
        ParamSpec("reconstruction.close_planar_tol_m", float, 0.002,
                  low=0.0, high=0.05, unit="m", short="planar_tol",
                  label="平面环容差",
                  help="边界环各点共面程度在此范围内才当作平面"),
    ),
)

specs = (CLOSE_SPEC,)
CLOSERS.add_spec(CLOSE_SPEC)
