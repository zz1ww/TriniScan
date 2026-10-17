"""配准与三角测量的参数声明。

这两个域的参数此前**全部硬编码在 pipeline 里**（``voxel_size``、
``icp_refine``、``edge_span`` …），只有写进 yaml 才生效，但没有任何
范围检查——``edge_span=0`` 会让位姿图断成孤立节点，``voxel_size=0``
会让 Open3D 除零。
"""
from __future__ import annotations

from triniscan.contracts import AlgorithmSpec, ParamSpec, REGISTRARS

from ..features.triangulate import triangulate
from .register import register_views

__all__ = ["SEQUENTIAL_SPEC", "TRIANGULATE_SPEC", "specs"]


SEQUENTIAL_SPEC = AlgorithmSpec(
    name="sequential",
    label="顺序/全局配准",
    func=register_views,
    help="把各视角按角度转到世界系后融合；可选全局对齐或位姿图",
    tags=("配准", "默认"),
    params=(
        ParamSpec("reconstruction.voxel_size", float, 0.0005,
                  low=1e-5, high=0.05, unit="m", short="voxel_size",
                  label="体素尺寸",
                  help="下采样与 ICP 的体素尺度；过大会丢细节，过小会变慢"),
        ParamSpec("reconstruction.icp_refine", bool, False, short="icp_refine",
                  label="ICP 精配准",
                  help="开启后用 ICP 修正角度误差，但慢且可能漂移"),
        ParamSpec("reconstruction.global_align", bool, False,
                  short="global_align",
                  label="全局对齐",
                  help="用全局优化替代顺序 ICP，消除累积误差"),
        ParamSpec("reconstruction.pose_graph", bool, False,
                  short="pose_graph",
                  label="位姿图优化",
                  help="全局对齐时优先用位姿图（失败自动回退迭代配准）"),
        ParamSpec("reconstruction.edge_span", int, 3,
                  low=1, high=36, short="edge_span",
                  label="连接邻居数",
                  help="每个视角向后连接几个邻居；1 会退化成链式"),
        ParamSpec("reconstruction.iterative_iters", int, 3,
                  low=1, high=20, short="iterative_iters",
                  label="迭代轮数",
                  help="兜底迭代配准的轮数"),
    ),
)

TRIANGULATE_SPEC = AlgorithmSpec(
    name="pixel_plane",
    label="像素-激光平面",
    func=triangulate,
    help="像素射线与激光平面求交，还原相机系三维点",
    tags=("几何", "默认"),
    params=(),
)

specs = (SEQUENTIAL_SPEC,)
REGISTRARS.add_spec(SEQUENTIAL_SPEC)
