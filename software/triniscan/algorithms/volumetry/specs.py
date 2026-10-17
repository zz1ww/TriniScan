"""体积计算的参数声明。

**默认值必须单一来源**：重构前 ``voxel_pitch=0.001`` 同时出现在
``pipeline.py``、``volume.py`` 和 ``default.yaml`` 三处，改一处忘一处
就是"配置不生效"。

封闭化的参数声明见 :mod:`triniscan.algorithms.closing.specs`。
"""
from __future__ import annotations

from triniscan.contracts import AlgorithmSpec, ParamSpec, VOLUMETERS

from .volume import compute_volume_divergence, compute_volume_voxel

__all__ = ["DIVERGENCE_SPEC", "VOXEL_SPEC", "specs"]


DIVERGENCE_SPEC = AlgorithmSpec(
    name="divergence",
    label="散度定理",
    func=compute_volume_divergence,
    help="对水密网格用散度定理求体积，精确且快，是默认方案",
    tags=("解析", "快", "默认"),
    params=(
        ParamSpec("volume.require_watertight", bool, True,
                  short="require_watertight",
                  label="要求水密",
                  help="强烈建议开启：非水密网格的散度体积没有物理意义"),
    ),
)


VOXEL_SPEC = AlgorithmSpec(
    name="voxel",
    label="体素法",
    func=compute_volume_voxel,
    help="体素化后数体素，与散度法交叉校核；结果略偏大",
    tags=("数值", "校核"),
    params=(
        ParamSpec("volume.voxel_pitch", float, 0.001,
                  low=1e-5, high=0.02, unit="m", short="pitch",
                  label="体素尺寸",
                  help="越小越准越慢；通常取平均点间距的 1~2 倍"),
    ),
)


specs = (DIVERGENCE_SPEC, VOXEL_SPEC)

VOLUMETERS.add_spec(DIVERGENCE_SPEC)
VOLUMETERS.add_spec(VOXEL_SPEC)
