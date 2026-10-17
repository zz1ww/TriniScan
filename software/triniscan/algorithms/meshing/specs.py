"""网格重建的参数声明。

默认值**只在这里写一次**——重构前 ``poisson_depth=9`` 同时存在于
``pipeline.py``、``mesh.py`` 和 ``default.yaml``，改一处忘一处就是
"配置不生效"。
"""
from __future__ import annotations

from triniscan.contracts import AlgorithmSpec, MESHERS, ParamSpec

from .mesh import reconstruct_mesh, reconstruct_mesh_ball_pivoting

__all__ = ["POISSON_SPEC", "BALL_PIVOTING_SPEC", "specs"]


POISSON_SPEC = AlgorithmSpec(
    name="poisson",
    label="泊松重建",
    func=reconstruct_mesh,
    help="基于有向点云的隐式曲面重建，抗噪好、能补小孔，是默认方案",
    tags=("隐式", "抗噪", "默认"),
    params=(
        ParamSpec("reconstruction.poisson_depth", int, 9,
                  low=4, high=14, short="poisson_depth",
                  label="泊松深度",
                  help="八叉树深度。每 +1 分辨率翻倍、内存约 ×4。"
                       "9~10 适合掌上物体；小件可到 11"),
        ParamSpec("reconstruction.density_quantile", float, 0.02,
                  low=0.0, high=0.5, short="density_quantile",
                  label="低密度裁剪分位",
                  help="裁掉密度最低的这比例顶点（Poisson 在远处会外扩）"),
        ParamSpec("reconstruction.adaptive_depth", bool, True,
                  short="adaptive_depth",
                  label="自适应深度",
                  help="按点数自动下调深度，避免小点云过拟合"),
        ParamSpec("reconstruction.orient_max_nn", int, 30,
                  low=5, high=200, short="orient_max_nn",
                  label="法向一致化近邻数",
                  help="越大越稳但越慢；点云很密时可减小"),
    ),
)

BALL_PIVOTING_SPEC = AlgorithmSpec(
    name="ball_pivoting",
    label="滚球法",
    func=reconstruct_mesh_ball_pivoting,
    help="沿点云滚球成面，保留原始细节、不外扩；但要求点云均匀、"
         "法向正确，难点云容易出洞",
    tags=("显式", "保细节"),
    params=(
        ParamSpec("reconstruction.ball_radii_mm", str, "1.5,3,6",
                  short="ball_radii_mm",
                  label="球半径序列(mm)",
                  help="逗号分隔的球半径，按顺序尝试"),
    ),
)

specs = (POISSON_SPEC, BALL_PIVOTING_SPEC)

for _spec in specs:
    MESHERS.add_spec(_spec)
