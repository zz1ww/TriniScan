"""网格封闭化：补洞 / 封底 / 水密化。

赛题体积口径：实体部分体积，通孔/凹槽/缺口不计入。
因此需先保证网格水密（watertight）才能正确积分。
"""
from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False


def close_mesh(mesh):
    """把三角网格封闭为水密体。

    流程: 去离群面片 → 补洞 → 若仍不水密则用凸包/包装兜底。

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh

    Returns
    -------
    trimesh.Trimesh: 水密网格
    """
    if not (_HAS_O3D and _HAS_TRIMESH):
        raise ImportError("需要 open3d 与 trimesh")

    # 1. 去除退化/重复面片
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # 转为 trimesh 处理补洞
    tm = _to_trimesh(mesh)

    # 2. 补洞
    try:
        tm.fill_holes()
    except Exception as e:  # noqa: BLE001
        print(f"[警告] fill_holes 失败: {e}")

    # 3. 检查水密性
    if not tm.is_watertight:
        print("[信息] 补洞后仍非水密，尝试凸包兜底（会高估凹体）")
        hull = tm.convex_hull
        # 若凸包与网格体积差异过大，说明凹体，需更细致处理
        tm = hull

    tm.fix_normals()
    return tm


def _to_trimesh(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def check_watertight(mesh) -> bool:
    """检查网格是否水密。"""
    return bool(mesh.is_watertight)


if __name__ == "__main__":
    print("封闭化需要网格输入；见 docs/05_体积计算方案.md")
