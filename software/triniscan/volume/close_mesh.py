"""网格封闭化：补洞 / 去噪 / 水密化。

赛题口径
--------
体积 = **实体部分**实际体积，通孔、凹槽、缺口等**空缺部分不计入**。
因此必须先把重建网格封闭为**水密（watertight）**体，才能用散度定理
正确积分。

流程
----
1. 清理退化/重复面片与非流形边；
2. 补洞（fill holes）；
3. 检查水密性；
4. 若仍不水密，按策略兜底：凸包（凸体）或体素封闭（凹体）。

注意
----
凸包兜底会把凹处填平，导致体积**高估**，仅适用于凸体或作为参考。
对含凹槽/通孔的复杂目标，应优先提高重建质量 + 精细补洞。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:  # pragma: no cover
    trimesh = None
    _HAS_TRIMESH = False

__all__ = ["close_mesh", "CloseMeshResult", "is_watertight",
           "mesh_from_open3d", "mesh_to_open3d"]


@dataclass
class CloseMeshResult:
    """封闭化结果。"""

    mesh: "trimesh.Trimesh"
    watertight: bool
    filled_holes: bool
    fallback_used: str        # "", "convex_hull", "voxel"
    num_components: int


def _require_trimesh() -> None:
    if not _HAS_TRIMESH:
        raise ImportError("需要 trimesh: pip install trimesh")


def mesh_from_open3d(o3d_mesh) -> "trimesh.Trimesh":
    """Open3D 三角网格 → trimesh。"""
    _require_trimesh()
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def mesh_to_open3d(tm: "trimesh.Trimesh"):
    """trimesh → Open3D 三角网格。"""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    mesh.compute_vertex_normals()
    return mesh


def is_watertight(mesh) -> bool:
    """判断网格是否水密（兼容 trimesh / Open3D）。"""
    if hasattr(mesh, "is_watertight"):
        return bool(mesh.is_watertight)
    if hasattr(mesh, "is_watertight"):  # pragma: no cover
        return bool(mesh.is_watertight())
    return False


def close_mesh(mesh,
               allow_convex_hull: bool = True,
               fallback: str = "convex_hull") -> CloseMeshResult:
    """把三角网格尽量封闭为水密体。

    Parameters
    ----------
    mesh : trimesh.Trimesh 或 Open3D TriangleMesh
    allow_convex_hull : 是否允许凸包兜底
    fallback : 兜底策略，``"convex_hull"`` 或 ``"voxel"``

    Returns
    -------
    CloseMeshResult
    """
    _require_trimesh()

    # 统一转为 trimesh（若输入是 Open3D）
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh_from_open3d(mesh)

    tm = mesh.copy()

    # 1. 清理（trimesh v4 中退化/重复面在 process 阶段处理；
    #    这里做安全的后处理）
    _safe_call(tm, "remove_unreferenced_vertices")
    _safe_call(tm, "merge_vertices")
    _safe_call(tm, "remove_infinite_values")
    if hasattr(tm, "_remove_degenerate_faces"):  # 老版本兼容
        _safe_call(tm, "_remove_degenerate_faces")
    _drop_degenerate_faces(tm)

    num_components = len(tm.split(only_watertight=False))

    # 2. 补洞
    filled = False
    try:
        before = len(tm.faces)
        tm.fill_holes()
        filled = len(tm.faces) > before
    except Exception as exc:  # noqa: BLE001
        log.warning("补洞失败: %s", exc)

    tm.fix_normals()

    # 3. 已水密则直接返回
    if tm.is_watertight:
        log.info("网格已封闭: %d 顶点, %d 面片",
                 len(tm.vertices), len(tm.faces))
        return CloseMeshResult(
            mesh=tm, watertight=True, filled_holes=filled,
            fallback_used="", num_components=num_components,
        )

    # 4. 兜底
    fallback_used = ""
    if fallback == "convex_hull" and allow_convex_hull:
        log.warning("补洞后仍非水密，使用凸包兜底（可能高估凹体体积）")
        tm = tm.convex_hull
        fallback_used = "convex_hull"
    elif fallback == "voxel":
        log.warning("补洞后仍非水密，使用体素封闭兜底")
        tm = _voxel_closed_mesh(tm)
        fallback_used = "voxel"
    else:
        log.warning("补洞后仍非水密，且未启用兜底策略")

    tm.fix_normals()
    return CloseMeshResult(
        mesh=tm,
        watertight=bool(tm.is_watertight),
        filled_holes=filled,
        fallback_used=fallback_used,
        num_components=num_components,
    )


def _safe_call(obj, method_name: str) -> None:
    """调用存在则调用，不存在则忽略（跨 trimesh 版本兼容）。"""
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            method()
        except Exception as exc:  # noqa: BLE001
            log.debug("%s 调用失败: %s", method_name, exc)


def _drop_degenerate_faces(tm: "trimesh.Trimesh") -> None:
    """删除面积为零（共线三点）的三角面片。"""
    if len(tm.faces) == 0:
        return
    areas = tm.area_faces
    keep = areas > 1e-15
    if not np.all(keep):
        removed = int((~keep).sum())
        tm.update_faces(keep)
        log.debug("删除 %d 个退化面片", removed)


def _voxel_closed_mesh(tm: "trimesh.Trimesh",
                       pitch: float | None = None) -> "trimesh.Trimesh":
    """体素占位法粗略封闭（对凹体比凸包更贴合）。"""
    if pitch is None:
        extents = tm.extents
        pitch = float(np.max(extents)) / 100.0
    vox = tm.voxelized(pitch=pitch).fill()
    try:
        return vox.marching_cubes
    except Exception:  # noqa: BLE001
        return vox.as_boxes()
