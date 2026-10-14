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
4. 若仍不水密，按策略兜底：

   - ``"bottom_plane"``（推荐）：把**边界环**用平面补片封闭。
     转台扫描时物体**底面天然缺失**，用平面封底比凸包更贴合真实形状，
     避免凸包把底面"撑圆"导致的体积高估；
   - ``"convex_hull"``：凸包兜底（仅适合凸体，会填平凹陷）；
   - ``"voxel"``：体素占位封闭（对凹体比凸包贴合，但较粗糙）。

若无兜底可用，则原样返回并标记非水密。
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    fallback_used: str        # "", "convex_hull", "voxel", "bottom_plane"
    num_components: int
    num_boundary_loops: int = 0
    capped_loops: int = 0
    notes: list[str] = field(default_factory=list)


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
    flag = getattr(mesh, "is_watertight", None)
    if flag is None:
        return False
    # trimesh 中是属性(bool)；Open3D 中是方法
    return bool(flag() if callable(flag) else flag)


def close_mesh(mesh,
               allow_convex_hull: bool = True,
               fallback: str = "bottom_plane",
               axis_dir=None,
               cap_all_planar_loops: bool = True,
               planar_tol: float = 0.002) -> CloseMeshResult:
    """把三角网格尽量封闭为水密体。

    Parameters
    ----------
    mesh : trimesh.Trimesh 或 Open3D TriangleMesh
    allow_convex_hull : 是否允许凸包兜底（仅当 fallback="convex_hull"）
    fallback : 兜底策略，``"bottom_plane"`` / ``"convex_hull"`` / ``"voxel"``
    axis_dir : 转轴方向（可选）。用于优先封最靠近底部的边界环。
    cap_all_planar_loops : 是否封闭所有近似平面的边界环（否则只封底部/顶部）
    planar_tol : 判定边界环"近似平面"的容差 (m)

    Returns
    -------
    CloseMeshResult
    """
    _require_trimesh()

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh_from_open3d(mesh)

    tm = mesh.copy()
    notes: list[str] = []

    # 1. 清理
    _safe_call(tm, "remove_unreferenced_vertices")
    _safe_call(tm, "merge_vertices")
    _safe_call(tm, "remove_infinite_values")
    if hasattr(tm, "_remove_degenerate_faces"):  # 老版本兼容
        _safe_call(tm, "_remove_degenerate_faces")
    _drop_degenerate_faces(tm)

    try:
        num_components = len(tm.split(only_watertight=False))
    except Exception:  # noqa: BLE001
        num_components = 1

    # 2. 补洞
    #    优先用**平面封底**（若请求 bottom_plane）：这样能把"底面天然缺失"
    #    用平面补片封闭，比 trimesh 的盲补洞更贴合真实形状；其余孔洞再交给
    #    trimesh.fill_holes。
    filled = False
    capped = 0
    use_bottom_plane = (fallback == "bottom_plane"
                        and not tm.is_watertight)

    if use_bottom_plane:
        tm, capped = _cap_boundary_loops(
            tm, axis_dir=axis_dir, planar_tol=planar_tol,
            all_planar=cap_all_planar_loops)
        filled = capped > 0

    # 剩余孔洞用 trimesh 常规补洞
    try:
        before = len(tm.faces)
        tm.fill_holes()
        filled = filled or (len(tm.faces) > before)
    except Exception as exc:  # noqa: BLE001
        log.warning("补洞失败: %s", exc)
        notes.append(f"fill_holes 失败: {exc}")

    _safe_call(tm, "fix_normals")

    num_loops = _count_boundary_loops(tm)

    # 3. 已水密则直接返回
    if tm.is_watertight:
        log.info("网格已封闭: %d 顶点, %d 面片%s",
                 len(tm.vertices), len(tm.faces),
                 "（平面封底）" if capped else "")
        return CloseMeshResult(
            mesh=tm, watertight=True, filled_holes=filled,
            fallback_used=("bottom_plane" if capped else ""),
            num_components=num_components,
            num_boundary_loops=0, capped_loops=capped, notes=notes,
        )

    # 4. 兜底
    fallback_used = ""
    if fallback == "bottom_plane":
        fallback_used = "bottom_plane"
        if capped:
            notes.append(f"平面封底：封闭 {capped} 个边界环")
        if not tm.is_watertight and allow_convex_hull:
            log.warning("平面封底后仍非水密，改用凸包兜底")
            tm = tm.convex_hull
            fallback_used = "bottom_plane+convex_hull"
    elif fallback == "convex_hull" and allow_convex_hull:
        log.warning("补洞后仍非水密，使用凸包兜底（可能高估凹体体积）")
        tm = tm.convex_hull
        fallback_used = "convex_hull"
    elif fallback == "voxel":
        log.warning("补洞后仍非水密，使用体素封闭兜底")
        tm = _voxel_closed_mesh(tm)
        fallback_used = "voxel"
    else:
        log.warning("补洞后仍非水密，且未启用兜底策略")

    _safe_call(tm, "fix_normals")
    return CloseMeshResult(
        mesh=tm,
        watertight=bool(tm.is_watertight),
        filled_holes=filled,
        fallback_used=fallback_used,
        num_components=num_components,
        num_boundary_loops=num_loops,
        capped_loops=capped,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 边界环处理
# ---------------------------------------------------------------------------
def _boundary_edges(tm: "trimesh.Trimesh") -> np.ndarray:
    """返回边界边 ``(M, 2)``（只被一个面引用的边）。

    实现说明：trimesh 的 ``edges_face`` 并不用于标记边界（其取值语义
    依赖版本），因此这里用**出现次数**统计：唯一化后的边若在全部三角
    面中出现恰好一次，即为边界边。稳健且跨版本一致。
    """
    try:
        faces = np.asarray(tm.faces, dtype=np.int64)
    except Exception:  # noqa: BLE001
        return np.empty((0, 2), dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    # 每个三角面三条边，排序后拼接
    e0 = np.sort(faces[:, [0, 1]], axis=1)
    e1 = np.sort(faces[:, [1, 2]], axis=1)
    e2 = np.sort(faces[:, [2, 0]], axis=1)
    edges = np.vstack([e0, e1, e2])

    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


def _boundary_loops(tm: "trimesh.Trimesh") -> list[list[int]]:
    """把边界边串成顶点环（每个环是顶点索引的有序列表）。"""
    edges = _boundary_edges(tm)
    if len(edges) == 0:
        return []

    # 邻接表
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []

    def _key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    for start in list(adj.keys()):
        for nxt in adj[start]:
            if _key(start, nxt) in visited_edges:
                continue
            loop = [start]
            prev, cur = start, nxt
            visited_edges.add(_key(prev, cur))
            ok = True
            while cur != start:
                loop.append(cur)
                candidates = [v for v in adj.get(cur, [])
                              if _key(cur, v) not in visited_edges]
                if not candidates:
                    ok = False
                    break
                prev, cur = cur, candidates[0]
                visited_edges.add(_key(prev, cur))
                if len(loop) > len(edges) + 2:   # 防御死循环
                    ok = False
                    break
            if ok and len(loop) >= 3:
                loops.append(loop)
    return loops


def _count_boundary_loops(tm: "trimesh.Trimesh") -> int:
    try:
        return len(_boundary_loops(tm))
    except Exception:  # noqa: BLE001
        return 0


def _cap_boundary_loops(tm: "trimesh.Trimesh",
                        axis_dir=None,
                        planar_tol: float = 0.002,
                        all_planar: bool = True):
    """用平面补片封闭边界环（优先底部）。

    对每个边界环：拟合平面，若残差在容差内（近似平面），
    则以环的质心为中心做扇形三角化补面。

    Returns
    -------
    (mesh, num_capped)
    """
    loops = _boundary_loops(tm)
    if not loops:
        return tm, 0

    verts = np.asarray(tm.vertices, dtype=np.float64)
    axis = None
    if axis_dir is not None:
        axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
        na = np.linalg.norm(axis)
        axis = axis / na if na > 1e-12 else None

    new_vertices = [verts]
    new_faces: list[tuple[int, int, int]] = []
    offset = len(verts)
    capped = 0

    # 若指定了轴，按"沿轴投影"排序，优先处理最底部的环
    def _loop_priority(loop):
        pts = verts[loop]
        if axis is not None:
            return float(pts.mean(axis=0) @ axis)
        return -float(pts[:, 2].mean())

    for loop in sorted(loops, key=_loop_priority):
        pts = verts[loop]
        if len(pts) < 3:
            continue
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        try:
            _, s, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        normal = vh[-1]
        planar_err = float(np.abs(centered @ normal).max())
        if all_planar is False and planar_err > planar_tol:
            continue
        if planar_err > max(planar_tol, 1e-6) * 10:
            # 明显非平面（可能是大开口），跳过，交给凸包/体素兜底
            continue

        # 以质心为枢纽做扇形三角化
        hub = offset
        new_vertices.append(centroid.reshape(1, 3))
        offset += 1

        # 判断环绕方向，保证补面法向与环一致（朝外）
        ring = np.asarray(loop, dtype=np.int64)
        # 用环的有向面积符号决定绕向
        order = _order_ring(pts, centroid, normal)
        m = len(order)
        for i in range(m):
            a = ring[order[i]]
            b = ring[order[(i + 1) % m]]
            new_faces.append((int(a), int(b), int(hub)))
        capped += 1

    if not new_faces:
        return tm, 0

    all_verts = np.vstack(new_vertices)
    all_faces = np.vstack(
        [np.asarray(tm.faces, dtype=np.int64),
         np.asarray(new_faces, dtype=np.int64)]
    )
    capped_mesh = trimesh.Trimesh(
        vertices=all_verts, faces=all_faces, process=False)
    _safe_call(capped_mesh, "remove_unreferenced_vertices")
    _safe_call(capped_mesh, "fix_normals")
    return capped_mesh, capped


def _order_ring(pts: np.ndarray, centroid: np.ndarray,
                normal: np.ndarray) -> list[int]:
    """把环上点按绕 ``normal`` 的角度排序，返回索引顺序。"""
    centered = pts - centroid
    u = centered[0]
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = u / nu
    v = np.cross(normal, u)
    ang = np.arctan2(centered @ v, centered @ u)
    return list(np.argsort(ang))


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
