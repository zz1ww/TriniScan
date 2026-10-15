"""多视角点云配准与拼接。

流程
----
1. 按转台角度，把每个视角的点云绕**旋转轴**变换到统一坐标系；
2. **全局对齐**（默认）：用多视角位姿图（pose graph）联合优化，
   消除逐视角顺序配准的累积误差；
3. 兜底：若位姿图不可用，退化为"迭代式重配准"（每视角对齐到当前
   融合结果，多轮迭代收敛）——比单向顺序 ICP 稳健得多；
4. 合并、下采样、去离群，得到完整点云。

坐标约定
--------
所有视角点云先在**相机坐标系**下（由三角测量得到），
变换到以转轴为参考的**世界坐标系**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..common.geometry import apply_transform, transform_about_axis
from ..common.logging_utils import get_logger

log = get_logger(__name__)

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:  # pragma: no cover
    o3d = None
    _HAS_O3D = False

__all__ = ["ViewCloud", "register_views", "to_open3d", "RegisterResult"]


@dataclass
class ViewCloud:
    """单个视角的点云与角度。"""

    angle_deg: float
    points: np.ndarray                  # (N, 3) 相机系
    meta: dict = field(default_factory=dict)


@dataclass
class RegisterResult:
    """配准结果（点云 + 质量信息）。"""

    pointcloud: "o3d.geometry.PointCloud"
    num_views: int
    num_points: int
    method: str                         # "pose_graph" / "iterative" / "none"
    mean_fitness: float = 0.0
    mean_rmse: float = 0.0
    num_edges: int = 0


def _require_open3d() -> None:
    if not _HAS_O3D:
        raise ImportError(
            "需要 open3d 进行点云配准: pip install open3d"
        )


def to_open3d(points: np.ndarray) -> "o3d.geometry.PointCloud":
    """NumPy 点集 → Open3D 点云。"""
    _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )
    return pcd


# ---------------------------------------------------------------------------
# 内部：ICP
# ---------------------------------------------------------------------------
def _ensure_normals(pcd, voxel_size: float, max_nn: int = 30) -> None:
    """为点云估计法向（总是在下采样之后调用，保证法向与几何一致）。

    点到面 ICP 只依赖法向所在的**直线**，并不要求全局一致朝向，因此
    朝向统一只是锦上添花。这里做两层容错：

    1. 法向方向用 ``orient_normals_towards_camera_location`` 快速统一
       （永不失败，代价低）；
    2. 更严格的 ``orient_normals_consistent_tangent_plane`` 是**可选**的
       精修，但它内部用 Qhull 做 Delaunay 三角化，遇到共面/共球点云
       （单视角近平面点云、下采样后高度规则的点云非常常见）会抛
       ``QH6239 ... cocircular or cospherical`` 之类的精度错误。
       因此这里用 try/except 包裹，失败则跳过，**绝不让它中断 ICP**。
    """
    radius = max(voxel_size * 2.0, 1e-6)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn)
    )

    # (1) 快速、健壮的朝向统一：指向相机原点（永不会失败）。
    try:
        pcd.orient_normals_towards_camera_location(
            np.zeros(3, dtype=np.float64))
    except Exception:  # pragma: no cover - 极端退化点云
        pass

    # (2) 可选精修：一致切平面朝向。共面/共球会触发 Qhull 精度错误，
    #     直接跳过即可（对点到面 ICP 无实质影响）。
    n = len(pcd.points)
    if n < 4:
        return
    try:
        pcd.orient_normals_consistent_tangent_plane(
            k=int(min(max_nn, max(n - 1, 3)))
        )
    except Exception as exc:  # Qhull 精度错误等，退化为上一步的朝向
        log.debug("orient_normals_consistent_tangent_plane 跳过: %s", exc)


def _icp(source: "o3d.geometry.PointCloud",
         target: "o3d.geometry.PointCloud",
         voxel_size: float,
         max_dist_factor: float = 2.0,
         init: Optional[np.ndarray] = None):
    """点到面 ICP 精配准。

    下采样后**重新估计法向**（避免用旧几何的法向），返回 Open3D 结果。
    """
    src = source.voxel_down_sample(voxel_size)
    tgt = target.voxel_down_sample(voxel_size)
    # 退化保护：下采样后点数过少（<6）无法可靠估法向/配准，直接返回恒等。
    if len(src.points) < 6 or len(tgt.points) < 6:
        return _identity_icp_result()
    _ensure_normals(src, voxel_size)
    _ensure_normals(tgt, voxel_size)

    # 自适应对应距离：至少覆盖几个体素
    max_dist = max(voxel_size * max_dist_factor, voxel_size * 2.0)
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_dist,
        init=(np.eye(4) if init is None else init),
        estimation_method=o3d.pipelines.registration
        .TransformationEstimationPointToPlane(),
    )
    return result


def _identity_icp_result():
    """构造一个"恒等变换 + 零适应度"的 ICP 结果（退化兜底）。"""
    r = o3d.pipelines.registration.RegistrationResult()
    r.transformation = np.eye(4)
    r.fitness = 0.0
    r.inlier_rmse = 0.0
    return r


# ---------------------------------------------------------------------------
# 全局对齐：位姿图优化
# ---------------------------------------------------------------------------
def _pose_graph_align(pcds: list, voxel_size: float,
                      edge_span: int, max_dist_factor: float):
    """多视角位姿图全局优化。

    以每个视角为节点，视角 ``i`` 与其后 ``edge_span`` 个视角建立
    里程计边（初值由转台角度给出），ICP 求相对位姿；再用
    ``global_optimization`` 联合优化所有节点位姿。

    Returns
    -------
    (poses, mean_fitness, mean_rmse, num_edges) 或 None（失败时）
    """
    n = len(pcds)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for pcd in pcds:
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.eye(4))
        )

    odometry = np.eye(4)
    fitnesses, rmses, num_edges = [], [], 0
    for s in range(n):
        for t in range(s + 1, min(s + edge_span + 1, n)):
            try:
                result = _icp(pcds[s], pcds[t], voxel_size,
                              max_dist_factor)
            except Exception as exc:  # noqa: BLE001
                log.debug("视角 %d→%d ICP 失败: %s", s, t, exc)
                continue
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcds[s].voxel_down_sample(voxel_size),
                pcds[t].voxel_down_sample(voxel_size),
                max(voxel_size * max_dist_factor, voxel_size * 2.0),
                result.transformation,
            )
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    s, t, result.transformation, info, uncertain=False)
            )
            fitnesses.append(result.fitness)
            rmses.append(result.inlier_rmse)
            num_edges += 1

    if num_edges == 0:
        return None

    try:
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max(voxel_size * max_dist_factor,
                                            voxel_size * 2.0),
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("位姿图全局优化失败，回退迭代配准: %s", exc)
        return None

    poses = [np.asarray(node.pose) for node in pose_graph.nodes]
    return poses, float(np.mean(fitnesses)), float(np.mean(rmses)), num_edges


def _iterative_align(pcds: list, voxel_size: float, iters: int,
                     max_dist_factor: float):
    """迭代式重配准兜底：每轮把各视角对齐到当前融合结果。

    相比单向顺序 ICP，双向迭代可显著减小累积误差（虽然不如全局优化）。
    """
    n = len(pcds)
    poses = [np.eye(4) for _ in range(n)]

    merged = None
    for i in range(n):
        if merged is None:
            merged = pcds[i]
        else:
            merged += pcds[i]

    fitnesses, rmses = [], []
    for _ in range(max(iters, 1)):
        updated = []
        for i in range(n):
            result = _icp(pcds[i], merged, voxel_size, max_dist_factor,
                          init=poses[i])
            poses[i] = result.transformation
            updated.append(pcds[i].transform(poses[i]))
            fitnesses.append(result.fitness)
            rmses.append(result.inlier_rmse)
        merged = updated[0]
        for p in updated[1:]:
            merged += p
        merged = merged.voxel_down_sample(voxel_size)
    return (poses, float(np.mean(fitnesses)) if fitnesses else 0.0,
            float(np.mean(rmses)) if rmses else 0.0, n)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def register_views(
    views: Sequence[ViewCloud],
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    voxel_size: float = 0.0005,
    icp_refine: bool = True,
    icp_max_dist_factor: float = 2.0,
    statistical_outlier_k: int = 20,
    statistical_outlier_std: float = 2.0,
    global_align: bool = True,
    pose_graph: bool = True,
    edge_span: int = 3,
    iterative_iters: int = 3,
    return_result: bool = False,
):
    """把各视角点云变换到统一坐标系并融合。

    Parameters
    ----------
    views : ViewCloud 序列
    axis_point, axis_dir : 转轴标定结果
    voxel_size : 下采样与 ICP 的体素尺度 (m)
    icp_refine : 是否做 ICP 精配准
    icp_max_dist_factor : ICP 最大对应距离 = factor × voxel_size
    statistical_outlier_k, statistical_outlier_std : 去离群参数
    global_align : 是否用全局对齐（位姿图/迭代）替代顺序 ICP
    pose_graph : 全局对齐时优先用位姿图优化（失败回退迭代配准）
    edge_span : 位姿图中每个视角向后连接的邻居数
    iterative_iters : 迭代配准的轮数（兜底路径）
    return_result : 为 True 时返回 :class:`RegisterResult`，否则仅返回点云

    Returns
    -------
    o3d.geometry.PointCloud 或 RegisterResult
    """
    _require_open3d()
    if not views:
        raise ValueError("没有可配准的视角")

    # 1. 按转台角度变换到世界坐标系
    world_pcds: list = []
    for view in views:
        pts = np.asarray(view.points, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            continue
        transform = transform_about_axis(
            axis_point, axis_dir, np.deg2rad(view.angle_deg)
        )
        pcd = to_open3d(apply_transform(transform, pts))
        world_pcds.append(pcd)

    if not world_pcds:
        raise ValueError("所有视角均为空，无法配准")

    method = "none"
    mean_fitness = mean_rmse = 0.0
    num_edges = 0

    if icp_refine and len(world_pcds) > 1 and global_align:
        poses = None
        if pose_graph:
            outcome = _pose_graph_align(world_pcds, voxel_size,
                                        edge_span, icp_max_dist_factor)
            if outcome is not None:
                poses, mean_fitness, mean_rmse, num_edges = outcome
                method = "pose_graph"
        if poses is None:
            poses, mean_fitness, mean_rmse, num_edges = _iterative_align(
                world_pcds, voxel_size, iterative_iters,
                icp_max_dist_factor)
            method = "iterative"
        world_pcds = [p.transform(T) for p, T in zip(world_pcds, poses)]
    elif icp_refine and len(world_pcds) > 1:
        # 顺序 ICP（旧行为，保留作对照）
        merged = world_pcds[0]
        for pcd in world_pcds[1:]:
            result = _icp(pcd, merged, voxel_size, icp_max_dist_factor)
            pcd = pcd.transform(result.transformation)
            merged += pcd
            mean_fitness += result.fitness
            mean_rmse += result.inlier_rmse
            num_edges += 1
        if num_edges:
            mean_fitness /= num_edges
            mean_rmse /= num_edges
        world_pcds = [merged]
        method = "sequential"

    merged = world_pcds[0]
    for pcd in world_pcds[1:]:
        merged += pcd

    merged = merged.voxel_down_sample(voxel_size)
    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=statistical_outlier_k,
        std_ratio=statistical_outlier_std,
    )
    log.info("点云配准完成[%s]: %d 点, 体素 %.1f mm, "
             "ICP 边 %d, 平均 fitness %.2f, RMSE %.2f mm",
             method, len(merged.points), voxel_size * 1e3, num_edges,
             mean_fitness, mean_rmse * 1e3)

    if return_result:
        return RegisterResult(
            pointcloud=merged, num_views=len(views),
            num_points=len(merged.points), method=method,
            mean_fitness=mean_fitness, mean_rmse=mean_rmse,
            num_edges=num_edges,
        )
    return merged
