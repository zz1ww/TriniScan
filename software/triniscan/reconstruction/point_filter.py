"""物理边界预过滤：剔除环境光/反光造成的"远点"与越界点。

为什么需要
----------
暗环境下扫描时，激光在物体上形成主条纹，但环境光（窗缝、指示灯、
反光贴纸、转台边缘镜面反射）也会在图像中留下**微弱亮斑**。这些亮斑
经三角测量后会得到**远离物体的虚假三维点**（典型表现：深度远大于
物体、或落在转台半径之外）。它们数量虽少，但旋转拼接后会形成
"悬空大环""张开的两张网"等悬浮结构，严重污染点云与后续网格。

成熟开源转台扫描器的通行做法**不是**统计离群（``statistical
outlier``），而是**物理约束硬门限**：

- **深度门限**：物体必然落在相机前方有限距离内，深度过大的点直接丢弃
  （本系统物体约 0.17~0.28 m，环境光远点可达 0.29~0.37 m）；
- **半径门限**：物体不可能超出转台半径（载物台边缘以外的点都是假的）；
- **轴向门限**：物体高度有限，沿转轴方向越界者丢弃。

物理门限的优点是**确定、可解释、不误伤**：阈值来自实物尺寸而非
点云统计，不会像统计离群那样在稀疏处误删真实表面。

连通域（聚类）去噪
------------------
物理门限管的是**"越界"**，但有一类噪声**既在界内、又互相紧邻**：
镜面反光簇、支架残片、激光打在背景上的孤立片。它们在空间上聚成
**小团**，团内点彼此邻近，因此 ``statistical outlier``（依赖"邻居数
少"）**抓不到**它们。

对这类噪声的正确工具是**欧氏聚类**：把点云按"距离 < eps"聚成若干
连通域，物体是**最大**的簇，其余小簇判为噪声。这正是 PCL
``EuclideanClusterExtraction`` / Open3D ``cluster_dbscan`` 的通行做法。
:func:`remove_small_clusters` 封装了这一逻辑（保留最大的 keep_top 个簇）。

坐标约定
--------
- ``filter_by_depth`` 作用于**相机坐标系**点（三角测量直接输出），
  取沿光轴（z）的深度；用在旋转拼接之前，可最早、最低成本地剔除远点。
- ``filter_by_bounds`` 作用于**世界坐标系**点（绕轴旋转拼接之后），
  以"距转轴的径向距离 + 沿转轴的轴向坐标"为约束。

本模块为纯函数库，不持有状态，便于单测与复用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["FilterResult", "filter_by_depth", "filter_by_bounds",
           "remove_small_clusters"]


@dataclass
class FilterResult:
    """一次过滤的结果（保留点 + 掩码 + 统计）。"""

    points: np.ndarray                  # (M, 3) 保留下来的点
    keep_mask: np.ndarray               # (N,) bool，True 表示保留
    num_input: int = 0
    num_removed: int = 0
    applied: bool = False
    reason: str = ""

    @property
    def removed_ratio(self) -> float:
        if self.num_input <= 0:
            return 0.0
        return self.num_removed / float(self.num_input)


def _empty(points: np.ndarray) -> FilterResult:
    n = len(points)
    return FilterResult(points, np.ones(n, dtype=bool), n, 0, False,
                        "无有效约束，未过滤")


def filter_by_depth(points: np.ndarray,
                    depth_min: Optional[float] = None,
                    depth_max: Optional[float] = None) -> FilterResult:
    """按**相机系深度**（沿光轴 z）剔除远/近点。

    Parameters
    ----------
    points : (N, 3) 相机坐标系点（三角测量输出）
    depth_min : 深度下限 (m)；None 或 <=0 表示不设下限
    depth_max : 深度上限 (m)；None 或 <=0 表示不设上限

    Returns
    -------
    FilterResult

    Notes
    -----
    只有上限/下限**至少一个为正**时才生效；否则原样返回
    （``applied=False``），避免误配成"全部丢弃"。
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n == 0:
        return _empty(points)

    lo = float(depth_min) if depth_min else 0.0
    hi = float(depth_max) if depth_max else 0.0
    if lo <= 0.0 and hi <= 0.0:
        return _empty(points)

    depth = points[:, 2]
    keep = np.ones(n, dtype=bool)
    if lo > 0.0:
        keep &= depth >= lo
    if hi > 0.0:
        keep &= depth <= hi

    removed = int((~keep).sum())
    result = FilterResult(
        points=points[keep], keep_mask=keep, num_input=n,
        num_removed=removed, applied=True,
        reason=(f"深度门限 [{lo if lo > 0 else '-∞'}, "
                f"{hi if hi > 0 else '+∞'}] m: 剔除 {removed} 点"
                f" ({removed / n * 100:.1f}%)"),
    )
    log.info("深度过滤: %s", result.reason)
    return result


def filter_by_bounds(points: np.ndarray,
                     axis_point: np.ndarray,
                     axis_dir: np.ndarray,
                     radius_max: Optional[float] = None,
                     z_min: Optional[float] = None,
                     z_max: Optional[float] = None) -> FilterResult:
    """按**世界系物理边界**（半径 + 轴向高度）剔除越界点。

    Parameters
    ----------
    points : (N, 3) 世界坐标系点（已绕轴拼接）
    axis_point : (3,) 轴上一点
    axis_dir : (3,) 单位转轴方向
    radius_max : 距转轴的最大径向距离 (m)；None 或 <=0 表示不限
    z_min, z_max : 沿转轴方向的坐标上下限 (m)；None 表示不限

    Returns
    -------
    FilterResult
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n == 0:
        return _empty(points)

    r_max = float(radius_max) if radius_max else 0.0
    lo = float(z_min) if z_min is not None else None
    hi = float(z_max) if z_max is not None else None
    if r_max <= 0.0 and lo is None and hi is None:
        return _empty(points)

    axis_point = np.asarray(axis_point, dtype=np.float64).reshape(3)
    axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    nrm = np.linalg.norm(axis)
    axis = axis / nrm if nrm > 1e-12 else axis

    rel = points - axis_point
    axial = rel @ axis                                   # 沿轴坐标
    radial = rel - np.outer(axial, axis)
    radial_dist = np.linalg.norm(radial, axis=1)

    keep = np.ones(n, dtype=bool)
    if r_max > 0.0:
        keep &= radial_dist <= r_max
    if lo is not None:
        keep &= axial >= lo
    if hi is not None:
        keep &= axial <= hi

    removed = int((~keep).sum())
    parts = []
    if r_max > 0.0:
        parts.append(f"半径<={r_max * 1e3:.0f}mm")
    if lo is not None or hi is not None:
        parts.append(f"轴向[{lo if lo is not None else '-∞'},"
                     f"{hi if hi is not None else '+∞'}]m")
    result = FilterResult(
        points=points[keep], keep_mask=keep, num_input=n,
        num_removed=removed, applied=True,
        reason=(f"边界门限({', '.join(parts)}): 剔除 {removed} 点"
                f" ({removed / n * 100:.1f}%)"),
    )
    log.info("边界过滤: %s", result.reason)
    return result


def remove_small_clusters(points: np.ndarray,
                          eps: float = 0.002,
                          min_points: int = 10,
                          keep_top: int = 1) -> FilterResult:
    """欧氏聚类去噪：剔除**小连通域**，只保留最大的若干簇。

    动机
    ----
    镜面反光簇、支架残片等噪声在空间上聚成**小团**，团内点互相邻近，
    因此"统计离群"（按邻居数判定）无效。改用欧氏聚类，把点云按
    "距离 < ``eps``" 聚成若干连通域，物体通常是**最大**的簇。

    实现使用 Open3D ``cluster_dbscan``（已在本项目依赖内，无需额外
    安装）。若 Open3D 不可用，则用基于 KD-Tree 的替代实现。

    Parameters
    ----------
    points : (N, 3) 点云（任意坐标系，通常为世界系）
    eps : DBSCAN 邻域半径 (m)，即两点视为"连通"的最大间距。
        应略大于点云的平均点间距；太小会把一个物体拆散，太大会把
        噪声并入物体。默认 2 mm。
    min_points : 成为核心点的最少邻居数（DBSCAN ``min_points``）。
        点云很密时应增大（如 10~30）。
    keep_top : 保留点数最多的前 k 个簇（默认 1，即只留物体）。

    Returns
    -------
    FilterResult : 保留点 + 掩码 + 统计

    Notes
    -----
    - DBSCAN 把**未归入任何簇**的点标为 ``-1``（噪声），本函数一律剔除；
    - 若聚类结果为空（参数过严），则原样返回（``applied=False``），
      避免把整片点云删光。
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n == 0:
        return _empty(points)
    k = max(1, int(keep_top))
    if n < max(int(min_points) * 2, 50):
        # 点太少，聚类无意义
        return _empty(points)

    labels = None
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        labels = np.asarray(pcd.cluster_dbscan(
            eps=float(eps), min_points=int(min_points),
            print_progress=False), dtype=np.int64)
    except Exception as exc:  # noqa: BLE001 - 退化为 KD-Tree 实现
        log.debug("Open3D 聚类不可用(%s)，改用 KD-Tree 近似", exc)
        labels = _dbscan_kdtree(points, eps=float(eps),
                                min_points=int(min_points))

    if labels is None or labels.size == 0:
        return _empty(points)

    uniq = [int(x) for x in np.unique(labels) if x >= 0]
    if not uniq:
        return _empty(points)

    # 按簇大小排序，保留前 keep_top 个
    sizes = np.array([int((labels == c).sum()) for c in uniq])
    order = np.argsort(sizes)[::-1]
    kept_labels = {uniq[i] for i in order[:k]}
    keep = np.isin(labels, list(kept_labels))

    removed = int((~keep).sum())
    if removed == 0:
        return _empty(points)

    small = len(uniq) - len(kept_labels)
    result = FilterResult(
        points=points[keep], keep_mask=keep, num_input=n,
        num_removed=removed, applied=True,
        reason=(f"连通域去噪(eps={eps*1e3:.1f}mm, min_pts={min_points}): "
                f"共 {len(uniq)} 簇，剔除 {small} 个小簇 + 噪声，"
                f"保留最大 {len(kept_labels)} 簇，共剔除 {removed} 点"
                f" ({removed / n * 100:.1f}%)"),
    )
    log.info("连通域过滤: %s", result.reason)
    return result


def _dbscan_kdtree(points: np.ndarray,
                   eps: float,
                   min_points: int) -> Optional[np.ndarray]:
    """无 Open3D 时的 DBSCAN 近似实现（基于 scipy cKDTree 的并查集）。

    用"半径邻域计数 + 并查集连通"近似 DBSCAN 的簇划分：
    对每个点取半径 ``eps`` 内的邻居数，邻居数 >= ``min_points`` 者
    作为核心点，核心点之间若互为邻居则合并。非核心点若邻接核心点
    则归入其簇，否则标为噪声（-1）。
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:  # noqa: BLE001
        return None

    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    tree = cKDTree(pts)
    neigh = tree.query_ball_point(pts, r=float(eps))
    core = np.array([len(nb) >= int(min_points) for nb in neigh], dtype=bool)

    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if not core[i]:
            continue
        for j in neigh[i]:
            if core[j]:
                union(i, j)

    labels = np.full(n, -1, dtype=np.int64)
    next_id = 0
    root_to_id: dict[int, int] = {}
    for i in range(n):
        if core[i]:
            r = find(i)
            if r not in root_to_id:
                root_to_id[r] = next_id
                next_id += 1
            labels[i] = root_to_id[r]
    # 边界点归入相邻核心点所属簇
    for i in range(n):
        if labels[i] != -1:
            continue
        for j in neigh[i]:
            if core[j]:
                labels[i] = labels[j]
                break
    return labels
