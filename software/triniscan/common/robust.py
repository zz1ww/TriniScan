"""稳健估计原语：鲁棒尺度、加权平面拟合、RANSAC。

为什么需要
----------
标定（尤其激光平面、转台圆）的最小二乘拟合对**离群点极敏感**：
一个反光坏点或一次误检就能把平面/圆拉偏，而传统 SVD 拟合没有任何
抵御能力。本模块提供可复用、无状态、可单测的稳健估计原语：

- :func:`mad_sigma`            基于 MAD 的稳健标准差（抗离群）；
- :func:`huber_weights`        Huber 权重（IRLS 用）；
- :func:`tukey_weights`        Tukey 双权（更激进，适合离群多）；
- :func:`circlish`             便捷：点到直线距离；
- :func:`irls_plane_fit`       加权迭代重加权最小二乘拟合平面；
- :func:`ransac_plane_fit`     RANSAC 拟合平面（对抗大比例离群）；
- :func:`robust_plane_fit`     统一入口：``"irls"`` / ``"ransac"`` / ``"svd"``。

约定
----
- 平面：``[a, b, c, d]``，满足 ``a*x + b*y + c*z + d = 0``，法向单位化。
- 点集：``(N, 3)`` ``float64``。
- 残差：点到平面的**带符号**距离（沿单位法向）。

所有函数均为纯函数，不修改输入，便于测试与线程安全复用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "mad_sigma",
    "huber_weights",
    "tukey_weights",
    "irls_plane_fit",
    "ransac_plane_fit",
    "robust_plane_fit",
    "robust_circle_3d_fit",
    "RobustPlaneResult",
    "RobustCircleResult",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 稳健尺度
# ---------------------------------------------------------------------------
def mad_sigma(values: np.ndarray, scale: float = 1.4826) -> float:
    """基于 MAD 的稳健标准差估计。

    ``sigma ≈ 1.4826 · median(|x - median(x)|)``，对离群点不敏感。

    Parameters
    ----------
    values : (N,) 残差/样本
    scale : MAD → σ 的一致性系数（正态分布下为 1.4826）

    Returns
    -------
    float
        稳健标准差；样本退化（全等/空）时返回 0.0。
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return float(scale * mad)


# ---------------------------------------------------------------------------
# 权重函数
# ---------------------------------------------------------------------------
def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Huber 权重：``|r|<=δ`` 取 1，否则 ``δ/|r|``。

    Parameters
    ----------
    residuals : (N,) 残差
    delta : 阈值（通常取 ``k · σ``，k≈1.345）

    Returns
    -------
    (N,) 权重，范围 (0, 1]。
    """
    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    if delta <= _EPS:
        return np.ones_like(residuals)
    abs_r = np.abs(residuals)
    return np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, _EPS))


def tukey_weights(residuals: np.ndarray, c: float) -> np.ndarray:
    """Tukey 双权：``|r|>c`` 直接置零（硬拒绝）。

    Parameters
    ----------
    residuals : (N,) 残差
    c : 阈值（通常取 ``4.685 · σ``）

    Returns
    -------
    (N,) 权重，范围 [0, 1]。
    """
    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    if c <= _EPS:
        return np.ones_like(residuals)
    u = residuals / c
    w = (1.0 - u ** 2) ** 2
    return np.where(np.abs(u) < 1.0, w, 0.0)


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
@dataclass
class RobustPlaneResult:
    """稳健平面拟合结果。"""

    plane: np.ndarray                 # (4,) 单位法向 [a,b,c,d]
    residual_rms: float               # 内点残差 RMS（无符号）
    inlier_mask: np.ndarray           # (N,) bool
    weights: np.ndarray               # (N,) 最终权重
    num_inliers: int
    num_points: int
    iterations: int
    method: str

    @property
    def outlier_ratio(self) -> float:
        if self.num_points <= 0:
            return 0.0
        return 1.0 - self.num_inliers / float(self.num_points)


# ---------------------------------------------------------------------------
# 内部：加权 SVD 平面拟合
# ---------------------------------------------------------------------------
def _weighted_plane_svd(points: np.ndarray,
                        weights: Optional[np.ndarray] = None) -> np.ndarray:
    """加权最小二乘平面拟合（加权质心 + 加权协方差 SVD）。"""
    if weights is None:
        w = np.ones(len(points), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
    total = float(w.sum())
    if total <= _EPS:
        w = np.ones(len(points), dtype=np.float64)
        total = float(len(points))

    centroid = (w[:, None] * points).sum(axis=0) / total
    centered = points - centroid
    # 加权协方差
    cov = (w[:, None] * centered).T @ centered / total
    _, _, vh = np.linalg.svd(cov, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm < _EPS:
        raise ValueError("平面法向退化（点共线或过少）")
    normal = normal / norm
    d = -float(normal @ centroid)
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


def _signed_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return points @ plane[:3] + plane[3]


# ---------------------------------------------------------------------------
# IRLS
# ---------------------------------------------------------------------------
def irls_plane_fit(
    points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    delta_k: float = 1.345,
    max_iter: int = 25,
    tol: float = 1e-10,
    weighting: str = "huber",
    min_sigma: float = 1e-9,
) -> RobustPlaneResult:
    """迭代重加权最小二乘（IRLS）拟合平面。

    每轮：用当前权重拟合平面 → 计算残差 → 由稳健尺度 σ 生成新权重，
    直至权重/平面收敛。

    Parameters
    ----------
    points : (N, 3)
    weights : 初始权重（可选，默认全 1）
    delta_k : Huber 的 k（阈值 = k·σ）
    max_iter : 最大迭代数
    tol : 平面参数变化收敛阈值
    weighting : ``"huber"`` 或 ``"tukey"``
    min_sigma : σ 下限（避免残差全零时除零）

    Returns
    -------
    RobustPlaneResult
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError("IRLS 拟合平面至少需要 3 个点")

    w = (np.ones(len(points), dtype=np.float64) if weights is None
         else np.asarray(weights, dtype=np.float64).ravel().copy())
    if len(w) != len(points):
        raise ValueError("初始权重长度与点数不一致")

    plane = _weighted_plane_svd(points, w)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        r = _signed_distance(points, plane)
        sigma = max(mad_sigma(r), min_sigma)
        if weighting == "tukey":
            w_new = tukey_weights(r, 4.685 * sigma)
        else:
            w_new = huber_weights(r, delta_k * sigma)
        # 防止全部权重归零导致拟合失败
        if w_new.sum() < 3:
            w_new = w
            break
        plane_new = _weighted_plane_svd(points, w_new)
        delta = float(np.max(np.abs(plane_new - plane)))
        plane = plane_new
        w = w_new
        if delta < tol:
            break

    # 符号统一：使相机原点(0,0,0)代入为负（法向背离相机、指向场景）
    if plane[3] > 0:
        plane = -plane

    r = np.abs(_signed_distance(points, plane))
    sigma = max(mad_sigma(_signed_distance(points, plane)), min_sigma)
    inlier = r <= max(3.0 * sigma, 1e-12)
    if inlier.sum() < 3:
        inlier = np.ones(len(points), dtype=bool)

    residual_rms = float(np.sqrt(np.mean(r[inlier] ** 2)))
    return RobustPlaneResult(
        plane=plane, residual_rms=residual_rms, inlier_mask=inlier,
        weights=w, num_inliers=int(inlier.sum()),
        num_points=len(points), iterations=iterations, method="irls",
    )


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------
def _plane_from_three_points(p0, p1, p2) -> Optional[np.ndarray]:
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return None  # 三点近共线
    normal = normal / norm
    d = -float(normal @ p0)
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


def ransac_plane_fit(
    points: np.ndarray,
    threshold: float,
    max_iter: int = 1000,
    confidence: float = 0.999,
    min_inliers: int = 3,
    seed: Optional[int] = 0,
) -> RobustPlaneResult:
    """RANSAC 拟合平面（可容忍大比例离群）。

    自适应停止：由当前内点率估计达到 ``confidence`` 所需迭代数。

    Parameters
    ----------
    points : (N, 3)
    threshold : 内点距离阈值（m）
    max_iter : 最大迭代数
    confidence : 目标置信度
    min_inliers : 最少内点数
    seed : 随机种子（保证可复现）

    Returns
    -------
    RobustPlaneResult
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n < 3:
        raise ValueError("RANSAC 拟合平面至少需要 3 个点")
    if threshold <= 0:
        raise ValueError("threshold 必须 > 0")

    rng = np.random.default_rng(seed)
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    iterations = 0

    for iterations in range(1, max_iter + 1):
        idx = rng.choice(n, size=3, replace=False)
        plane = _plane_from_three_points(
            points[idx[0]], points[idx[1]], points[idx[2]])
        if plane is None:
            continue
        dist = np.abs(_signed_distance(points, plane))
        inliers = dist < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            # 自适应更新迭代上限
            ratio = count / n
            if 0.0 < ratio < 1.0:
                denom = np.log(max(1.0 - ratio ** 3, _EPS))
                need = np.log(max(1.0 - confidence, _EPS)) / denom
                max_iter = min(max_iter, max(iterations + 1, int(np.ceil(need))))

    if best_inliers is None or best_count < min_inliers:
        # 退化：退回 IRLS（若连 3 内点都没有，IRLS 会抛错）
        log.warning("RANSAC 未找到满足条件的内点集，退回 IRLS")
        return irls_plane_fit(points)

    # 用全部内点做一次精修（最小二乘）
    in_pts = points[best_inliers]
    plane = _weighted_plane_svd(in_pts)
    if plane[3] > 0:
        plane = -plane
    r = np.abs(_signed_distance(points, plane))
    inlier = r < threshold
    residual_rms = float(np.sqrt(np.mean(r[inlier] ** 2)))

    w = inlier.astype(np.float64)
    return RobustPlaneResult(
        plane=plane, residual_rms=residual_rms, inlier_mask=inlier,
        weights=w, num_inliers=int(inlier.sum()),
        num_points=n, iterations=iterations, method="ransac",
    )


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def robust_plane_fit(
    points: np.ndarray,
    method: str = "irls",
    outlier_threshold: Optional[float] = None,
    **kwargs,
) -> RobustPlaneResult:
    """稳健平面拟合统一入口。

    Parameters
    ----------
    points : (N, 3)
    method : ``"irls"`` / ``"ransac"`` / ``"svd"``（普通最小二乘）
    outlier_threshold : RANSAC 的距离阈值；None 时取 ``0.5·\" 尺度``
    **kwargs : 透传给具体算法

    Returns
    -------
    RobustPlaneResult
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    key = method.lower()

    if key == "svd":
        plane = _weighted_plane_svd(points)
        if plane[3] > 0:
            plane = -plane
        r = np.abs(_signed_distance(points, plane))
        residual_rms = float(np.sqrt(np.mean(r ** 2)))
        return RobustPlaneResult(
            plane=plane, residual_rms=residual_rms,
            inlier_mask=np.ones(len(points), dtype=bool),
            weights=np.ones(len(points), dtype=np.float64),
            num_inliers=len(points), num_points=len(points),
            iterations=1, method="svd",
        )

    if key == "ransac":
        if outlier_threshold is None:
            # 用 IRLS 估计尺度后取 3σ，给一个自适应默认阈值
            base = irls_plane_fit(points)
            sigma = max(mad_sigma(_signed_distance(points, base.plane)), 1e-9)
            outlier_threshold = 3.0 * sigma
        return ransac_plane_fit(points, threshold=outlier_threshold,
                                **kwargs)

    if key == "irls":
        return irls_plane_fit(points, **kwargs)

    raise ValueError(
        f"未知的稳健拟合方法 '{method}'，可选: svd / irls / ransac")


# ---------------------------------------------------------------------------
# 三维圆拟合（转轴标定用）
# ---------------------------------------------------------------------------
@dataclass
class RobustCircleResult:
    """稳健圆拟合结果。"""

    center: np.ndarray                # (3,) 圆心
    normal: np.ndarray                # (3,) 圆面单位法向（即旋转轴方向）
    radius: float
    residual_rms: float               # 内点径向残差 RMS
    inlier_mask: np.ndarray           # (N,) bool
    num_inliers: int
    num_points: int
    method: str

    @property
    def outlier_ratio(self) -> float:
        if self.num_points <= 0:
            return 0.0
        return 1.0 - self.num_inliers / float(self.num_points)


def _fit_circle_lsq(points: np.ndarray):
    """最小二乘拟合空间圆，返回 ``(center, normal, radius)``。

    先由 SVD 确定圆面（最小奇异向量为法向），再在平面内做代数圆拟合。
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    nn = np.linalg.norm(normal)
    if nn < _EPS:
        raise ValueError("圆面法向退化")
    normal = normal / nn
    u, v = vh[0], vh[1]

    uv = centered @ np.column_stack([u, v])
    A = np.column_stack([2.0 * uv[:, 0], 2.0 * uv[:, 1],
                         np.ones(len(uv))])
    b = (uv ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    radius = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    center = centroid + cx * u + cy * v
    return center, normal, radius


def _circle_residuals(points: np.ndarray, center: np.ndarray,
                      normal: np.ndarray, radius: float) -> np.ndarray:
    """点到拟合圆周的径向偏差（去掉沿轴分量后的半径差）。"""
    rel = points - center
    axial = rel @ normal
    radial_vec = rel - np.outer(axial, normal)
    dist = np.linalg.norm(radial_vec, axis=1)
    return dist - radius


def robust_circle_3d_fit(
    points: np.ndarray,
    method: str = "ransac",
    threshold: Optional[float] = None,
    min_inliers: int = 3,
    max_iter: int = 1000,
    confidence: float = 0.999,
    seed: Optional[int] = 0,
) -> RobustCircleResult:
    """稳健拟合三维空间圆（用于转轴）。

    RANSAC 通过三点确定一个圆来采样；``threshold=None`` 时先用全量
    最小二乘估计尺度，取 3σ 作为内点阈值。

    Parameters
    ----------
    points : (N, 3), N >= 3
    method : ``"ransac"`` 或 ``"lsq"``
    threshold : 内点径向残差阈值（m）
    min_inliers : 最少内点数
    max_iter, confidence, seed : RANSAC 参数

    Returns
    -------
    RobustCircleResult
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n < 3:
        raise ValueError("拟合圆至少需要 3 个点")

    key = method.lower()
    if key == "lsq":
        center, normal, radius = _fit_circle_lsq(points)
        r = _circle_residuals(points, center, normal, radius)
        return RobustCircleResult(
            center=center, normal=normal, radius=radius,
            residual_rms=float(np.sqrt(np.mean(r ** 2))),
            inlier_mask=np.ones(n, dtype=bool),
            num_inliers=n, num_points=n, method="lsq",
        )

    if key != "ransac":
        raise ValueError(
            f"未知的圆拟合方法 '{method}'，可选: lsq / ransac")

    # 自适应默认阈值
    if threshold is None:
        try:
            c0, n0, r0 = _fit_circle_lsq(points)
            sigma = max(mad_sigma(_circle_residuals(points, c0, n0, r0)),
                        1e-9)
            threshold = 3.0 * sigma
        except Exception:  # noqa: BLE001
            threshold = 1e-3

    rng = np.random.default_rng(seed)
    best_mask: Optional[np.ndarray] = None
    best_count = -1
    iterations = 0
    budget = max_iter

    for iterations in range(1, budget + 1):
        idx = rng.choice(n, size=3, replace=False)
        try:
            center, normal, radius = _fit_circle_lsq(points[idx])
        except Exception:  # noqa: BLE001 - 三点近共线
            continue
        r = np.abs(_circle_residuals(points, center, normal, radius))
        mask = r < threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
            ratio = count / n
            if 0.0 < ratio < 1.0:
                denom = np.log(max(1.0 - ratio ** 3, _EPS))
                need = np.log(max(1.0 - confidence, _EPS)) / denom
                budget = min(max_iter, max(iterations + 1,
                                           int(np.ceil(need))))

    if best_mask is None or best_count < min_inliers:
        log.warning("圆 RANSAC 未找到足够内点，退回最小二乘")
        center, normal, radius = _fit_circle_lsq(points)
        r = _circle_residuals(points, center, normal, radius)
        return RobustCircleResult(
            center=center, normal=normal, radius=radius,
            residual_rms=float(np.sqrt(np.mean(r ** 2))),
            inlier_mask=np.ones(n, dtype=bool),
            num_inliers=n, num_points=n, method="ransac-lsq",
        )

    # 用内点精修
    inliers = points[best_mask]
    center, normal, radius = _fit_circle_lsq(inliers)
    r = np.abs(_circle_residuals(points, center, normal, radius))
    mask = r < threshold
    if mask.sum() < min_inliers:
        mask = best_mask
    residual_rms = float(np.sqrt(np.mean(
        _circle_residuals(points[mask], center, normal, radius) ** 2)))
    return RobustCircleResult(
        center=center, normal=normal, radius=radius,
        residual_rms=residual_rms, inlier_mask=mask,
        num_inliers=int(mask.sum()), num_points=n, method="ransac",
    )
