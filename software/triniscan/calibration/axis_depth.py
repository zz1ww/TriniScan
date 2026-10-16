"""方案 A：球心高度约束反解深度 → 转轴标定。

原理
----
相机看到球 -> 轮廓圆心像素 (u,v) -> 球心必在过该像素的视线上：
    P_i = t_i * d_i        (d_i 为单位方向, t_i>0 为深度参数)

球心随转台绕**固定空间圆**运动。设圆：
    圆心 a (3),  单位法向 n (3, |n|=1),  半径 R
对每帧球心 P_i 必须满足：
    (P_i - a) · n = 0            (共面)
    |P_i - a|^2 = R^2            (在圆上)

未知数: n(2 自由度, 用参数化) + a(3) + R(1) + t_i(N)
方程:   2N 个 (共面 + 等半径)
N>=3 可解, N>=6 稳健。

关键点：**不使用球的像素半径**，因此彻底避开
"半径抖动 -> 深度 Z 抖动" 的老问题。
"""
from __future__ import annotations

import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["assemble_axis_points_from_pixels",
           "calibrate_axis_from_pixels",
           "write_axis_npz"]


def write_axis_npz(path: str, center, direction, radius,
                   residual_rms_m: float, num_points: int,
                   num_inliers: int | None = None) -> None:
    """把转轴标定写成系统统一格式的 .npz（字段与 axis_calib 一致）。"""
    import datetime
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                exist_ok=True)
    meta = {
        "format_version": 1,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "type": "turntable_axis",
        "method": "height_constraint",
    }
    np.savez(
        path,
        point=np.asarray(center, dtype=np.float64),
        direction=np.asarray(direction, dtype=np.float64),
        radius=float(radius),
        residual_rms=float(residual_rms_m),
        num_points=int(num_points),
        num_inliers=int(num_inliers if num_inliers is not None
                        else num_points),
        outlier_ratio=0.0,
        _meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    log.info("已保存转轴标定: %s (残差 %.3f mm)", path,
             residual_rms_m * 1e3)


def _unit_rays(pixels: np.ndarray, K: np.ndarray,
               dist=None) -> np.ndarray:
    """像素 -> 相机系单位方向向量。"""
    from .laser_calib import pixel_rays
    rays = np.asarray(pixel_rays(pixels, K, dist), dtype=np.float64)
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return rays / norms


def assemble_axis_points_from_pixels(
        pixels: np.ndarray, K: np.ndarray, dist=None,
        n_init=None, a_init=None, R_init=None,
        vertical_hint=None, height_known=None,
        radius_known=None, radius_px=None) -> dict:
    """由球心像素 + 圆约束反解每个球心的 3D 坐标。

    单目尺度歧义必须由**外部尺度锚**打破，二选一：
    - ``radius_known`` + ``radius_px`` : 已知球半径(m) + 各帧像素半径，
      用"平均像素半径"锚定深度尺度（**对逐帧半径抖动不敏感**）；
    - ``height_known`` : 已知球心在相机系的平均 Z（高度）。

    Returns
    -------
    dict: points(N,3), t(N,), direction(3), center(3), radius, rms_mm
    """
    from scipy.optimize import least_squares

    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    N = len(pixels)
    if N < 3:
        raise ValueError(f"至少需要 3 个视角，当前 {N}")
    d = _unit_rays(pixels, K, dist)          # (N,3) 单位方向
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    # 初值：用固定"名义深度"先给一组球心，再拟合圆拿初值
    z_nom = 0.5
    P0 = np.array([z_nom * d[i] / d[i, 2] for i in range(N)])

    if vertical_hint is None:
        vertical_hint = np.array([0.0, 0.0, 1.0])  # 相机系近似竖直
    vertical_hint = np.asarray(vertical_hint, dtype=np.float64)
    vertical_hint = vertical_hint / np.linalg.norm(vertical_hint)

    if n_init is None:
        # 用 SVD 拟合 P0 所在平面的法向（粗略），若不稳则回退到竖直提示
        c0 = P0.mean(axis=0)
        U, S, Vt = np.linalg.svd(P0 - c0)
        n_init = Vt[2] if S[2] > 1e-6 * max(S[0], 1e-12) else vertical_hint
        n_init = np.asarray(n_init, dtype=np.float64)
        if np.dot(n_init, vertical_hint) < 0:
            n_init = -n_init
    if a_init is None:
        a_init = P0.mean(axis=0)
    if R_init is None:
        R_init = float(np.linalg.norm(P0 - a_init, axis=1).mean())

    # 参数化：n = 用两个角度；但更稳的是直接优化 n(3) + 归一化约束
    # 这里用 n(3) 直接参数 + 惩罚项保持单位长度，简单可靠。
    a0 = np.asarray(a_init, dtype=np.float64)
    n0 = np.asarray(n_init, dtype=np.float64)
    n0 = n0 / np.linalg.norm(n0)
    R0 = float(R_init)
    # t 初值：由 P0 深度沿 d 反推
    t0 = np.array([
        float(np.dot(P0[i], d[i]) / max(np.dot(d[i], d[i]), 1e-12))
        for i in range(N)
    ])
    t0 = np.clip(t0, 1e-3, None)

    def unpack(x):
        a = x[0:3]
        n = x[3:6]
        R = x[6]
        t = x[7:7 + N]
        return a, n, R, t

    # 绝对尺度约束：单目只有方向、没有尺度，若不固定尺度，t_i 会整体
    # 缩到 0（退化）。两种锚定方式（优先用半径，因它最直接）：
    #   radius_known + radius_px_per_view：Z_i ≈ f·R / r_px_i 给每帧尺度
    #   height_known：所有球心高度相同（相机系 Z 均值）
    h_known = None
    if height_known is not None:
        h_known = float(height_known)
    r_anchor = None
    if radius_known is not None and radius_px is not None:
        r_anchor = (float(radius_known),
                    np.asarray(radius_px, dtype=np.float64).reshape(-1))

    def residuals(x):
        a, n, R, t = unpack(x)
        nn = np.linalg.norm(n)
        P = t[:, None] * d                      # (N,3) 当前球心
        rel = P - a
        coplanar = rel @ n                      # (N,) 应=0
        radial = np.linalg.norm(rel, axis=1) - R  # (N,) 应=0
        pen = (nn - 1.0) * 10.0                 # 单位长度惩罚
        parts = [coplanar, radial, [pen]]
        if h_known is not None:
            parts.append([(P[:, 2].mean() - h_known) * 10.0])
        if r_anchor is not None:
            # 逐帧半径→逐帧 Z 的**软**约束（权重低）：只用它定尺度，
            # 不让单帧抖动主导。
            Rm, rpx = r_anchor
            f = 0.5 * (K[0, 0] + K[1, 1])
            z_i = f * Rm / np.maximum(rpx, 1e-6)
            parts.append((P[:, 2] - z_i) * 0.5)
        return np.concatenate(parts)

    x0 = np.concatenate([a0, n0, [R0], t0])
    res = least_squares(residuals, x0, method="lm", max_nfev=20000)
    a, n, R, t = unpack(res.x)
    n = n / np.linalg.norm(n)
    P = t[:, None] * d
    if np.all(np.abs(P) < 1e-9) or abs(R) < 1e-6:
        # 掉进"全部缩到原点"的平凡解：回退到尺度搜索法
        log.warning("联合优化退化，回退到尺度搜索法")
        return _scale_search(pixels, K, dist, d, vertical_hint,
                             radius_known, radius_px, height_known)

    if np.dot(n, vertical_hint) < 0:
        n = -n

    # 残差（点到拟合圆平面的 RMS, mm）
    rel = P - a
    cop = rel @ n
    rad = np.linalg.norm(rel, axis=1) - R
    rms = float(np.sqrt(np.mean(cop ** 2 + rad ** 2))) * 1e3

    return {
        "points": P,
        "t": t,
        "direction": n,
        "center": a,
        "radius": float(abs(R)),
        "rms_mm": rms,
    }


def _scale_search(pixels, K, dist, d, vertical_hint,
                  radius_known, radius_px, height_known):
    """尺度搜索法：避免高维联合优化掉进"全部缩到原点"的平凡解。

    做法：对候选深度尺度 s（即球心平均深度），构造球心
        P_i = s * d_i / (d_i · e_z)       (使各点深度 ~ s)
    然后**解析拟合圆**（:func:`fit_circle_3d`），取残差最小的 s。
    最后一维搜索，稳健且无退化。
    """
    from ..common.geometry import fit_circle_3d

    N = len(pixels)
    # 尺度锚定：球半径 + 平均像素半径给出名义深度
    f = 0.5 * (K[0, 0] + K[1, 1])
    if radius_known is not None and radius_px is not None:
        s0 = f * float(radius_known) / max(float(np.mean(radius_px)), 1e-6)
    elif height_known is not None:
        s0 = float(height_known)
    else:
        s0 = 0.4

    def circle_at(s):
        P = (s * d) / np.maximum(d[:, 2:3], 1e-6)
        center, normal, radius = fit_circle_3d(P)
        rel = P - np.asarray(center)
        rms = float(np.sqrt(np.mean(
            (rel @ np.asarray(normal)) ** 2 +
            (np.linalg.norm(rel, axis=1) - radius) ** 2))) * 1e3
        return P, np.asarray(center), np.asarray(normal), float(radius), rms

    # 在 s0 附近对数网格搜索 + 细化
    best = None
    for lo, hi, steps in [(s0 * 0.3, s0 * 3.0, 60),
                          (0, 0, 0)]:
        if steps == 0:
            break
        for s in np.geomspace(max(lo, 1e-3), hi, steps):
            try:
                P, c, n, R, rms = circle_at(s)
            except Exception:
                continue
            if best is None or rms < best[5]:
                best = (P, c, n, R, rms, s)

    P, c, n, R, rms, s = best
    if np.dot(n, vertical_hint) < 0:
        n = -n
    log.info("尺度搜索: s=%.3f m, 半径=%.1f mm, 残差=%.3f mm",
             s, R * 1e3, rms)
    # 对应深度参数 t
    t = (s * np.ones(N)) / np.maximum(d[:, 2], 1e-6)
    return {
        "points": P,
        "t": t,
        "direction": n,
        "center": c,
        "radius": float(abs(R)),
        "rms_mm": rms,
    }


def calibrate_axis_from_pixels(
        pixels: np.ndarray, K: np.ndarray, dist=None,
        out_path: str | None = None, angles_deg=None) -> dict:
    """端到端：像素 -> (高度约束反解 3D 点) -> 转轴参数。"""
    r = assemble_axis_points_from_pixels(pixels, K, dist=dist)
    if out_path:
        try:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                        exist_ok=True)
            np.savez(
                out_path,
                point=r["center"], direction=r["direction"],
                radius=r["radius"], residual_rms=r["rms_mm"] / 1e3,
                num_points=len(r["points"]),
                num_inliers=len(r["points"]), outlier_ratio=0.0,
            )
            log.info("已保存转轴标定: %s (残差 %.3f mm)", out_path,
                     r["rms_mm"])
        except Exception as exc:  # noqa: BLE001
            log.warning("保存转轴标定失败: %s", exc)
    return r
