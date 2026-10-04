"""Steger 条纹中心提取的核心实现（Hessian + 高斯导数）。

方法摘要
--------
1. 用解析高斯导数核计算 ``r_x, r_y, r_xx, r_xy, r_yy``；
2. 对每个候选像素求 Hessian 特征向量得到条纹法向 ``(nx, ny)``；
3. 沿法向做二阶泰勒展开求亚像素偏移::

       t = - r_n / r_nn
       r_n  = r_x·nx + r_y·ny
       r_nn = r_xx·nx² + 2 r_xy nx ny + r_yy·ny²

4. 亚像素中心 = ``(x + t·nx, y + t·ny)``，仅当 |t| <= 0.5 且 r_nn < 0 有效；
5. 按**列**归并（同一列的多个候选点取法向极值最强者），
   得到每列一个中心，方便后续按列组织点云。

这种"每列一个中心"的输出与灰度重心法一致，便于在三角测量中统一处理。
"""
from __future__ import annotations

import numpy as np


def _gaussian_kernels(sigma: float, truncate: float = 4.0):
    """生成一维高斯导数核：g, g', g''（已含 sigma 归一化）。"""
    radius = max(1, int(round(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    gp = -x / (sigma ** 2) * g
    gpp = ((x ** 2 - sigma ** 2) / (sigma ** 4)) * g
    return g, gp, gpp


def _convolve1d_separable(img: np.ndarray, kx: np.ndarray,
                          ky: np.ndarray) -> np.ndarray:
    """分离卷积：等价于与二维核 ``kx ⊗ ky`` **卷积**。

    注意：``cv2.filter2D`` 实现的是**相关**而非卷积，因此需把核翻转，
    否则一阶导（梯度）方向会反号。
    """
    import cv2
    kx = kx[::-1]
    ky = ky[::-1]
    out = cv2.filter2D(img, cv2.CV_64F, kx.reshape(1, -1),
                       borderType=cv2.BORDER_REPLICATE)
    out = cv2.filter2D(out, cv2.CV_64F, ky.reshape(-1, 1),
                       borderType=cv2.BORDER_REPLICATE)
    return out


def steger_response(gray: np.ndarray, sigma: float):
    """计算 Steger 法所需的全部导数图。

    Returns
    -------
    rx, ry, rxx, rxy, ryy : 与输入同尺寸的 float64 数组
    """
    img = gray.astype(np.float64)
    g, gp, gpp = _gaussian_kernels(sigma)
    rx = _convolve1d_separable(img, gp, g)
    ry = _convolve1d_separable(img, g, gp)
    rxx = _convolve1d_separable(img, gpp, g)
    ryy = _convolve1d_separable(img, g, gpp)
    rxy = _convolve1d_separable(img, gp, gp)
    return rx, ry, rxx, rxy, ryy


def _normal_and_offset(hxx, hxy, hyy, gx, gy):
    """由 Hessian 分量求法向与亚像素偏移（逐像素，返回数组）。"""
    tmp = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy ** 2)
    lam1 = 0.5 * (hxx + hyy + tmp)   # 较大特征值
    lam2 = 0.5 * (hxx + hyy - tmp)   # 较小特征值

    # 取绝对值较大的特征值，用其**特征向量**作为条纹法向。
    # 对特征值 λ，Hessian 的特征向量为 (hxy, λ - hxx)（另一等价形式
    # 为 (λ - hyy, hxy)，两者仅差比例，方向一致或相差 π）。
    use1 = np.abs(lam1) >= np.abs(lam2)
    lam_sel = np.where(use1, lam1, lam2)
    nx = hxy
    ny = lam_sel - hxx

    norm = np.hypot(nx, ny)
    safe = norm > 1e-12
    nx = np.where(safe, nx / np.where(safe, norm, 1.0), 0.0)
    ny = np.where(safe, ny / np.where(safe, norm, 1.0), 0.0)

    r_nn = hxx * nx * nx + 2.0 * hxy * nx * ny + hyy * ny * ny
    r_n = gx * nx + gy * ny
    t = np.zeros_like(r_n)
    denom_ok = np.abs(r_nn) > 1e-12
    t[denom_ok] = -r_n[denom_ok] / r_nn[denom_ok]
    return nx, ny, r_nn, t, safe


def steger_center(gray: np.ndarray, sigma: float,
                  threshold: float,
                  subpixel: bool = True,
                  max_offset: float = 0.5) -> np.ndarray:
    """Steger 法提取亚像素条纹中心（每列一个中心）。

    Parameters
    ----------
    gray : 灰度图
    sigma : 高斯微分尺度（与条纹半宽同量级最佳）
    threshold : 灰度阈值（中心像素灰度须大于此值）
    subpixel : 是否做亚像素修正
    max_offset : 亚像素偏移上限（像素）

    Returns
    -------
    centers : (N, 2) 亚像素 (u, v)，按列升序
    """
    rx, ry, rxx, rxy, ryy = steger_response(gray, sigma)
    height, width = gray.shape

    nx, ny, r_nn, t, safe = _normal_and_offset(
        rxx, rxy, ryy, rx, ry
    )

    # 有效条件：法向有效、二阶导为负（亮脊）、亚像素偏移在范围内、
    # 中心像素灰度超阈值
    valid = safe & (r_nn < 0)
    if subpixel:
        valid &= (np.abs(t) <= max_offset)
    else:
        t = np.zeros_like(t)
    valid &= gray > threshold

    col_idx = np.arange(width)
    centers = []
    for u in range(width):
        rows = np.nonzero(valid[:, u])[0]
        if rows.size == 0:
            continue
        # 同一列可能多个候选（噪声/双峰），取"最像中心"者：
        # 即沿法向二阶导最负（曲率最强）的点
        best = rows[np.argmin(r_nn[rows, u])]
        if subpixel:
            v = best + t[best, u] * ny[best, u]
            uu = u + t[best, u] * nx[best, u]
        else:
            uu, v = float(u), float(best)
        centers.append((float(uu), float(v)))

    if not centers:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64)
