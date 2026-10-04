"""激光中心线亚像素提取。

提供两种方法:
    - gray_centroid: 逐列灰度重心法（快，先做）
    - steger:        Hessian 矩阵法（精度高，慢）
"""
from __future__ import annotations

import cv2
import numpy as np


def _preprocess(img: np.ndarray, threshold: int,
                sigma: float, roi) -> np.ndarray:
    """转灰度 + 高斯平滑 + 归一化。"""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if roi is not None:
        x, y, w, h = roi
        mask = np.zeros_like(gray)
        mask[y:y + h, x:x + w] = 255
        gray = cv2.bitwise_and(gray, mask)

    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    return gray


def extract_gray_centroid(gray: np.ndarray, threshold: int,
                          min_pixels: int = 1) -> np.ndarray:
    """逐列灰度重心法提取中心线。

    对每一列，取灰度 > threshold 的像素，按灰度加权求重心行坐标。

    Returns
    -------
    centers : (N, 2) 每列的 (u, v) 亚像素中心
    """
    h, w = gray.shape
    centers = []
    for u in range(w):
        col = gray[:, u].astype(np.float64)
        idx = np.where(col > threshold)[0]
        if len(idx) < min_pixels:
            continue
        weights = col[idx] - threshold
        if weights.sum() <= 0:
            continue
        v = (idx * weights).sum() / weights.sum()
        centers.append((float(u), float(v)))
    return np.array(centers, dtype=np.float64)


def extract_steger(gray: np.ndarray, threshold: int,
                   sigma: float = 1.5,
                   subpixel: bool = True) -> np.ndarray:
    """Steger 法（Hessian 矩阵）提取中心线。

    较精确，计算量大。先灰度重心法，需要更高精度时再用。
    """
    img = gray.astype(np.float64)

    # 高斯一阶/二阶导
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=3)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1, ksize=3)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1, ksize=3)

    h, w = img.shape
    centers = []
    for v in range(h):
        for u in range(w):
            if img[v, u] < threshold:
                continue
            hxx, hxy, hyy = gxx[v, u], gxy[v, u], gyy[v, u]
            # Hessian 特征值方向
            tmp = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
            lam1 = 0.5 * (hxx + hyy + tmp)
            lam2 = 0.5 * (hxx + hyy - tmp)
            # 法向：对应绝对值最大的特征值
            if abs(lam1) >= abs(lam2):
                nx, ny = hxy, lam1 - hxx
            else:
                nx, ny = lam1 - hyy, hxy
            norm = np.hypot(nx, ny)
            if norm < 1e-9:
                continue
            nx, ny = nx / norm, ny / norm

            if subpixel:
                # 沿法向泰勒展开求亚像素
                rx = gx[v, u] * nx + gy[v, u] * ny
                rxx = hxx * nx * nx + 2 * hxy * nx * ny + hyy * ny * ny
                if abs(rxx) < 1e-9:
                    continue
                t = -rx / rxx
                if abs(t) > 0.5:
                    continue
                uu, vv = u + t * nx, v + t * ny
            else:
                uu, vv = float(u), float(v)
            centers.append((uu, vv))
    return np.array(centers, dtype=np.float64)


def extract_centerline(img: np.ndarray, cfg) -> np.ndarray:
    """统一入口：按配置选择方法。

    Parameters
    ----------
    img : 输入图像
    cfg : Config（triniscan.core.config.Config）

    Returns
    -------
    centers : (N, 2) 亚像素中心点 (u, v)
    """
    method = cfg.get("method", "gray_centroid")
    threshold = cfg.get("threshold", 60)
    sigma = cfg.get("smooth_sigma", 1.0)
    roi = cfg.get("roi", None)

    gray = _preprocess(img, threshold, sigma, roi)

    if method == "steger":
        return extract_steger(gray, threshold, sigma)
    return extract_gray_centroid(gray, threshold)


if __name__ == "__main__":
    import sys
    from triniscan.core.config import Config

    if len(sys.argv) < 2:
        print("用法: python centerline.py <image>")
        sys.exit(1)
    image = cv2.imread(sys.argv[1])
    cfg = Config.load().extraction
    pts = extract_centerline(image, cfg)
    print(f"提取到 {len(pts)} 个中心线点")
    vis = image.copy()
    for (u, v) in pts:
        cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0, 0, 255), -1)
    cv2.imshow("centerline", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
