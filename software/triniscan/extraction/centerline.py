"""激光中心线亚像素提取。

本模块实现线结构光（光切法）中**激光条纹中心**的亚像素定位，
是三角测量精度的第一道关口。条纹中心定位偏差会 1:1 传递到三维点。

列扫描 vs 行扫描
----------------
"逐列"与"逐行"是**同一套权重逻辑**的两个扫描方向，选哪个取决于
激光条纹在图像中的走向：

- **逐列 (column)**：每列沿 v 方向求一个中心，适合**水平条纹**；
- **逐行 (row)**：每行沿 u 方向求一个中心，适合**竖直条纹**。

本系统的激光线在图像中近似**竖直**（跨越多行），因此**逐行扫描**
是更自然的选择：图像高 1080 行 → 每帧最多 1080 个采样，且分辨率
不被条纹宽度限制。逐列扫描在同样的数据上每个有效列只能给出一个点，
采样密度低得多。这也是成熟开源转台扫描器的通行做法。

方法清单
--------
1. **灰度重心法 (gray centroid)** — 按灰度加权求重心，列/行两个方向
   均有实现。实现简单、速度快，适合条纹宽度中等、亮度分布近似对称
   的情形。
2. **Steger 法 (Hessian)** — 基于 Hessian 矩阵求条纹法向，再沿法向
   做泰勒展开求亚像素极值点。精度最高、抗噪最好，适合高精度场合，
   计算量较大。对条纹走向不敏感（不依赖扫描方向）。
3. **抛物线插值法 (parabola)** — 峰值 + 三点抛物线插值，最快的亚像素
   方法。
4. **极值法 (peak)** — 取最大灰度位置。最快，精度最低，仅用于预览。

默认方法为 ``row_peak``（逐行峰值 + 抛物线亚像素），可用 ``steger``
在精度敏感时对比替换。

设计说明
--------
- 所有方法共享同一套预处理（转灰度 → ROI → 平滑）。
- 每个方法均返回 ``(N, 2)`` 的 ``(u, v)`` 亚像素坐标，按扫描方向排序。
- 通过 :func:`extract_centerline` 统一入口，按配置字符串选择方法，
  便于二次修改与替换（新增方法只需注册到 ``_METHODS``）。
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "extract_centerline",
    "extract_gray_centroid",
    "extract_gray_centroid_fast",
    "extract_parabola",
    "extract_steger",
    "extract_peak",
    "extract_row_peak",
    "extract_row_centroid",
    "preprocess",
    "available_methods",
]


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------
def preprocess(image: np.ndarray,
               roi: tuple[int, int, int, int] | list[int] | None = None,
               sigma: float = 1.0,
               invert: bool = False,
               channel: str = "gray") -> np.ndarray:
    """统一预处理：色彩通道分离、ROI 裁剪、高斯平滑。

    Parameters
    ----------
    image : 输入图像（灰度或 BGR）
    roi : ``(x, y, w, h)`` 感兴趣区域；None 表示全图。
          ROI 外的像素被置零。
    sigma : 高斯平滑标准差；<=0 表示不平滑。
    invert : 若条纹为暗背景上的亮线用 False（默认）；
             若为亮背景上的暗线则设 True。
    channel : 色彩通道选择，用于**抑制与激光不同色的干扰**：

        - ``"gray"``      : 普通灰度（默认）；
        - ``"r"``/``"g"``/``"b"`` : 单通道；
        - ``"red_laser"`` : ``R - max(G, B)``，突出红色激光、抑制
          黑白棋盘格与灰白背景（黑白处 R≈G≈B，相减≈0）；
        - ``"green_laser"``: ``G - max(R, B)``，绿激光对应。

        对红色激光（650nm），强烈建议用 ``"red_laser"``：棋盘格边缘、
        木框、手等灰白景象会被大幅压暗，中心线提取只保留激光本身。

    Returns
    -------
    gray : float64 灰度图

    Notes
    -----
    **黑白（单通道）相机自适应**：若 ``image.ndim == 2``（相机直出灰度），
    则无论 ``channel`` 配置为何一律按 ``"gray"`` 处理——此时颜色差分
    （``red_laser`` / ``green_laser``）无意义，若不降级会静默失效或
    报错。这样黑白相机与彩色相机可**共用同一套配置与代码路径**。
    """
    if image.ndim == 3:
        ch = (channel or "gray").lower()
        if ch in ("red_laser", "green_laser"):
            b, g, r = (image[..., i].astype(np.float64)
                       for i in range(3))
            if ch == "red_laser":
                gray = r - np.maximum(g, b)
            else:
                gray = g - np.maximum(r, b)
            gray = np.clip(gray, 0.0, None)
            # 颜色差分结果动态范围较小，拉伸到 0~255 便于统一阈值语义
            mx = float(gray.max())
            if mx > 0:
                gray = gray / mx * 255.0
        elif ch == "r":
            gray = image[..., 2].astype(np.float64)
        elif ch == "g":
            gray = image[..., 1].astype(np.float64)
        elif ch == "b":
            gray = image[..., 0].astype(np.float64)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # 单通道（黑白相机直出）：颜色差分通道自动降级为 gray
        if (channel or "gray").lower() not in ("gray", ""):
            log.debug("单通道图像：提取通道 '%s' 自动降级为 gray", channel)
        gray = image.copy()

    if roi is not None:
        x, y, w, h = (int(v) for v in roi)
        mask = np.zeros(gray.shape, dtype=gray.dtype)
        mask[y:y + h, x:x + w] = 255
        gray = cv2.bitwise_and(gray, mask)

    gray = gray.astype(np.float64)

    if sigma and sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), float(sigma))

    if invert:
        gray = gray.max() - gray

    return gray


def _column_roi(gray: np.ndarray, u: int,
                threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """取第 u 列中超过阈值的像素索引与权重。"""
    col = gray[:, u]
    idx = np.nonzero(col > threshold)[0]
    if idx.size == 0:
        return idx, np.empty(0)
    weights = col[idx] - threshold
    return idx, weights


# ---------------------------------------------------------------------------
# 方法 1：灰度重心法（逐列，参考实现，清晰但较慢）
# ---------------------------------------------------------------------------
def extract_gray_centroid(gray: np.ndarray,
                          threshold: float = 60.0,
                          min_pixels: int = 1) -> np.ndarray:
    """逐列灰度重心法（可读性优先的参考实现）。

    对每一列 u，取灰度大于 ``threshold`` 的像素，以 ``(灰度 - 阈值)``
    为权重求行坐标重心，得到该列亚像素中心。

    Parameters
    ----------
    gray : 预处理后的灰度图
    threshold : 分割阈值（绝对灰度）
    min_pixels : 视为有效条纹的最少像素数

    Returns
    -------
    centers : (N, 2) 亚像素 (u, v)，按列升序
    """
    height, width = gray.shape
    centers = []
    for u in range(width):
        idx, weights = _column_roi(gray, u, threshold)
        if idx.size < min_pixels or weights.sum() <= 0:
            continue
        v = float((idx * weights).sum() / weights.sum())
        centers.append((float(u), v))
    return np.asarray(centers, dtype=np.float64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# 方法 1b：灰度重心法（向量化加速版，生产使用）
# ---------------------------------------------------------------------------
def extract_gray_centroid_fast(gray: np.ndarray,
                               threshold: float = 60.0,
                               min_pixels: int = 1) -> np.ndarray:
    """逐列灰度重心法（**全向量化**，生产推荐）。

    与 :func:`extract_gray_centroid` 结果一致，但用纯 NumPy 一次性对整个
    图像做加权求和，**没有 Python 逐列循环**，速度约为逐列版的 5~10 倍
    （1920×1080 实测约 5~10 ms）。

    算法（对每一列 u 独立进行）::

        w[v]  = max(gray[v, u] - threshold, 0)      # 权重
        v(u)  = Σ v·w[v] / Σ w[v]                    # 灰度重心

    仅当该列有效像素数 ≥ ``min_pixels`` 且权重和 > 0 时输出一个中心。

    Parameters
    ----------
    gray : 预处理后的灰度图（float 或 uint8 均可）
    threshold : 分割阈值（绝对灰度）
    min_pixels : 视为有效条纹的最少像素数

    Returns
    -------
    centers : (N, 2) 亚像素 (u, v)，按列升序
    """
    g = np.asarray(gray, dtype=np.float64)
    height, width = g.shape

    # 权重：低于阈值的像素置零（等价于 (col - threshold) 取正部分）
    weights = np.maximum(g - float(threshold), 0.0)
    if min_pixels > 1:
        counts = (g > threshold).sum(axis=0)
        weights[:, counts < min_pixels] = 0.0

    col_sum = weights.sum(axis=0)                      # (W,)
    valid = col_sum > 0.0
    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)

    rows = np.arange(height, dtype=np.float64)[:, None]
    row_sum = (weights * rows).sum(axis=0)             # (W,)

    u = np.nonzero(valid)[0].astype(np.float64)
    v = row_sum[valid] / col_sum[valid]
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# 方法 1c：抛物线插值法（向量化，最快的亚像素方法）
# ---------------------------------------------------------------------------
def extract_parabola(gray: np.ndarray,
                     threshold: float = 60.0,
                     min_pixels: int = 1) -> np.ndarray:
    """逐列"峰值 + 三点抛物线插值"亚像素提取（最快）。

    思路（激光三角测量中的经典做法）：对每列取灰度最大值所在行 ``p``，
    再用其上下邻居 ``y[p-1], y[p], y[p+1]`` 拟合抛物线，取顶点作为亚像素
    中心::

        delta = 0.5 * (y[p-1] - y[p+1]) / (y[p-1] - 2·y[p] + y[p+1])

    全程向量化（``argmax`` + 花式索引），无 Python 循环，速度最快
    （1920×1080 实测约 3~6 ms），适合对速度敏感的预览/粗扫描。

    精度说明：当条纹剖面接近高斯/抛物线时精度可达 ~0.1 px；对边缘陡峭
    或非对称剖面会略差于灰度重心法。若需最高精度请用 ``steger``。

    Returns
    -------
    centers : (N, 2) 亚像素 (u, v)，按列升序
    """
    g = np.asarray(gray, dtype=np.float64)
    height, width = g.shape
    if height < 3:
        return np.empty((0, 2), dtype=np.float64)

    peak = np.argmax(g, axis=0)                        # (W,)
    cols = np.arange(width)
    y0 = g[peak, cols]                                 # 峰值列

    # 有效：峰值超阈值、且峰不在边界（否则无法取两点邻居）
    valid = (y0 > float(threshold)) & (peak > 0) & (peak < height - 1)
    if min_pixels > 1:
        valid &= (g > threshold).sum(axis=0) >= min_pixels
    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)

    ym1 = g[peak - 1, cols]
    yp1 = g[peak + 1, cols]
    denom = ym1 - 2.0 * y0 + yp1                       # 抛物线二阶系数×2
    delta = np.zeros(width, dtype=np.float64)
    nz = np.abs(denom) > 1e-12
    delta[nz] = 0.5 * (ym1[nz] - yp1[nz]) / denom[nz]
    delta = np.clip(delta, -1.0, 1.0)                  # 顶点必须落在中心格

    u = cols[valid].astype(np.float64)
    v = peak[valid].astype(np.float64) + delta[valid]
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# 方法 1d：逐行扫描（适配竖直线激光，本系统默认）
# ---------------------------------------------------------------------------
def extract_row_peak(gray: np.ndarray,
                     threshold: float = 60.0,
                     min_pixels: int = 1,
                     subpixel: bool = True) -> np.ndarray:
    """逐行"峰值 + 三点抛物线插值"亚像素提取（**竖直线激光首选**）。

    与 :func:`extract_parabola` 完全相同的权重逻辑，只是把扫描方向从
    "逐列（沿 v）"换成"逐行（沿 u）"——因为本系统的激光条纹近似竖直，
    每一行都有且仅有一个明确的激光位置。

    对每一行 v：

    1. 沿 u 取灰度最大值所在列 ``p``（``argmax``，**对零散亮点免疫**：
       只有全场最亮的像素才会被选中，不会像重心法那样被周边暗噪声拉偏）；
    2. 若峰值 > ``threshold`` 且 ``p`` 不在边界，用左右邻居
       ``y[p-1], y[p], y[p+1]`` 拟合抛物线取顶点，得到亚像素偏移::

           delta = 0.5 * (y[p-1] - y[p+1]) / (y[p-1] - 2·y[p] + y[p+1])

    全程向量化（``argmax(axis=1)`` + 花式索引），无 Python 循环。

    Parameters
    ----------
    gray : 预处理后的灰度图
    threshold : 分割阈值（绝对灰度）
    min_pixels : 该行超过阈值的像素数下限（用于过滤仅一两个噪声点的行）
    subpixel : 是否做抛物线亚像素插值；False 时返回整数峰值列

    Returns
    -------
    centers : (N, 2) 亚像素 (u, v)，按行（v）升序
    """
    g = np.asarray(gray, dtype=np.float64)
    height, width = g.shape
    if height < 1 or width < 3:
        return np.empty((0, 2), dtype=np.float64)

    peak = np.argmax(g, axis=1)                        # (H,) 每行最亮列
    rows = np.arange(height)
    y0 = g[rows, peak]                                 # 每行峰值

    valid = y0 > float(threshold)
    if min_pixels > 1:
        valid &= (g > threshold).sum(axis=1) >= min_pixels
    if subpixel:
        # 峰在边界时无法取左右邻居 → 该行放弃亚像素（退化为整数）
        valid &= (peak > 0) & (peak < width - 1)

    # 连续性校验：真激光是一条**连续亮带**，其峰值两侧相邻列也应足够亮；
    # 而环境光/反光的"飞点"通常是**孤立**的个别像素。要求峰值邻域
    # （在边界内）亮度超过阈值的一半，可有效剔除孤立亮点。
    interior = (peak > 0) & (peak < width - 1)
    if interior.any():
        idx = np.nonzero(interior)[0]
        nb = np.minimum(g[idx, peak[idx] - 1], g[idx, peak[idx] + 1])
        ok = nb > float(threshold) * 0.5
        bad = idx[~ok]
        valid[bad] = False

    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)

    u = peak[valid].astype(np.float64)
    if subpixel:
        ym1 = g[rows[valid], peak[valid] - 1]
        yp1 = g[rows[valid], peak[valid] + 1]
        yy = y0[valid]
        denom = ym1 - 2.0 * yy + yp1
        delta = np.zeros(u.shape, dtype=np.float64)
        nz = np.abs(denom) > 1e-12
        delta[nz] = 0.5 * (ym1[nz] - yp1[nz]) / denom[nz]
        delta = np.clip(delta, -1.0, 1.0)              # 顶点须落在中心格
        u = u + delta

    v = rows[valid].astype(np.float64)
    return np.column_stack([u, v])


def extract_row_centroid(gray: np.ndarray,
                         threshold: float = 60.0,
                         min_pixels: int = 1) -> np.ndarray:
    """逐行灰度重心法（竖直线激光的亚像素参考实现）。

    与 :func:`extract_gray_centroid_fast` 逻辑一致，只是改为按**行**
    做加权重心（全向量化）。相比 :func:`extract_row_peak`，重心法对
    亮度分布更敏感、精度略高，但**对行内零散亮点更脆弱**（会被拉偏）。
    适合条纹干净、背景抑制良好的情形。

    Returns
    -------
    centers : (N, 2) 亚像素 (u, v)，按行（v）升序
    """
    g = np.asarray(gray, dtype=np.float64)
    height, width = g.shape

    weights = np.maximum(g - float(threshold), 0.0)    # (H, W)
    if min_pixels > 1:
        counts = (g > threshold).sum(axis=1)
        weights[counts < min_pixels, :] = 0.0

    row_sum = weights.sum(axis=1)                      # (H,)
    valid = row_sum > 0.0
    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)

    cols = np.arange(width, dtype=np.float64)[None, :]
    col_sum = (weights * cols).sum(axis=1)             # (H,)

    u = col_sum[valid] / row_sum[valid]
    v = np.nonzero(valid)[0].astype(np.float64)
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# 方法 2：Steger 法（Hessian）
# ---------------------------------------------------------------------------
def extract_steger(gray: np.ndarray,
                   threshold: float = 60.0,
                   sigma: float = 1.5,
                   subpixel: bool = True,
                   max_offset: float = 0.5,
                   merge: str = "column",
                   min_sep_px: float = 1.0) -> np.ndarray:
    """Steger 法（Hessian 矩阵）提取条纹中心。

    实现委托给 :mod:`triniscan.extraction._steger`，使用解析高斯导数核，
    亚像素精度优于 0.01 px（见单元测试）。

    Parameters
    ----------
    gray : 预处理后的灰度图
    threshold : 最小灰度阈值
    sigma : 微分尺度（与条纹半宽同量级最佳）
    subpixel : 是否做亚像素修正
    max_offset : 亚像素偏移的容许范围（像素）
    merge : 归并策略，``"column"``（默认，每列一个）或 ``"normal"``
        （按法向非极大抑制，支持倾斜/大曲率条纹）
    min_sep_px : ``"normal"`` 策略下两中心最小间距（像素）

    Returns
    -------
    centers : (N, 2) 亚像素 (u, v)
    """
    from ._steger import steger_center
    return steger_center(gray, sigma=sigma, threshold=threshold,
                         subpixel=subpixel, max_offset=max_offset,
                         merge=merge, min_sep_px=min_sep_px)


# ---------------------------------------------------------------------------
# 方法 3：极值法（快速预览）
# ---------------------------------------------------------------------------
def extract_peak(gray: np.ndarray, threshold: float = 60.0) -> np.ndarray:
    """逐列取最大灰度位置（整数像素，最快，精度最低）。"""
    height, width = gray.shape
    centers = []
    for u in range(width):
        col = gray[:, u]
        idx = int(np.argmax(col))
        if col[idx] > threshold:
            centers.append((float(u), float(idx)))
    return np.asarray(centers, dtype=np.float64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# 方法注册与统一入口
# ---------------------------------------------------------------------------
_METHODS: dict[str, Callable[..., np.ndarray]] = {
    "gray_centroid": extract_gray_centroid_fast,
    "parabola": extract_parabola,
    "steger": extract_steger,
    "peak": extract_peak,
    "row_peak": extract_row_peak,
    "row_centroid": extract_row_centroid,
}

#: 需要 ``subpixel`` 参数的方法（逐行峰值）
_ROW_PEAK_METHODS = {"row_peak"}


def available_methods() -> list[str]:
    """返回已注册的中心线提取方法名。"""
    return sorted(_METHODS.keys())


def register_method(name: str, func: Callable[..., np.ndarray]) -> None:
    """注册新的中心线提取方法（便于二次扩展）。"""
    _METHODS[name] = func
    log.info("注册中心线提取方法: %s", name)


def extract_centerline(image: np.ndarray, config) -> np.ndarray:
    """统一入口：按配置选择方法并提取中心线。

    Parameters
    ----------
    image : 输入图像（BGR 或灰度）
    config : 支持 ``.get(key, default)`` 的配置对象
        （``triniscan.core.config.Config`` 或普通 dict 包装均可）

    Returns
    -------
    centers : (N, 2) 亚像素 (u, v)
    """
    method = _cfg_get(config, "method", "gray_centroid")
    if method not in _METHODS:
        raise ValueError(
            f"未知的中心线提取方法 '{method}'，可选: {available_methods()}"
        )

    roi = _cfg_get(config, "roi", None)
    threshold = float(_cfg_get(config, "threshold", 60.0))
    invert = bool(_cfg_get(config, "invert", False))
    min_pixels = int(_cfg_get(config, "min_pixels", 1))
    channel = str(_cfg_get(config, "channel", "gray"))

    # Steger 法自带高斯微分平滑，无需额外预处理平滑（避免过度模糊）；
    # 其余方法用 smooth_sigma 预处理。
    if method == "steger":
        sigma_pre = float(_cfg_get(config, "smooth_sigma", 0.0))
    else:
        sigma_pre = float(_cfg_get(config, "smooth_sigma", 1.0))

    gray = preprocess(image, roi=roi, sigma=sigma_pre, invert=invert,
                      channel=channel)

    # 可选：图像级条纹增强（双阈值平滑 + 峰值窗口掩膜）。
    # 默认关闭，保证不影响既有行为；开启见 extraction.enhance_enable。
    # **只作用于灰度图，不改变任何提取算法**。
    #
    # 注：配置项采用**扁平键**（``enhance_enable`` / ``enhance_threshold``
    # 等）而非嵌套字典，这样 UI 配置面板才能自动渲染成"是/否"下拉框
    # 与数值输入框。兼容旧的嵌套写法 ``enhance: {enable: ...}``。
    _enh = _cfg_get(config, "enhance", None)
    if isinstance(_enh, dict):                       # 旧嵌套写法（向后兼容）
        def _ev(k, d):
            return _enh.get(k, d)
    else:                                            # 扁平键（推荐）
        def _ev(k, d):
            return _cfg_get(config, "enhance_" + k, d)

    if bool(_ev("enable", False)):
        from .stripe_enhance import enhance_stripe
        use_scale = bool(_ev("scale_to_width", True))
        gray = enhance_stripe(
            gray,
            threshold=float(_ev("threshold", threshold)),
            blur_ksize=int(_ev("blur_ksize", 5)),
            window=int(_ev("window", 9)),
            enable_window=bool(_ev("window_enable", True)),
            enable_double_threshold=bool(
                _ev("double_threshold_enable", True)),
            scale_to_width=(gray.shape[1] if use_scale else None),
            min_peak=float(_ev("min_peak", 0.0)),
        )

    func = _METHODS[method]
    if method == "steger":
        steger_sigma = float(
            _cfg_get(config, "steger_sigma",
                     _cfg_get(config, "smooth_sigma", 1.5))
        )
        steger_merge = str(_cfg_get(config, "steger_merge", "column"))
        steger_min_sep = float(_cfg_get(config, "steger_min_sep_px", 1.0))
        centers = func(gray, threshold=threshold, sigma=steger_sigma,
                       merge=steger_merge, min_sep_px=steger_min_sep)
    elif method == "peak":
        centers = func(gray, threshold=threshold)
    elif method in _ROW_PEAK_METHODS:
        subpixel = bool(_cfg_get(config, "subpixel", True))
        centers = func(gray, threshold=threshold, min_pixels=min_pixels,
                       subpixel=subpixel)
    else:  # gray_centroid / parabola / row_centroid（向量化，支持 min_pixels）
        centers = func(gray, threshold=threshold, min_pixels=min_pixels)

    log.debug("中心线提取[%s]: %d 点", method, len(centers))
    return centers


def _cfg_get(config, key: str, default):
    """兼容 Config 对象与普通 dict 的取值。"""
    if hasattr(config, "get"):
        return config.get(key, default)
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
