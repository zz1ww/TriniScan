"""中心线提取算法 —— **从主项目复制而来**，仅做最小改动。

来源
----
所有算法实现逐字复制自
``software/triniscan/algorithms/extraction/centerline.py``
（以及 ``_steger.py``），**不重写算法本身**。

改动只有三处：

1. 去掉 ``extract_`` 前缀，方法名更短（``row_peak``）；
2. 每个函数上方加 ``@register(...)``，登记到 :data:`METHODS` 供命令行选择；
3. 去掉对主程序包（``triniscan.*``）的 import，改为本目录内相对导入。

**算法代码本身一字未改**，包括阈值语义、连续性校验、抛物线插值、
Steger 的解析高斯导数核等细节。这样实验台的结果与主流程一致，
调出来的参数可以直接搬回 ``config/default.yaml``。

加算法
------
1. 从 ``centerline.py`` 复制函数（改名、加 ``@register``）；
2. 若它依赖 ``_steger`` 之类的外部模块，同样复制到本目录；
3. 命令行立刻可用 ``--method 名字``。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["Spec", "METHODS", "register", "get", "names", "describe_all",
           "preprocess"]


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    """一个算法的声明。"""

    func: Callable[..., np.ndarray]
    label: str = ""
    #: 算法专属参数 -> (默认值, 说明)。命令行用 ``--param k=v`` 覆盖。
    params: dict[str, tuple[Any, str]] = field(default_factory=dict)

    def defaults(self) -> dict[str, Any]:
        return {k: v[0] for k, v in self.params.items()}

    def call(self, gray: np.ndarray, **kwargs: Any) -> np.ndarray:
        """按函数真实签名过滤参数后调用。

        各算法签名并不统一（``steger`` 没有 ``min_pixels``，``peak``
        没有 ``subpixel``）。这里只传函数真正接受的键，避免为了"统一
        接口"而改动复制的算法。
        """
        import inspect
        accepted = set(inspect.signature(self.func).parameters)
        return self.func(gray, **{k: v for k, v in kwargs.items()
                                  if k in accepted})


METHODS: dict[str, Spec] = {}


def register(name: str, label: str = "",
             **params: tuple[Any, str]) -> Callable:
    def deco(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        if name in METHODS:
            raise ValueError(f"算法名重复: {name}")
        METHODS[name] = Spec(func=func, label=label or func.__name__,
                             params=dict(params))
        return func
    return deco


def get(name: str) -> Spec:
    if name not in METHODS:
        raise KeyError(f"未知算法 '{name}'，可选: {', '.join(names())}")
    return METHODS[name]


def names() -> list[str]:
    return sorted(METHODS)


def describe_all() -> str:
    lines = []
    for name in names():
        spec = METHODS[name]
        lines.append(f"  {name:<16} {spec.label}")
        for key, (default, help_text) in spec.params.items():
            lines.append(f"      --param {key}={default}    {help_text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 预处理 —— 复制自 centerline.py::preprocess
# ---------------------------------------------------------------------------
import cv2  # noqa: E402

#: 主项目用 logging；实验台直接 print，保持零依赖
log = type("_Log", (), {
    "debug": staticmethod(lambda *a, **k: None),
    "info": staticmethod(lambda *a, **k: None),
    "warning": staticmethod(lambda *a, **k: None),
})()


def preprocess(image: np.ndarray,
               roi=None,
               sigma: float = 1.0,
               invert: bool = False,
               channel: str = "gray") -> np.ndarray:
    """统一预处理：色彩通道分离、ROI 裁剪、高斯平滑。

    **逐字复制自** ``centerline.py``，仅把 ``logging`` 换成空实现。
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
# 方法 1：灰度重心（逐列参考实现）—— 复制自 extract_gray_centroid
# ---------------------------------------------------------------------------
@register("gray_centroid", label="逐列灰度重心（逐列循环，参考实现）")
def gray_centroid(gray: np.ndarray, threshold: float = 60.0,
                  min_pixels: int = 1) -> np.ndarray:
    """逐列灰度重心法（可读性优先的参考实现）。"""
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
# 灰色重心（向量化）—— 复制自 extract_gray_centroid_fast
#
# 注意：主项目的 ``_METHODS`` **没有注册**它（``gray_centroid`` 实际
# 指向的正是这个向量化版本）。这里复制过来只为保持来源完整，
# 不单独注册，避免实验台出现主流程没有的方法名。
# ---------------------------------------------------------------------------
def gray_centroid_fast(gray: np.ndarray, threshold: float = 60.0,
                       min_pixels: int = 1) -> np.ndarray:
    """逐列灰度重心法（全向量化，生产推荐）。"""
    g = np.asarray(gray, dtype=np.float64)
    height, width = g.shape

    weights = np.maximum(g - float(threshold), 0.0)
    if min_pixels > 1:
        counts = (g > threshold).sum(axis=0)
        weights[:, counts < min_pixels] = 0.0

    col_sum = weights.sum(axis=0)
    valid = col_sum > 0.0
    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)

    rows = np.arange(height, dtype=np.float64)[:, None]
    row_sum = (weights * rows).sum(axis=0)

    u = np.nonzero(valid)[0].astype(np.float64)
    v = row_sum[valid] / col_sum[valid]
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# 方法 3：抛物线插值（逐列）—— 复制自 extract_parabola
# ---------------------------------------------------------------------------
@register("parabola", label="逐列峰值 + 抛物线插值（最快）")
def parabola(gray: np.ndarray, threshold: float = 60.0,
             min_pixels: int = 1) -> np.ndarray:
    """逐列"峰值 + 三点抛物线插值"亚像素提取。"""
    g = np.asarray(gray, dtype=np.float64)
    height, width = g.shape
    if height < 3:
        return np.empty((0, 2), dtype=np.float64)

    peak = np.argmax(g, axis=0)
    cols = np.arange(width)
    y0 = g[peak, cols]

    valid = (y0 > float(threshold)) & (peak > 0) & (peak < height - 1)
    if min_pixels > 1:
        valid &= (g > threshold).sum(axis=0) >= min_pixels
    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)

    # 【lab 修正】原实现在这里直接用全部列的 peak 取 peak±1，
    # 当某列峰值落在第 0 行或最后一行时会 IndexError
    # （5120×800 图像实测：9 列峰值在第 799 行 → 崩溃）。
    # 这里只在 valid 的列上取邻居，语义与原意一致。
    pv = peak[valid]
    cv_ = cols[valid]
    ym1 = g[pv - 1, cv_]
    yp1 = g[pv + 1, cv_]
    yy = y0[valid]
    denom = ym1 - 2.0 * yy + yp1
    delta = np.zeros(pv.shape, dtype=np.float64)
    nz = np.abs(denom) > 1e-12
    delta[nz] = 0.5 * (ym1[nz] - yp1[nz]) / denom[nz]
    delta = np.clip(delta, -1.0, 1.0)

    u = cv_.astype(np.float64)
    v = pv.astype(np.float64) + delta
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# 方法 4：逐行峰值（竖直线激光首选）—— 复制自 extract_row_peak
# ---------------------------------------------------------------------------
@register("row_peak", label="逐行峰值 + 抛物线插值（竖直线首选）",
          subpixel=(True, "是否做抛物线亚像素插值"))
def row_peak(gray: np.ndarray, threshold: float = 60.0,
             min_pixels: int = 1,
             subpixel: bool = True) -> np.ndarray:
    """逐行"峰值 + 三点抛物线插值"亚像素提取（**竖直线激光首选**）。

    对每一行 v：

    1. 沿 u 取灰度最大值所在列 ``p``；
    2. 若峰值 > ``threshold`` 且 ``p`` 不在边界，用左右邻居拟合抛物线；
    3. **连续性校验**：峰值两侧邻域的亮度须超过阈值的一半，
       用于剔除环境光/反光的孤立亮点。

    注意第 3 条：这是原实现的关键细节，缺了它会把噪点当激光。
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


# ---------------------------------------------------------------------------
# 方法 5：逐行灰度重心 —— 复制自 extract_row_centroid
# ---------------------------------------------------------------------------
@register("row_centroid", label="逐行灰度重心（向量化）")
def row_centroid(gray: np.ndarray, threshold: float = 60.0,
                 min_pixels: int = 1) -> np.ndarray:
    """逐行灰度重心法（竖直线激光的亚像素参考实现）。"""
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
# 方法 6：Steger（Hessian）—— 复制自 extract_steger
# ---------------------------------------------------------------------------
@register("steger", label="Steger 法（Hessian 亚像素）",
          sigma=(1.5, "微分尺度（与条纹半宽同量级最佳）"),
          max_offset=(0.5, "亚像素偏移容许范围（像素）"),
          merge=("column", "归并策略：column（每列一个）/ normal（法向 NMS）"),
          min_sep_px=(1.0, "normal 策略下两中心最小间距（像素）"))
def steger(gray: np.ndarray, threshold: float = 60.0,
           sigma: float = 1.5, subpixel: bool = True,
           max_offset: float = 0.5, merge: str = "column",
           min_sep_px: float = 1.0) -> np.ndarray:
    """Steger 法（Hessian 矩阵）提取条纹中心。

    实现委托给本目录的 ``_steger`` 模块（复制自主项目的同名模块），
    使用解析高斯导数核。
    """
    from _steger import steger_center
    return steger_center(gray, sigma=sigma, threshold=threshold,
                         subpixel=subpixel, max_offset=max_offset,
                         merge=merge, min_sep_px=min_sep_px)


# ---------------------------------------------------------------------------
# 方法 7：极值（逐列取整数峰）—— 复制自 extract_peak
# ---------------------------------------------------------------------------
@register("peak", label="逐列最大灰度（整数像素，最快）")
def peak(gray: np.ndarray, threshold: float = 60.0,
         min_pixels: int = 1) -> np.ndarray:
    """逐列取最大灰度位置（整数像素，最快，精度最低）。"""
    height, width = gray.shape
    centers = []
    for u in range(width):
        col = gray[:, u]
        idx = int(np.argmax(col))
        if col[idx] > threshold:
            centers.append((float(u), float(idx)))
    return np.asarray(centers, dtype=np.float64).reshape(-1, 2)
