"""激光条纹增强（提取前的图像级预处理）。

定位
----
本模块位于 :func:`triniscan.extraction.centerline.preprocess` **之后**、
中心线提取算法**之前**，只做**图像级**操作，**不触碰任何中心线提取
算法**。这样既吸收了成熟开源转台扫描器的经验，又保持提取算法本身的
稳定与可回归。

为什么需要
----------
线结构光扫描中，激光主条纹之外常有**污染**：

1. **镜面反光**：激光打在光滑表面/转台边缘，在图像中形成固定位置的
   亮斑，其亮度甚至超过激光本身，逐行 ``argmax`` 会被引到亮斑上；
2. **零散噪点**：传感器噪声与微弱点状环境光。

成熟开源扫描器（Cyclop/Horus，见 ``Jesus89/3DScanScience`` 第 3.1 节）
的处理流程是：**阈值 → 平滑 → 再阈值 → 峰值窗口掩膜**。本模块把它
拆成四个单一职责函数，便于单测与组合。

方法
----
- :func:`threshold_to_zero`：低于阈值的像素置零（去噪，保留强度）；
- :func:`blur_odd`：奇数核平滑（把连续条纹连成亮带，削弱孤立亮斑）；
- :func:`double_threshold_smooth`：阈值 → 平滑 → 阈值（Cyclop 流程）；
- :func:`peak_window_mask`：**每行取峰值列，只保留其 ±window 内的
  像素**，把远处的反光簇整行清零；
- :func:`enhance_stripe`：统一入口，按配置组合上述步骤。

分辨率无关
----------
``window`` 与 ``blur_ksize`` 的默认值按 1920 px 宽标定；对其它宽度
（如 1280）用 :func:`scale_px_param` 按 ``width / 1920`` 缩放，保证
不同分辨率下"物理尺度"一致。
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "threshold_to_zero",
    "blur_odd",
    "double_threshold_smooth",
    "peak_window_mask",
    "scale_px_param",
    "enhance_stripe",
]

#: 默认参数按 1920 px 宽带标定
_REFERENCE_WIDTH = 1920


def scale_px_param(value: float, width: int,
                   reference_width: int = _REFERENCE_WIDTH) -> int:
    """把按参考宽度标定的像素参数缩放到目标宽度。

    Parameters
    ----------
    value : 参考宽度下的像素数（可为小数）
    width : 目标图像宽度（像素）
    reference_width : 参考宽度（默认 1920）

    Returns
    -------
    int : 缩放后取整（至少 1）

    Examples
    --------
    >>> scale_px_param(9, 1920)
    9
    >>> scale_px_param(9, 1280)
    6
    """
    if reference_width <= 0 or width <= 0:
        return max(1, int(round(value)))
    return max(1, int(round(float(value) * float(width)
                            / float(reference_width))))


def _odd_ksize(k: int) -> int:
    """把核尺寸规范成 >=1 的奇数（平滑核要求）。"""
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def threshold_to_zero(gray: np.ndarray, threshold: float) -> np.ndarray:
    """低于阈值的像素置零，其余保留原值（``THRESH_TOZERO``）。

    Parameters
    ----------
    gray : (H, W) 灰度图（float）
    threshold : 阈值

    Returns
    -------
    ndarray : 同形状，``gray > threshold`` 处保留、其余为 0
    """
    g = np.asarray(gray, dtype=np.float64)
    out = g.copy()
    out[out <= float(threshold)] = 0.0
    return out


def blur_odd(gray: np.ndarray, ksize: int) -> np.ndarray:
    """奇数核均值平滑（核尺寸 <=1 时原样返回）。

    平滑的作用不是"变好看"，而是**把连贯的激光条带连成亮带、把孤立
    的几像素亮斑平均掉**——这是后续峰值/窗口掩膜能选中真激光的前提。
    """
    k = _odd_ksize(ksize)
    if k <= 1:
        return np.asarray(gray, dtype=np.float64)
    return cv2.blur(np.asarray(gray, dtype=np.float64), (k, k))


def double_threshold_smooth(gray: np.ndarray,
                            threshold: float,
                            ksize: int) -> np.ndarray:
    """Cyclop 流程：**阈值 → 平滑 → 阈值**。

    第一次阈值去孤立噪点；平滑把条纹连成亮带；第二次阈值去掉平滑
    带来的低强度晕圈。净效果是"干净的连续亮带"。
    """
    g = threshold_to_zero(gray, threshold)
    g = blur_odd(g, ksize)
    g = threshold_to_zero(g, threshold)
    return g


def peak_window_mask(gray: np.ndarray,
                     window: int,
                     min_peak: float = 0.0) -> np.ndarray:
    """峰值窗口掩膜：每行只保留**峰值列 ± window** 内的像素。

    原理
    ----
    激光在图像中是**一条连续曲线**，因此每一行**必然**有一个明确的
    激光位置；而镜面反光簇是与激光**分离**的局部亮斑。对每一行取
    峰值列 ``p``，只保留 ``[p-window, p+window]`` 区间，可把远处的
    反光簇**整行清零**。窗口**自适应**跟随激光，无需硬编码 ROI。

    这是 Cyclop（``Jesus89/3DScanScience`` 3.1 节，"Valid window
    mask"）的做法。

    Parameters
    ----------
    gray : (H, W) 灰度图（建议先经 :func:`double_threshold_smooth`）
    window : 半窗宽（像素）；<=0 表示不掩膜
    min_peak : 峰值低于此值的行整行清零（该行无激光）

    Returns
    -------
    ndarray : 同形状，窗口外像素为 0

    Notes
    -----
    若反光簇与激光**紧邻**（间距 < window），窗口无法分离——此时需
    依靠平滑后的峰值判据。本函数对"反光与激光相距较远"的场景最有效。
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or g.size == 0:
        return g.copy()
    w = int(window)
    if w <= 0:
        return g.copy()

    h, width = g.shape
    peak = np.argmax(g, axis=1)                       # (H,) 每行峰值列
    peak_val = g[np.arange(h), peak]
    valid = peak_val > float(min_peak)

    cols = np.arange(width)[None, :]
    lo = (peak - w)[:, None]
    hi = (peak + w)[:, None]
    mask = (cols >= lo) & (cols <= hi) & valid[:, None]
    return np.where(mask, g, 0.0)


def enhance_stripe(gray: np.ndarray,
                   *,
                   threshold: float = 60.0,
                   blur_ksize: int = 5,
                   window: int = 9,
                   enable_window: bool = True,
                   enable_double_threshold: bool = True,
                   scale_to_width: Optional[int] = None,
                   min_peak: float = 0.0) -> np.ndarray:
    """激光条纹增强统一入口（图像级，不改变提取算法）。

    Parameters
    ----------
    gray : (H, W) 已预处理灰度图（如 ``preprocess`` 的输出）
    threshold : 阈值（0~255 语义）
    blur_ksize : 平滑核（按参考宽度标定，自动奇数化）
    window : 峰值掩膜半窗宽（按参考宽度标定）
    enable_window : 是否做峰值窗口掩膜
    enable_double_threshold : 是否做阈值→平滑→阈值
    scale_to_width : 按此宽度缩放 ``blur_ksize`` / ``window``；
        None 表示用 ``gray.shape[1]``（自动按实际宽度缩放）
    min_peak : 峰值窗口掩膜的最小峰值

    Returns
    -------
    ndarray : 增强后的灰度图（float64）

    Examples
    --------
    >>> gray = np.zeros((100, 200)); gray[:, 50] = 200; gray[:, 10] = 255
    >>> out = enhance_stripe(gray, threshold=60, blur_ksize=1, window=5)
    >>> float(out[:, 10].max())
    0.0
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or g.size == 0:
        return g.copy()

    if scale_to_width:
        target_w = int(scale_to_width)
        k = scale_px_param(blur_ksize, target_w)
        win = scale_px_param(window, target_w)
    else:
        k = _odd_ksize(blur_ksize)
        win = int(window)

    if enable_double_threshold:
        g = double_threshold_smooth(g, threshold, k)
    if enable_window:
        g = peak_window_mask(g, win, min_peak=min_peak)
    return g
