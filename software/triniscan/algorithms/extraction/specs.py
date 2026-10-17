"""中心线提取的参数声明。

``extract_centerline(image, config)`` 接收一个"支持 .get 的配置对象"，
内部用 ``_cfg_get`` 手工取值（见 ``centerline.py``）。这里把这些键
**声明**出来，供 UI 枚举与配置校验使用。

注意：``extract_centerline`` **不接受 kwargs**，它只认配置对象。因此本
spec 的 ``read()`` 结果用于"生成子配置视图 + 校验"，而不是直接当
kwargs 展开：见 :meth:`centerline_view`。
"""
from __future__ import annotations

from triniscan.contracts import AlgorithmSpec, EXTRACTORS, ParamSpec

from .centerline import extract_centerline

__all__ = ["EXTRACT_SPEC", "specs", "centerline_view"]


EXTRACT_SPEC = AlgorithmSpec(
    name="centerline",
    label="激光中心线提取",
    func=extract_centerline,
    help="逐行/逐列求激光条亚像素中心；method 决定具体算法",
    tags=("几何", "预处理"),
    params=(
        ParamSpec("extraction.method", str, "row_peak",
                  choices=("row_peak", "row_centroid", "steger",
                           "gray_centroid", "parabola", "peak"),
                  short="method",
                  label="提取方法",
                  help="row_peak=逐行峰值+抛物线(默认,竖直线激光最优)；"
                       "steger=最精确但最慢；parabola=逐列最快"),
        ParamSpec("extraction.threshold", float, 60.0,
                  low=0.0, high=255.0, short="threshold",
                  label="亮度阈值",
                  help="低于此亮度视为无激光；过低会引入噪点"),
        ParamSpec("extraction.smooth_sigma", float, 1.5,
                  low=0.0, high=10.0, short="smooth_sigma",
                  label="预处理高斯 σ",
                  help="不依赖 Steger 内部平滑；0 表示不平滑"),
        ParamSpec("extraction.subpixel", bool, True, short="subpixel",
                  label="亚像素",
                  help="逐行峰值法是否做抛物线插值"),
        ParamSpec("extraction.min_pixels", int, 1,
                  low=1, high=1000, short="min_pixels",
                  label="单视角最少像素",
                  help="少于该值的视角视为无效"),
        ParamSpec("extraction.invert", bool, False, short="invert",
                  label="亮暗反转",
                  help="亮背景暗激光时开启"),
        ParamSpec("extraction.channel", str, "red_laser",
                  choices=("red_laser", "gray"), short="channel",
                  label="颜色通道",
                  help="red_laser=彩色相机用颜色差分；gray=黑白相机"),
        ParamSpec("extraction.steger_sigma", float, 1.5,
                  low=0.5, high=6.0, short="steger_sigma",
                  label="Steger σ"),
        ParamSpec("extraction.steger_merge", str, "normal",
                  choices=("normal", "column", "none"),
                  short="steger_merge", label="Steger 合并方式"),
        ParamSpec("extraction.steger_min_sep_px", float, 1.0,
                  low=0.0, high=50.0, short="steger_min_sep_px",
                  label="Steger 最小间距(px)"),
    ),
)

specs = (EXTRACT_SPEC,)
EXTRACTORS.add_spec(EXTRACT_SPEC)


def centerline_view(cfg, **overrides) -> dict:
    """构造 ``extract_centerline`` 需要的**扁平子配置**。

    ``extract_centerline`` 期望的是 ``{"method": ..., "threshold": ...}``
    这种去掉 ``extraction.`` 前缀的扁平字典，而配置树里键是点号全路径。
    本函数完成这个翻译，顺带做**范围校验**（重构前完全没有校验）。
    """
    return EXTRACT_SPEC.read(cfg, overrides=overrides)
