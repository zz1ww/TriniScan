"""点云过滤（可插拔）。

可用过滤器
----------
==================  ==========  ==========================================
注册名              顺序敏感    作用
==================  ==========  ==========================================
``depth``           否          相机系深度门限（三角测量后立即执行）
``bounds``          否          世界系半径/轴向物理边界
``cluster``         否          欧氏聚类去重（只留最大 k 个簇）
``statistical``     否          统计离群 SOR（默认不启用）
``turntable``       是          去转盘表面（axis_height / ransac）
``axial``           是          轴向裁剪（自动主体区间 / 硬门限）
==================  ==========  ==========================================

"顺序敏感"表示该过滤器依赖**当前点云的整体分布**（如估计台面高度、
统计高度直方图），因此不能随意调换顺序。

配置
----
推荐写法（新式，显式列出顺序）::

    reconstruction:
      filter_chain:
        - {name: depth}
        - {name: bounds, enable: false}
        - {name: cluster, eps_m: 0.002, keep_top: 1}
        - {name: turntable, method: axis_height}
        - {name: axial, auto_trim: true}

若没写 ``filter_chain``，会自动回退到旧式扁平键
（``depth_filter`` / ``bounds_radius_m`` / ``cluster_filter`` /
``remove_turntable`` / ``axial.*``），行为与重构前一致。

用法::

    from triniscan.algorithms.filtering import FilterChain
    chain = FilterChain.from_config(cfg)
    cloud, masks = chain.apply(cloud, ctx)
    print(masks.summary())          # 每层删了多少、为什么
"""
from __future__ import annotations

# 导入即注册（顺序即默认推荐顺序）
from . import axial, bounds, cluster, depth, statistical, turntable  # noqa: F401
from .axial import AxialTrimFilter, dominant_band                   # noqa: F401
from .base import FilterChain, FilterPlugin                         # noqa: F401
from .bounds import BoundsFilter                                    # noqa: F401
from .cluster import ClusterFilter                                  # noqa: F401
from .depth import DepthFilter                                      # noqa: F401
from .statistical import StatisticalOutlierFilter                   # noqa: F401
from .turntable import TurntableFilter                              # noqa: F401

# 兼容：旧的纯函数实现仍可直接调用
from .point_filter import (FilterResult, filter_by_bounds,          # noqa: F401
                           filter_by_depth, remove_small_clusters)
from .turntable_removal import (TurntableRemovalResult,             # noqa: F401
                                estimate_table_height,
                                remove_turntable_axis_height,
                                remove_turntable_surface)

__all__ = [
    # 插件
    "FilterPlugin", "FilterChain",
    "DepthFilter", "BoundsFilter", "ClusterFilter",
    "StatisticalOutlierFilter", "TurntableFilter", "AxialTrimFilter",
    "dominant_band",
    # 纯函数（兼容）
    "FilterResult", "filter_by_depth", "filter_by_bounds",
    "remove_small_clusters",
    "TurntableRemovalResult", "estimate_table_height",
    "remove_turntable_axis_height", "remove_turntable_surface",
]
