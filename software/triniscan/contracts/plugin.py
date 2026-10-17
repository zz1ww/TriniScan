"""插件协议：所有可替换算法共享同一个"形状"。

问题
----
重构前 5 个环节的调用签名各不相同：

.. code-block:: python

    extract_centerline(image, cfg) -> (N,2)            # 返回数组
    remove_small_clusters(pts, eps, k) -> FilterResult # 返回 dataclass
    ScanPipeline._filter_bounds(pcd) -> (pcd, info)    # 返回元组
    register_views(clouds, ...) -> RegisterResult      # 返回结果对象

调用方必须**逐个记住**每种签名，无法统一遍历、统一记录、统一开关。
这正是 `build_pointcloud` 长得像 4 段复制粘贴的原因。

方案
----
统一为 ``run(item, ctx) -> (item_out, StageReport)``，并约定：

- 插件类属性声明 ``name / label / params / depends_on_current``；
- 输入输出是"数据项"（框架不关心具体类型，由 stage 保证）；
- **掩膜类插件**（过滤器）额外实现 ``mask(item, ctx) -> (mask, info)``，
  以便 MaskStack 汇总。

这样 ``pipeline`` 只需要:

.. code-block:: python

    for plugin in chain:
        data, report = plugin.run(data, ctx)
        ctx.report.add(report)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from .params import ParamSpec
from .report import MaskStack, ScanReport, StageReport

__all__ = ["Plugin", "RunContext", "PluginBase"]


# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    """一次扫描的上下文：配置、报告、掩膜、标定、日志前缀。

    插件只通过这个对象访问外部世界，**不直接读全局配置、不直接写文件**
    ——这保证了插件可单测（构造一个假的 ctx 即可）。
    """

    cfg: Any                                     # Config（点号访问）
    report: ScanReport = field(default_factory=ScanReport)
    masks: Optional[MaskStack] = None            # 由过滤 stage 建立
    calib: Any = None                            # CalibrationBundle
    axis_point: Optional[np.ndarray] = None
    axis_dir: Optional[np.ndarray] = None
    extras: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.cfg.get(key, default)

    def log_prefix(self, name: str) -> str:
        return f"[{name}] "

    def new_mask_stack(self, n: int) -> MaskStack:
        self.masks = MaskStack(n, dry_run=self.dry_run)
        return self.masks


# ---------------------------------------------------------------------------
@runtime_checkable
class Plugin(Protocol):
    """所有可替换算法的统一接口（结构化协议，鸭子类型即可满足）。"""

    name: str                                    # 注册名
    label: str                                   # 中文名
    params: tuple[ParamSpec, ...]                # 参数声明

    def run(self, item: Any, ctx: RunContext) -> tuple[Any, StageReport]:
        ...


class PluginBase:
    """插件基类：提供参数读取、计时、日志前缀等公共能力。

    具体插件继承它，只需实现 ``execute()``。

    Attributes
    ----------
    params : 子类覆盖，声明本插件用到的**全部**参数
    depends_on_current : True 表示"本插件必须基于上一层的输出点云判断"
        （如去转盘的高度估计、轴向裁剪的直方图），因此**不能与其他
        层交换顺序**。MaskStack 会在报告里标记。
    """

    name: str = ""
    label: str = ""
    params: tuple[ParamSpec, ...] = ()
    depends_on_current: bool = False

    def __init__(self) -> None:
        self.last_report: Optional[StageReport] = None

    # ------------------------------------------------------------------
    def read_params(self, cfg) -> dict[str, Any]:
        """按 ``params`` 声明读取配置（缺省回填默认值，非法即报错）。"""
        out: dict[str, Any] = {}
        for spec in self.params:
            raw = cfg.get(spec.name, spec.default)
            out[spec.key] = spec.coerce(raw)
        return out

    # ------------------------------------------------------------------
    def run(self, item: Any, ctx: RunContext) -> tuple[Any, StageReport]:
        """模板方法：计时 → 执行 → 记录报告。子类不覆盖此方法。"""
        t0 = time.perf_counter()
        params = self.read_params(ctx.cfg) if self.params else {}
        report = StageReport(stage=self.name, params=params)
        try:
            out, report = self.execute(item, ctx, params, report)
        finally:
            report.elapsed_s = time.perf_counter() - t0
            self.last_report = report
        return out, report

    def execute(self, item: Any, ctx: RunContext,
                params: dict[str, Any],
                report: StageReport) -> tuple[Any, StageReport]:
        """子类实现的实际逻辑。"""
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 execute()")


def _short(dotted: str) -> str:
    """``"a.b.c"`` → ``"c"``（参数快照用短名，报告更易读）。

    已被 :attr:`ParamSpec.key` 取代，仅为兼容旧引用保留。
    """
    return dotted.rsplit(".", 1)[-1]
