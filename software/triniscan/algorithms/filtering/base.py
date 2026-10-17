"""过滤器基类与滤镜链（FilterChain）。

设计要点
--------
**过滤器不再自己删点，而是产出掩膜。**

重构前的写法是"串行删除"：

.. code-block:: python

    pcd, _ = self._filter_bounds(pcd)      # 换成新对象，索引丢失
    pcd, _ = self._filter_clusters(pcd)    # 看不到原始索引了
    ...

改后由 :class:`FilterChain` 统一管理**原始索引**：

.. code-block:: python

    chain = FilterChain.from_config(cfg)
    cloud, stack = chain.apply(cloud, ctx)
    print(stack.summary())    # 每层删了多少、为什么删

每个插件拿到的永远是"上一层的剩余点"，但它返回的掩膜会在链内被
**映射回原始点云的长度**，因此可以安全地汇总到 :class:`MaskStack`。

关于"顺序"
----------
大部分过滤（深度/边界/连通域/统计离群）是**纯局部判断**，只依赖点
本身。但去转盘（要估计台面高度）和轴向裁剪（要算高度直方图）依赖
**当前点云的整体分布**，因此它们的结果与顺序有关。这类插件把
``depends_on_current = True``，报告里会标出来，提醒你换顺序时要谨慎。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...contracts import (Cloud, FILTERS, MaskStack, PluginBase, RunContext,
                          StageReport)

__all__ = ["FilterPlugin", "FilterChain"]


class FilterPlugin(PluginBase):
    """过滤器插件基类。

    子类只需实现 :meth:`compute_mask`，返回"保留掩膜 + 诊断信息"。

    Attributes
    ----------
    overrides : 滤镜链项中就地覆盖的参数（如 ``{"eps_m": 0.003}``），
        优先于全局配置。默认空 dict 时全部走配置。
    """

    #: 是否必须基于**上一层过滤后的点云**判断（会与顺序耦合）
    depends_on_current: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.overrides: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def read_params(self, cfg) -> dict[str, Any]:
        """按 ``params`` 声明读取配置，再用 ``overrides`` 覆盖。"""
        out = super().read_params(cfg)
        for key, value in self.overrides.items():
            spec = next((s for s in self.params if s.key == key), None)
            out[key] = spec.coerce(value) if spec is not None else value
        return out

    # ------------------------------------------------------------------
    def compute_mask(self, cloud: Cloud,
                     ctx: RunContext) -> tuple[np.ndarray, dict]:
        """计算保留掩膜。

        Parameters
        ----------
        cloud : 当前输入点云（长度可能小于原始点云）
        ctx : 运行上下文

        Returns
        -------
        (keep_mask, info)
            ``keep_mask`` 长度必须等于 ``len(cloud)``；
            ``info`` 至少含 ``reason`` 与 ``applied``，可含任意诊断键。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 compute_mask()")

    # ------------------------------------------------------------------
    def execute(self, cloud: Cloud, ctx: RunContext,
                params: dict, report: StageReport
                ) -> tuple[Cloud, StageReport]:
        keep, info = self.compute_mask(cloud, ctx)
        keep = np.asarray(keep, dtype=bool).ravel()
        if keep.shape != (len(cloud),):
            raise ValueError(
                f"[{self.name}] 掩膜形状 {keep.shape} 与输入点数 "
                f"({len(cloud)},) 不符")
        applied = bool(info.get("applied", True))
        report.n_in = len(cloud)
        report.n_out = int(keep.sum())
        report.extra = {k: v for k, v in info.items()
                        if k not in ("reason", "applied")}
        report.extra["applied"] = applied
        if info.get("reason"):
            report.extra["reason"] = info["reason"]
        self.last_mask = keep
        return cloud.subset(keep), report


# ---------------------------------------------------------------------------
class FilterChain:
    """按配置顺序执行一串过滤器，并把掩膜汇总到 MaskStack。"""

    def __init__(self, plugins: list[FilterPlugin], dry_run: bool = False):
        self.plugins = plugins
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg, dry_run: bool = False) -> "FilterChain":
        """从配置构造滤镜链。

        优先读取新式配置 ``reconstruction.filter_chain``（列表）；
        若不存在，则回退到**旧式扁平键**（``depth_filter`` /
        ``bounds_radius_m`` / ``cluster_filter`` / ``remove_turntable`` /
        ``axial.*``），保证旧 yaml 行为完全不变。
        """
        items = cfg.get("reconstruction.filter_chain", None)
        if items is None:
            items = _legacy_chain(cfg)
        plugins: list[FilterPlugin] = []
        for item in items:
            if isinstance(item, str):
                item = {"name": item}
            if not isinstance(item, dict):
                raise ValueError(f"filter_chain 项格式非法: {item!r}")
            name = str(item.get("name", "")).strip()
            if not name:
                raise ValueError(f"filter_chain 项缺少 name: {item!r}")
            if not item.get("enable", True):
                continue
            plugin = FILTERS.get(name)()
            plugin.overrides = {k: v for k, v in item.items()
                                if k not in ("name", "enable")}
            plugins.append(plugin)
        return cls(plugins, dry_run=dry_run)

    # ------------------------------------------------------------------
    def apply(self, cloud: Cloud, ctx: RunContext
              ) -> tuple[Cloud, Optional[MaskStack]]:
        """依次执行，返回过滤后的点云 + 掩膜汇总。

        每个插件看到的是"上一层的剩余点"（长度随过滤递减），但它
        返回的掩膜会在链内**展开回原始长度**再登记，因此
        ``MaskStack`` 的每层都是 ``(N_original,)``，可直接对比。
        """
        n0 = len(cloud)
        stack = ctx.new_mask_stack(n0) if ctx.masks is None else ctx.masks
        idx = np.arange(n0)                     # 当前剩余点的原始索引
        report = ctx.report

        for plugin in self.plugins:
            sub = Cloud(cloud.points[idx], cloud.frame, cloud.source)
            out, stage_report = plugin.run(sub, ctx)

            keep_local = getattr(plugin, "last_mask", None)
            if keep_local is None or len(keep_local) != len(sub):
                keep_local = np.ones(len(sub), dtype=bool)
            keep_local = np.asarray(keep_local, dtype=bool)

            full = np.zeros(n0, dtype=bool)
            full[idx] = keep_local
            reason = stage_report.extra.get("reason", "")
            applied = bool(stage_report.extra.get("applied", True))
            stack.add(plugin.name, full, reason=reason,
                      params=stage_report.params, applied=applied,
                      extra=stage_report.extra,
                      depends_on_previous=plugin.depends_on_current)
            report.add(stage_report)

            idx = idx[keep_local]
            if idx.size == 0:
                report.add(StageReport(
                    stage="filter_chain",
                    warnings=["所有点已被过滤掉，后续过滤跳过"]))
                break

        if self.dry_run:
            return cloud, stack
        return Cloud(cloud.points[idx], cloud.frame, cloud.source), stack

    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        return [p.name for p in self.plugins]


# ---------------------------------------------------------------------------
def _legacy_chain(cfg) -> list[dict]:
    """把旧式扁平配置翻译成滤镜链（**保持原有行为与顺序**）。"""
    chain: list[dict] = []

    if bool(cfg.get("reconstruction.depth_filter", False)):
        chain.append({"name": "depth"})

    # 旧代码：三个门限全空 → 直接返回，等价于不启用
    r_max = float(cfg.get("reconstruction.bounds_radius_m", 0.0) or 0.0)
    z_min = cfg.get("reconstruction.bounds_z_min_m", None)
    z_max = cfg.get("reconstruction.bounds_z_max_m", None)
    if r_max > 0 or z_min is not None or z_max is not None:
        chain.append({"name": "bounds"})

    if bool(cfg.get("reconstruction.cluster_filter", False)):
        chain.append({"name": "cluster"})

    if bool(cfg.get("reconstruction.remove_turntable", True)):
        chain.append({"name": "turntable"})

    axial_auto = bool(cfg.get("reconstruction.axial.auto_trim", False))
    h_min = cfg.get("reconstruction.axial.height_min_m", None)
    h_max = cfg.get("reconstruction.axial.height_max_m", None)
    r_lim = cfg.get("reconstruction.axial.radius_max_m", None)
    if axial_auto or h_min is not None or h_max is not None or r_lim is not None:
        chain.append({"name": "axial"})

    return chain
