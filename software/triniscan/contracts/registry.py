"""插件注册表：把"哪个名字对应哪个实现"从 if/elif 挪到一张表。

问题
----
重构前，项目的 5 个可替换环节用了 5 种不同接法：

==================  ==========================================
环节                接法
==================  ==========================================
中心线提取          ``_METHODS`` 字典（**唯一做对的**）
去转盘              ``if method == "ransac": ... else: ...``
过滤                ``pipeline`` 里 4 个写死顺序的私有方法
网格重建            ``if method == "poisson": ... else: ...``
配准                ``register_views`` 内部 ``if icp_refine and ...``
==================  ==========================================

每加一个算法，就要去 ``pipeline.py`` 找分支、加 ``elif``、去
``default.yaml`` 加键、去 ``ui/labels.py`` 加标签、去
``ui/panels.py`` 加下拉项 —— **改 6 个文件，且容易漏**。

方案
----
一个极简的注册表：``dict[str, type]`` + 装饰器。注册发生在 **import
时**（各 ``algorithms/<domain>/__init__.py`` 负责 import 自己的模块）。

使用::

    @FILTERS.register("cluster")
    class ClusterFilter(FilterPlugin): ...

    cls = FILTERS.get("cluster")      # 取实现
    FILTERS.names()                   # 用于配置校验 / UI 下拉
"""
from __future__ import annotations

from typing import Any, Callable, Iterator

from .errors import PluginError

__all__ = ["Registry", "FILTERS", "EXTRACTORS", "REGISTRARS", "MESHERS",
           "CLOSERS", "VOLUMETERS", "CALIBRATORS"]


class Registry:
    """一个领域一个注册表。

    Attributes
    ----------
    domain : 领域名（用于报错信息，如 ``"filtering"``）
    """

    def __init__(self, domain: str, description: str = "") -> None:
        self.domain = domain
        self.description = description
        self._items: dict[str, type] = {}
        self._specs: dict[str, object] = {}

    # ------------------------------------------------------------------
    def register(self, name: str) -> Callable[[type], type]:
        """装饰器：把类注册到 ``name`` 下。

        重复注册同名会**直接报错**（而不是静默覆盖）——这是刻意设计的，
        因为静默覆盖意味着"你以为是 A 实现，实际跑的是 B"。
        """
        def deco(cls: type) -> type:
            if name in self._items and self._items[name] is not cls:
                raise PluginError(
                    self.domain,
                    f"'{name}' 已被 {self._items[name].__qualname__} 注册，"
                    f"不能再注册 {cls.__qualname__}")
            self._items[name] = cls
            # 反向记录：让插件知道自己的注册名（日志/报告要用）
            cls.plugin_name = name            # type: ignore[attr-defined]
            return cls
        return deco

    def add(self, name: str, cls: type) -> None:
        """函数式注册（不想用装饰器时）。"""
        self.register(name)(cls)

    # ------------------------------------------------------------------
    def get(self, name: str) -> type:
        if name not in self._items:
            raise PluginError(
                self.domain, f"未知实现 '{name}'", self._items.keys())
        return self._items[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """取实现并实例化。"""
        return self.get(name)(*args, **kwargs)

    def has(self, name: str) -> bool:
        return name in self._items

    # ------------------------------------------------------------------
    # 声明式实现（AlgorithmSpec）
    # ------------------------------------------------------------------
    def add_spec(self, spec) -> None:
        """登记一个 :class:`~triniscan.contracts.algorithm.AlgorithmSpec`。

        与 :meth:`add` 共用同一张表：``get`` 返回 ``spec.func``，
        ``spec(name)`` 返回声明本身。这样"取实现"和"取参数声明"
        两个动作走同一个名字空间，不会出现两处名字对不上。
        """
        name = spec.name
        if name in self._items and self._items[name] is not spec.func:
            raise PluginError(
                self.domain,
                f"'{name}' 已被 {self._items[name]!r} 占用，"
                f"不能再登记 {spec.func!r}")
        self._items[name] = spec.func
        self._specs[name] = spec
        if hasattr(spec.func, "plugin_name"):
            return
        try:
            spec.func.plugin_name = name       # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass                                # 内建函数不可写，忽略

    def spec(self, name: str):
        """取 :class:`AlgorithmSpec`（未登记则报错）。"""
        if name not in self._specs:
            raise PluginError(
                self.domain,
                f"'{name}' 没有登记参数声明（AlgorithmSpec）",
                self._specs.keys())
        return self._specs[name]

    def specs(self) -> Iterator[tuple[str, object]]:
        return iter(sorted(self._specs.items()))

    @property
    def spec_names(self) -> list[str]:
        """已登记参数声明的实现名。"""
        return sorted(self._specs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> Iterator[tuple[str, type]]:
        return iter(sorted(self._items.items()))

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry('{self.domain}', {self.names()})"


# ---------------------------------------------------------------------------
# 全局注册表（一个领域一个）
# ---------------------------------------------------------------------------
EXTRACTORS = Registry("extraction", "激光条纹中心线提取")
FILTERS = Registry("filtering", "点云过滤")
REGISTRARS = Registry("registration", "多视角配准")
MESHERS = Registry("meshing", "表面重建")
CLOSERS = Registry("closing", "网格封闭化")
VOLUMETERS = Registry("volumetry", "体积积分")
CALIBRATORS = Registry("calibration", "标定")
