"""算法描述：让"纯函数型算法"也能被声明、枚举、校验。

问题
----
``FilterChain`` 里的过滤器是 :class:`~triniscan.contracts.plugin.PluginBase`
对象，参数有 ``ParamSpec`` 声明、有默认值、能校验。

但提取 / 三角测量 / 配准 / 网格 / 封闭 / 体积这 6 个域仍是**纯函数**，
它们从配置里手工取值：

.. code-block:: python

    # 三处默认值，各写各的
    poisson_depth = int(self.cfg.get("reconstruction.poisson_depth", 9))
    mesh_result = reconstruct_mesh(pcd, poisson_depth=poisson_depth, ...)

由此出现的问题：

1. **默认值没有单一来源**：``9`` 同时出现在 pipeline、mesh.py 和
   ``default.yaml``，改一处忘一处就是"配置不生效"。
2. **参数无法枚举**：UI 配置面板靠 ``labels.py`` 手工维护键名，
   加一个参数要改三个文件。
3. **越界值静默通过**：``poisson_depth=0`` 会跑到 Open3D 里才报错，
   错误信息与配置项无关。

方案
----
用 :class:`AlgorithmSpec` 把"注册名 + 中文名 + 参数声明 + 调用入口"
打包在一起，登记进注册表。**函数本身一字不改**：

.. code-block:: python

    MESHERS.add("poisson", AlgorithmSpec(
        name="poisson", label="泊松重建",
        params=(ParamSpec("reconstruction.poisson_depth", int, 9, low=4, high=14),),
        func=reconstruct_mesh,
    ))

调用方改为"先读参数、再调用"：

.. code-block:: python

    spec = MESHERS.spec("poisson")
    kwargs = spec.read(cfg)                    # 读取 + 校验 + 回填默认值
    mesh = spec.func(pcd, **kwargs)

为什么不把它们改写成对象插件？
------------------------------
纯函数没有上下文需求（不产生掩膜、不累积报告），改成对象只会退化成

.. code-block:: python

    class PoissonMesher(PluginBase):
        def execute(self, pcd, ctx, params, report):
            return reconstruct_mesh(pcd, **params), report      # 一行转发

——多出一个类，行为完全一样。**声明式描述**已经拿到了全部收益
（单一默认值来源、可枚举、可校验），又没有打破现有调用方。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .params import ParamSpec

__all__ = ["AlgorithmSpec"]


@dataclass
class AlgorithmSpec:
    """一个算法实现的声明。

    Attributes
    ----------
    name : 注册名（如 ``"poisson"``）
    label : 中文名（UI 显示）
    params : 参数声明。``ParamSpec.name`` 写**全局点号键**，
        因此 ``read(cfg)`` 能直接从配置树取值。
    func : 实际实现（纯函数或类）。调用方负责以
        ``func(*positional, **spec.read(cfg))`` 的方式使用。
    help : 一句话说明（UI 提示）。
    tags : 自由标签（如 ``("几何法", "快")``），供 UI 分组/过滤。
    """

    name: str
    label: str
    func: Callable[..., Any]
    params: tuple[ParamSpec, ...] = ()
    help: str = ""
    tags: tuple[str, ...] = ()
    #: 额外信息（实现者自填，不参与校验）
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def read(self, cfg, *, overrides: Optional[dict] = None) -> dict[str, Any]:
        """从配置读取全部参数（短名作键），可选就地覆盖。

        Parameters
        ----------
        cfg : 支持 ``.get(key, default)`` 的对象
        overrides : 以 ``ParamSpec.key``（短名）为键的覆盖值，
            同样经过类型转换与范围校验。

        Raises
        ------
        ConfigError
            任一参数超出范围 / 类型不对（由 ``ParamSpec.coerce`` 抛出）。
        """
        out: dict[str, Any] = {}
        for spec in self.params:
            raw = cfg.get(spec.name, spec.default)
            out[spec.key] = spec.coerce(raw)
        for key, value in (overrides or {}).items():
            spec = self._spec(key)
            if spec is None:
                out[key] = value
            else:
                out[key] = spec.coerce(value)
        return out

    # ------------------------------------------------------------------
    def keys(self) -> list[str]:
        """全部参数的短名（与 :meth:`read` 的键一致）。"""
        return [spec.key for spec in self.params]

    def config_keys(self) -> list[str]:
        """全部参数的全局点号键（用于 UI / 文档生成）。"""
        return [spec.name for spec in self.params]

    def validate(self, cfg) -> list[str]:
        """只校验不抛异常，返回错误列表（UI 用）。"""
        errors: list[str] = []
        for spec in self.params:
            raw = cfg.get(spec.name, spec.default)
            errors.extend(spec.validate(raw))
        return errors

    def _spec(self, key: str) -> Optional[ParamSpec]:
        for spec in self.params:
            if spec.key == key:
                return spec
        return None

    def __repr__(self) -> str:              # pragma: no cover
        return (f"<AlgorithmSpec {self.name!r} {self.label!r} "
                f"params={len(self.params)}>")
