"""算法层：所有可插拔实现（每个域一个子包 + 一个全局注册表）。

用法::

    from triniscan.algorithms import load_all, describe
    load_all()                       # 触发全部注册（只需一次）
    print(describe())                # 人可读的实现清单

    from triniscan.contracts import FILTERS, MESHERS
    FILTERS.names()                  # ['axial', 'bounds', 'cluster', ...]
    MESHERS.spec("poisson").read(cfg)   # 读参数（带校验与默认值回填）

两种可插拔性，**故意分开**
--------------------------
1. **过滤器**（``FILTERS``）是真正的插件对象（:class:`PluginBase`
   子类，``run(item, ctx) -> (item, report)``）——它们需要共享
   ``RunContext``、汇总掩膜、记录参数快照。

2. **其余算法域**（提取 / 三角测量 / 配准 / 网格 / 封闭 / 体积）
   是**纯函数**，签名稳定、无状态。它们以
   :class:`~triniscan.contracts.AlgorithmSpec`（声明式）登记：
   参数有唯一默认值、能被 UI 枚举、能校验，但**函数本身一字不改**。

   为什么不强行对象化？纯函数没有上下文需求，改成对象只会退化成
   "一行转发"的样板类，行为完全一样。

局限
----
- 每个子包的 ``specs.py`` 在 import 时完成登记（幂等）；
- :func:`load_all` 把全部子包 import 一遍，保证注册表完整；
- 单独 import 某个子包（如 ``triniscan.algorithms.filtering``）时
  只注册该域，启动更快。
"""
from __future__ import annotations

__all__ = ["load_all", "loaded_domains", "describe", "spec_table"]


def load_all() -> list[str]:
    """导入全部算法子包并登记声明。返回已加载的域列表。"""
    # 先 import 子包（触发过滤器的装饰器注册）
    from . import (calibration, closing, extraction, filtering,   # noqa: F401
                   meshing, registration, volumetry)
    from .features import triangulate                             # noqa: F401
    # 再 import 声明模块（触发 AlgorithmSpec 登记）
    _register_specs()
    return loaded_domains()


def loaded_domains() -> list[str]:
    """返回当前已加载（已注册）的域，便于自检。

    注意：本函数**不**导入任何子包。若 ``FILTERS`` 为空，是调用方还没
    ``import triniscan.algorithms.filtering``（或调用 :func:`load_all`）。
    """
    from ..contracts import (CALIBRATORS, CLOSERS, EXTRACTORS, FILTERS,
                             MESHERS, REGISTRARS, VOLUMETERS)
    return [r.domain for r in (EXTRACTORS, FILTERS, REGISTRARS, MESHERS,
                               CLOSERS, VOLUMETERS, CALIBRATORS)
            if len(r)]


def describe() -> str:
    """人可读的实现清单（日志/文档/``--list`` 用）。

    同时标出"是否登记了参数声明"——没有声明的实现无法被 UI 配置。
    """
    from ..contracts import (CALIBRATORS, CLOSERS, EXTRACTORS, FILTERS,
                             MESHERS, REGISTRARS, VOLUMETERS)
    lines = []
    for reg in (EXTRACTORS, FILTERS, REGISTRARS, MESHERS,
                CLOSERS, VOLUMETERS, CALIBRATORS):
        names = reg.names()
        declared = set(_spec_registry_names(reg))
        # 过滤器不是 AlgorithmSpec，它们用 PluginBase.params 声明参数
        for n in names:
            try:
                if getattr(reg.get(n), "params", ()):
                    declared.add(n)
            except Exception:           # noqa: BLE001 - 仅用于展示
                pass
        cells = []
        for n in names:
            cells.append(n if n in declared else f"{n}(无声明)")
        lines.append(f"{reg.domain:<14s}{'、'.join(cells) if cells else '（未加载）'}")
    return "\n".join(lines)


def spec_table() -> dict[str, list[dict]]:
    """导出全部参数声明（UI 生成配置面板 / 文档用）。

    Returns
    -------
    ``{domain: [{"name", "label", "help", "tags", "params": [...]}]}``
    """
    from ..contracts import (CALIBRATORS, CLOSERS, EXTRACTORS, FILTERS,
                             MESHERS, REGISTRARS, VOLUMETERS)
    out: dict[str, list[dict]] = {}
    for reg in (EXTRACTORS, FILTERS, REGISTRARS, MESHERS,
                CLOSERS, VOLUMETERS, CALIBRATORS):
        entries = []
        for name, spec in reg.specs():
            entries.append({
                "name": name,
                "label": spec.label,
                "help": spec.help,
                "tags": list(spec.tags),
                "params": [{
                    "key": p.name, "short": p.key, "type": p.type.__name__,
                    "default": p.default, "low": p.low, "high": p.high,
                    "choices": list(p.choices) if p.choices else None,
                    "unit": p.unit, "label": p.label, "help": p.help,
                } for p in spec.params],
            })
        out[reg.domain] = entries
    return out


# ---------------------------------------------------------------------------
def _spec_registry_names(registry) -> set[str]:
    try:
        return set(registry.spec_names)
    except AttributeError:              # pragma: no cover - 旧注册表
        return set()


def _register_specs() -> None:
    """导入各域的 ``specs`` 模块（幂等，模块级 import 只执行一次）。"""
    from .calibration import specs as _calibration_specs  # noqa: F401
    from .closing import specs as _closing_specs          # noqa: F401
    from .extraction import specs as _extraction_specs    # noqa: F401
    from .meshing import specs as _meshing_specs          # noqa: F401
    from .registration import specs as _registration_specs  # noqa: F401
    from .volumetry import specs as _volumetry_specs      # noqa: F401
