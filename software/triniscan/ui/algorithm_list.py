"""算法实现列表：从注册表**动态生成**，不在界面里写死。

问题
----
重构前界面有两处硬编码的"算法清单"：

- ``panels._FIELD_CHOICES`` 手写 ``"extraction.method": [...6 个方法名...]``；
- 标定页手写 ``"peak" / "sphere"`` 两个 Radiobutton。

后果是：**新增实现界面看不见，删掉实现界面仍提供**，用户选中后才在
运行时报错。而算法层其实已经用 ``AlgorithmSpec`` 声明了每个实现的
``name`` / ``label`` / ``help`` / ``params``（含 ``choices``）。

方案
----
界面只读这些声明：

.. code-block:: python

    from triniscan.ui.algorithm_list import entries, choices_for_key
    entries("MESHERS")                       # [AlgorithmEntry(...), ...]
    choices_for_key("extraction.method")     # ['row_peak', 'steger', ...]

单一来源由此成立：算法改了，界面自动跟着变。

注意
----
- 本模块**只读**注册表；``load_all()`` 幂等，重复调用无副作用；
- 过滤器（``FILTERS``）是插件对象，同样经 ``add_spec`` 登记了声明，
  因此也能被枚举。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["AlgorithmEntry", "DOMAIN_TITLES", "REGISTRY_NAMES",
           "entries", "entry_of", "spec_for", "choices_for_key",
           "all_domains_summary"]


#: 界面里可枚举的算法域 → 中文标题
DOMAIN_TITLES: dict[str, str] = {
    "EXTRACTORS": "中心线提取",
    "FILTERS": "点云过滤",
    "REGISTRARS": "多视角配准",
    "MESHERS": "网格重建",
    "CLOSERS": "网格封闭",
    "VOLUMETERS": "体积计算",
    "CALIBRATORS": "标定",
}

#: 注册表变量名（与 ``triniscan.contracts`` 里的全局对象同名）
REGISTRY_NAMES = tuple(DOMAIN_TITLES)


@dataclass(frozen=True)
class AlgorithmEntry:
    """界面上的一行算法（从注册表读出，不可变）。"""

    domain: str          # "MESHERS"
    name: str            # "poisson"
    label: str           # "泊松重建"
    help: str = ""
    tags: tuple[str, ...] = ()
    #: 是否登记了参数声明；未登记时 UI 只显示名字、不展示参数
    has_params: bool = False

    @property
    def display(self) -> str:
        """下拉框显示文本：``name — 中文名``（便于对照 yaml）。"""
        return f"{self.name} — {self.label}" if self.label else self.name


def _registry(domain: str):
    """按名字取注册表；未加载时先 ``load_all()``。"""
    from ..contracts import (CALIBRATORS, CLOSERS, EXTRACTORS, FILTERS,
                             MESHERS, REGISTRARS, VOLUMETERS)
    table = {
        "EXTRACTORS": EXTRACTORS, "FILTERS": FILTERS,
        "REGISTRARS": REGISTRARS, "MESHERS": MESHERS,
        "CLOSERS": CLOSERS, "VOLUMETERS": VOLUMETERS,
        "CALIBRATORS": CALIBRATORS,
    }
    reg = table.get(domain)
    if reg is None:
        raise KeyError(f"未知算法域: {domain}")
    if not len(reg):
        from ..algorithms import load_all
        load_all()
    return reg


def entries(domain: str) -> list[AlgorithmEntry]:
    """某域的全部实现（按名字排序，注册表本身已排序）。"""
    reg = _registry(domain)
    spec_names = set(getattr(reg, "spec_names", []) or [])
    out: list[AlgorithmEntry] = []
    for name in reg.names():
        spec = reg.spec(name) if name in spec_names else None
        out.append(AlgorithmEntry(
            domain=domain,
            name=name,
            label=getattr(spec, "label", "") or name,
            help=getattr(spec, "help", "") or "",
            tags=tuple(getattr(spec, "tags", ()) or ()),
            has_params=spec is not None,
        ))
    return out


def entry_of(domain: str, name: str) -> Optional[AlgorithmEntry]:
    """按名字查一个实现；不存在返回 None（界面据此提示"实现缺失"）。"""
    for e in entries(domain):
        if e.name == name:
            return e
    return None


def spec_for(domain: str, name: str):
    """取 ``AlgorithmSpec`` 本身（需要参数声明时用）；不存在返回 None。"""
    reg = _registry(domain)
    if name not in set(getattr(reg, "spec_names", []) or []):
        return None
    return reg.spec(name)


def choices_for_key(config_key: str) -> Optional[list]:
    """给定配置项全路径，反查它在算法声明里的**候选值**。

    遍历所有域的参数声明，找 ``ParamSpec.name == config_key`` 且声明了
    ``choices`` 的那一个。找不到返回 None。

    ``ConfigPanel`` 因此不必再维护 ``_FIELD_CHOICES``。
    """
    try:
        from ..algorithms import load_all
        load_all()
    except Exception:  # noqa: BLE001
        return None
    for domain in REGISTRY_NAMES:
        try:
            reg = _registry(domain)
        except Exception:  # noqa: BLE001
            continue
        spec_names = set(getattr(reg, "spec_names", []) or [])
        for name in reg.names():
            if name not in spec_names:
                continue
            spec = reg.spec(name)
            for p in getattr(spec, "params", ()) or ():
                if p.name == config_key:
                    choices = getattr(p, "choices", None)
                    if choices:
                        return list(choices)
    return None


def all_domains_summary() -> str:
    """全部域可用实现的清单（启动时写日志，便于确认注册表状态）。"""
    lines = []
    for domain, title in DOMAIN_TITLES.items():
        try:
            names = "、".join(e.name for e in entries(domain))
        except Exception as exc:  # noqa: BLE001
            names = f"（加载失败: {exc}）"
        lines.append(f"{title}: {names}")
    return "\n".join(lines)
