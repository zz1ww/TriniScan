"""界面读写配置的统一出口：参数表 + 算法选择列表。

问题
----
重构前"配置项"在界面里是**散落的**：

1. ``ConfigPanel`` 遍历 ``cfg.raw`` 建控件，控件类型靠 ``isinstance``
   猜（``bool`` / ``None`` / ``int`` / ``float``），范围**完全不校验**；
2. ``_FIELD_CHOICES`` 手写下拉候选值，与算法层的 ``ParamSpec.choices``
   各写一份；
3. 标定页自己维护 ``_sphere_radius_mm`` 等 StringVar，与配置面板
   各自解析、各自校验；
4. 类型转换散在 ``ConfigPanel._coerce``、``calibration._read_axis_params``
   等多处，规则不一致（比如布尔接受哪些写法）。

后果是同一份配置在界面上有三个"说法"，改一处忘了另一处就是"改了没用"。

方案
----
把"某个配置项该长什么样"收在一处——:class:`FieldSpec`：

.. code-block:: python

    fields = all_fields(cfg)
    f = fields["scan.num_views"]
    f.kind          # "int"
    f.choices       # [180, 150, 120, 90, 60]
    f.coerce("72")  # 72（int）
    f.validate(0)   # ["scan.num_views 小于下限 1"]

来源优先级
----------
1. **算法声明**（``AlgorithmSpec.ParamSpec``）——最权威，带范围与候选值；
2. ``Config.TYPES``（若配置模块提供了已知类型表）；
3. 按当前值类型推断（兜底，永不失败）。

这样面板只负责"摆控件"，不再负责"怎么解析"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .labels import field_hint, field_label, group_label

__all__ = ["FieldSpec", "all_fields", "field_groups", "algorithm_choice_key"]


#: ``kind`` 取值
_KIND_BOOL = "bool"
_KIND_INT = "int"
_KIND_FLOAT = "float"
_KIND_STR = "str"
_KIND_NONE = "none"


def _kind_of(value: Any) -> str:
    if isinstance(value, bool):
        return _KIND_BOOL
    if isinstance(value, int):
        return _KIND_INT
    if isinstance(value, float):
        return _KIND_FLOAT
    if value is None:
        return _KIND_NONE
    return _KIND_STR


@dataclass
class FieldSpec:
    """一个配置项在界面上应有的样子。

    Attributes
    ----------
    key : 全局点号键（``"scan.num_views"``）
    label / group : 中文短名与分组名
    hint : 悬停说明
    kind : ``"bool" / "int" / "float" / "str" / "none"``
    choices : 下拉候选值（None 表示自由文本）
    low / high : 数值范围（None 表示不限）
    unit : 单位（``"m" / "mm" / "deg" / "px"``）
    """

    key: str
    label: str
    group: str
    hint: str = ""
    kind: str = _KIND_STR
    choices: Optional[list] = None
    low: Optional[float] = None
    high: Optional[float] = None
    unit: str = ""
    #: 是否来自算法声明（True 时说明该参数在某处被真正使用）
    from_algorithm: bool = False

    # ------------------------------------------------------------------
    @property
    def is_numeric(self) -> bool:
        return self.kind in (_KIND_INT, _KIND_FLOAT)

    @property
    def group_key(self) -> str:
        return self.key.split(".", 1)[0]

    # ------------------------------------------------------------------
    def coerce(self, raw: Any) -> Any:
        """把界面上的原始输入转成配置里应有的 Python 类型。

        失败时**原样返回**（保留用户输入，交由 :meth:`validate` 报错），
        不在这里弹框——界面层不该决定交互方式。
        """
        if self.choices is not None:
            # 候选项可能带类型（如 180 是 int），按候选值本身的类型还原
            for c in self.choices:
                if str(c) == str(raw).strip():
                    return c
            # 不在候选里：按 kind 尝试一次
        text = "" if raw is None else str(raw).strip()

        if self.kind == _KIND_BOOL:
            low = text.lower()
            if low in ("是", "true", "1", "yes", "on", "y"):
                return True
            if low in ("否", "false", "0", "no", "off", "n", ""):
                return False
            return raw
        if self.kind == _KIND_NONE:
            return None if text == "" else text
        if self.kind == _KIND_INT:
            try:
                return int(float(text))
            except (TypeError, ValueError):
                return raw
        if self.kind == _KIND_FLOAT:
            try:
                return float(text)
            except (TypeError, ValueError):
                return raw
        return text

    def validate(self, value: Any) -> list[str]:
        """校验并返回错误信息列表（空列表 = 通过）。"""
        errors: list[str] = []
        if self.choices is not None and value not in self.choices:
            errors.append(f"{self.key} 必须是 {self.choices} 之一，"
                          f"当前 {value!r}")
            return errors
        if isinstance(value, bool):
            return errors
        if self.low is not None and isinstance(value, (int, float)) \
                and value < self.low:
            errors.append(f"{self.key} 小于下限 {self.low}（当前 {value}）")
        if self.high is not None and isinstance(value, (int, float)) \
                and value > self.high:
            errors.append(f"{self.key} 大于上限 {self.high}（当前 {value}）")
        if self.kind == _KIND_INT and isinstance(value, float) \
                and self.choices is None:
            errors.append(f"{self.key} 必须是整数（当前 {value}）")
        return errors

    def to_text(self, value: Any) -> str:
        """配置值 → 控件文本。"""
        if isinstance(value, bool):
            return "是" if value else "否"
        if value is None:
            return ""
        return str(value)


# ---------------------------------------------------------------------------
def _collect_algorithm_fields() -> dict[str, Any]:
    """从算法声明里收集 ``config_key → ParamSpec``。"""
    out: dict[str, Any] = {}
    try:
        from ..algorithms import load_all
        load_all()
    except Exception:  # noqa: BLE001
        return out

    from ..contracts import (CALIBRATORS, CLOSERS, EXTRACTORS, FILTERS,
                             MESHERS, REGISTRARS, VOLUMETERS)
    for reg in (EXTRACTORS, FILTERS, REGISTRARS, MESHERS, CLOSERS,
                VOLUMETERS, CALIBRATORS):
        spec_names = set(getattr(reg, "spec_names", []) or [])
        for name in reg.names():
            if name not in spec_names:
                continue
            try:
                spec = reg.spec(name)
            except Exception:  # noqa: BLE001
                continue
            for p in getattr(spec, "params", ()) or ():
                # 同一配置项可能被多个实现声明（如 camera.index）；
                # 保留第一个带范围/候选值的，信息更全。
                prev = out.get(p.name)
                if prev is None or (getattr(prev, "choices", None) is None
                                    and getattr(p, "choices", None)):
                    out[p.name] = p
    return out


def all_fields(cfg) -> dict[str, FieldSpec]:
    """按当前配置树构造全部 ``FieldSpec``（键为全局点号键）。

    Parameters
    ----------
    cfg : ``Config``
    """
    algo = _collect_algorithm_fields()
    fields: dict[str, FieldSpec] = {}

    for group, params in (getattr(cfg, "raw", {}) or {}).items():
        if str(group).startswith("_") or not isinstance(params, dict):
            continue
        for key, value in params.items():
            key_path = f"{group}.{key}"
            p = algo.get(key_path)
            if p is not None:
                kind = {
                    bool: _KIND_BOOL, int: _KIND_INT,
                    float: _KIND_FLOAT, str: _KIND_STR,
                }.get(getattr(p, "type", type(value)), _kind_of(value))
                fs = FieldSpec(
                    key=key_path,
                    label=field_label(key_path),
                    group=group_label(group),
                    hint=field_hint(key_path) or getattr(p, "help", ""),
                    kind=kind,
                    choices=(list(p.choices)
                             if getattr(p, "choices", None) else None),
                    low=getattr(p, "low", None),
                    high=getattr(p, "high", None),
                    unit=getattr(p, "unit", "") or "",
                    from_algorithm=True,
                )
            else:
                fs = FieldSpec(
                    key=key_path,
                    label=field_label(key_path),
                    group=group_label(group),
                    hint=field_hint(key_path),
                    kind=_kind_of(value),
                )
            fields[key_path] = fs
    return fields


def field_groups(fields: dict[str, FieldSpec]) -> dict[str, list[FieldSpec]]:
    """按分组聚合（保持配置树中的顺序）。"""
    out: dict[str, list[FieldSpec]] = {}
    for fs in fields.values():
        out.setdefault(fs.group, []).append(fs)
    return out


def algorithm_choice_key(key_path: str) -> bool:
    """该配置项是否是"算法实现选择"（值是某注册表里的名字）。

    用于在界面上把"选哪个实现"与"实现的参数"分开展示。
    """
    name = key_path.split(".", 1)[-1]
    return name in ("method", "mesh_method", "close_method", "volume_method",
                    "register_method", "fallback")
