"""参数规格（ParamSpec）：让"一个参数"只在一个地方声明。

问题
----
重构前，**同一个参数的含义散落在四个地方**：

1. ``default.yaml`` 里写一个值；
2. ``pipeline.py`` 里 ``cfg.get("...", 0.002)`` 再写一个默认值；
3. ``ui/labels.py`` 里手写中文名；
4. ``ui/panels.py`` 的 ``_FIELD_CHOICES`` 里手写下拉选项。

四处不同步 → 出现 ``turntable_band_below_m`` yaml 写 0.0015、代码默认
0.002 的静默漂移。用户删掉 yaml 那一行，行为就悄悄变了。

方案
----
用 :class:`ParamSpec` 把「类型 + 默认值 + 范围 + 中文说明」打包成一个
不可变对象。插件在类属性 ``params`` 里列出自己用到的所有参数，
再由 ``config.schema`` 汇总成全局 schema，用于：

- 启动校验（类型/范围/枚举/未知键）
- 默认值回填（**唯一来源**，代码里不再写第二份默认值）
- UI 控件生成（后续可做，当前 UI 仍手写）
- 日志里的参数快照
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import ConfigError

__all__ = ["ParamSpec", "ParamSet", "coerce"]


@dataclass(frozen=True)
class ParamSpec:
    """单个参数的规格声明。

    Parameters
    ----------
    name : 参数键（点号路径，如 ``reconstruction.cluster_eps_m``）。
           **建议写全路径**，这样注册表可以直接拼出全局 schema，
           不用再猜属于哪个插件。
    type : 期望类型（``int``/``float``/``bool``/``str``）。
    default : 默认值。**这是全项目唯一的默认值来源。**
    low, high : 数值允许范围（闭区间）；``None`` 表示不限制。
    choices : 枚举可选值（``type`` 为 ``str`` 时用）。
    unit : 单位（``"m"``/``"mm"``/``"px"``/``"deg"``），仅用于展示。
    label : 中文短名（UI 用）。
    help : 中文说明（UI 提示 + 文档用）。
    ui_visible : 是否在配置面板显示。
    short : 报告/插件内使用的短名。默认取 ``name`` 的最后一段，
        但**扁平键**（如 ``reconstruction.cluster_eps_m``）最后一段是
        ``cluster_eps_m``，读起来啰嗦且与插件内代码不符，此时应显式指定
        （如 ``"eps_m"``）。**同一插件内 short 必须唯一。**
    """

    name: str
    type: type
    default: Any
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    unit: str = ""
    label: str = ""
    help: str = ""
    ui_visible: bool = True
    short: str = ""

    @property
    def key(self) -> str:
        """插件内使用的键名（``short`` 优先）。"""
        return self.short or self.name.rsplit(".", 1)[-1]

    # ------------------------------------------------------------------
    def coerce(self, value: Any) -> Any:
        """把外部值（YAML/UI 输入）转换成规范类型并校验范围。

        Raises
        ------
        ConfigError
            无法转换、超出范围或不在枚举内。
        """
        return coerce(self, value)

    def validate(self, value: Any) -> list[str]:
        """只检查不抛异常的版本，返回问题描述列表（供批量校验用）。"""
        try:
            self.coerce(value)
        except ConfigError as exc:
            return [str(exc)]
        return []

    def __post_init__(self) -> None:
        # 默认值本身必须合法，否则 schema 一建好就是错的
        if self.choices is not None and self.default not in self.choices:
            raise ValueError(
                f"ParamSpec('{self.name}') 的默认值 {self.default!r} "
                f"不在 choices {self.choices} 内")
        if self.low is not None and self.high is not None \
                and self.low > self.high:
            raise ValueError(
                f"ParamSpec('{self.name}') 的 low={self.low} > high={self.high}")


def coerce(spec: ParamSpec, value: Any) -> Any:
    """按 spec 转换并校验单个值。

    YAML 有几个经典坑，这里统一兜住：

    - ``"60"``（字符串）vs ``60``（数字）；
    - ``"true"``/``"yes"``/``"on"`` vs ``True``；
    - ``null`` 对 ``Optional`` 参数是合法的（由 ``default is None`` 指示）。
    """
    # null：仅当默认值就是 None（可空参数）时允许
    if value is None:
        if spec.default is None:
            return None
        raise ConfigError("不能为空", spec.name)

    try:
        if spec.type is bool:
            out = _to_bool(value)
        elif spec.type is int:
            out = int(round(float(value))) \
                if not isinstance(value, bool) else int(value)
        elif spec.type is float:
            out = float(value)
            if not math.isfinite(out):
                raise ConfigError(f"必须是有限数，收到 {value!r}", spec.name)
        elif spec.type is str:
            out = str(value)
        else:                     # 其它类型（列表、路径等）原样返回
            out = value
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"无法转成 {spec.type.__name__}，收到 {value!r}", spec.name) from exc

    if isinstance(out, (int, float)) and not isinstance(out, bool):
        if spec.low is not None and out < spec.low:
            raise ConfigError(
                f"值 {out}{spec.unit} 小于下限 {spec.low}{spec.unit}",
                spec.name)
        if spec.high is not None and out > spec.high:
            raise ConfigError(
                f"值 {out}{spec.unit} 大于上限 {spec.high}{spec.unit}",
                spec.name)

    if spec.choices is not None:
        if out not in spec.choices:
            raise ConfigError(
                f"取值 {out!r} 不在允许列表 "
                f"{list(spec.choices)} 内", spec.name)

    return out


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "y"):
        return True
    if text in ("0", "false", "no", "off", "n", ""):
        return False
    raise ConfigError(f"无法转成 bool，收到 {value!r}")


@dataclass
class ParamSet:
    """一组 ParamSpec 的集合，提供批量宽松校验。

    与 :class:`ParamSpec` 的区别：ParamSpec 是**声明**，ParamSet 是
    **声明 + 实际取值**的组合，用于"把这个插件的参数全部读出来"。
    """

    specs: tuple[ParamSpec, ...] = ()
    _by_name: dict[str, ParamSpec] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {s.name: s for s in self.specs}

    @classmethod
    def of(cls, specs: Iterable[ParamSpec]) -> "ParamSet":
        return cls(tuple(specs))

    def get_spec(self, name: str) -> ParamSpec | None:
        return self._by_name.get(name)

    def read(self, cfg, prefix: str = "") -> dict[str, Any]:
        """按 spec 从配置对象读出全部参数（已转换校验）。

        缺失的键用 ``spec.default`` 回填；非法值直接抛 :class:`ConfigError`
        ——**宁可启动即失败，也不要跑 6 分钟后崩**。

        Parameters
        ----------
        cfg : 支持 ``.get(key, default)`` 的配置对象
        prefix : 给 ``spec.name`` 加的前缀（当 spec 只写了短名时用）
        """
        out: dict[str, Any] = {}
        for spec in self.specs:
            key = f"{prefix}{spec.name}" if prefix else spec.name
            raw = cfg.get(key, spec.default)
            out[spec.name] = spec.coerce(raw)
        return out

    def problems(self, cfg) -> list[str]:
        """宽松版：返回问题列表而不抛异常（供 `triniscan config-check`）。"""
        issues: list[str] = []
        for spec in self.specs:
            raw = cfg.get(spec.name, spec.default)
            issues.extend(spec.validate(raw))
        return issues
