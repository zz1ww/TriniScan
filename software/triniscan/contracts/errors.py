"""统一异常体系。

为什么需要
----------
重构前项目里的错误处理是**混着来的**：有的地方 ``raise ValueError``，
有的地方 ``return None``，有的地方 ``except Exception: pass``（静默吞）。
调用方无法区分"参数写错"、"数据不够"、"硬件没接"这三种完全不同的
故障，只能一律兜底，于是 bug 被层层掩盖。

本模块定义一棵**浅的**异常树，只区分调用方真正会做不同处理的三类：

- :class:`ConfigError`  —— 配置写错。**启动时就该失败**，不要跑一半再崩。
- :class:`InputError`   —— 输入数据不足/不合法（如有效视角太少）。
- :class:`HardwareError`—— 相机/转台/串口问题（可以提示用户重试）。

算法内部的数值问题仍然用 ``ValueError``/``RuntimeError``，不必全部改造
——过度细分异常会让代码难读。这里只保证"**跨层传播的故障**"有明确类型。
"""
from __future__ import annotations

__all__ = [
    "TriniScanError",
    "ConfigError",
    "InputError",
    "HardwareError",
    "PluginError",
]


class TriniScanError(Exception):
    """项目所有自定义异常的基类。

    便于调用方写 ``except TriniScanError`` 一次性捕获"我自己的问题"，
    而不会误吞 ``KeyboardInterrupt`` / ``SystemExit`` / 第三方库异常。
    """


class ConfigError(TriniScanError):
    """配置错误：缺键、类型不符、越界、组合非法。

    Attributes
    ----------
    key : str
        出错的配置键（点号路径），便于直接定位到 yaml 那一行。
    """

    def __init__(self, message: str, key: str = "") -> None:
        self.key = key
        super().__init__(f"[配置] {key}: {message}" if key else f"[配置] {message}")


class InputError(TriniScanError):
    """输入数据不足或非法（视角太少、点云为空、网格不闭合等）。"""


class HardwareError(TriniScanError):
    """硬件相关故障（相机打不开、转台无响应、串口超时）。"""


class PluginError(TriniScanError):
    """插件未注册、接口不符等开发期错误。"""

    def __init__(self, domain: str, message: str, available=()) -> None:
        self.domain = domain
        self.available = tuple(available)
        hint = f"；已注册: {sorted(self.available)}" if self.available else ""
        super().__init__(f"[{domain}] {message}{hint}")
