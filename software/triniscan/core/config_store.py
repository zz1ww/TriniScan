"""配置存储：界面与业务共享的**唯一配置来源**。

设计目标
--------
界面上修改的任何参数都必须：**内存立即生效** + **可显式落盘**，
且**测量、标定等所有模块读到的是同一份配置**。

原实现的问题（已修复）
----------------------
- ``MainWindow.cfg`` 与 ``ConfigPanel._cfg`` 是两份对象：界面改了值，
  主窗口更新了，但面板刷新时又用旧的显示回去，用户以为"改了没用"。
- 改动只存在内存，从不写回 ``default.yaml``，重启即丢失。

本模块用 :class:`ConfigStore` 统一管理，所有面板持有**同一个**
store 实例，通过 ``store.cfg`` 取配置、``store.update()`` 改值、
``store.save()`` 落盘。改值后通过回调通知界面刷新"未保存"标记。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from ..common.logging_utils import get_logger
from .config import Config

log = get_logger(__name__)

__all__ = ["ConfigStore"]


class ConfigStore:
    """配置的唯一来源，负责加载 / 修改 / 保存 / 通知。

    Parameters
    ----------
    cfg : 初始配置（通常由 ``Config.load()`` 得到）。
    config_path : 绑定的 YAML 文件路径（默认取 ``cfg`` 记录的原始路径）。

    Attributes
    ----------
    cfg : Config
        当前配置。**所有模块都应通过它读取**，不要各自缓存。
    dirty : bool
        是否有尚未保存的改动。
    """

    def __init__(self, cfg: Config,
                 config_path: Optional[str] = None) -> None:
        self._cfg = cfg
        self._path: Optional[str] = (
            config_path
            or cfg.get("_meta.config_path")
            or None
        )
        self._dirty = False
        self._listeners: list[Callable[[ConfigStore, str, Any], None]] = []

    # ------------------------------------------------------------------
    # 只读访问
    # ------------------------------------------------------------------
    @property
    def cfg(self) -> Config:
        """当前配置（只读使用；修改请走 :meth:`update`）。"""
        return self._cfg

    @property
    def path(self) -> Optional[str]:
        """绑定的 YAML 文件路径。"""
        return self._path

    @property
    def dirty(self) -> bool:
        """是否有未保存的改动。"""
        return self._dirty

    # ------------------------------------------------------------------
    # 通知
    # ------------------------------------------------------------------
    def add_listener(
            self,
            fn: Callable[["ConfigStore", str, Any], None]) -> None:
        """注册改动监听器 ``fn(store, key_path, value)``。"""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(
            self,
            fn: Callable[["ConfigStore", str, Any], None]) -> None:
        """移除监听器。"""
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self, key_path: str, value: Any) -> None:
        for fn in list(self._listeners):
            try:
                fn(self, key_path, value)
            except Exception as exc:  # noqa: BLE001 - 单个监听器出错不阻断
                log.warning("配置监听器异常 (%s=%r): %s",
                            key_path, value, exc)

    # ------------------------------------------------------------------
    # 修改
    # ------------------------------------------------------------------
    def update(self, key_path: str, value: Any) -> None:
        """修改一个参数：内存立即生效并通知监听器，但不写盘。

        真正的落盘由 :meth:`save` 完成（界面"保存配置"按钮或
        退出时确认）。
        """
        self._cfg.set(key_path, value)
        self._dirty = True
        self._notify(key_path, value)

    def update_many(self, values: dict) -> None:
        """批量修改（一次通知，避免多次重绘）。"""
        changed = {}
        for key_path, value in values.items():
            self._cfg.set(key_path, value)
            changed[key_path] = value
        if changed:
            self._dirty = True
            for key_path, value in changed.items():
                self._notify(key_path, value)

    def replace(self, cfg: Config,
                config_path: Optional[str] = None) -> None:
        """整体替换配置（用于"打开配置…"）。"""
        self._cfg = cfg
        if config_path is not None:
            self._path = config_path
        else:
            self._path = cfg.get("_meta.config_path") or self._path
        self._dirty = False
        self._notify("", None)

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> str:
        """把当前配置写回 YAML 文件。

        Parameters
        ----------
        path : 目标路径；None 时写回绑定路径。

        Returns
        -------
        str
            实际写入的文件路径。

        Raises
        ------
        ValueError
            无可用路径时。
        """
        target = path or self._path
        if not target:
            raise ValueError("未指定保存路径，且配置中无原始路径记录")
        target = os.path.abspath(target)
        written = self._cfg.save(target)
        if written != self._path:
            # 首次"另存"后，把新路径设为绑定路径，后续改动直接覆盖它
            self._path = written
        self._dirty = False
        log.info("配置已保存: %s", written)
        return written

    def __repr__(self) -> str:
        return (f"ConfigStore(path={self._path!r}, "
                f"dirty={self._dirty})")
