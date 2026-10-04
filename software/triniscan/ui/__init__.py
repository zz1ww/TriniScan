"""图形界面模块。

基于标准库 **Tkinter**（无需额外依赖，随 Python 附带），
配合 matplotlib 做点云/图像预览。

设计
----
- 界面与业务解耦：所有耗时操作通过 :class:`~triniscan.ui.worker.Worker`
  放到后台线程执行，UI 线程只负责刷新，避免卡死；
- 配置直接读写 :class:`~triniscan.core.config.Config`，
  与命令行共用同一套参数，保证一致性；
- 模块化：主窗口 :class:`~triniscan.ui.app.MainWindow` + 各面板。

入口::

    python -m triniscan.ui
"""
from __future__ import annotations

__all__ = ["MainWindow", "launch"]


def launch() -> int:
    """启动图形界面（延迟导入，避免无显示环境下报错）。"""
    from .app import launch as _launch
    return _launch()
