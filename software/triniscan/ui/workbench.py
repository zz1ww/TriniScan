"""工作台基类：左侧可折叠面板 + 右侧主视图 + 底部工具条。

为什么要有"工作台"
------------------
旧界面只有两页（测量 / 标定），靠顶部单选按钮 + ``place()`` 叠放切换。
页数再多就会失控：每加一个功能都要在主窗口里加一个 ``_page_xxx`` 字段、
一条 ``lift/lower`` 分支，以及对 ``on_hide`` 的特殊处理。

借鉴 Horus 的结构，把"一个完整的操作场景"封装成一个 **Workbench**：

.. code-block:: python

    class MeasureWorkbench(Workbench):
        title = "测量"

        def add_panels(self):
            self.add_panel("camera", CameraPanel, title="相机", expanded=True)
            self.add_panel("extract", ExtractPanel, title="提取")

        def add_pages(self):
            self.add_page("video", self.video)

子类只声明**有什么**，基类负责布局、滚动、挂载与生命周期分发。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

__all__ = ["CollapsiblePanel", "Workbench"]


class CollapsiblePanel(ttk.Frame):
    """可折叠的参数分组（标题栏点击展开/收起）。

    Parameters
    ----------
    master : 父控件
    title : 标题
    expanded : 初始是否展开
    on_toggle : 展开状态变化回调 ``fn(expanded: bool)``

    Notes
    -----
    内容区使用 ``grid_remove()`` 隐藏而非 ``destroy()``：折叠后控件
    仍然存在，保存的值与绑定的事件都不会丢失，展开时无需重建。
    """

    _ARROW_OPEN = "▾"
    _ARROW_CLOSED = "▸"

    def __init__(self, master: tk.Misc, title: str, *,
                 expanded: bool = False,
                 on_toggle: Optional[Callable[[bool], None]] = None,
                 **kw) -> None:
        super().__init__(master, **kw)
        self.title = title
        self._expanded = bool(expanded)
        self._on_toggle = on_toggle

        self.columnconfigure(0, weight=1)

        self._header = ttk.Button(
            self, style="Collapsible.TButton",
            command=self.toggle, takefocus=False)
        self._header.grid(row=0, column=0, sticky="ew")

        self.body = ttk.Frame(self)
        self.body.grid(row=1, column=0, sticky="ew", padx=(8, 4),
                       pady=(0, 4))
        self.body.columnconfigure(1, weight=1)

        self._sync()

    # ------------------------------------------------------------------
    @property
    def expanded(self) -> bool:
        return self._expanded

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._sync()
        if self._on_toggle is not None:
            self._on_toggle(expanded)

    def _sync(self) -> None:
        arrow = self._ARROW_OPEN if self._expanded else self._ARROW_CLOSED
        try:
            self._header.configure(text=f"{arrow}  {self.title}")
            if self._expanded:
                self.body.grid()
            else:
                self.body.grid_remove()
        except tk.TclError:                 # pragma: no cover - 已销毁
            pass


class Workbench(ttk.Frame):
    """一个完整操作场景（左侧面板 + 右侧主视图）。

    子类需要实现
    ------------
    - :meth:`add_panels`：声明左侧面板
    - :meth:`add_pages`：声明右侧视图

    可选覆盖
    --------
    - :attr:`title`：工作台名称（菜单显示）
    - :meth:`setup_engine`：连接硬件后的初始化
    - :meth:`on_show` / :meth:`on_hide`：进入 / 离开本工作台
    """

    #: 工作台名称（菜单与窗口标题使用）
    title: str = ""

    #: 左侧面板列宽（像素）
    panel_width: int = 340

    def __init__(self, master: tk.Misc, store, **kw) -> None:
        super().__init__(master, **kw)
        self.store = store
        self._panels: dict[str, CollapsiblePanel] = {}
        self._factories: list[tuple[str, type, str, bool]] = []
        self._pages: dict[str, ttk.Widget] = {}
        self._current_page: Optional[str] = None
        self._built = False
        # 日志出口：由主窗口注入（``fn(text)``），默认写标准日志
        self._logger: Optional[Callable[[str], None]] = None

        # 右视图容器必须**先于**子类构造视图创建：子类在 __init__ 里
        # 就要用 ``Workbench.view_parent`` 作为 master，否则视图会挂在
        # 工作台根上，place(relwidth=1) 时盖住左侧面板与工具条。
        self._right = ttk.Frame(self)

        self._build()

    # ------------------------------------------------------------------
    # 配置访问
    # ------------------------------------------------------------------
    @property
    def cfg(self):
        """当前配置（始终取 store 中的最新值）。"""
        return self.store.cfg

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    @property
    def view_parent(self) -> ttk.Frame:
        """右视图的父容器。

        子类构造右侧视图时应使用 ``Workbench.view_parent(master)`` 而非
        直接传 ``master``，否则视图会挂在工作台根上并铺满整个工作台。
        更简单的写法是在 ``add_pages`` 里再构造视图（此时可传
        ``self.view_parent``），见 :meth:`add_page` 的说明。
        """
        return self._right

    def set_logger(self, fn: Optional[Callable[[str], None]]) -> None:
        """注入日志出口（主窗口传 ``append_log``，线程安全）。"""
        self._logger = fn

    def log(self, text: str) -> None:
        """向界面日志追加一行（可安全地在 UI 线程调用）。"""
        if self._logger is not None:
            try:
                self._logger(text)
                return
            except Exception:               # noqa: BLE001
                pass
        from ..common.logging_utils import get_logger
        get_logger(__name__).info("%s", text)

    # ------------------------------------------------------------------
    # 子类接口
    # ------------------------------------------------------------------
    def add_panels(self) -> None:            # pragma: no cover - 抽象
        """子类在此声明左侧面板。"""

    def add_pages(self) -> None:             # pragma: no cover - 抽象
        """子类在此声明右侧视图。"""

    def setup_engine(self) -> None:
        """连接硬件 / 加载标定后的初始化钩子。"""

    def on_show(self) -> None:
        """进入工作台（开始预览、恢复定时器等）。"""

    def on_hide(self) -> None:
        """离开工作台（停止预览、释放相机等）。"""

    def refresh(self) -> None:
        """配置变化后刷新各面板。"""

        for panel in self._panels.values():
            fn = getattr(panel, "refresh", None)
            if callable(fn):
                fn()

    # ------------------------------------------------------------------
    # 基类实现
    # ------------------------------------------------------------------
    def _build(self) -> None:
        if self._built:
            return
        self._built = True

        # pack 的分配顺序 = 调用顺序：每个 slave 依次从"剩余空腔"里取空间。
        # 因此**先 pack 底部工具条与左侧面板**（它们有固定尺寸需求），
        # 最后 pack 右视图（fill=both + expand=True 吃掉剩下的全部）。
        # 若反过来先 pack 右视图，工具栏会被挤到无处可放
        # （实测 toolbar 宽高退化为 1×1，按钮整排消失）。
        self._toolbar = ttk.Frame(self)
        self._toolbar.pack(side="bottom", fill="x", padx=6, pady=(4, 4))

        self._left = ttk.Frame(self, width=self.panel_width)
        self._left.pack(side="left", fill="y")
        # 锁定宽度，避免内容撑宽
        self._left.pack_propagate(False)

        self._scroller = _VerticalScroller(self._left)
        self._scroller.pack(fill="both", expand=True)

        # 右视图最后 pack：拿走剩余空间（容器在 __init__ 中已创建）
        self._right.pack(side="right", fill="both", expand=True,
                         padx=(6, 0))

        self.add_panels()
        self.add_pages()

        if self._current_page is None:
            raise RuntimeError(
                f"{type(self).__name__}.add_pages() 必须至少调用一次 "
                f"add_page()")

    # ------------------------------------------------------------------
    @property
    def toolbar(self) -> ttk.Frame:
        """底部工具条容器（横跨整个工作台）。"""
        return self._toolbar

    def add_toolbar(self) -> ttk.Frame:
        """子类调用以声明"我要工具条"；返回的 Frame 已就位。"""
        return self._toolbar

    # ------------------------------------------------------------------
    def add_panel(self, key: str, factory: type, *, title: str = "",
                  expanded: bool = False) -> None:
        """声明一个左侧折叠面板。

        Parameters
        ----------
        key : 唯一标识（用于 :meth:`panel` 取回实例）
        factory : 面板类，签名为 ``factory(parent, workbench)``
        title : 标题（默认取 ``factory.title`` 或 ``key``）
        expanded : 初始是否展开
        """
        title = title or getattr(factory, "title", "") or key
        if key in self._panels or any(k == key for k, *_ in self._factories):
            raise ValueError(f"重复的面板 key: {key!r}")
        self._factories.append((key, factory, title, expanded))

        container = CollapsiblePanel(self._scroller.body, title,
                                     expanded=expanded)
        container.pack(fill="x", padx=4, pady=2)
        panel = factory(container.body, self)
        panel.pack(fill="x")
        self._panels[key] = container
        # 让 container 代理 refresh 到真实面板
        container.refresh = getattr(panel, "refresh", lambda: None)  # type: ignore[attr-defined]
        setattr(container, "content_widget", panel)

    def add_page(self, key: str, widget: ttk.Widget,
                 *, show: bool = True) -> None:
        """声明一个右侧视图（互斥显示）。

        Notes
        -----
        子类通常在 ``__init__`` 里构造视图，此时 ``_right`` 还不存在，
        视图的 master 只能是工作台本身。若直接 ``place(relwidth=1)``
        它会铺满**整个工作台**，把左侧面板和底部工具条一起盖住。
        因此这里先把视图 ``pack_forget()`` 并从原父容器摘下，
        再挂到 ``_right`` 之下，最后在 ``_right`` 内部叠放。
        """
        if key in self._pages:
            raise ValueError(f"重复的页面 key: {key!r}")
        # 把视图挂到右视图容器下（若原 master 不是 _right）
        if widget.master is not self._right:
            raise ValueError(
                f"页面 {key!r} 的父容器必须是 Workbench.view_parent，"
                f"实际是 {widget.master}。请在 add_pages() 中构造视图，"
                f"或把 master 传为 self.view_parent。")
        self._pages[key] = widget
        if show or self._current_page is None:
            self.show_page(key)

    def show_page(self, key: str) -> None:
        """切换到指定视图。"""
        if key not in self._pages:
            raise KeyError(f"未知页面: {key!r}")
        if self._current_page == key:
            return
        if self._current_page is not None:
            self._pages[self._current_page].place_forget()
        self._pages[key].place(relx=0, rely=0, relwidth=1, relheight=1)
        self._current_page = key

    # ------------------------------------------------------------------
    def panel(self, key: str):
        """取回面板实例（如 ``self.panel("camera")``）。"""
        container = self._panels[key]
        return getattr(container, "content_widget")

    def collapse_all(self) -> None:
        """收起全部面板（扫描时腾出空间）。"""
        for panel in self._panels.values():
            panel.set_expanded(False)

    def set_panels_visible(self, visible: bool) -> None:
        """显示 / 隐藏整个左侧面板区。"""
        if visible:
            self._left.pack(side="left", fill="y")
        else:
            self._left.pack_forget()


class _VerticalScroller(ttk.Frame):
    """竖向滚动容器（内容高度超出时出现滚动条）。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(self, highlightthickness=0,
                                 borderwidth=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self._bar = ttk.Scrollbar(self, orient="vertical",
                                  command=self._canvas.yview)
        self._bar.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=self._bar.set)

        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window(
            (0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", self._on_body)
        self._canvas.bind("<Configure>", self._on_canvas)

    def _on_body(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
