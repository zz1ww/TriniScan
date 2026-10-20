"""新版主窗口：工作台容器（``--ui new`` 启用）。

与旧版 :mod:`triniscan.ui.app` 并存，互不影响：

- 旧版：两页 ``place()`` 叠放，``ConfigPanel`` 平铺全部参数；
- 新版：顶部工作台选择器 + 工作台实例，左面板 / 右视图分离。

当前注册三个工作台（见 :data:`WORKBENCHES`）：**测量 / 标定 / 硬件**，
顶部下拉切换。新增工作台只需实现 :class:`Workbench` 子类并追加到该
元组，主窗口无需改动。
"""
from __future__ import annotations

import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from ..common.logging_utils import get_logger
from ..core.config import Config
from ..core.config_store import ConfigStore
from .panels import LogPanel
from .workbench import Workbench
from .workbench_calib import CalibrationWorkbench
from .workbench_hardware import HardwareWorkbench
from .workbench_measure import MeasureWorkbench

log = get_logger(__name__)

APP_TITLE = "TriniScan — 激光三角测量三维体积测量系统"

__all__ = ["NewMainWindow", "launch"]

#: 工作台注册顺序（P3 追加标定台 / 硬件台）
WORKBENCHES: tuple[type[Workbench], ...] = (
    MeasureWorkbench, CalibrationWorkbench, HardwareWorkbench,
)


class NewMainWindow(ttk.Frame):
    """新版主窗口：工作台容器 + 菜单 + 状态栏 + 日志。"""

    def __init__(self, master: tk.Tk,
                 cfg: Config | ConfigStore) -> None:
        super().__init__(master)
        self.master = master
        self.store = cfg if isinstance(cfg, ConfigStore) \
            else ConfigStore(cfg)
        self.store.add_listener(self._on_store_change)

        self._workbenches: dict[str, Workbench] = {}
        self._current: Optional[str] = None
        self._log_queue: "queue.Queue[str]" = queue.Queue()

        self.pack(fill="both", expand=True)
        self._setup_style()
        self._build_menu()
        self._build_toolbar()
        # 状态栏必须先于 _build_body()：后者会选中首个工作台，
        # 而 _select() 会更新状态栏文字。
        self._build_statusbar()
        self._build_body()
        self._poll()

    # ------------------------------------------------------------------
    @property
    def cfg(self) -> Config:
        return self.store.cfg

    # ------------------------------------------------------------------
    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:                 # pragma: no cover
            pass
        style.configure("Collapsible.TButton", anchor="w",
                        font=("", 9, "bold"), foreground="#2c5aa0",
                        padding=(4, 3))
        style.configure("Status.TLabel", anchor="w")
        # 主动作按钮（「开始测量」）：蓝色底白字，与普通按钮区分，
        # 避免用户在一排灰色按钮里找不到入口。
        style.configure("Accent.TButton", foreground="#ffffff",
                        background="#2c5aa0", padding=(8, 4),
                        font=("", 9, "bold"))
        style.map("Accent.TButton",
                  background=[("active", "#3d6fbd"),
                              ("disabled", "#a8b8cc")],
                  foreground=[("disabled", "#e8eef5")])

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.master)

        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="打开配置…", command=self._open_config)
        m_file.add_command(label="保存配置", command=self._save_config,
                           accelerator="Ctrl+S")
        m_file.add_separator()
        m_file.add_command(label="退出", command=self.master.destroy)
        menubar.add_cascade(label="文件", menu=m_file)
        self.master.bind("<Control-s>", lambda _e: self._save_config())

        m_view = tk.Menu(menubar, tearoff=0)
        m_view.add_command(label="展开全部面板",
                           command=self._expand_all)
        m_view.add_command(label="收起全部面板",
                           command=self._collapse_all)
        menubar.add_cascade(label="视图", menu=m_view)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="关于", command=self._about)
        menubar.add_cascade(label="帮助", menu=m_help)

        self.master.config(menu=menubar)

    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=(6, 0))

        ttk.Label(bar, text="工作台:").pack(side="left")
        self._workbench_var = tk.StringVar()
        names = [w.title or w.__name__ for w in WORKBENCHES]
        self._combo = ttk.Combobox(
            bar, textvariable=self._workbench_var, values=names,
            state="readonly", width=14)
        self._combo.pack(side="left", padx=(4, 12))
        self._combo.bind("<<ComboboxSelected>>", self._on_workbench_change)

        self._fps_label = ttk.Label(bar, text="", foreground="#7f8c8d")
        self._fps_label.pack(side="right", padx=6)

    # ------------------------------------------------------------------
    def _build_body(self) -> None:
        body = ttk.Panedwindow(self, orient="vertical")
        body.pack(fill="both", expand=True, padx=6, pady=6)

        # 上：当前工作台
        self._stage = ttk.Frame(body)
        body.add(self._stage, weight=4)

        # 下：日志（可拖动分隔）
        log_frame = ttk.Frame(body)
        self.log_panel = LogPanel(log_frame)
        self.log_panel.pack(fill="both", expand=True)
        body.add(log_frame, weight=1)

        for cls in WORKBENCHES:
            wb = cls(self._stage, self.store)
            wb.set_logger(self.append_log)
            wb.pack_forget()
            self._workbenches[wb.title or cls.__name__] = wb

        first = next(iter(self._workbenches))
        self._select(first)

    # ------------------------------------------------------------------
    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self)
        # side=bottom 且先于 body 创建，保证它始终贴底
        bar.pack(fill="x", side="bottom")
        self.status = ttk.Label(bar, text="就绪", relief="sunken",
                                style="Status.TLabel")
        self.status.pack(fill="x")

    # ------------------------------------------------------------------
    # 工作台切换
    # ------------------------------------------------------------------
    def _select(self, name: str) -> None:
        if name not in self._workbenches:
            return
        if self._current is not None:
            old = self._workbenches[self._current]
            old.pack_forget()
            try:
                old.on_hide()
            except Exception:               # noqa: BLE001
                log.exception("离开工作台失败: %s", self._current)
        wb = self._workbenches[name]
        wb.pack(fill="both", expand=True)
        self._current = name
        self._workbench_var.set(name)
        self.status.configure(text=name)
        try:
            wb.on_show()
        except Exception:                   # noqa: BLE001
            log.exception("进入工作台失败: %s", name)

    def _on_workbench_change(self, _event: tk.Event) -> None:
        self._select(self._workbench_var.get())

    # ------------------------------------------------------------------
    def _expand_all(self) -> None:
        pass                                # 由各工作台自行处理

    def _collapse_all(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def _on_store_change(self, _store, _key: str, _value) -> None:
        wb = self._workbenches.get(self._current or "")
        if wb is not None:
            try:
                wb.refresh()
            except Exception:               # noqa: BLE001
                log.exception("刷新工作台失败")

    def _open_config(self) -> None:
        path = filedialog.askopenfilename(
            title="打开配置", filetypes=[("YAML", "*.yaml *.yml")])
        if not path:
            return
        try:
            new_cfg = Config.load(path)
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("打开失败", str(exc))
            return
        self.store.replace(new_cfg, path)
        self.status.configure(text=f"已加载 {path}")

    def _save_config(self) -> None:
        try:
            path = self.store.save()
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("保存失败", str(exc))
            return
        self.status.configure(text=f"已保存 {path}")

    def _about(self) -> None:
        messagebox.showinfo(
            "关于",
            f"{APP_TITLE}\n\n工作台界面（新版）\n"
            "配置：ConfigStore 单一来源")

    # ------------------------------------------------------------------
    # 日志轮询
    # ------------------------------------------------------------------
    def append_log(self, text: str) -> None:
        """线程安全地追加一行日志。"""
        self._log_queue.put(text)

    def _poll(self) -> None:
        try:
            while True:
                self.log_panel.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        except Exception:                   # noqa: BLE001
            log.exception("日志刷新失败")
        try:
            wb = self._workbenches.get(self._current or "")
            fps = getattr(wb, "fps", 0.0) if wb is not None else 0.0
            self._fps_label.configure(
                text=f"{fps:.1f} fps" if fps > 0 else "")
        except Exception:                   # noqa: BLE001
            pass
        try:
            self.after(500, self._poll)
        except Exception:                   # noqa: BLE001 - 窗口已销毁
            pass


# ---------------------------------------------------------------------------
def launch(config_path: Optional[str] = None,
           log_level: str = "INFO") -> int:
    """启动新版界面。"""
    from ..common.logging_utils import set_level

    set_level(log_level)
    cfg = Config.load(config_path) if config_path else Config.load()

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1360x860")
    root.minsize(1024, 640)

    window = NewMainWindow(root, cfg)

    def _on_close() -> None:
        for wb in window._workbenches.values():
            try:
                wb.on_hide()
            except Exception:               # noqa: BLE001
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
    return 0
