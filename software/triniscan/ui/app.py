"""主窗口：把所有面板、菜单、后台任务串起来。

配置管理
--------
所有面板（参数配置、标定）与业务模块（测量、标定）共用**同一个**
:class:`~triniscan.core.config_store.ConfigStore`：

- 界面改参数 → ``store.update()`` → 内存立即生效，并通知各面板刷新；
- 点"保存配置" → ``store.save()`` → 写回 ``default.yaml``；
- 测量 / 标定读取「``store.cfg``」，因此**总是拿到最新值**。

界面布局::

    ┌──────────────────────────────────────────────────────────┐
    │ 菜单栏: 文件 | 工具 | 帮助                                 │
    ├───────────────┬──────────────────────────────────────────┤
    │ 参数配置       │ 预览（激光图像 / 点云）                    │
    │ (可编辑)       ├──────────────────────────────────────────┤
    │               │ 测量结果                                  │
    ├───────────────┴──────────────────────────────────────────┤
    │ 控制条: [检查标定][预览单帧][开始测量][停止][保存配置]      │
    ├──────────────────────────────────────────────────────────┤
    │ 运行日志                                                  │
    └──────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import os
import queue
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from ..core.config import Config, find_project_root
from ..core.config_store import ConfigStore
from ..common.logging_utils import get_logger, set_level
from .panels import ConfigPanel, LogPanel, ResultPanel
from .preview import PreviewPanel
from .worker import TaskMessage, Worker

log = get_logger(__name__)

APP_TITLE = "TriniScan — 激光三角测量三维体积测量系统"


class MainWindow(ttk.Frame):
    """应用主窗口。"""

    def __init__(self, master: tk.Tk,
                 cfg: Config | ConfigStore) -> None:
        super().__init__(master)
        self.master = master
        # 兼容直接传 Config 的调用方（如测试/旧脚本）
        self.store = cfg if isinstance(cfg, ConfigStore) \
            else ConfigStore(cfg)
        self.store.add_listener(self._on_store_change)

        self._queue: "queue.Queue[TaskMessage]" = queue.Queue()
        self._worker: Optional[Worker] = None
        self._last_result = None

        self.pack(fill="both", expand=True)
        self._setup_style()
        self._build_menu()
        self._build_layout()
        self._build_statusbar()
        self._switch_page()
        self._update_dirty_hint()
        self._poll()

    # ------------------------------------------------------------------
    # 配置访问
    # ------------------------------------------------------------------
    @property
    def cfg(self) -> Config:
        """当前配置（始终取 store 里的最新值）。"""
        return self.store.cfg

    # ------------------------------------------------------------------
    # 外观
    # ------------------------------------------------------------------
    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Group.TLabel", font=("", 9, "bold"),
                        foreground="#2c5aa0")
        style.configure("Key.TLabel", font=("", 9))
        style.configure("Value.TLabel", font=("", 11, "bold"),
                        foreground="#1a7f37")
        style.configure("Title.TLabel", font=("", 13, "bold"))

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.master)

        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="打开配置…", command=self._open_config)
        m_file.add_command(label="另存配置…", command=self._save_config_as)
        m_file.add_separator()
        m_file.add_command(label="退出", command=self.master.destroy)
        menubar.add_cascade(label="文件", menu=m_file)

        m_tools = tk.Menu(menubar, tearoff=0)
        m_tools.add_command(label="检查标定", command=self._on_check)
        m_tools.add_command(label="切换到标定界面",
                            command=self._goto_calibration)
        m_tools.add_command(label="预览单帧图像…",
                            command=self._on_preview_file)
        m_tools.add_command(label="开始测量", command=self._on_run)
        m_tools.add_separator()
        m_tools.add_command(label="打开数据目录",
                            command=self._open_output_dir)
        menubar.add_cascade(label="工具", menu=m_tools)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="关于", command=self._about)
        menubar.add_cascade(label="帮助", menu=m_help)

        self.master.config(menu=menubar)

    def _build_layout(self) -> None:
        """搭建可切换的两页界面：测量页与标定页。"""
        # 顶部：界面切换条
        switch = ttk.Frame(self)
        switch.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(switch, text="界面:", style="Key.TLabel").pack(
            side="left")
        self._page = tk.StringVar(value="measure")
        ttk.Radiobutton(switch, text="测量", value="measure",
                        variable=self._page,
                        command=self._switch_page).pack(side="left", padx=4)
        ttk.Radiobutton(switch, text="标定", value="calibration",
                        variable=self._page,
                        command=self._switch_page).pack(side="left")

        # 页面容器
        self._container = ttk.Frame(self)
        self._container.pack(fill="both", expand=True)

        self._page_measure = ttk.Frame(self._container)
        self._page_calib = ttk.Frame(self._container)
        for page in (self._page_measure, self._page_calib):
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._build_measure_page(self._page_measure)
        self._build_calib_page(self._page_calib)

    def _build_measure_page(self, parent: ttk.Frame) -> None:
        """测量页：参数配置 + 预览 + 结果 + 日志（原布局）。"""
        paned = ttk.Panedwindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        # 左：配置
        left = ttk.Frame(paned)
        self.config_panel = ConfigPanel(left, self.cfg,
                                        on_change=self._on_config_change)
        self.config_panel.pack(fill="both", expand=True)
        paned.add(left, weight=1)

        # 右：预览 + 结果
        right = ttk.Frame(paned)
        self.preview_panel = PreviewPanel(right)
        self.preview_panel.pack(fill="both", expand=True)
        self.result_panel = ResultPanel(right)
        self.result_panel.pack(fill="x", pady=(6, 0))
        paned.add(right, weight=2)

        # 控制条
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=6, pady=6)
        self.btn_check = ttk.Button(ctrl, text="检查标定",
                                    command=self._on_check)
        self.btn_check.pack(side="left")
        self.btn_preview = ttk.Button(ctrl, text="预览单帧",
                                      command=self._on_preview)
        self.btn_preview.pack(side="left", padx=4)
        self.btn_run = ttk.Button(ctrl, text="▶ 开始测量",
                                  command=self._on_run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(ctrl, text="■ 停止",
                                   command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left")

        self.btn_save_cfg = ttk.Button(ctrl, text="💾 保存配置",
                                       command=self._on_save_config)
        self.btn_save_cfg.pack(side="left", padx=(12, 4))

        self.progress = ttk.Progressbar(ctrl, mode="determinate",
                                        length=260)
        self.progress.pack(side="right", padx=6)
        self.progress_label = ttk.Label(ctrl, text="就绪")
        self.progress_label.pack(side="right")

        # 日志
        self.log_panel = LogPanel(parent)
        self.log_panel.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def _build_calib_page(self, parent: ttk.Frame) -> None:
        """标定页：实时拍摄 + 质量检测 + 自动标定。"""
        from .calibration import CalibrationPanel

        self.calib_panel = CalibrationPanel(parent, self.store)
        self.calib_panel.pack(fill="both", expand=True)

    def _switch_page(self) -> None:
        """切换测量页 / 标定页。"""
        if self._page.get() == "calibration":
            self._page_measure.lower()
            self._page_calib.lift()
            self.status.configure(text="标定界面")
        else:
            # 离开标定页时停止实时取景，释放相机
            try:
                self.calib_panel.on_hide()
            except Exception:  # noqa: BLE001
                pass
            self._page_calib.lower()
            self._page_measure.lift()
            if self.store.dirty:
                self.status.configure(text="有未保存的配置改动")
            else:
                self.status.configure(text="就绪")

    def _build_statusbar(self) -> None:
        self.status = ttk.Label(self, text="就绪", relief="sunken",
                                anchor="w")
        self.status.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # 配置回调（store 为唯一来源）
    # ------------------------------------------------------------------
    def _on_config_change(self, key_path: str, value) -> None:
        """参数配置面板改了一个值 → 写入 store（内存立即生效）。"""
        self.store.update(key_path, value)

    def _on_store_change(self, store: ConfigStore,
                         key_path: str, value) -> None:
        """store 变化监听：同步面板显示 + 刷新"未保存"标记 + 记日志。"""
        if key_path:
            self._append_log(f"[配置] {key_path} = {value!r}（未保存）")
            # 同步参数面板相应字段（用户正在编辑的字段值本就一致，
            # 这里主要覆盖"标定页改了参数"等外部改动的情况）
            panel = getattr(self, "config_panel", None)
            if panel is not None:
                try:
                    panel.set_value(key_path, value)
                except Exception as exc:  # noqa: BLE001
                    log.warning("同步参数面板失败 (%s): %s", key_path, exc)
        self._update_dirty_hint()

    def _update_dirty_hint(self) -> None:
        """在状态栏/按钮上提示"有未保存改动"。"""
        btn = getattr(self, "btn_save_cfg", None)
        if btn is not None:
            btn.configure(
                text="💾 保存配置*" if self.store.dirty else "💾 保存配置")
        status = getattr(self, "status", None)
        if status is not None and self.store.dirty \
                and self._page.get() == "measure":
            status.configure(text="有未保存的配置改动")

    def _on_save_config(self) -> None:
        """按钮/菜单：把当前配置写回 YAML 文件。"""
        self.save_config_to_file()

    def save_config_to_file(self, path: Optional[str] = None,
                            quiet: bool = False) -> bool:
        """把当前配置写入 YAML 文件（默认写回原文件）。

        Returns
        -------
        bool
            是否保存成功。
        """
        try:
            written = self.store.save(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存配置失败", str(exc))
            return False
        self._update_dirty_hint()
        if not quiet:
            messagebox.showinfo(
                "配置已保存",
                f"已写入：\n{written}\n\n"
                "测量与标定都会立即使用该配置；下次启动也自动加载。")
        return True

    # ------------------------------------------------------------------
    # 菜单/按钮动作
    # ------------------------------------------------------------------
    def _open_config(self) -> None:
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            filetypes=[("YAML 配置", "*.yaml *.yml"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            cfg = Config.load(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", str(exc))
            return
        self.store.replace(cfg, config_path=os.path.abspath(path))
        self._refresh_all_panels()
        self._append_log(f"已加载配置: {path}")

    def _save_config_as(self) -> None:
        """菜单：另存为新文件，并把绑定路径切到新文件。"""
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML 配置", "*.yaml"), ("所有文件", "*.*")])
        if not path:
            return
        self.save_config_to_file(path)

    def _refresh_all_panels(self) -> None:
        """让所有面板按当前 store.cfg 重新显示。"""
        try:
            self.config_panel.refresh(self.store.cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("刷新参数面板失败: %s", exc)
        try:
            self.calib_panel.on_config_changed()
        except Exception as exc:  # noqa: BLE001
            log.warning("刷新标定面板失败: %s", exc)

    def _confirm_discard(self) -> bool:
        """有未保存改动时询问；返回 True 表示可以继续并已处理。"""
        if not self.store.dirty:
            return True
        ans = messagebox.askyesnocancel(
            "未保存的配置",
            "当前有未保存的配置改动。\n\n"
            "「是」= 先保存   「否」= 放弃改动   「取消」= 什么都不做")
        if ans is None:
            return False
        if ans:
            return self.save_config_to_file(quiet=True)
        return True

    def _goto_calibration(self) -> None:
        """菜单动作：切到标定界面。"""
        self._page.set("calibration")
        self._switch_page()

    def _open_output_dir(self) -> None:
        path = self.cfg.resolve("output.result_dir", self.cfg.root)
        path = os.path.dirname(path) or self.cfg.root
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"无法打开目录: {exc}")

    def _about(self) -> None:
        messagebox.showinfo(
            "关于 TriniScan",
            "TriniScan 0.2.0\n"
            "激光三角测量三维体积测量系统\n\n"
            "第十五届全国大学生光电设计竞赛 · 赛题二\n"
            "基于光电技术的复杂目标物体体积高精度测量",
        )

    # ------------------------------------------------------------------
    # 后台任务
    # ------------------------------------------------------------------
    def _busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for btn in (self.btn_check, self.btn_preview, self.btn_run):
            btn.configure(state=state)
        self.btn_stop.configure(state="normal" if busy else "disabled")

    def _start_task(self, func, title: str) -> None:
        if self._worker is not None and self._worker.running:
            messagebox.showinfo("提示", "已有任务在运行")
            return
        self._busy(True)
        self.progress.configure(value=0)
        self._append_log(f"—— {title} ——")
        self._worker = Worker(func, self._queue)
        self._worker.start()

    def _on_check(self) -> None:
        cfg = self.cfg

        def job(report):
            keys = ("calibration.camera_file",
                    "calibration.laser_file",
                    "calibration.axis_file")
            lines, ok = [], True
            for k in keys:
                path = cfg.resolve(k)
                exists = os.path.exists(path)
                ok &= exists
                lines.append(f"[{'OK' if exists else '缺失'}] {path}")
                report(f"检查 {os.path.basename(path)}")
            return ok, lines

        self._start_task(job, "检查标定文件")

    def _on_preview(self) -> None:
        cfg = self.cfg
        port = str(cfg.get("turntable.port", ""))

        def job(report):
            from ..camera import Camera
            report("打开相机…", 0.3)
            with Camera(cfg.raw["camera"]) as cam:
                report("抓取图像…", 0.7)
                image = cam.grab()
            # 顺手把该帧中心线画出来
            try:
                from ..extraction import extract_centerline
                pixels = extract_centerline(image,
                                            cfg.raw.get("extraction", {}))
                return image, pixels
            except Exception:  # noqa: BLE001
                return image, None

        def on_done(payload):
            image, pixels = payload
            self.preview_panel.set_image(image)
            n = 0 if pixels is None else len(pixels)
            self._append_log(f"预览: 图像 {image.shape}, 中心线 {n} 点")

        self._start_task_then(job, "预览单帧", on_done)

    def _on_preview_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("图像", "*.png *.jpg *.jpeg *.bmp"),
                       ("所有文件", "*.*")])
        if not path:
            return
        if self.preview_panel.load_image_file(path):
            self._append_log(f"已加载图像: {path}")
            # 计算中心线
            try:
                import cv2
                from ..extraction import extract_centerline
                img = cv2.imread(path)
                pixels = extract_centerline(
                    img, self.cfg.raw.get("extraction", {}))
                self._append_log(f"中心线: {len(pixels)} 点")
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"中心线提取失败: {exc}")
        else:
            messagebox.showerror("失败", "无法读取图像")

    def _on_run(self) -> None:
        cfg = self.cfg

        def job(report):
            import time

            from ..core.pipeline import ScanPipeline

            pipe = ScanPipeline(cfg)
            t0 = time.time()
            report("载入标定…", 0.05)
            pipe.load_calibrations()
            report("采集（转-停-拍）…", 0.2)
            views = pipe.acquire()
            report("中心线提取 + 三角测量…", 0.5)
            clouds = pipe.reconstruct_views(views)
            if not clouds:
                raise RuntimeError("没有有效视角，测量失败")
            report("多视角配准…", 0.7)
            pcd, pc_path = pipe.build_pointcloud(clouds)
            report("重建 + 体积…", 0.9)
            vol, _closed, mesh_path = pipe.measure_volume(pcd)
            points = None
            try:
                import numpy as np
                points = np.asarray(pcd.points)
            except Exception:  # noqa: BLE001
                points = None
            info = {
                "elapsed_s": time.time() - t0,
                "num_views": len(clouds),
                "num_points": int(sum(len(c.points) for c in clouds)),
                "mesh_path": mesh_path,
                "points_path": pc_path,
            }
            return vol, points, info

        def on_done(payload):
            vol, points, info = payload
            result = _make_result(vol, info)
            self._last_result = result
            self.result_panel.update_result(result)
            if points is not None:
                self.preview_panel.set_points(points)
            self._append_log(
                f"测量完成: {result.volume_cm3:.2f} cm³ "
                f"({result.num_points} 点)")
            if result.mesh_path:
                self._append_log(f"网格: {result.mesh_path}")
            if result.points_path:
                self._append_log(f"点云: {result.points_path}")

        self._start_task_then(job, "执行测量", on_done)

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._append_log("已请求停止（协作式，当前阶段结束后退出）")

    def _start_task_then(self, func, title, on_done) -> None:
        self._pending_done = on_done
        self._start_task(func, title)

    # ------------------------------------------------------------------
    # 消息轮询
    # ------------------------------------------------------------------
    def _poll(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg.kind == "progress":
                    if msg.payload is not None:
                        self.progress.configure(value=msg.payload * 100)
                    self.progress_label.configure(text=msg.text or "处理中")
                    self.status.configure(text=msg.text or "处理中")
                elif msg.kind == "log":
                    self._append_log(msg.text)
                elif msg.kind == "done":
                    self._handle_done(msg.payload)
                elif msg.kind == "error":
                    self._handle_error(msg)
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _handle_done(self, payload) -> None:
        self._busy(False)
        self.progress.configure(value=100)
        self.progress_label.configure(text="完成")
        self.status.configure(text="就绪")

        on_done = getattr(self, "_pending_done", None)
        self._pending_done = None

        if on_done is not None:
            try:
                on_done(payload)
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"处理结果失败: {exc}")
            return

        # 检查标定的返回值
        if isinstance(payload, tuple) and len(payload) == 2 \
                and isinstance(payload[1], list):
            ok, lines = payload
            for line in lines:
                self._append_log(line)
            self._append_log("检查通过" if ok else "存在缺失")

    def _handle_error(self, msg: TaskMessage) -> None:
        self._busy(False)
        self.progress.configure(value=0)
        self.progress_label.configure(text="失败")
        self.status.configure(text="失败")
        self._append_log(msg.text)
        messagebox.showerror("任务失败", str(msg.payload))

    # ------------------------------------------------------------------
    def _append_log(self, text: str) -> None:
        panel = getattr(self, "log_panel", None)
        if panel is not None:
            panel.append(text)


# ---------------------------------------------------------------------------
def _make_result(vol, info=None):
    """由 ``VolumeResult``（+ 运行信息）组装界面展示用的轻量结果对象。

    Parameters
    ----------
    vol : VolumeResult
        体积计算结果。
    info : dict, optional
        运行信息，键可为 ``elapsed_s`` / ``num_views`` / ``num_points`` /
        ``mesh_path`` / ``points_path``。缺省时以 0/None 填充。
    """
    from ..core.pipeline import ScanResult
    info = info or {}
    return ScanResult(
        volume_cm3=vol.volume_cm3,
        volume_m3=vol.volume_m3,
        elapsed_s=float(info.get("elapsed_s", 0.0)),
        num_views=int(info.get("num_views", 0)),
        num_points=int(info.get("num_points", 0)),
        mesh_watertight=bool(vol.watertight),
        cross_check_cm3=(vol.cross_check_m3 * 1e6
                         if vol.cross_check_m3 else None),
        stages_s={},
        mesh_path=info.get("mesh_path"),
        points_path=info.get("points_path"),
    )


# ---------------------------------------------------------------------------
def launch(config_path: Optional[str] = None,
           log_level: str = "INFO") -> int:
    """启动图形界面。"""
    set_level(log_level)
    try:
        cfg = Config.load(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"加载配置失败: {exc}")
        cfg = Config({}, root=find_project_root())

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"无法初始化图形界面（可能无显示环境）: {exc}")
        return 1

    root.title(APP_TITLE)
    root.geometry("1180x760")
    root.minsize(900, 600)

    app = MainWindow(root, cfg)
    # 关闭窗口时，若有未保存配置则询问
    root.protocol("WM_DELETE_WINDOW",
                  lambda: _on_close(root, app))
    app._append_log(f"项目根: {cfg.root}")
    app._append_log(f"配置文件: {app.store.path}")
    app._append_log("就绪。可在左侧编辑参数，改完点『💾 保存配置』"
                    "写回文件；测量/标定始终使用当前配置。")
    root.mainloop()
    return 0


def _on_close(root: tk.Tk, app: "MainWindow") -> None:
    """窗口关闭钩子：提示保存未落盘的配置改动。"""
    if app.store.dirty:
        ans = messagebox.askyesnocancel(
            "退出前保存？",
            "有未保存的配置改动。\n\n"
            "「是」= 保存并退出   「否」= 直接退出   「取消」= 返回")
        if ans is None:
            return
        if ans and not app.save_config_to_file(quiet=True):
            return
    root.destroy()
