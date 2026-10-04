"""主窗口：把所有面板、菜单、后台任务串起来。

界面布局::

    ┌──────────────────────────────────────────────────────────┐
    │ 菜单栏: 文件 | 工具 | 帮助                                 │
    ├───────────────┬──────────────────────────────────────────┤
    │ 参数配置       │ 预览（激光图像 / 点云）                    │
    │ (可编辑)       ├──────────────────────────────────────────┤
    │               │ 测量结果                                  │
    ├───────────────┴──────────────────────────────────────────┤
    │ 控制条: [检查标定][预览单帧][开始测量][停止] 进度条         │
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
from ..common.logging_utils import get_logger, set_level
from .panels import ConfigPanel, LogPanel, ResultPanel
from .preview import PreviewPanel
from .worker import TaskMessage, Worker

log = get_logger(__name__)

APP_TITLE = "TriniScan — 激光三角测量三维体积测量系统"


class MainWindow(ttk.Frame):
    """应用主窗口。"""

    def __init__(self, master: tk.Tk, cfg: Config) -> None:
        super().__init__(master)
        self.master = master
        self.cfg = cfg
        self._queue: "queue.Queue[TaskMessage]" = queue.Queue()
        self._worker: Optional[Worker] = None
        self._last_result = None

        self.pack(fill="both", expand=True)
        self._setup_style()
        self._build_menu()
        self._build_layout()
        self._build_statusbar()
        self._poll()

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
        m_file.add_command(label="另存配置…", command=self._save_config)
        m_file.add_separator()
        m_file.add_command(label="退出", command=self.master.destroy)
        menubar.add_cascade(label="文件", menu=m_file)

        m_tools = tk.Menu(menubar, tearoff=0)
        m_tools.add_command(label="检查标定", command=self._on_check)
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
        paned = ttk.Panedwindow(self, orient="horizontal")
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
        ctrl = ttk.Frame(self)
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

        self.progress = ttk.Progressbar(ctrl, mode="determinate",
                                        length=260)
        self.progress.pack(side="right", padx=6)
        self.progress_label = ttk.Label(ctrl, text="就绪")
        self.progress_label.pack(side="right")

        # 日志
        self.log_panel = LogPanel(self)
        self.log_panel.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def _build_statusbar(self) -> None:
        self.status = ttk.Label(self, text="就绪", relief="sunken",
                                anchor="w")
        self.status.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # 配置回调
    # ------------------------------------------------------------------
    def _on_config_change(self, key_path: str, value) -> None:
        self.cfg = self.cfg.with_overrides(_nest(key_path, value))
        self._append_log(f"[配置] {key_path} = {value!r}")

    # ------------------------------------------------------------------
    # 菜单/按钮动作
    # ------------------------------------------------------------------
    def _open_config(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("YAML 配置", "*.yaml *.yml"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            self.cfg = Config.load(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", str(exc))
            return
        self.config_panel.refresh(self.cfg)
        self._append_log(f"已加载配置: {path}")

    def _save_config(self) -> None:
        import yaml
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML 配置", "*.yaml"), ("所有文件", "*.*")])
        if not path:
            return
        data = {k: v for k, v in self.cfg.raw.items()
                if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        self._append_log(f"已保存配置: {path}")

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
            from ..core.pipeline import ScanPipeline

            pipe = ScanPipeline(cfg)
            report("载入标定…", 0.05)
            pipe.load_calibrations()
            report("采集（转-停-拍）…", 0.2)
            views = pipe.acquire()
            report("中心线提取 + 三角测量…", 0.5)
            clouds = pipe.reconstruct_views(views)
            if not clouds:
                raise RuntimeError("没有有效视角，测量失败")
            report("多视角配准…", 0.7)
            pcd, _ = pipe.build_pointcloud(clouds)
            report("重建 + 体积…", 0.9)
            vol, _closed, _mesh = pipe.measure_volume(pcd)
            points = None
            try:
                import numpy as np
                points = np.asarray(pcd.points)
            except Exception:  # noqa: BLE001
                points = None
            return vol, points

        def on_done(payload):
            vol, points = payload
            result = _make_result(vol)
            self._last_result = result
            self.result_panel.update_result(result)
            if points is not None:
                self.preview_panel.set_points(points)
            self._append_log(
                f"测量完成: {result.volume_cm3:.2f} cm³ "
                f"({result.num_points} 点)")

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
        self.log_panel.append(text)


# ---------------------------------------------------------------------------
def _nest(key_path: str, value) -> dict:
    """把 ``"a.b.c"`` 展开为 ``{"a": {"b": {"c": value}}}``。"""
    parts = key_path.split(".")
    node: dict = {}
    cur = node
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node


def _make_result(vol):
    """由 ``VolumeResult`` 组装一个可在界面上展示的轻量结果对象。"""
    from ..core.pipeline import ScanResult
    return ScanResult(
        volume_cm3=vol.volume_cm3,
        volume_m3=vol.volume_m3,
        elapsed_s=0.0,
        num_views=0,
        num_points=0,
        mesh_watertight=bool(vol.watertight),
        cross_check_cm3=(vol.cross_check_m3 * 1e6
                         if vol.cross_check_m3 else None),
        stages_s={},
        mesh_path=None,
        points_path=None,
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
    app._append_log(f"项目根: {cfg.root}")
    app._append_log("就绪。可在左侧编辑参数，然后点『检查标定』"
                    "或『开始测量』。")
    root.mainloop()
    return 0
