"""标定界面：实时拍摄 + 质量检测 + 自动标定。

设计
----
- 本面板只负责"显示与交互"，业务逻辑在
  :mod:`triniscan.ui.capture_session`（可独立测试）；
- 相机采集与标定都在后台线程执行，界面通过队列刷新，
  避免 Tk 卡死（复用 :class:`~triniscan.ui.worker.Worker` 的思路）；
- 三种标定模式：相机内参 / 激光平面 / 转台转轴。

界面布局::

    ┌──────────────────────────────────────────────────────────┐
    │ 模式: (●)相机内参 ( )激光平面 ( )转台转轴                  │
    ├───────────────────────────────┬──────────────────────────┤
    │ 实时画面 / 预览                │ 拍摄进度                 │
    │                               │  ✓ 质量合格 ...          │
    │                               │  ✗ 图像模糊 ...          │
    ├───────────────────────────────┴──────────────────────────┤
    │ [开始拍摄] [拍摄] [撤销] [清空] [开始标定]  状态          │
    └──────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

from ..common.logging_utils import get_logger
from .capture_session import CalibMode, CaptureSession
from .preview import PreviewPanel

log = get_logger(__name__)

__all__ = ["CalibrationPanel"]

#: 实时预览的最大边长（像素）——降低刷新开销，保证界面流畅
_PREVIEW_MAX_SIDE = 720
#: 实时预览目标帧率下的轮询间隔（毫秒）
_LIVE_INTERVAL_MS = 60


class CalibrationPanel(ttk.Frame):
    """标定面板：实时拍摄 + 质量检测 + 自动标定。"""

    def __init__(self, master: tk.Misc, cfg) -> None:
        super().__init__(master)
        self.cfg = cfg
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._session: Optional[CaptureSession] = None
        self._camera = None
        self._latest_frame: Optional[np.ndarray] = None
        self._capturing = False          # 实时取景是否开启
        self._live_thread: Optional[threading.Thread] = None
        self._live_stop = threading.Event()
        self._busy = False               # 标定进行中

        self._mode = tk.StringVar(value=CalibMode.CAMERA.value)
        # 转轴现场标定的参数（可在界面配置）
        self._sphere_radius_mm = tk.StringVar(value="20.0")
        self._axis_method = tk.StringVar(value="peak")
        self._axis_views = tk.StringVar(value="12")
        self._axis_sweep_active = False
        self._axis_stop = False
        self._build_ui()
        self._poll()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # 顶部：模式选择
        top = ttk.LabelFrame(self, text="标定模式")
        top.pack(fill="x", padx=6, pady=(6, 0))
        for m in CalibMode:
            ttk.Radiobutton(
                top, text=m.label, value=m.value,
                variable=self._mode, command=self._on_mode_change,
            ).pack(side="left", padx=8, pady=4)

        # 转轴参数（仅转轴模式显示）
        self.axis_opts = ttk.LabelFrame(self, text="转轴标定设置")

        r = ttk.Frame(self.axis_opts)
        r.pack(fill="x", padx=8, pady=(4, 2))
        ttk.Label(r, text="标记球半径 (mm):").pack(side="left")
        self.ent_radius = ttk.Entry(r, textvariable=self._sphere_radius_mm,
                                    width=8)
        self.ent_radius.pack(side="left", padx=4)
        ttk.Label(r, text="  采样角度数:").pack(side="left")
        self.ent_views = ttk.Entry(r, textvariable=self._axis_views,
                                   width=6)
        self.ent_views.pack(side="left", padx=4)

        m = ttk.Frame(self.axis_opts)
        m.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Label(m, text="取点方法:").pack(side="left")
        ttk.Radiobutton(m, text="最亮极点法（关环境光）", value="peak",
                        variable=self._axis_method).pack(side="left",
                                                         padx=4)
        ttk.Radiobutton(m, text="球心法（亮背景）", value="sphere",
                        variable=self._axis_method).pack(side="left",
                                                         padx=4)

        self.lbl_axis_hint = ttk.Label(
            self.axis_opts, foreground="#8a5a00", wraplength=760,
            justify="left",
            text=("操作：① 先完成「相机内参」与「激光平面」标定；"
                  "② 把涂黑的小球固定在转台偏心位置，确保全程在视野内；"
                  "③ 点击「开始转轴标定」，程序将自动转动转台、逐角度"
                  "拍摄并拟合转轴。"))
        self.lbl_axis_hint.pack(fill="x", padx=8, pady=(0, 6))

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=6, pady=6)

        # 左：实时画面
        self.preview = PreviewPanel(mid)
        self.preview.pack(side="left", fill="both", expand=True)

        # 右：进度与判定
        right = ttk.LabelFrame(mid, text="拍摄进度")
        right.pack(side="left", fill="both", expand=False, padx=(6, 0))

        self.lbl_progress = ttk.Label(right, text="0 / 0 张",
                                      style="Value.TLabel")
        self.lbl_progress.pack(anchor="w", padx=8, pady=(6, 2))
        self.bar_progress = ttk.Progressbar(right, mode="determinate",
                                            length=220)
        self.bar_progress.pack(fill="x", padx=8, pady=(0, 6))

        ttk.Label(right, text="最近判定").pack(anchor="w", padx=8)
        self.txt_verdict = tk.Text(right, width=34, height=10, wrap="word",
                                   state="disabled", font=("Consolas", 9))
        self.txt_verdict.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        # 底部：操作按钮
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=6, pady=(0, 6))
        self.btn_live = ttk.Button(ctrl, text="开始拍摄",
                                   command=self._on_toggle_live)
        self.btn_live.pack(side="left")
        self.btn_shoot = ttk.Button(ctrl, text="拍摄", state="disabled",
                                    command=self._on_shoot)
        self.btn_shoot.pack(side="left", padx=4)
        self.btn_undo = ttk.Button(ctrl, text="撤销", state="disabled",
                                   command=self._on_undo)
        self.btn_undo.pack(side="left")
        self.btn_clear = ttk.Button(ctrl, text="清空", state="disabled",
                                    command=self._on_clear)
        self.btn_clear.pack(side="left", padx=4)
        self.btn_calib = ttk.Button(ctrl, text="开始标定", state="disabled",
                                    command=self._on_calibrate)
        self.btn_calib.pack(side="left")

        self.lbl_status = ttk.Label(ctrl, text="就绪")
        self.lbl_status.pack(side="right")

    # ------------------------------------------------------------------
    # 会话与模式
    # ------------------------------------------------------------------
    def _current_mode(self) -> CalibMode:
        return CalibMode(self._mode.get())

    def _ensure_session(self) -> CaptureSession:
        mode = self._current_mode()
        if self._session is None or self._session.mode is not mode:
            self._session = CaptureSession(
                mode, self.cfg,
                save_images=(mode is CalibMode.CAMERA),
                on_log=None,
            )
        return self._session

    def _on_mode_change(self) -> None:
        self._stop_live()
        self._session = None
        mode = self._current_mode()
        self._ensure_session()
        self._refresh_ui()
        self._append_verdict(f"已切换到：{mode.label}")

        # 转轴模式的专属设置面板与按钮
        if mode is CalibMode.AXIS:
            self.axis_opts.pack(fill="x", padx=6, pady=(4, 0),
                                before=self.preview.master)
            self.btn_calib.configure(text="开始转轴标定")
            self.btn_shoot.configure(state="disabled")
            self.btn_live.configure(state="disabled")
            self._append_verdict(
                "转轴模式：请放好标记球，点击「开始转轴标定」全自动完成。")
        else:
            self.axis_opts.pack_forget()
            self.btn_calib.configure(text="开始标定")
            self.btn_live.configure(state="normal")

    # ------------------------------------------------------------------
    # 实时取景
    # ------------------------------------------------------------------
    def _on_toggle_live(self) -> None:
        if self._capturing:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self) -> None:
        if self._capturing:
            return
        self._ensure_session()
        self._live_stop.clear()
        self._capturing = True
        self.btn_live.configure(text="停止拍摄")
        self.btn_shoot.configure(state="normal")

        def _run() -> None:
            try:
                from ..camera import Camera
                with Camera(self.cfg.raw["camera"]) as cam:
                    self._camera = cam
                    while not self._live_stop.is_set():
                        img = cam.grab()
                        self._queue.put(("frame", img))
            except Exception as exc:  # noqa: BLE001
                self._queue.put(("error", f"相机错误: {exc}"))
            finally:
                self._camera = None
                self._queue.put(("live_stopped", None))

        self._live_thread = threading.Thread(
            target=_run, daemon=True, name="triniscan-live")
        self._live_thread.start()

    def _stop_live(self) -> None:
        if not self._capturing:
            return
        self._live_stop.set()
        self._capturing = False
        self.btn_live.configure(text="开始拍摄")
        self.btn_shoot.configure(state="disabled", text="拍摄")

    # ------------------------------------------------------------------
    # 拍摄
    # ------------------------------------------------------------------
    def _on_shoot(self) -> None:
        # 实时取景中：从最近一帧抓取；否则同步抓一帧
        if self._capturing:
            self._queue.put(("shoot", None))
        else:
            self._shoot_once()

    def _shoot_once(self) -> None:
        try:
            from ..camera import Camera
            with Camera(self.cfg.raw["camera"]) as cam:
                img = cam.grab()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("拍摄失败", str(exc))
            return
        self._handle_frame(img, do_eval=True)

    def _handle_frame(self, img: np.ndarray,
                      do_eval: bool = False) -> None:
        """处理一帧：更新预览；可选地做质量检测并收集。"""
        self.preview.set_image(img)
        if not do_eval:
            return
        session = self._ensure_session()
        verdict = session.add_frame(img)
        self._append_verdict(verdict.summary)
        self._refresh_ui()

    # ------------------------------------------------------------------
    # 其他操作
    # ------------------------------------------------------------------
    def _on_undo(self) -> None:
        session = self._ensure_session()
        if session.remove_last():
            self._append_verdict("已撤销最后一张")
            self._refresh_ui()

    def _on_clear(self) -> None:
        session = self._ensure_session()
        session.clear()
        self._append_verdict("已清空所有采集")
        self._refresh_ui()

    def _on_calibrate(self) -> None:
        if self._current_mode() is CalibMode.AXIS:
            self._start_axis_sweep()
            return

        session = self._ensure_session()
        if not session.ready:
            messagebox.showinfo(
                "提示",
                f"至少需要 {session.mode.min_views} 张有效图像，"
                f"当前 {session.count} 张。")
            return
        self._set_busy(True)
        self._append_verdict("开始标定…")

        def _run() -> None:
            outcome = session.run_calibration()
            self._queue.put(("calib_done", outcome))

        threading.Thread(target=_run, daemon=True,
                         name="triniscan-calib").start()

    # ------------------------------------------------------------------
    # 转轴：全自动扫描
    # ------------------------------------------------------------------
    def _read_axis_params(self) -> bool:
        """读取并校验界面上的转轴参数，写回配置。"""
        try:
            radius_mm = float(self._sphere_radius_mm.get())
            views = int(self._axis_views.get())
        except ValueError:
            messagebox.showerror("参数错误", "球半径与角度数必须是数字。")
            return False
        if radius_mm <= 0 or views < 3:
            messagebox.showerror("参数错误", "球半径需 > 0，角度数需 >= 3。")
            return False
        try:
            self.cfg.set("axis_calib.sphere_radius_m", radius_mm / 1000.0)
            self.cfg.set("axis_calib.num_views", views)
            self.cfg.set("axis_calib.method", self._axis_method.get())
        except Exception:  # noqa: BLE001 - 配置对象可能只读
            log.debug("配置对象不支持 set，使用默认参数")
        return True

    def _missing_prereqs(self) -> list:
        """返回缺失的前置标定文件说明（空列表表示齐全）。"""
        import os
        missing = []
        for key, label in (("calibration.camera_file", "相机内参"),
                           ("calibration.laser_file", "激光平面")):
            path = self.cfg.get(key, None)
            if path and not os.path.isabs(path):
                path = os.path.join(getattr(self.cfg, "root", "."), path)
            if not path or not os.path.exists(path):
                missing.append(label + "（" + str(path) + "）")
        return missing

    def _start_axis_sweep(self) -> None:
        if self._axis_sweep_active:
            return
        if not self._read_axis_params():
            return
        missing = self._missing_prereqs()
        if missing:
            messagebox.showwarning(
                "前置标定未完成",
                "转轴标定需要先完成：\n  - " + "\n  - ".join(missing)
                + "\n\n请先切换到对应模式完成标定。")
            return

        self._axis_sweep_active = True
        self._axis_stop = False
        self._set_busy(True)
        self._stop_live()
        self._append_verdict("开始转轴自动标定（转台将自动转动）…")

        def _progress(i: int, total: int, msg: str) -> None:
            self._queue.put(("axis_progress", (i, total, msg)))

        def _run() -> None:
            from .capture_session import sweep_axis_calibration
            result = sweep_axis_calibration(
                self.cfg,
                on_progress=_progress,
                stop_flag=lambda: self._axis_stop,
            )
            self._queue.put(("axis_done", result))

        threading.Thread(target=_run, daemon=True,
                         name="triniscan-axis").start()

    # ------------------------------------------------------------------
    # 状态刷新
    # ------------------------------------------------------------------
    def _refresh_ui(self) -> None:
        session = self._ensure_session()
        self.lbl_progress.configure(text=session.progress_text())
        rec = max(session.mode.recommend_views, 1)
        frac = min(session.count / rec, 1.0)
        self.bar_progress.configure(value=frac * 100)
        if self._current_mode() is CalibMode.AXIS:
            self.btn_undo.configure(state="disabled")
            self.btn_clear.configure(state="disabled")
            self.btn_calib.configure(
                state="disabled" if self._busy else "normal")
            return

        state = "normal" if session.count else "disabled"
        self.btn_undo.configure(state=state)
        self.btn_clear.configure(state=state)
        self.btn_calib.configure(
            state="normal" if session.ready and not self._busy
            else "disabled")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.lbl_status.configure(text="标定中…" if busy else "就绪")
        self.btn_calib.configure(state="disabled" if busy else "normal")
        if busy:
            self.btn_shoot.configure(state="disabled")
            self.btn_live.configure(state="disabled")
        elif self._current_mode() is not CalibMode.AXIS:
            self.btn_live.configure(state="normal")
            if self._capturing:
                self.btn_shoot.configure(state="normal")

    def _append_verdict(self, text: str) -> None:
        self.txt_verdict.configure(state="normal")
        self.txt_verdict.insert("end", text.rstrip() + "\n")
        self.txt_verdict.see("end")
        self.txt_verdict.configure(state="disabled")

    # ------------------------------------------------------------------
    # 队列轮询（UI 线程）
    # ------------------------------------------------------------------
    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "frame":
                    self.preview.set_image(payload)
                    # 实时取景下按固定节奏存"最新帧"，供拍摄按钮取用
                    self._latest_frame = payload
                elif kind == "shoot":
                    if self._latest_frame is not None:
                        self._handle_frame(self._latest_frame,
                                           do_eval=True)
                elif kind == "live_stopped":
                    self._capturing = False
                    self.btn_live.configure(text="开始拍摄")
                    self.btn_shoot.configure(state="disabled")
                elif kind == "calib_done":
                    self._set_busy(False)
                    self._refresh_ui()
                    self._append_verdict(payload.message)
                    if payload.ok:
                        messagebox.showinfo("标定完成", payload.message)
                    else:
                        messagebox.showerror("标定失败", payload.message)
                elif kind == "axis_progress":
                    i, total, msg = payload
                    self._append_verdict(f"[{i}/{total}] {msg}")
                    self.bar_progress.configure(
                        value=100.0 * i / max(total, 1))
                elif kind == "axis_done":
                    self._axis_sweep_active = False
                    self._set_busy(False)
                    self._append_verdict(payload.message)
                    if payload.ok:
                        detail = (
                            payload.message + "\n\n"
                            "转轴方向: " + str(np.round(payload.direction, 5))
                            + "\n轴上一点: " + str(np.round(payload.point, 5))
                            + "\n现场点存档: " + str(payload.points_path))
                        messagebox.showinfo("转轴标定完成", detail)
                    else:
                        messagebox.showerror("转轴标定失败",
                                             payload.message)
                elif kind == "error":
                    self._append_verdict(payload)
        except queue.Empty:
            pass
        self.after(_LIVE_INTERVAL_MS, self._poll)

    # ------------------------------------------------------------------
    def on_hide(self) -> None:
        """切换到其它界面时调用：停止实时取景，释放相机。"""
        self._stop_live()

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        return self._latest_frame
