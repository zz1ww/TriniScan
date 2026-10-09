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
        self._ensure_session()
        self._refresh_ui()
        self._append_verdict(f"已切换到：{self._current_mode().label}")
        if self._current_mode() is CalibMode.AXIS:
            self._append_verdict(
                "转轴标定需要特征点三维坐标，本界面提供接口；"
                "请先用「转轴」采集各角度图像后由外部流程计算。")

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
        session = self._ensure_session()
        if not session.ready:
            messagebox.showinfo(
                "提示",
                f"至少需要 {session.mode.min_views} 张有效图像，"
                f"当前 {session.count} 张。")
            return
        if self._current_mode() is CalibMode.AXIS:
            messagebox.showinfo(
                "提示",
                "转轴标定不接受图像，请通过接口传入特征点三维坐标。")
            return
        self._set_busy(True)
        self._append_verdict("开始标定…")

        def _run() -> None:
            outcome = session.run_calibration()
            self._queue.put(("calib_done", outcome))

        threading.Thread(target=_run, daemon=True,
                         name="triniscan-calib").start()

    # ------------------------------------------------------------------
    # 状态刷新
    # ------------------------------------------------------------------
    def _refresh_ui(self) -> None:
        session = self._ensure_session()
        self.lbl_progress.configure(text=session.progress_text())
        rec = max(session.mode.recommend_views, 1)
        frac = min(session.count / rec, 1.0)
        self.bar_progress.configure(value=frac * 100)
        state = "normal" if session.count else "disabled"
        self.btn_undo.configure(state=state)
        self.btn_clear.configure(state=state)
        self.btn_calib.configure(
            state="normal" if session.ready and not self._busy
            else "disabled")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.lbl_status.configure(text="标定中…" if busy else "就绪")
        self.btn_calib.configure(
            state="disabled" if busy else "normal")
        if busy:
            self.btn_shoot.configure(state="disabled")
        elif self._capturing:
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
