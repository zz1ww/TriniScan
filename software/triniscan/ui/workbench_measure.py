"""测量工作台：调参 + 实时预览 + 完整重建。

布局::

    ┌──────────────────┬────────────────────────────────────┐
    │ ▾ 相机           │                                    │
    │ ▾ 扫描计划       │                                    │
    │ ▾ 提取   [算法▼] │        实时画面  /  重建结果         │
    │ ▾ 过滤链         │                                    │
    │ ▾ 配准           │                                    │
    │ ▾ 网格           │                                    │
    │ ▾ 封闭           │                                    │
    │ ▾ 体积  [算法▼]  │                                    │
    ├──────────────────┴────────────────────────────────────┤
    │ [▶预览][■停止] ☑中心线 │ [⚙ 从图像重建]  12.3 fps      │
    └───────────────────────────────────────────────────────┘

面板与流水线的对应关系（**每个面板都必须真正被 pipeline 读取**）
---------------------------------------------------------------
===========  ==========================================
面板         流水线读取点
===========  ==========================================
相机          ``Camera(cfg.raw["camera"])``
扫描计划      ``acquire`` → ``resolve_scan_plan(cfg)``
提取          ``reconstruct_views`` → ``centerline_view(cfg)``
过滤链        ``build_pointcloud`` → ``_run_filter_chain``
配准          ``build_pointcloud`` → ``SEQUENTIAL_SPEC.read(cfg)``
网格          ``measure_volume`` → ``POISSON_SPEC.read(cfg)``
封闭          ``measure_volume`` → ``CLOSE_SPEC.read(cfg)``
体积          ``measure_volume`` → ``volume.method`` / 两个 spec
===========  ==========================================

网格面板**不提供 poisson/ball_pivoting 下拉**：``MESHERS`` 虽然登记了
两种，但 ``measure_volume`` 固定调用 ``POISSON_SPEC``，提供下拉就是
假开关。面板里会以文字说明这一点。
"""
from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import ttk
from queue import Empty, Queue
from typing import Any, Optional

import numpy as np

from .algorithm_panel import AlgorithmPanel, _ChoiceSelector
from .filter_chain_ui import FilterChainEditor
from .params import ParamGroup, SpecPanel, bind_tooltip
from .rebuild_job import RebuildJob, RebuildOutcome
from .result_view import ResultView
from .scan_plan_panel import ScanPanel
from ..common.logging_utils import get_logger
from .video import LiveCamera, VideoView
from .workbench import Workbench
from .worker import TaskMessage, Worker

log = get_logger(__name__)

__all__ = ["MeasureWorkbench"]


# ---------------------------------------------------------------------------
class CameraPanel(ttk.Frame):
    """相机参数。"""

    title = "相机"

    KEYS = (
        "camera.index", "camera.backend", "camera.width", "camera.height",
        "camera.fps", "camera.exposure", "camera.gain",
        "camera.auto_exposure", "camera.mono",
    )

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        super().__init__(master)
        self.workbench = workbench
        self.group = ParamGroup(self, on_change=self._on_change)
        self.group.pack(fill="x")

        for key in self.KEYS:
            fs = workbench.field_spec(key)
            if fs is not None:
                self.group.add_row(fs, workbench.cfg.get(key))

        ttk.Label(self, text="修改后需重新开始预览才生效",
                  foreground="#7f8c8d").pack(anchor="w", pady=(4, 0))

    def _on_change(self, key_path: str, value: Any) -> None:
        self.workbench.store.update(key_path, value)

    def refresh(self) -> None:
        for key in self.group.keys:
            self.group.set_value(key, self.workbench.cfg.get(key))


class ExtractPanel(ttk.Frame):
    """中心线提取：算法 + 该算法的参数。"""

    title = "提取"

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        super().__init__(master)
        self.panel = AlgorithmPanel(
            self, "EXTRACTORS", workbench.store,
            method_key="extraction.method",
            on_change=lambda k, v: workbench.store.update(k, v))
        self.panel.pack(fill="x")
        self.panel.refresh()

    def refresh(self) -> None:
        self.panel.refresh()


class FilterChainPanel(ttk.Frame):
    """过滤链。"""

    title = "过滤链"

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        super().__init__(master)
        self.workbench = workbench
        self.editor = FilterChainEditor(self, workbench.store)
        self.editor.pack(fill="x")
        self.editor._on_change = self._on_chain_change

        self._summary = ttk.Label(self, text="", foreground="#7f8c8d",
                                  wraplength=290, justify="left")
        self._summary.pack(anchor="w", pady=(6, 0))
        self._update_summary()

    def _on_chain_change(self, _chain: list) -> None:
        self._update_summary()

    def _update_summary(self) -> None:
        chain = self.workbench.cfg.get("reconstruction.filter_chain", [])
        if not isinstance(chain, list):
            self._summary.configure(text="⚠ 配置不是列表，请重新添加过滤器")
            return
        active = [str(i.get("name")) for i in chain
                  if isinstance(i, dict) and i.get("enable", True)]
        self._summary.configure(
            text=f"启用 {len(active)} 项：{' → '.join(active) or '（无）'}")

    def refresh(self) -> None:
        self.editor.rebuild()
        self._update_summary()


class _SpecMixin:
    """固定算法面板的公共实现（配准 / 网格 / 封闭）。"""

    SPEC = None            # 子类覆盖
    NOTE = ""

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        super().__init__(master)
        self.panel = SpecPanel(
            self, self.SPEC, workbench.store,
            on_change=lambda k, v: workbench.store.update(k, v),
            note=self.NOTE)
        self.panel.pack(fill="x")

    def refresh(self) -> None:
        self.panel.refresh()


class RegisterPanel(_SpecMixin, ttk.Frame):
    """配准（SEQUENTIAL_SPEC）。"""

    title = "配准"

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        from ..algorithms.registration.specs import SEQUENTIAL_SPEC
        self.SPEC = SEQUENTIAL_SPEC
        self.NOTE = "按转台角度顺序拼接各视角点云"
        super().__init__(master, workbench)


class MeshPanel(_SpecMixin, ttk.Frame):
    """网格（POISSON_SPEC）。"""

    title = "网格"

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        from ..algorithms.meshing.specs import POISSON_SPEC
        self.SPEC = POISSON_SPEC
        self.NOTE = ("当前流水线固定使用泊松重建；滚球法虽已实现，"
                     "但未接入 measure_volume，故此处不提供切换。")
        super().__init__(master, workbench)


class ClosePanel(_SpecMixin, ttk.Frame):
    """封闭（CLOSE_SPEC）。"""

    title = "封闭"

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        from ..algorithms.closing.specs import CLOSE_SPEC
        self.SPEC = CLOSE_SPEC
        self.NOTE = "封闭失败会降级到兜底策略，体积只能作参考"
        super().__init__(master, workbench)


class VolumePanel(ttk.Frame):
    """体积：method 下拉 + 该方法的参数。

    ``volume.method`` 不属于任何 ``AlgorithmSpec`` 的参数列表
    （它是两个 spec 之外的调度键），因此不能直接用 ``AlgorithmPanel``。
    这里显式建下拉，并在切换时切换参数表单。
    """

    title = "体积"

    METHODS = ("divergence", "voxel")

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        super().__init__(master)
        self.workbench = workbench
        from ..algorithms.volumetry.specs import (DIVERGENCE_SPEC,
                                                  VOXEL_SPEC)
        self._specs = {"divergence": DIVERGENCE_SPEC, "voxel": VOXEL_SPEC}

        ttk.Label(self, text="计算方法", style="Key.TLabel").pack(anchor="w")
        self._var = tk.StringVar()
        self.combo = ttk.Combobox(self, textvariable=self._var,
                                  state="readonly",
                                  values=list(self.METHODS), width=16)
        self.combo.pack(fill="x", pady=(2, 6))
        self.combo.bind("<<ComboboxSelected>>", self._on_method)

        self._holder = ttk.Frame(self)
        self._holder.pack(fill="x")

        self._cross = ParamGroup(
            self, on_change=lambda k, v: workbench.store.update(k, v))
        self._cross.pack(fill="x", pady=(6, 0))
        fs = workbench.field_spec("volume.cross_check")
        if fs is not None:
            self._cross.add_row(fs, workbench.cfg.get("volume.cross_check"))

        self._note = ttk.Label(self, text="", foreground="#7f8c8d",
                               wraplength=290, justify="left")
        self._note.pack(anchor="w", pady=(6, 0))
        self._current = None
        self.refresh()

    # ------------------------------------------------------------------
    def _on_method(self, _event: tk.Event) -> None:
        self.workbench.store.update("volume.method", self._var.get())
        self._rebuild()

    def _rebuild(self) -> None:
        for child in self._holder.winfo_children():
            child.destroy()
        method = self._var.get()
        spec = self._specs.get(method)
        if spec is None:
            self._note.configure(text=f"未知方法 {method!r}")
            self._current = None
            return
        panel = SpecPanel(
            self._holder, spec, self.workbench.store,
            on_change=lambda k, v: self.workbench.store.update(k, v))
        panel.pack(fill="x")
        self._current = panel
        self._note.configure(
            text=("散度定理：对封闭网格做体积积分，要求水密"
                  if method == "divergence"
                  else "体素法：按体素占据计数，对网格缺陷更宽容"))

    def refresh(self) -> None:
        self._var.set(str(self.workbench.cfg.get("volume.method",
                                                 "divergence")))
        self._rebuild()
        for key in self._cross.keys:
            self._cross.set_value(key, self.workbench.cfg.get(key))


# ---------------------------------------------------------------------------
class MeasureWorkbench(Workbench):
    """测量台。"""

    title = "测量"
    panel_width = 360

    def __init__(self, master: tk.Misc, store, **kw) -> None:
        # 右侧视图在 add_pages() 里构造：那里的父容器已是 view_parent，
        # 避免视图挂在工作台根上盖住面板与工具条。
        self.video: Optional[VideoView] = None
        self.result_view: Optional[ResultView] = None

        self._live: Optional[LiveCamera] = None
        self._overlay_enabled = True
        self._fps = 0.0
        self._fps_t0 = time.perf_counter()
        self._fps_count = 0

        self._worker: Optional[Worker] = None
        self._msg_queue: Queue = Queue()
        self._rebuilding = False
        self._last_outcome: Optional[RebuildOutcome] = None
        self._limit_var = tk.StringVar(value="")
        super().__init__(master, store, **kw)

    # ------------------------------------------------------------------
    # 声明
    # ------------------------------------------------------------------
    def add_panels(self) -> None:
        self.add_panel("camera", CameraPanel, expanded=True)
        self.add_panel("scan", ScanPanel, expanded=False)
        self.add_panel("extract", ExtractPanel, expanded=True)
        self.add_panel("filter", FilterChainPanel, expanded=False)
        self.add_panel("register", RegisterPanel, expanded=False)
        self.add_panel("mesh", MeshPanel, expanded=False)
        self.add_panel("close", ClosePanel, expanded=False)
        self.add_panel("volume", VolumePanel, expanded=False)

    def add_pages(self) -> None:
        parent = self.view_parent
        self.video = VideoView(parent, placeholder="（等待实时画面…）")
        self.result_view = ResultView(parent)
        self.add_page("video", self.video)
        self.add_page("result", self.result_view, show=False)
        self._build_toolbar()

    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        """底部工具条：开始测量 + 预览控制 + 重建入口 + 视图切换。"""
        bar = self.add_toolbar()

        # 「开始测量」是主动作，放在最左、最醒目。
        # 完整流程：归零 → 逐视角转动/抓图 → 重建 → 体积。
        self.btn_measure = ttk.Button(bar, text="▶ 开始测量", width=11,
                                      style="Accent.TButton",
                                      command=self.rebuild_full)
        self.btn_measure.pack(side="left")

        ttk.Separator(bar, orient="vertical").pack(
            side="left", fill="y", padx=8)

        self.btn_preview = ttk.Button(bar, text="▶ 预览", width=9,
                                      command=self.toggle_preview)
        self.btn_preview.pack(side="left")

        self._overlay_var = tk.BooleanVar(value=self._overlay_enabled)
        ttk.Checkbutton(bar, text="中心线", variable=self._overlay_var,
                        command=self._on_overlay_toggle).pack(
            side="left", padx=(8, 0))

        ttk.Separator(bar, orient="vertical").pack(
            side="left", fill="y", padx=10)

        ttk.Button(bar, text="显示画面", width=9,
                   command=lambda: self.show_page("video")).pack(side="left")
        ttk.Button(bar, text="显示结果", width=9,
                   command=lambda: self.show_page("result")).pack(
            side="left", padx=(4, 0))

        ttk.Separator(bar, orient="vertical").pack(
            side="left", fill="y", padx=10)

        ttk.Label(bar, text="图像上限").pack(side="left")
        ttk.Entry(bar, textvariable=self._limit_var, width=6).pack(
            side="left", padx=(4, 4))
        self._limit_var.set("")
        lbl = ttk.Label(bar, text="（空=全部）", foreground="#7f8c8d")
        lbl.pack(side="left")
        bind_tooltip(lbl, "调试时可只跑前 N 张图，快速验证参数")

        self.btn_rebuild = ttk.Button(bar, text="⚙ 从图像重建",
                                      command=self.rebuild_from_images)
        self.btn_rebuild.pack(side="left", padx=(8, 0))

    def _on_overlay_toggle(self) -> None:
        self.set_overlay_enabled(bool(self._overlay_var.get()))

    def setup_engine(self) -> None:
        self._fps_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    def field_spec(self, key: str):
        from .field_spec import all_fields
        try:
            return all_fields(self.cfg).get(key)
        except Exception:                   # noqa: BLE001
            log.exception("构造 FieldSpec 失败: %s", key)
            return None

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        self.start_preview()

    def on_hide(self) -> None:
        self.stop_preview()

    # ------------------------------------------------------------------
    # 实时画面
    # ------------------------------------------------------------------
    @property
    def previewing(self) -> bool:
        return self._live is not None and self._live.running

    @property
    def fps(self) -> float:
        return self._fps

    def start_preview(self) -> None:
        if self.previewing:
            return
        from ..hardware.camera import Camera

        config = dict(self.cfg.raw.get("camera", {}))
        self._live = LiveCamera(
            self.video,
            lambda: Camera(config).open(),
            on_error=self._on_camera_error,
            on_frame=self._on_frame,
        )
        self.video.set_overlay(self._draw_overlay)
        self._live.start()
        self._sync_toolbar()

    def stop_preview(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self.video.clear()
        self._sync_toolbar()

    def toggle_preview(self) -> None:
        if self.previewing:
            self.stop_preview()
        else:
            self.start_preview()
        self._sync_toolbar()

    def _sync_toolbar(self) -> None:
        btn = getattr(self, "btn_preview", None)
        if btn is not None:
            btn.configure(text="■ 停止" if self.previewing else "▶ 预览")
        btn_rb = getattr(self, "btn_rebuild", None)
        if btn_rb is not None:
            btn_rb.configure(
                state="disabled" if self._rebuilding else "normal")

    def _on_camera_error(self, message: str) -> None:
        """相机打不开时，把原因**显示在画面上**（而不是只写日志）。"""
        log.warning("%s", message)
        index = self.cfg.get("camera.index")
        backend = self.cfg.get("camera.backend")
        self.video.set_placeholder(
            f"⚠ 无法打开相机\n\n{message}\n\n"
            f"当前配置：index={index}, backend={backend}\n"
            f"请在「相机」面板调整后点「重新连接」")
        self._notify(f"相机未就绪: {message}")

    def _on_frame(self, _frame: np.ndarray) -> None:
        self._fps_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t0 = now

    # ------------------------------------------------------------------
    # 叠加层
    # ------------------------------------------------------------------
    def set_overlay_enabled(self, enabled: bool) -> None:
        self._overlay_enabled = bool(enabled)

    def _draw_overlay(self, bgr: np.ndarray) -> np.ndarray:
        if not self._overlay_enabled:
            return bgr
        try:
            points = self.extract_points(bgr)
        except Exception:                   # noqa: BLE001
            log.debug("中心线叠加失败", exc_info=True)
            return bgr
        if points is None or len(points) == 0:
            return bgr
        import cv2

        for x, y in np.asarray(points, dtype=np.float64).reshape(-1, 2):
            cv2.circle(bgr, (int(round(x)), int(round(y))), 1,
                       (0, 255, 0), -1, lineType=cv2.LINE_AA)
        return bgr

    def extract_points(self, bgr: np.ndarray):
        """跑一遍中心线提取（只做二维，不三角测量）。"""
        from ..algorithms.extraction import centerline as _cl

        extract_cfg = self.cfg.raw.get("extraction", {}) or {}
        return _cl.extract_centerline(bgr, dict(extract_cfg))

    # ------------------------------------------------------------------
    # 重建
    # ------------------------------------------------------------------
    @property
    def rebuilding(self) -> bool:
        return self._rebuilding

    @property
    def last_outcome(self) -> Optional[RebuildOutcome]:
        return self._last_outcome

    def set_limit(self, text: str) -> None:
        self._limit_var.set(text)

    def rebuild_from_images(self) -> None:
        """从 ``output.raw_dir`` 中已采集的图像重建（不需要硬件）。"""
        if self._rebuilding:
            return
        raw_dir = str(self.cfg.resolve("output.raw_dir"))
        limit = self._parse_limit()
        self._start_rebuild(raw_dir, limit)

    def rebuild_full(self) -> None:
        """完整测量（含转台采集）。"""
        if self._rebuilding:
            return
        self._start_rebuild(None, None)

    def _parse_limit(self) -> Optional[int]:
        text = self._limit_var.get().strip()
        if not text:
            return None
        try:
            n = int(text)
        except ValueError:
            log.warning("图像上限 %r 不是整数，忽略", text)
            return None
        return n if n > 0 else None

    def _start_rebuild(self, raw_dir: Optional[str],
                       limit: Optional[int]) -> None:
        if raw_dir is not None and not os.path.isdir(raw_dir):
            self._notify(f"图像目录不存在: {raw_dir}")
            return

        self._rebuilding = True
        self._sync_toolbar()
        self._last_outcome = None
        self.result_view.show_placeholder("正在重建…")
        self.show_page("result")
        self._notify("开始重建…")
        self.stop_preview()          # 释放相机，避免与采集争用

        job = RebuildJob(self.cfg, raw_dir, limit,
                         progress=self._on_progress)
        self._drain(self._msg_queue)
        self._worker = Worker(job.run, self._msg_queue)
        self._worker.start()
        self.after(100, self._poll_rebuild)

    def _on_progress(self, stage: str, fraction: float, text: str) -> None:
        """Worker 线程调用：只写队列，绝不碰控件。"""
        self._msg_queue.put(
            TaskMessage("progress", payload=fraction, text=text))

    @staticmethod
    def _drain(queue: Queue) -> None:
        try:
            while True:
                queue.get_nowait()
        except Empty:
            pass

    def _poll_rebuild(self) -> None:
        finished = False
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                if msg.kind == "progress":
                    if msg.text:
                        self._notify(msg.text)
                elif msg.kind == "done":
                    finished = True
                    self._on_rebuild_done(msg.payload)
                elif msg.kind == "error":
                    finished = True
                    self._on_rebuild_error(msg.payload)
        except Empty:
            pass
        except tk.TclError:                 # pragma: no cover
            return

        if finished:
            self._rebuilding = False
            self._sync_toolbar()
            return
        try:
            self.after(120, self._poll_rebuild)
        except tk.TclError:                 # pragma: no cover
            pass

    def _on_rebuild_done(self, outcome: RebuildOutcome) -> None:
        self._last_outcome = outcome
        self.result_view.show(outcome)
        if outcome.ok:
            r = outcome.result
            self._notify(f"重建完成：{r.volume_cm3:.2f} cm³，"
                         f"{r.num_points:,} 点，{outcome.elapsed_s:.1f} s")
        else:
            self._notify(f"重建失败：{outcome.error}")

    def _on_rebuild_error(self, exc: Any) -> None:
        self._last_outcome = RebuildOutcome(error=str(exc))
        self.result_view.show(self._last_outcome)
        self._notify(f"✗ 重建失败: {exc}")

    def _notify(self, text: str) -> None:
        self.log(text)
