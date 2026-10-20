"""标定台：包装既有 :class:`CalibrationPanel`，补上依赖校验与实时渲染。

为什么不重写
------------
``ui/calibration.py`` 与 ``ui/capture_session.py`` 是用户正在使用且**未
提交**的工作，里面已经沉淀了拍摄节奏、质量判定、球心/棋盘格两套转轴
标定等大量实战逻辑。重写等于把风险重新引入。

因此本模块采用**子类包装**，只解决三个既有缺陷，其余行为逐行保持：

1. **无界队列**：``CalibrationPanel._queue`` 是 ``queue.Queue()``，相机
   线程以 30 fps 写入而 UI 只取最新帧。连续取景十分钟即堆积上万帧
   （1280×800×3 ≈ 3 MB/帧）。这里替换成单槽位队列（配合原有的
   ``get_nowait`` 轮循环，语义完全兼容）。
2. **matplotlib 预览**：85 ms/帧（12 fps）。这里把 ``PreviewPanel`` 换成
   :class:`~triniscan.ui.video.VideoView`（PPM 路径 21 ms/帧）。为此
   提供 ``set_image`` 适配器，父类的调用点无需改动。
3. **标定失效无人知晓**：改内参后激光/转轴标定不再成立，但界面照旧
   显示"已标定"。这里在面板顶部加**标定状态卡**，比对
   ``calib_*.npz`` 中记录的 ``camera_md5``，不一致即标红提示。

依赖关系
--------
激光平面建立在相机内参之上（像素→射线），转轴建立在两者之上（点云
坐标系）。因此旧标定必须记住"生成时的内参指纹"，这是
:mod:`triniscan.calib_manifest` 的职责。
"""
from __future__ import annotations

import queue
import tkinter as tk
from tkinter import ttk
from typing import Any, Optional

import numpy as np

from ..calib_manifest import CalibrationStatus, inspect_calibrations
from ..common.logging_utils import get_logger
from .calibration import CalibrationPanel
from .video import VideoView
from .workbench import Workbench

log = get_logger(__name__)

__all__ = ["CalibrationWorkbench", "CalibrationStatusPanel", "LatestFrameQueue"]


# ---------------------------------------------------------------------------
class LatestFrameQueue(queue.Queue):
    """单槽位队列：写入新帧时丢弃尚未消费的旧帧。

    接口与 :class:`queue.Queue` 兼容（父类只用 ``put`` /
    ``get_nowait`` / ``empty``），但容量恒为 1，因此内存不会随运行
    时间增长。**必须保留 ``task_done`` 语义**：父类不使用，这里也不
    实现，避免与 ``join`` 不一致。
    """

    def __init__(self) -> None:
        super().__init__(maxsize=0)      # 容量自己管，不脏写后 throw
        self._lock_free = None

    def put(self, item: Any, block: bool = True,
            timeout: Optional[float] = None) -> None:
        with self.mutex:
            self.queue.clear()           # 丢旧帧，只留最新
        super().put(item, block=False)

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)


# ---------------------------------------------------------------------------
class _VideoPreviewAdapter(ttk.Frame):
    """把 :class:`VideoView` 包装成父类期望的 ``PreviewPanel`` 接口。

    ``CalibrationPanel`` 只调用 ``self.preview.set_image(img)``，
    因此只需实现这一个方法即可整体替换渲染后端。
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.view = VideoView(self)
        self.view.pack(fill="both", expand=True)

    def set_image(self, image: Any) -> None:
        try:
            self.view.show(image)
        except Exception:                   # noqa: BLE001
            log.debug("标定预览渲染失败", exc_info=True)

    def clear(self) -> None:
        self.view.clear()


# ---------------------------------------------------------------------------
class CalibrationStatusPanel(ttk.Frame):
    """标定状态卡：三项标定各自的质量与"是否已过期"。"""

    def __init__(self, master: tk.Misc, workbench: "CalibrationWorkbench") -> None:
        super().__init__(master)
        self.workbench = workbench
        self._rows: dict[str, tuple[ttk.Label, ttk.Label]] = {}

        for key, title in (("camera", "相机内参"),
                           ("laser", "激光平面"),
                           ("axis", "转台转轴")):
            row = ttk.Frame(self)
            row.pack(fill="x", pady=1)
            name = ttk.Label(row, text=title, width=8, anchor="w")
            name.pack(side="left")
            state = ttk.Label(row, text="—", foreground="#7f8c8d",
                              anchor="w")
            state.pack(side="left", fill="x", expand=True)
            self._rows[key] = (name, state)

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Button(bar, text="重新检查", width=9,
                   command=self.refresh).pack(side="left")
        self._note = ttk.Label(self, text="", foreground="#c0392b",
                               wraplength=290, justify="left")
        self._note.pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            status = self.workbench.status()
        except Exception:                   # noqa: BLE001
            log.exception("检查标定状态失败")
            self._note.configure(text="⚠ 无法读取标定文件")
            return

        for key, (name, state) in self._rows.items():
            item = status.item(key)
            mark = {"ok": "✓", "warn": "!", "fail": "✗",
                    "missing": "—", "stale": "⟳"}[item.level]
            state.configure(text=f"{mark} {item.text}",
                            foreground=item.color)

        stale = status.stale_items()
        if stale:
            self._note.configure(
                text="⚠ " + "、".join(i.title for i in stale)
                     + " 建立于旧内参之上，建议重新标定")
        elif status.needs_recalibration:
            self._note.configure(text="")
        else:
            self._note.configure(text="")


# ---------------------------------------------------------------------------
class CalibrationWorkbench(Workbench):
    """标定工作台：状态卡 + 完整标定面板。"""

    title = "标定"
    panel_width = 380

    def add_panels(self) -> None:
        self.add_panel("status", CalibrationStatusPanel, expanded=True)
        self.add_panel("summary", _CalibSummaryPanel, expanded=False)

    def add_pages(self) -> None:
        self.panel_obj = _CalibrationPanelV2(self.view_parent, self.store)
        self.add_page("calib", self.panel_obj)

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        # 标定面板不会自动开始取景（避免一进来就抢相机）；
        # 这里只确保两张状态卡是最新的。
        self.refresh()

    def on_hide(self) -> None:
        try:
            self.panel_obj.on_hide()
        except Exception:                   # noqa: BLE001
            log.debug("停止标定取景失败", exc_info=True)

    def refresh(self) -> None:
        if self.panel("status") is not None:
            self.panel("status").refresh()

    # ------------------------------------------------------------------
    def status(self) -> CalibrationStatus:
        return inspect_calibrations(self.cfg)


# ---------------------------------------------------------------------------
class _CalibrationPanelV2(CalibrationPanel):
    """``CalibrationPanel`` 的最小改动子类（只换队列与预览后端）。"""

    def __init__(self, master: tk.Misc, cfg) -> None:
        super().__init__(master, cfg)
        # 1) 单槽位队列：抑制长时间取景的内存增长
        self._queue = LatestFrameQueue()
        # 2) 预览后端：把 matplotlib PreviewPanel 换成 VideoView
        self._replace_preview()

    def _replace_preview(self) -> None:
        old = getattr(self, "preview", None)
        if old is None:                     # pragma: no cover
            return
        master = old.master
        try:
            info = old.pack_info()
        except tk.TclError:                 # pragma: no cover
            info = {"side": "top", "fill": "both", "expand": True,
                    "padx": 0, "pady": 0, "ipadx": 0, "ipady": 0,
                    "anchor": "center"}
        old.destroy()
        adapter = _VideoPreviewAdapter(master)
        adapter.pack(**{k: v for k, v in info.items() if v != ""})
        self.preview = adapter             # type: ignore[assignment]

    def on_hide(self) -> None:
        super().on_hide()
        preview = getattr(self, "preview", None)
        if preview is not None and hasattr(preview, "clear"):
            try:
                preview.clear()
            except tk.TclError:             # pragma: no cover
                pass


class _CalibSummaryPanel(ttk.Frame):
    """标定数值摘要（只读）。"""

    def __init__(self, master: tk.Misc, workbench: Workbench) -> None:
        super().__init__(master)
        self.workbench = workbench
        self._text = tk.Text(self, height=12, width=36, wrap="word",
                             relief="flat", background="#f7f8f9",
                             foreground="#2c3e50", font=("Consolas", 9))
        self._text.pack(fill="both", expand=True)
        self._text.configure(state="disabled")
        self.refresh()

    def refresh(self) -> None:
        try:
            status = self.workbench.status()
        except Exception:                   # noqa: BLE001
            lines = ["无法读取标定文件"]
        else:
            lines = status.summary_lines()
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", "\n".join(lines))
        self._text.configure(state="disabled")
