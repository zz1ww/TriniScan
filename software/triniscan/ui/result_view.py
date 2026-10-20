"""重建结果页：体积卡片 + 三维点云预览 + 阶段耗时。

为什么用 matplotlib 而不用实时管线
----------------------------------
实时画面走 PPM 直绘（48 fps），但这里是**一次性结果**展示，需要
旋转、缩放、颜色映射——matplotlib 的 ``Axes3D`` 现成可用，且不需要
额外依赖。渲染一次约 0.3 s，对结果页完全够用。

点云规模可能有几十万点，直接 ``scatter`` 会卡死。因此按点数分档降采样
（>50 万降到 5 万），并保持**均匀抽稀**而非取前 N 个——否则只看到
局部。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np

from ..common.logging_utils import get_logger
from .rebuild_job import RebuildOutcome

log = get_logger(__name__)

__all__ = ["ResultView"]

#: 点云渲染上限（超过则等间隔抽稀）
MAX_RENDER_POINTS = 50_000
#: 默认视角（仰角, 方位角）
VIEW_ANGLES = (22, -60)


class ResultView(ttk.Frame):
    """结果页：左侧数值卡片，右侧 3D 点云。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._card = _ResultCard(self)
        self._card.grid(row=0, column=0, sticky="ns", padx=(6, 8), pady=6)

        self._plot_holder = ttk.Frame(self)
        self._plot_holder.grid(row=0, column=1, sticky="nsew",
                               padx=(0, 6), pady=6)

        self._canvas = None
        self._ax = None
        self._outcome: Optional[RebuildOutcome] = None
        self.show_placeholder()

    # ------------------------------------------------------------------
    def show_placeholder(self, text: str = "尚无重建结果") -> None:
        self._card.reset()
        self._clear_plot()
        ttk.Label(self._plot_holder, text=text, foreground="#7f8c8d",
                  font=("", 12)).place(relx=0.5, rely=0.5, anchor="center")

    def show(self, outcome: RebuildOutcome) -> None:
        """展示一次重建结果。"""
        self._outcome = outcome
        self._clear_plot()
        self._card.update_from(outcome)
        points = outcome.points
        if points is None or len(points) == 0:
            ttk.Label(self._plot_holder, text="无点云数据",
                      foreground="#e67e22", font=("", 12)).place(
                relx=0.5, rely=0.5, anchor="center")
            return
        try:
            self._render(points)
        except Exception as exc:            # noqa: BLE001
            log.exception("点云渲染失败")
            ttk.Label(self._plot_holder, text=f"点云渲染失败: {exc}",
                      foreground="#c0392b", wraplength=420).place(
                relx=0.5, rely=0.5, anchor="center")

    def clear(self) -> None:
        self._outcome = None
        self.show_placeholder()

    # ------------------------------------------------------------------
    def _clear_plot(self) -> None:
        for child in self._plot_holder.winfo_children():
            child.destroy()
        self._canvas = None
        self._ax = None

    def _render(self, points: np.ndarray) -> None:
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from matplotlib.figure import Figure

        pts = _subsample(points, MAX_RENDER_POINTS)
        fig = Figure(figsize=(5.2, 4.2), dpi=100, facecolor="#1e1e1e")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#1e1e1e")

        color = pts[:, 2] * 1000.0          # 按高度着色（mm）
        ax.scatter(pts[:, 0] * 1000, pts[:, 1] * 1000, pts[:, 2] * 1000,
                   c=color, cmap="viridis", s=0.6, linewidths=0)
        ax.set_xlabel("X (mm)", color="#c8c8c8", fontsize=8)
        ax.set_ylabel("Y (mm)", color="#c8c8c8", fontsize=8)
        ax.set_zlabel("Z (mm)", color="#c8c8c8", fontsize=8)
        ax.tick_params(colors="#8a8a8a", labelsize=7)
        ax.view_init(*VIEW_ANGLES)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("#262626")
            axis.pane.set_edgecolor("#3a3a3a")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._plot_holder)
        toolbar = NavigationToolbar2Tk(canvas, self._plot_holder,
                                       pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw_idle()
        self._canvas = canvas
        self._ax = ax

        # 点数提示（清楚说明显示的是抽稀后的数据）
        n = len(points)
        if n > len(pts):
            ttk.Label(self._plot_holder,
                      text=f"显示 {len(pts):,}/{n:,} 点（等间隔抽稀）",
                      foreground="#7f8c8d").pack(side="bottom")

    # ------------------------------------------------------------------
    @property
    def outcome(self) -> Optional[RebuildOutcome]:
        return self._outcome


# ---------------------------------------------------------------------------
def _subsample(points: np.ndarray, limit: int) -> np.ndarray:
    """等间隔抽稀（不是取前 N 个，否则只显示局部）。"""
    n = len(points)
    if n <= limit:
        return points
    idx = np.linspace(0, n - 1, limit).astype(np.int64)
    return points[idx]


# ---------------------------------------------------------------------------
class _ResultCard(ttk.Frame):
    """结果数值卡片。"""

    FIELDS = (
        ("volume", "体积"),
        ("cross", "交叉校核"),
        ("hull", "凸包体积"),
        ("points", "点数"),
        ("views", "有效视角"),
        ("angle", "每视角转角"),
        ("filtered", "过滤后点数"),
        ("watertight", "封闭"),
        ("close", "封闭策略"),
        ("time", "耗时"),
    )

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._labels: dict[str, ttk.Label] = {}
        self._value_labels: dict[str, ttk.Label] = {}

        ttk.Label(self, text="重建结果", style="Key.TLabel").pack(
            anchor="w", pady=(0, 6))

        grid = ttk.Frame(self)
        grid.pack(fill="x")
        for i, (key, title) in enumerate(self.FIELDS):
            ttk.Label(grid, text=title, foreground="#7f8c8d").grid(
                row=i, column=0, sticky="w", pady=1)
            value = ttk.Label(grid, text="—", anchor="e")
            value.grid(row=i, column=1, sticky="e", padx=(8, 0))
            self._value_labels[key] = value

        self._warn = ttk.Label(self, text="", foreground="#e67e22",
                               wraplength=230, justify="left")
        self._warn.pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        for label in self._value_labels.values():
            label.configure(text="—")
        self._warn.configure(text="")

    def update_from(self, outcome: RebuildOutcome) -> None:
        self.reset()
        if not outcome.ok:
            self._warn.configure(text=f"✗ {outcome.error}")
            return

        r = outcome.result
        v = self._value_labels
        v["volume"].configure(
            text=f"{r.volume_cm3:.2f} cm³", font=("", 12, "bold"))
        v["cross"].configure(
            text=(f"{r.cross_check_cm3:.2f} cm³"
                  if r.cross_check_cm3 is not None else "—"))
        v["hull"].configure(
            text=(f"{r.hull_volume_cm3:.2f} cm³"
                  if r.hull_volume_cm3 is not None else "—"))
        v["points"].configure(text=f"{r.num_points:,}")
        v["views"].configure(text=str(r.num_views))
        v["angle"].configure(
            text=(f"{outcome.angle_step_deg:.6f}°"
                  if outcome.angle_step_deg else "—"))
        v["filtered"].configure(
            text=(f"{r.points_after_filter:,} / {r.points_before_filter:,}"
                  if r.points_before_filter else "—"))
        v["watertight"].configure(
            text="是" if r.mesh_watertight else "否",
            foreground="#27ae60" if r.mesh_watertight else "#c0392b")
        v["close"].configure(text=r.close_fallback or "—")
        v["time"].configure(text=f"{outcome.elapsed_s:.1f} s")

        notes = []
        if outcome.plan_warning:
            notes.append(f"⚠ {outcome.plan_warning}")
        if r.cross_check_diff is not None:
            notes.append(f"交叉校核偏差 {r.cross_check_diff * 100:.1f}%")
        if r.close_degraded:
            notes.append(f"⚠ 封闭降级：{r.close_degrade_reason}")
        if r.turntable_removed:
            notes.append(f"去除转盘点 {r.turntable_removed:,}")
        self._warn.configure(text="\n".join(notes))
