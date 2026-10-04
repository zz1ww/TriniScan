"""预览面板：激光图像 + 点云，基于 matplotlib 嵌入 Tk。

matplotlib 为可选依赖；缺失时面板退化为提示文字，
不影响其它功能（便于在没装 matplotlib 的机器上跑）。
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np

def _setup_cjk_font() -> None:
    """让 matplotlib 能显示中文（Windows/Linux 常见中文字体）。"""
    from matplotlib import font_manager, rcParams

    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.sans-serif"] = [name] + rcParams[
                "font.sans-serif"]
            break
    rcParams["axes.unicode_minus"] = False


try:  # matplotlib 为可选依赖
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure

    _setup_cjk_font()
    _HAS_MPL = True
except Exception:  # noqa: BLE001
    _HAS_MPL = False


class PreviewPanel(ttk.LabelFrame):
    """图像 / 点云预览面板。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="预览")
        self._mode = tk.StringVar(value="image")

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Radiobutton(bar, text="激光图像", value="image",
                        variable=self._mode,
                        command=self._switch).pack(side="left")
        ttk.Radiobutton(bar, text="点云", value="cloud",
                        variable=self._mode,
                        command=self._switch).pack(side="left", padx=6)

        if not _HAS_MPL:
            ttk.Label(self, text="（未安装 matplotlib，预览不可用；\n"
                                 "  pip install matplotlib）").pack(pady=20)
            self._fig = None
            return

        self._fig = Figure(figsize=(4, 3), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_axis_off()
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, self,
                                             pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(fill="x")

        self._image = None       # 最近的激光图像 (H, W, 3) uint8
        self._points = None      # 最近的点云 (N, 3) m
        self._draw_image()

    # ------------------------------------------------------------------
    def _switch(self) -> None:
        if self._mode.get() == "image":
            self._draw_image()
        else:
            self._draw_cloud()

    def set_image(self, image: np.ndarray) -> None:
        """更新激光图像（BGR 或灰度）。"""
        self._image = image
        if self._mode.get() == "image":
            self._draw_image()

    def set_points(self, points: np.ndarray) -> None:
        """更新点云 (N, 3)，单位 m。"""
        self._points = points
        if self._mode.get() == "cloud":
            self._draw_cloud()

    def load_image_file(self, path: str) -> bool:
        """从文件加载图像并显示。"""
        if not os.path.exists(path):
            return False
        try:
            import cv2
            img = cv2.imread(path)
        except Exception:  # noqa: BLE001
            return False
        if img is None:
            return False
        self.set_image(img)
        return True

    # ------------------------------------------------------------------
    def _draw_image(self) -> None:
        if self._fig is None:
            return
        self._ax.clear()
        self._ax.set_axis_off()
        if self._image is not None:
            img = self._image
            if img.ndim == 3:
                img = img[:, :, ::-1]  # BGR -> RGB
            self._ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            self._ax.set_title("激光图像")
        else:
            self._ax.text(0.5, 0.5, "暂无图像", ha="center", va="center")
        self._canvas.draw_idle()

    def _draw_cloud(self) -> None:
        if self._fig is None:
            return
        self._ax.clear()
        if self._points is not None and len(self._points):
            pts = self._points
            # 依高度 z 着色，直观看出形状
            z = pts[:, 2]
            sc = self._ax.scatter(pts[:, 0], pts[:, 1], c=z, s=0.5,
                                  cmap="viridis")
            self._fig.colorbar(sc, ax=self._ax, shrink=0.7, label="z (m)")
            self._ax.set_aspect("equal", adjustable="datalim")
            self._ax.set_title(f"点云 {len(pts)} 点")
            self._ax.set_xlabel("x (m)")
            self._ax.set_ylabel("y (m)")
        else:
            self._ax.text(0.5, 0.5, "暂无点云", ha="center", va="center")
            self._ax.set_axis_off()
        self._fig.tight_layout()
        self._canvas.draw_idle()
