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
            self._cbar = None
            return

        self._fig = Figure(figsize=(4, 3), dpi=100)
        # 两个坐标区：2D（图像）与 3D（点云），按需切换显示
        self._ax_img = self._fig.add_subplot(111)
        self._ax_img.set_axis_off()
        self._ax_cloud = self._fig.add_subplot(111, projection="3d")
        self._ax_cloud.set_visible(False)
        self._cbar = None        # 点云色条句柄（刷新时先移除，避免叠加）

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
        self._ax_cloud.set_visible(False)
        self._ax_img.set_visible(True)
        self._ax_img.clear()
        self._ax_img.set_axis_off()
        if self._image is not None:
            img = self._image
            if img.ndim == 3:
                img = img[:, :, ::-1]  # BGR -> RGB
            self._ax_img.imshow(img, cmap=None if img.ndim == 3 else "gray")
            self._ax_img.set_title("激光图像")
        else:
            self._ax_img.text(0.5, 0.5, "暂无图像",
                              ha="center", va="center")
        self._canvas.draw_idle()

    def _draw_cloud(self) -> None:
        if self._fig is None:
            return
        self._ax_img.set_visible(False)
        self._ax_cloud.set_visible(True)
        ax = self._ax_cloud
        # 先移除旧色条（必须在 clear 之前，否则句柄失效）
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        ax.clear()
        if self._points is not None and len(self._points):
            ax.set_axis_on()
            pts = np.asarray(self._points, dtype=float)
            # 依高度 z 着色，直观看出形状；3D 视图可用鼠标左键
            # 拖拽旋转、右键缩放（工具栏亦提供快捷按钮）。
            z = pts[:, 2]
            sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=z, s=1.0,
                            cmap="viridis", depthshade=False)
            self._cbar = self._fig.colorbar(sc, ax=ax, shrink=0.6,
                                            pad=0.1, label="z (m)")
            # 三轴等比例，避免形状被拉伸
            _set_axes_equal(ax, pts)
            ax.set_title(f"点云 {len(pts)} 点")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            ax.view_init(elev=20, azim=-60)
        else:
            ax.text2D(0.5, 0.5, "暂无点云", transform=ax.transAxes,
                      ha="center", va="center")
            ax.set_axis_off()
        self._fig.tight_layout()
        self._canvas.draw_idle()


def _set_axes_equal(ax, pts: np.ndarray) -> None:
    """让 3D 三轴按相同比例显示，保证形状不被拉伸。"""
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max()) / 2.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 0.05
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
