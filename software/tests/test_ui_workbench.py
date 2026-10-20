"""新版工作台界面的测试（P1：骨架 + 实时画面组件）。

不依赖真实相机：用假的 ``open_camera`` 返回预置帧。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from triniscan.core.config import Config
from triniscan.ui.video import FrameSource, VideoView, _to_rgb


def _cfg() -> Config:
    path = os.path.join(os.path.dirname(__file__), "..", "config",
                        "default.yaml")
    return Config.load(os.path.abspath(path))


@pytest.fixture
def root():
    import tkinter as tk
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("无显示环境，跳过界面测试")
    r.withdraw()
    yield r
    r.destroy()


class FakeCamera:
    """按需返回帧的假相机；记录被关闭过。"""

    def __init__(self, shape=(48, 64)):
        self.shape = shape
        self.closed = False
        self.reads = 0

    def read(self):
        self.reads += 1
        frame = np.full((*self.shape, 3), self.reads % 256, dtype=np.uint8)
        time.sleep(0.002)
        return frame

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# 图像规范化
# ---------------------------------------------------------------------------
def test_to_rgb_grayscale_is_repeated():
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    rgb = _to_rgb(gray)
    assert rgb.shape == (2, 2, 3)
    assert np.array_equal(rgb[..., 0], rgb[..., 1])


def test_to_rgb_bgr_is_reversed():
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0] = (1, 2, 3)                    # B, G, R
    assert tuple(_to_rgb(bgr)[0, 0]) == (3, 2, 1)


def test_to_rgb_rejects_bad_shape():
    with pytest.raises(ValueError):
        _to_rgb(np.zeros((2, 2, 5), dtype=np.uint8))


# ---------------------------------------------------------------------------
# FrameSource：单槽位、丢旧帧、正常关闭
# ---------------------------------------------------------------------------
def test_frame_source_yields_frames_and_closes():
    cam = FakeCamera()
    src = FrameSource(lambda: cam)
    try:
        src.start()
        deadline = time.time() + 3.0
        frame = None
        while frame is None and time.time() < deadline:
            frame = src.latest()
            time.sleep(0.01)
        assert frame is not None, "3 秒内没有收到帧"
        assert frame.shape == (48, 64, 3)
        assert src.is_open()
    finally:
        src.stop()
    assert cam.closed, "停止后相机必须被关闭"
    assert not src.running


def test_frame_source_single_slot_keeps_latest():
    """槽位容量为 1：连续写入后只能取到一帧，且是最新的。"""
    src = FrameSource(lambda: FakeCamera())
    for i in range(5):
        src._put(np.full((1, 1), i, dtype=np.uint8))
    first = src.latest()
    assert int(first[0, 0]) == 4, "应保留最新一帧"
    assert src.latest() is None, "槽位应已清空"


def test_frame_source_reports_open_failure():
    errors = []

    def _fail():
        raise RuntimeError("no camera")

    src = FrameSource(_fail, on_error=errors.append)
    src.start()
    src._thread.join(timeout=3.0)
    assert errors and "no camera" in errors[0]


# ---------------------------------------------------------------------------
# VideoView
# ---------------------------------------------------------------------------
def test_video_view_shows_and_clears(root):
    view = VideoView(root)
    view.pack(fill="both", expand=True)
    root.geometry("400x300")
    root.update()

    view.show(np.zeros((120, 160, 3), dtype=np.uint8))
    root.update()
    assert view._photo is not None, "PhotoImage 必须被持有，否则画面空白"
    assert view.frame is not None

    view.clear()
    root.update()
    assert view.frame is None
    assert view._photo is None


def test_video_view_accepts_grayscale(root):
    view = VideoView(root)
    view.pack(fill="both", expand=True)
    root.geometry("300x200")
    root.update()
    view.show(np.full((60, 80), 128, dtype=np.uint8))
    root.update()
    assert view._photo is not None


def test_video_view_overlay_is_applied(root):
    view = VideoView(root)
    view.pack(fill="both", expand=True)
    root.geometry("300x200")
    root.update()

    called = []

    def overlay(bgr):
        called.append(bgr.shape)
        return bgr

    view.set_overlay(overlay)
    view.show(np.zeros((60, 80, 3), dtype=np.uint8))
    root.update()
    assert called == [(60, 80, 3)]


def test_video_view_overlay_error_does_not_crash(root):
    view = VideoView(root)
    view.pack(fill="both", expand=True)
    root.geometry("300x200")
    root.update()

    def bad(_bgr):
        raise RuntimeError("boom")

    view.set_overlay(bad)
    view.show(np.zeros((60, 80, 3), dtype=np.uint8))
    root.update()                           # 不应抛异常
    assert view._photo is not None


# ---------------------------------------------------------------------------
# Workbench 骨架
# ---------------------------------------------------------------------------
def test_workbench_builds_panels_and_pages(root):
    from triniscan.ui.workbench_measure import MeasureWorkbench

    wb = MeasureWorkbench(root, _CfgStore())
    try:
        root.update()
        assert "camera" in wb._panels
        assert wb._current_page == "video"
        assert wb.panel("camera") is not None
    finally:
        wb.stop_preview()


def test_workbench_panel_toggle(root):
    from triniscan.ui.workbench_measure import MeasureWorkbench

    wb = MeasureWorkbench(root, _CfgStore())
    panel = wb._panels["camera"]
    if not panel.expanded:
        panel.set_expanded(True)
    assert panel.expanded
    panel.set_expanded(False)
    assert not panel.expanded


def _CfgStore():
    from triniscan.core.config_store import ConfigStore
    return ConfigStore(_cfg())
