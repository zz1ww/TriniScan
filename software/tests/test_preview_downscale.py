"""预览面板的降采样测试（需图形环境，无则跳过）。

1080p 全尺寸逐帧重绘是实时取景卡顿的主因之一；预览必须先降采样
再交给 matplotlib，同时不能改变检测用的原图。
"""
import numpy as np
import pytest
import tkinter as tk


@pytest.fixture
def panel():
    import triniscan.ui.preview as P
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无显示环境，跳过界面测试")
    root.withdraw()
    try:
        yield P.PreviewPanel(root)
    finally:
        root.destroy()


def test_large_image_downscaled(panel):
    from triniscan.ui.preview import _PREVIEW_MAX_SIDE
    img = np.zeros((1080, 1920, 3), np.uint8)
    panel.set_image(img)
    assert max(panel._image.shape[:2]) <= _PREVIEW_MAX_SIDE


def test_custom_max_side(panel):
    img = np.zeros((1080, 1920, 3), np.uint8)
    panel.set_image(img, max_side=200)
    assert max(panel._image.shape[:2]) <= 200


def test_small_image_not_upscaled(panel):
    img = np.zeros((300, 400, 3), np.uint8)
    panel.set_image(img)
    assert panel._image.shape[:2] == (300, 400)


def test_aspect_ratio_preserved(panel):
    img = np.zeros((1080, 1920, 3), np.uint8)
    panel.set_image(img, max_side=720)
    h, w = panel._image.shape[:2]
    assert abs(w / h - 1920 / 1080) < 0.02


def test_grayscale_supported(panel):
    img = np.zeros((1080, 1920), np.uint8)
    panel.set_image(img)
    assert panel._image.ndim == 2
    assert max(panel._image.shape[:2]) <= 720
