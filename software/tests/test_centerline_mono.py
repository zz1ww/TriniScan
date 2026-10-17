"""黑白相机 + 单通道图像下的中心线提取测试。

覆盖：单通道图像走 gray 分支（即使 channel=red_laser 也不报错/静默失效）、
多分辨率（720p/1080p）下提取精度一致。
"""
import numpy as np
import pytest

from triniscan.extraction.centerline import (
    extract_centerline,
    extract_row_peak,
    preprocess,
)


class _Cfg:
    def __init__(self, d):
        self._d = d

    def get(self, k, default=None):
        return self._d.get(k, default)


def test_preprocess_single_channel_is_gray():
    img = np.full((20, 20), 123, dtype=np.uint8)
    out = preprocess(img, sigma=0, channel="gray")
    assert out.shape == (20, 20)
    assert out.dtype == np.float64
    assert np.allclose(out, 123.0)


@pytest.mark.parametrize("channel", ["red_laser", "green_laser", "r", "g", "b"])
def test_preprocess_single_channel_ignores_color_channel(channel):
    """单通道图像给颜色通道配置时，应自动降级为 gray（含红激光差分）。"""
    img = np.full((20, 20), 100, dtype=np.uint8)
    out = preprocess(img, sigma=0, channel=channel)
    assert out.shape == (20, 20)
    # 降级后应等于原灰度值（而不是颜色差分得到的 0）
    assert np.allclose(out, 100.0)


def test_preprocess_bgr_still_uses_red_laser():
    """三通道（彩色）图像下 red_laser 差分仍正常工作。"""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 2] = 200                 # R
    img[..., 1] = 50                  # G
    img[..., 0] = 50                  # B
    out = preprocess(img, sigma=0, channel="red_laser")
    assert out.max() > 0


def _synth_vertical_stripe(width=1920, height=200, center_u=900.3,
                           sigma=2.0, amplitude=200.0, background=10.0):
    """生成竖直线激光条纹（逐行扫描对应场景），返回 (图像, 真实中心 u)。"""
    u = np.arange(width, dtype=np.float64)[None, :]
    img = background + amplitude * np.exp(
        -(u - center_u) ** 2 / (2 * sigma ** 2)
    )
    img = np.repeat(img, height, axis=0)
    return np.clip(img, 0, 255).astype(np.uint8), center_u


@pytest.mark.parametrize("width", [1280, 1920])
def test_row_peak_multires_accuracy(width):
    """720p / 1080p 下逐行峰值的亚像素精度都应在 0.3 px 以内。"""
    scale = width / 1920.0
    center = 900.3 * scale
    img, truth = _synth_vertical_stripe(width=width, center_u=center)
    centers = extract_row_peak(img, threshold=60.0, subpixel=True)
    assert len(centers) > 0
    err = np.abs(centers[:, 0] - truth)
    assert float(np.median(err)) < 0.3


def test_extract_centerline_mono_image_via_config():
    """黑白相机：单通道图像 + channel=red_laser 配置，应正常出点。"""
    img, _ = _synth_vertical_stripe(width=1920)
    cfg = _Cfg({
        "method": "row_peak",
        "threshold": 60.0,
        "smooth_sigma": 1.5,
        "channel": "red_laser",       # 黑白相机下应自动降级
        "subpixel": True,
    })
    centers = extract_centerline(img, cfg)
    assert len(centers) > 0


def test_extract_centerline_enhance_disabled_by_default():
    """未配置 enhance 时，行为与未加增强一致（零回归）。"""
    img, truth = _synth_vertical_stripe(width=1280)
    cfg = _Cfg({
        "method": "row_peak",
        "threshold": 60.0,
        "smooth_sigma": 1.5,
        "channel": "gray",
        "subpixel": True,
    })
    centers = extract_centerline(img, cfg)
    assert len(centers) > 0
    err = float(np.median(np.abs(centers[:, 0] - truth)))
    assert err < 0.3


def test_extract_centerline_enhance_enabled_still_works():
    """开启条纹增强（扁平键）后，干净场景仍正常出点且精度不劣化。"""
    img, truth = _synth_vertical_stripe(width=1280)
    cfg = _Cfg({
        "method": "row_peak",
        "threshold": 60.0,
        "smooth_sigma": 1.5,
        "channel": "gray",
        "subpixel": True,
        "enhance_enable": True,
        "enhance_threshold": 60.0,
        "enhance_blur_ksize": 5,
        "enhance_window": 9,
    })
    centers = extract_centerline(img, cfg)
    assert len(centers) > 0
    err = float(np.median(np.abs(centers[:, 0] - truth)))
    assert err < 0.5


def test_extract_centerline_enhance_legacy_nested_compat():
    """旧嵌套写法 ``enhance: {enable: ...}`` 仍应向后兼容。"""
    img, truth = _synth_vertical_stripe(width=1280)
    cfg = _Cfg({
        "method": "row_peak",
        "threshold": 60.0,
        "smooth_sigma": 1.5,
        "channel": "gray",
        "subpixel": True,
        "enhance": {"enable": True, "threshold": 60.0,
                    "blur_ksize": 5, "window": 9},
    })
    centers = extract_centerline(img, cfg)
    assert len(centers) > 0
    err = float(np.median(np.abs(centers[:, 0] - truth)))
    assert err < 0.5


def test_flat_and_nested_enhance_equivalent():
    """扁平键与旧嵌套写法应产生完全一致的结果。"""
    img, _ = _synth_vertical_stripe(width=1280)
    common = {"method": "row_peak", "threshold": 60.0,
              "smooth_sigma": 1.5, "channel": "gray", "subpixel": True}
    flat = extract_centerline(img, _Cfg(dict(common, **{
        "enhance_enable": True, "enhance_threshold": 60.0,
        "enhance_blur_ksize": 5, "enhance_window": 9})))
    nested = extract_centerline(img, _Cfg(dict(common, **{
        "enhance": {"enable": True, "threshold": 60.0,
                    "blur_ksize": 5, "window": 9}})))
    assert np.allclose(flat, nested)
