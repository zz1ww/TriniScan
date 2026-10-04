"""中心线提取单元测试（用合成高斯条纹验证亚像素精度）。"""
import numpy as np
import pytest

from triniscan.extraction.centerline import (
    available_methods,
    extract_centerline,
    extract_gray_centroid_fast,
    extract_peak,
    extract_steger,
    preprocess,
)


def _synth_stripe(height=100, width=80, center_v=50.3, sigma=2.0,
                  amplitude=200.0, background=10.0):
    """生成一条水平高斯条纹（列方向），返回灰度图与真实中心。"""
    v = np.arange(height, dtype=np.float64)[:, None]
    img = background + amplitude * np.exp(
        -(v - center_v) ** 2 / (2 * sigma ** 2)
    )
    img = np.repeat(img, width, axis=1)
    return np.clip(img, 0, 255).astype(np.uint8), center_v


def test_preprocess_gray():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out = preprocess(img, sigma=0)
    assert out.shape == (10, 10)
    assert out.dtype == np.float64


def test_preprocess_roi_masks():
    img = np.full((20, 20), 255, dtype=np.uint8)
    out = preprocess(img, roi=(0, 0, 10, 10), sigma=0)
    assert out[5, 5] > 0
    assert out[15, 15] == 0


def test_gray_centroid_accuracy():
    img, true_v = _synth_stripe(center_v=50.3, sigma=2.0)
    pts = extract_gray_centroid_fast(img, threshold=50)
    assert len(pts) == img.shape[1]
    v_values = pts[:, 1]
    # 亚像素精度：均值误差 < 0.1 px
    assert abs(v_values.mean() - true_v) < 0.1


def test_steger_accuracy():
    # Steger 法应达到优于 0.05 px 的亚像素精度
    img, true_v = _synth_stripe(center_v=50.3, sigma=2.0)
    pts = extract_steger(img, threshold=50, sigma=2.0)
    assert len(pts) == img.shape[1]      # 每列一个中心
    assert abs(pts[:, 1].mean() - true_v) < 0.05


@pytest.mark.parametrize("center_v", [48.1, 49.7, 50.0, 50.3, 50.85, 51.2])
def test_steger_subpixel_various_centers(center_v):
    img, true_v = _synth_stripe(center_v=center_v, sigma=2.0)
    pts = extract_steger(img, threshold=50, sigma=2.0)
    assert abs(pts[:, 1].mean() - true_v) < 0.05


def test_steger_beats_peak():
    img, true_v = _synth_stripe(center_v=50.3)
    steger = extract_steger(img, threshold=50, sigma=2.0)
    peak = extract_peak(img, threshold=50)
    err_steger = abs(steger[:, 1].mean() - true_v)
    err_peak = abs(peak[:, 1].mean() - true_v)
    assert err_steger < err_peak


def test_peak_coarse():
    img, true_v = _synth_stripe(center_v=50.3)
    pts = extract_peak(img, threshold=50)
    assert len(pts) == img.shape[1]
    assert np.all(pts[:, 1] == 50)  # 整数峰值


def test_extract_centerline_dispatch():
    img, _ = _synth_stripe()
    cfg = {"method": "gray_centroid", "threshold": 50, "smooth_sigma": 1.0}
    pts = extract_centerline(img, cfg)
    assert pts.ndim == 2 and pts.shape[1] == 2


def test_extract_centerline_bad_method():
    img, _ = _synth_stripe()
    with pytest.raises(ValueError):
        extract_centerline(img, {"method": "nope"})


def test_available_methods():
    methods = available_methods()
    assert "steger" in methods
    assert "gray_centroid" in methods
    assert "peak" in methods


def test_empty_image():
    img = np.zeros((10, 10), dtype=np.uint8)
    pts = extract_gray_centroid_fast(img, threshold=50)
    assert pts.shape == (0, 2)
