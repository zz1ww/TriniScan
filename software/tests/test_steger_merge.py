"""中心线 Steger 归并策略测试（column vs normal）。"""
import numpy as np

from triniscan.extraction._steger import steger_center
from triniscan.extraction.centerline import extract_steger


def _synth_stripe(height=120, width=90, center_v=60.3, sigma=2.0,
                  amplitude=200.0, background=10.0):
    v = np.arange(height, dtype=np.float64)[:, None]
    img = background + amplitude * np.exp(
        -(v - center_v) ** 2 / (2 * sigma ** 2))
    img = np.repeat(img, width, axis=1)
    return np.clip(img, 0, 255).astype(np.uint8), center_v


def test_column_merge_one_per_column():
    img, true_v = _synth_stripe()
    pts = steger_center(img, sigma=2.0, threshold=50, merge="column")
    assert len(pts) == img.shape[1]
    assert abs(pts[:, 1].mean() - true_v) < 0.05


def test_normal_merge_default_equals_column_for_horizontal():
    img, _ = _synth_stripe()
    col = steger_center(img, sigma=2.0, threshold=50, merge="column")
    nor = steger_center(img, sigma=2.0, threshold=50, merge="normal")
    # 水平条纹下，两种策略都应给出每列一个中心
    assert len(nor) == len(col)


def test_normal_merge_handles_two_lines():
    """同一列出现两条水平条纹时，normal 策略应同时保留两条线的中心。"""
    height, width = 100, 40
    v = np.arange(height, dtype=np.float64)[:, None]
    img = (10 + 200 * np.exp(-(v - 30.0) ** 2 / 8.0)
           + 200 * np.exp(-(v - 70.0) ** 2 / 8.0))
    img = np.repeat(img, width, axis=1).astype(np.uint8)

    col = steger_center(img, sigma=2.0, threshold=50, merge="column")
    nor = steger_center(img, sigma=2.0, threshold=50, merge="normal",
                        min_sep_px=3.0)
    # column 每列只能保留一个中心（丢一条线）
    assert len(col) == width
    # normal 应同时保留上、下两条线：存在接近 v=30 与 v=70 的两簇中心
    rows = nor[:, 1]
    assert np.any(np.abs(rows - 30.0) < 2.0)
    assert np.any(np.abs(rows - 70.0) < 2.0)


def test_extract_steger_merge_param():
    img, true_v = _synth_stripe()
    pts = extract_steger(img, threshold=50, sigma=2.0, merge="normal")
    assert len(pts) >= 1
    assert abs(pts[:, 1].mean() - true_v) < 0.5
