"""激光条纹增强（双阈值平滑 + 峰值窗口掩膜）测试。

覆盖：基础算子、分辨率缩放、反光簇抑制、以及 720p/1080p 一致性。
"""
import numpy as np
import pytest

from triniscan.extraction.stripe_enhance import (
    blur_odd,
    double_threshold_smooth,
    enhance_stripe,
    peak_window_mask,
    scale_px_param,
    threshold_to_zero,
)


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------
def test_threshold_to_zero_keeps_above():
    g = np.array([[0.0, 50.0, 100.0, 200.0]])
    out = threshold_to_zero(g, 60.0)
    assert out.tolist() == [[0.0, 0.0, 100.0, 200.0]]


def test_blur_odd_returns_same_for_small_kernel():
    g = np.ones((5, 5)) * 10.0
    out = blur_odd(g, 1)
    assert out.shape == g.shape
    assert np.allclose(out, g)


def test_blur_odd_makes_even_kernel_odd():
    g = np.ones((7, 7)) * 10.0
    out = blur_odd(g, 4)          # 4 -> 5
    assert out.shape == g.shape
    # 常量图平滑后仍为常量
    assert np.allclose(out, 10.0)


def test_blur_odd_averages_isolated_bright_pixel():
    g = np.zeros((11, 11))
    g[5, 5] = 121.0               # 单亮点
    out = blur_odd(g, 5)          # 5x5 均值 -> 121/25 ≈ 4.84
    assert out[5, 5] < 10.0


def test_double_threshold_smooth_chain():
    g = np.zeros((20, 20))
    g[:, 10] = 200.0
    out = double_threshold_smooth(g, 60.0, 3)
    assert out.shape == g.shape
    assert out[:, 10].max() > 0


# ---------------------------------------------------------------------------
# 分辨率缩放（720 vs 1080）
# ---------------------------------------------------------------------------
def test_scale_px_param_1080():
    assert scale_px_param(9, 1920) == 9
    assert scale_px_param(5, 1920) == 5


def test_scale_px_param_720():
    assert scale_px_param(9, 1280) == 6
    assert scale_px_param(5, 1280) == 3


def test_scale_px_param_min_one():
    assert scale_px_param(1, 100) == 1
    assert scale_px_param(0.1, 10) == 1


def test_scale_px_param_invalid_width():
    assert scale_px_param(9, 0) == 9


# ---------------------------------------------------------------------------
# 峰值窗口掩膜
# ---------------------------------------------------------------------------
def test_peak_window_mask_keeps_peak_neighbourhood():
    # 源图中只有峰值列(40)有值；掩膜只决定"哪些列被保留"，
    # 不改变像素值。因此验证 35..45 被保留、之外被清零。
    g = np.zeros((50, 100))
    g[:, 40] = 200.0
    out = peak_window_mask(g, 5)
    assert out[:, 40].max() == 200.0
    nz = np.nonzero(out[0])[0]
    assert nz.min() == 40 and nz.max() == 40     # 仅 40 列有非零值
    # 掩膜本身：单峰(40)时，38/42 在 ±5 窗口内应保留，33/47 在外应清零
    g2 = np.zeros((50, 100))
    g2[:, 38] = 100.0
    g2[:, 40] = 200.0        # 单峰在 40
    g2[:, 42] = 100.0
    g2[:, 33] = 100.0
    g2[:, 47] = 100.0
    out2 = peak_window_mask(g2, 5)
    assert out2[:, 38].max() == 100.0
    assert out2[:, 42].max() == 100.0
    assert out2[:, 33].max() == 0.0
    assert out2[:, 47].max() == 0.0


def test_peak_window_mask_disabled():
    g = np.ones((10, 10)) * 5.0
    assert np.allclose(peak_window_mask(g, 0), g)


def test_peak_window_mask_min_peak_zeroes_dim_rows():
    g = np.zeros((10, 20))
    g[:, 5] = 3.0                 # 远低于 min_peak
    out = peak_window_mask(g, 5, min_peak=50.0)
    assert out.max() == 0.0


# ---------------------------------------------------------------------------
# 端到端：抑制反光簇
# ---------------------------------------------------------------------------
def _scene(width=1920, height=200, laser_col=48, blob=None):
    """激光为贯穿全图的竖直亮带；blob 为可选的 (rows, cols, val) 反光块。"""
    g = np.zeros((height, width), dtype=np.float64)
    g[:, laser_col:laser_col + 5] = 180.0
    if blob is not None:
        r0, r1, c0, c1, val = blob
        g[r0:r1, c0:c1] = val
    return g


def test_enhance_suppresses_dim_wide_reflection():
    """较宽但比激光暗的反光块，应被峰值窗口掩膜清除。"""
    g = _scene(blob=(60, 90, 900, 930, 120.0))
    out = enhance_stripe(g, threshold=60.0, blur_ksize=5, window=9)
    assert out[0, 50] > 0                 # 激光保留
    assert out[75, 915] == 0.0            # 反光被清除


def test_enhance_reflection_near_laser_is_not_separable():
    """已知局限：反光块与激光间距 < window 时，窗口掩膜无法分离。

    这属于算法边界（Cyclop 同样如此），需要靠物理门限/连通域兜底。
    本测试把该局限固定下来，避免误以为窗口掩膜万能。
    """
    g = _scene(laser_col=48, blob=(60, 90, 62, 82, 250.0))  # 间距 ~14px
    out = enhance_stripe(g, threshold=60.0, blur_ksize=5, window=9)
    assert out[0, 50] > 0                 # 激光仍在
    assert out[75, 70] > 0                # 紧邻的反光无法被窗口剔除（已知）


def test_enhance_laser_survives_at_all_rows():
    """无论是否存在反光，激光各行都应保留。"""
    g = _scene(blob=(60, 90, 900, 930, 120.0))
    out = enhance_stripe(g, threshold=60.0, blur_ksize=5, window=9)
    assert (out[:, 50] > 0).all()


# ---------------------------------------------------------------------------
# 720 / 1080 分辨率一致性（变态采样）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1280, 1920])
def test_enhance_multires_keeps_laser_removes_reflection(width):
    """同一几何场景在 720p 与 1080p 下，增强结果语义一致。"""
    scale = width / 1920.0
    laser_col = int(round(48 * scale))
    blob = (60, 90, int(round(900 * scale)), int(round(930 * scale)), 120.0)
    g = _scene(width=width, height=200, laser_col=laser_col, blob=blob)
    out = enhance_stripe(g, threshold=60.0, blur_ksize=5, window=9,
                         scale_to_width=width)
    assert out[0, laser_col + 2] > 0
    assert out[75, int(round(915 * scale))] == 0.0


def test_enhance_disable_window_keeps_reflection():
    """关闭窗口掩膜后，反光块应仍在（对照实验）。"""
    g = _scene(blob=(60, 90, 900, 930, 250.0))
    out = enhance_stripe(g, threshold=60.0, blur_ksize=1, window=9,
                         enable_window=False, enable_double_threshold=False)
    assert out[75, 915] == 250.0


def test_enhance_empty_image():
    out = enhance_stripe(np.zeros((0, 0)))
    assert out.size == 0
