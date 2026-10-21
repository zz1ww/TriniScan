"""双图差分激光标定测试（合成数据）。

覆盖两个层面：
1. ``diff_laser_images`` 对配对可信度的校验（真/假配对）；
2. ``extract_centerline_from_diff`` 的几何精度与鲁棒性。
"""
import cv2
import numpy as np
import pytest

from triniscan.algorithms.calibration import (
    diff_laser_images,
    extract_centerline_from_diff,
)


# ----------------------------------------------------------------------
# 合成工具
# ----------------------------------------------------------------------
def _checkerboard(w=1280, h=800, square=40, seed=0):
    """合成一张带轻微噪声的黑白棋盘（灰度）。"""
    rng = np.random.default_rng(seed)
    n = ((np.indices((h // square + 1, w // square + 1)).sum(0) % 2)
         * 200 + 25).astype(np.uint8)
    img = cv2.resize(n, (w, h), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (3, 3), 1)
    noisy = img.astype(np.int16) + rng.normal(0, 1.5, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _with_line(img, slope=0.3, x0=200, x1=1100, y0=400, value=255):
    """在图上叠加一条斜直线（模拟激光）。"""
    out = img.copy()
    for x in range(x0, x1):
        y = int(y0 + slope * (x - x0))
        cv2.circle(out, (x, y), 2, value, -1)
    return cv2.GaussianBlur(out, (3, 3), 1)


# ----------------------------------------------------------------------
# 真配对
# ----------------------------------------------------------------------
def test_true_pair_is_accepted_and_line_recovered():
    dark = _checkerboard()
    lit = _with_line(dark, slope=0.3)
    res = diff_laser_images(dark, lit)

    assert res.ok, res.reason
    assert res.diff is not None
    # 激光只占极少像素
    assert res.bright_ratio < 0.02

    line = extract_centerline_from_diff(res.diff, threshold=60)
    assert len(line) > 500

    a = np.polyfit(line[:, 0], line[:, 1], 1)
    assert a[0] == pytest.approx(0.3, abs=0.02)
    resid = np.abs(np.polyval(a, line[:, 0]) - line[:, 1])
    assert np.median(resid) < 1.5


def test_diff_removes_static_background():
    """差分必须把棋盘完全扣掉：差分图里不应再看到棋盘结构。"""
    dark = _checkerboard()
    lit = _with_line(dark)
    res = diff_laser_images(dark, lit)
    assert res.ok

    # 差分图的中位数亮度应接近 0（棋盘被扣干净）
    assert float(np.median(res.diff)) == 0
    ok, _ = cv2.findChessboardCornersSB(
        res.diff, (11, 8),
        flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    assert not ok, "差分图中残留了棋盘结构"


# ----------------------------------------------------------------------
# 假配对必须被拦下
# ----------------------------------------------------------------------
def test_identical_pair_rejected():
    dark = _checkerboard()
    res = diff_laser_images(dark, dark.copy())
    assert not res.ok
    assert "几乎相同" in res.reason


def test_brightness_drift_rejected():
    """自动曝光重新收敛会让整体亮度偏移，差分只剩弥散残影。"""
    dark = _checkerboard()
    fake = np.clip(dark.astype(np.float32) * 1.05 + 5, 0, 255).astype(np.uint8)
    res = diff_laser_images(dark, fake)
    assert not res.ok
    assert "亮度" in res.reason


def test_board_moved_rejected():
    """拍两张之间板移动：差分亮区远超一条线。"""
    dark = _checkerboard()
    lit = _with_line(dark)
    moved = np.roll(lit, 30, axis=1)
    res = diff_laser_images(dark, moved)
    assert not res.ok
    assert res.bright_ratio > 0.05


def test_size_mismatch_rejected():
    dark = _checkerboard(1280, 800)
    lit = _checkerboard(1280, 720)
    res = diff_laser_images(dark, lit)
    assert not res.ok
    assert "尺寸不一致" in res.reason


def test_empty_rejected():
    assert not diff_laser_images(None, None).ok


# ----------------------------------------------------------------------
# 提取器鲁棒性
# ----------------------------------------------------------------------
def test_extractor_ignores_thick_blobs():
    """块状物体（手/木框）不能被当成激光线。"""
    diff = np.zeros((400, 600), dtype=np.uint8)
    # 一列细线
    for x in range(50, 550):
        diff[int(200 + 0.1 * (x - 50)), x] = 255
    # 一个大方块
    diff[50:200, 400:560] = 255

    line = extract_centerline_from_diff(diff, threshold=60)
    assert len(line) > 100
    # 方块占据 y=50..200，细线在 y≈205..250；被方块污染的点会落到
    # y<200 的区域。提取结果不应有任何点进入方块内部。
    assert (line[:, 1] >= 200).all(), "方块区域贡献了点"
    # 且提取的是一条直线（方块没把中心拉偏）
    a = np.polyfit(line[:, 0], line[:, 1], 1)
    assert a[0] == pytest.approx(0.1, abs=0.02)


def test_extractor_handles_broken_line():
    """激光穿过黑白格会碎成多段，仍应拼回同一条线。"""
    diff = np.zeros((300, 900), dtype=np.uint8)
    for x in range(0, 900, 1):
        if (x // 60) % 2 == 0:      # 每隔 60 列断一次
            diff[int(150 + 0.05 * x), x] = 255
    line = extract_centerline_from_diff(diff, threshold=60)
    # 只允许丢掉断裂处的一小部分
    assert len(line) > 300


def test_extractor_empty_when_no_line():
    diff = np.zeros((200, 200), dtype=np.uint8)
    assert len(extract_centerline_from_diff(diff)) == 0
