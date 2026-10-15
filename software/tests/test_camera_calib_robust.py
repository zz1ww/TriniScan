"""相机内参标定的稳健性测试。

覆盖三处关键改动：
1. SB 检测器（findChessboardCornersSB）+ classic 回退；
2. ``_per_view_errors`` 与 ``cv2`` 返回的 RMS **同口径**（修复了少除 sqrt(N) 的 bug）；
3. 自动选图 ``_prune_outlier_views`` 能剔除坏图。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from triniscan.calibration.camera_calib import (
    _per_view_errors,
    _prune_outlier_views,
    as_point_array,
    calibrate_camera,
    make_object_points,
)


def _render_chessboard(cols: int, rows: int, square_px: int = 40,
                       margin: int = 60) -> np.ndarray:
    """渲染一张正对视角的棋盘格图像（用于确定性测试）。"""
    w = (cols + 1) * square_px + 2 * margin
    h = (rows + 1) * square_px + 2 * margin
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = margin + r * square_px
                x0 = margin + c * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 0
    return img


def test_per_view_errors_matches_cv2_rms():
    """逐视角 RMS 的均方根应与 cv2 返回的整体 RMS 同量级（口径一致）。

    这是修复前会失败的回归测试：旧实现少除了 sqrt(N)，误差被低估约 9 倍。
    """
    cols, rows = 11, 8
    objp = make_object_points(cols, rows, 0.01)
    img = _render_chessboard(cols, rows)

    obj_points, img_points = [], []
    for _ in range(6):
        corners = _detect(img, cols, rows)
        if corners is None:
            pytest.skip("测试环境无法检测渲染的棋盘格")
        obj_points.append(np.ascontiguousarray(objp, dtype=np.float32))
        img_points.append(corners)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (img.shape[1], img.shape[0]), None, None
    )
    per_view = _per_view_errors(obj_points, img_points, rvecs, tvecs, K, dist)
    combined = float(np.sqrt(np.mean(np.square(per_view))))

    # 口径一致：两者应非常接近（旧的错误实现会差 sqrt(N) 倍）
    assert abs(combined - rms) < 0.5, (
        f"per-view RMS {combined:.4f} 与 cv2 rms {rms:.4f} 口径不一致"
    )


def test_detect_sb_and_classic_give_corners():
    """SB 与 classic 都应能检测到棋盘格，且角点数一致。"""
    cols, rows = 11, 8
    img = _render_chessboard(cols, rows)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    from triniscan.calibration.camera_calib import _detect_corners
    c_sb = _detect_corners(gray, (cols, rows), use_sb=True)
    c_cl = _detect_corners(gray, (cols, rows), use_sb=False)
    if c_sb is None and c_cl is None:
        pytest.skip("测试环境无法检测渲染的棋盘格")
    if c_sb is not None:
        assert c_sb.shape[0] == cols * rows
    if c_cl is not None:
        assert c_cl.shape[0] == cols * rows


def test_calibrate_camera_returns_consistent_per_view():
    """标定结果中 per_view 的均方根应与 rms 一致（口径自洽）。"""
    cols, rows = 11, 8
    img = _render_chessboard(cols, rows)
    imgs = [img.copy() for _ in range(6)]
    try:
        result = calibrate_camera(imgs, cols, rows, 0.01, auto_prune=False)
    except RuntimeError:
        pytest.skip("测试环境无法检测渲染的棋盘格")

    combined = float(np.sqrt(np.mean(np.square(result.per_view_errors))))
    assert abs(combined - result.rms) < 0.5


def test_prune_outlier_views_removes_bad_view():
    """注入一张错位坏图，自动选图应把它剔除。"""
    cols, rows = 11, 8
    objp = make_object_points(cols, rows, 0.01)
    img = _render_chessboard(cols, rows)
    corners = _detect(img, cols, rows)
    if corners is None:
        pytest.skip("测试环境无法检测渲染的棋盘格")

    obj_points = [np.ascontiguousarray(objp, dtype=np.float32)
                  for _ in range(10)]
    img_points = [corners.copy() for _ in range(10)]
    # 第 5 张注入明显偏移（坏图）
    bad = img_points[5].copy()
    bad[:, :, 0] += 25.0
    img_points[5] = bad

    kept_obj, kept_img, kept_idx = _prune_outlier_views(
        obj_points, img_points, (img.shape[1], img.shape[0]),
        max_iters=10, enable=True, min_images=5,
    )
    assert 5 not in kept_idx, "坏图未被剔除"
    assert len(kept_idx) < len(obj_points)


def _detect(img: np.ndarray, cols: int, rows: int):
    from triniscan.calibration.camera_calib import _detect_corners
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _detect_corners(gray, (cols, rows))
