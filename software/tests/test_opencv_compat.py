"""OpenCV 4.x / 5.x 点集形状兼容性测试。

背景
----
``cv2.norm`` / ``cv2.projectPoints`` 等函数要求二维点集为
``(N, 1, 2)``（``CV_32FC2``）。若某组点被拉平成 ``(2N,)``，OpenCV 会把
它当作 ``CV_32FC1``；与标准点集比较时报：

    Input type mismatch (expected: '_src1.type() == _src2.type()'),
    '_src1.type()' is 5 (CV_32FC1) must be equal to '_src2.type()' is 37 (CV_32FC2)

OpenCV 4.x 多数场景能容忍，5.x 严格校验。这里验证归一化函数
``as_point_array`` 与改造后的 ``_per_view_errors`` 在两种输入下都稳定。
"""
import cv2
import numpy as np
import pytest

from triniscan.calibration import camera_calib as CC
from triniscan.calibration.camera_calib import (
    as_point_array,
    make_object_points,
)


# ---------------------------------------------------------------------------
# as_point_array 归一化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(88, 2), (88, 1, 2), (176,), (88, 1, 2, 1)])
def test_as_point_array_normalises_shape(shape):
    n = int(np.prod(shape) // 2)
    arr = np.arange(n * 2, dtype=np.float32).reshape(shape)
    out = as_point_array(arr)
    assert out.shape == (n, 1, 2)
    assert out.dtype == np.float32


def test_as_point_array_accepts_list():
    out = as_point_array([[1.0, 2.0], [3.0, 4.0]])
    assert out.shape == (2, 1, 2)
    assert out.dtype == np.float32


def test_as_point_array_accepts_int_input():
    """整数输入也应被规范为 float32。"""
    out = as_point_array(np.array([[10, 20], [30, 40]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.reshape(-1, 2),
                               [[10, 20], [30, 40]])


def test_as_point_array_is_contiguous():
    out = as_point_array(np.arange(176, dtype=np.float32).reshape(88, 2)[::2])
    assert out.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# cv2.norm 通道一致性（回归）
# ---------------------------------------------------------------------------
def _chessboard_view(cols, rows, shift, size=(900, 700), sq=70):
    h, w = size[1], size[0]
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y = 40 + shift + r * sq
                x = 60 + c * sq
                img[y:y + sq, x:x + sq] = 0
    found, corners = cv2.findChessboardCorners(img, (cols, rows))
    if not found:
        return None
    return cv2.cornerSubPix(
        img, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


def test_norm_rejects_mismatched_channels():
    """确认 cv2.norm 对 C1/C2 不匹配确实会报错（问题本身）。"""
    a = np.zeros((10, 1, 2), np.float32)        # CV_32FC2
    b = np.zeros((20,), np.float32)             # CV_32FC1
    with pytest.raises(cv2.error):
        cv2.norm(a, b, cv2.NORM_L2)


def test_as_point_array_prevents_mismatch():
    """归一化后两者同为 CV_32FC2，cv2.norm 不再报错。"""
    a = np.zeros((10, 1, 2), np.float32)
    b_flat = np.zeros((20,), np.float32)
    # 归一化前会炸
    with pytest.raises(cv2.error):
        cv2.norm(a, b_flat, cv2.NORM_L2)
    # 归一化后正常
    val = cv2.norm(as_point_array(a), as_point_array(b_flat), cv2.NORM_L2)
    assert val == 0.0


# ---------------------------------------------------------------------------
# _per_view_errors 端到端（含畸形输入）
# ---------------------------------------------------------------------------
def _calibrate_views(n_views=5, cols=11, rows=8, flatten_idx=None):
    objs, imgs = [], []
    for k in range(n_views):
        c = _chessboard_view(cols, rows, shift=-k * 4)
        if c is None:
            continue
        objs.append(make_object_points(cols, rows, 0.01))
        imgs.append(c)
    if flatten_idx is not None and len(imgs) > flatten_idx:
        imgs[flatten_idx] = imgs[flatten_idx].reshape(-1)   # -> CV_32FC1
    return objs, imgs


def test_per_view_errors_normal_input():
    objs, imgs = _calibrate_views()
    assert len(objs) >= 3
    size = (900, 700)
    objs_n = [np.ascontiguousarray(o, np.float32).reshape(-1, 3)
              for o in objs]
    imgs_n = [as_point_array(p) for p in imgs]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objs_n, imgs_n, size, None, None)
    errs = CC._per_view_errors(objs_n, imgs_n, rvecs, tvecs, K, dist)
    assert len(errs) == len(objs_n)
    assert all(np.isfinite(e) and e >= 0 for e in errs)


def test_per_view_errors_flattened_input_does_not_crash():
    """畸形（被拉平）的角点输入也不应导致 cv2.norm 报错（回归）。"""
    objs, imgs = _calibrate_views(flatten_idx=2)
    assert len(objs) >= 3
    size = (900, 700)
    objs_n = [np.ascontiguousarray(o, np.float32).reshape(-1, 3)
              for o in objs]
    imgs_n = [as_point_array(p) for p in imgs]      # 归一化吸收畸变
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objs_n, imgs_n, size, None, None)
    errs = CC._per_view_errors(objs_n, imgs_n, rvecs, tvecs, K, dist)
    assert len(errs) == len(objs_n)
    assert all(np.isfinite(e) for e in errs)


def test_calibrate_camera_accepts_flattened_corner_list(monkeypatch):
    """即使传入的角点被拉平，calibrate_camera 也应正常完成。"""
    cols, rows = 11, 8
    images = []
    for k in range(5):
        h, w = 700, 900
        img = np.full((h, w), 255, np.uint8)
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    y = 40 - k * 4 + r * 70
                    x = 60 + c * 70
                    img[y:y + 70, x:x + 70] = 0
        images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    calls = {"n": 0}
    orig = CC._detect_corners

    def patched(gray, pattern_size):
        out = orig(gray, pattern_size)
        calls["n"] += 1
        if out is not None and calls["n"] == 2:
            return out.reshape(-1)          # 故意拉平一个视角
        return out

    monkeypatch.setattr(CC, "_detect_corners", patched)
    result = CC.calibrate_camera(images, cols, rows, 0.01)
    assert result.num_images >= 3
    assert np.isfinite(result.rms)
    assert all(np.isfinite(e) for e in result.per_view_errors)
