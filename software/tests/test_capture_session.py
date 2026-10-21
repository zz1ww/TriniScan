"""交互式标定拍摄会话的单元测试（不依赖相机与界面）。"""
import numpy as np
import cv2
import pytest

from triniscan.ui.capture_session import (
    CalibMode,
    CaptureSession,
    QualityVerdict,
)


class _Cfg:
    """最小配置替身，避免依赖真实 yaml。"""

    root = "."

    def __init__(self, data=None):
        self._d = data or {}

    def get(self, key, default=None):
        return self._d.get(key, default)


def _make_session(mode=CalibMode.CAMERA, save_images=False, data=None):
    return CaptureSession(mode, _Cfg(data), save_images=save_images)


def _chessboard(cols=9, rows=6, square=40, margin=50, angle=0.0,
                scale=1.0, blur=0):
    w = (cols + 1) * square + 2 * margin
    h = (rows + 1) * square + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                cv2.rectangle(
                    img,
                    (margin + c * square, margin + r * square),
                    (margin + c * square + square,
                     margin + r * square + square),
                    0, -1)
    if angle or scale != 1.0:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        img = cv2.warpAffine(img, m, (w, h), borderValue=255)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img


# ---------------------------------------------------------------------------
# 模式元数据
# ---------------------------------------------------------------------------
def test_mode_metadata():
    assert CalibMode.CAMERA.min_views == 6
    assert CalibMode.LASER.min_views == 2
    assert CalibMode.AXIS.min_views == 3
    for m in CalibMode:
        assert m.recommend_views >= m.min_views
        assert isinstance(m.label, str) and m.label


# ---------------------------------------------------------------------------
# 质量判定
# ---------------------------------------------------------------------------
def test_reject_blank_image():
    s = _make_session()
    v = s.evaluate(np.full((200, 200), 128, np.uint8))
    assert isinstance(v, QualityVerdict)
    assert not v.ok
    assert "棋盘格" in v.reason


def test_accept_clean_chessboard():
    s = _make_session()
    v = s.add_frame(_chessboard())
    assert v.ok
    assert v.num_corners == 54
    assert s.count == 1


def test_reject_too_small():
    s = _make_session()
    # 棋盘格极小：整体缩小
    small = cv2.resize(_chessboard(), None, fx=0.25, fy=0.25)
    v = s.evaluate(small)
    assert not v.ok


def test_reject_blurry():
    s = _make_session()
    v = s.evaluate(_chessboard(angle=3, blur=41))
    assert not v.ok
    assert "模糊" in v.reason


def test_reject_duplicate():
    s = _make_session()
    assert s.add_frame(_chessboard()).ok
    v = s.evaluate(_chessboard())      # 完全相同
    assert not v.ok
    assert "相似" in v.reason


def test_novelty_allows_varied_poses():
    s = _make_session()
    for a in range(0, 30, 3):
        s.add_frame(_chessboard(angle=a))
    assert s.count >= 5


# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------
def test_progress_and_ready():
    s = _make_session()
    assert not s.ready
    for a in range(0, 30, 3):
        s.add_frame(_chessboard(angle=a))
    if s.count >= s.mode.min_views:
        assert s.ready
    assert "张" in s.progress_text()


def test_clear_and_undo():
    s = _make_session()
    s.add_frame(_chessboard())
    s.add_frame(_chessboard(angle=5))
    n = s.count
    assert n == 2
    assert s.remove_last()
    assert s.count == n - 1
    s.clear()
    assert s.count == 0
    assert not s.remove_last()


def test_save_images_creates_files(tmp_path):
    data = {
        "output.calib_dir": str(tmp_path),
        "calibration.camera_file": str(tmp_path / "cam.npz"),
    }
    s = _make_session(save_images=True, data=data)
    s.add_frame(_chessboard())
    out = tmp_path / "camera" / "000.png"
    assert out.exists()


# ---------------------------------------------------------------------------
# 标定流程
# ---------------------------------------------------------------------------
def test_camera_calibration_end_to_end(tmp_path):
    data = {
        "camera_calib.pattern_cols": 9,
        "camera_calib.pattern_rows": 6,
        "camera_calib.square_size_m": 0.02,
        "calibration.camera_file": str(tmp_path / "calib_camera.npz"),
    }
    s = _make_session(data=data)
    for a in range(0, 30, 3):
        s.add_frame(_chessboard(angle=a))
    assert s.ready
    outcome = s.run_calibration()
    assert outcome.ok, outcome.message
    assert (tmp_path / "calib_camera.npz").exists()
    assert "px" in outcome.quality


def test_laser_calibration_requires_camera_file(tmp_path):
    data = {"calibration.camera_file": str(tmp_path / "missing.npz")}
    s = _make_session(CalibMode.LASER, data=data)
    s.images.append(_chessboard())     # 直接塞一帧绕过质量检测
    s.images.append(_chessboard())
    outcome = s.run_calibration()
    assert not outcome.ok
    assert "相机标定" in outcome.message


def test_axis_from_points(tmp_path):
    data = {"calibration.axis_file": str(tmp_path / "calib_axis.npz")}
    s = _make_session(CalibMode.AXIS, data=data)
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([0.05 * np.cos(theta), 0.05 * np.sin(theta),
                           np.full_like(theta, 0.03)])
    outcome = s.calibrate_axis_from_points(pts)
    assert outcome.ok, outcome.message
    assert (tmp_path / "calib_axis.npz").exists()


# ---------------------------------------------------------------------------
# 双图差分模式（激光标定 use_diff）
# ---------------------------------------------------------------------------
def _diff_session():
    return _make_session(CalibMode.LASER, data={
        "laser_calib.use_diff": True,
        "calibration.camera_file": "config/calib_camera.npz",
    })


def _with_laser(img, y=None):
    """在棋盘图上叠一条激光线（模拟"开激光"那一张）。"""
    out = img.copy()
    h, w = out.shape[:2]
    yy = h // 2 if y is None else y
    cv2.line(out, (w // 6, yy), (w * 5 // 6, yy), 255, 3)
    return out


def test_diff_pair_second_frame_accepted():
    """同一位姿的"关激光 → 开激光"必须能配对（曾被误判为过于相似）。"""
    s = _diff_session()
    dark = _chessboard()
    assert s.add_frame(dark).ok
    v = s.add_frame(_with_laser(dark))
    assert v.ok, v.reason
    assert s.count == 2
    assert "配对" in v.reason


def test_diff_pair_requires_pose_change_between_pairs():
    """不同对之间位姿必须不同：同姿态再起一对要被拒。"""
    s = _diff_session()
    dark = _chessboard()
    assert s.add_frame(dark).ok
    assert s.add_frame(_with_laser(dark)).ok
    v = s.evaluate(dark)          # 下一对的关激光帧，姿态没变
    assert not v.ok
    assert "更换棋盘格姿态" in v.reason


def test_diff_pair_rejects_board_moved_inside_pair():
    """一对之内棋盘被碰动，第 2 张必须被拒。"""
    s = _diff_session()
    dark = _chessboard()
    assert s.add_frame(dark).ok
    moved = _with_laser(_chessboard(angle=8))
    v = s.evaluate(moved)
    assert not v.ok, v.reason
    assert "碰动" in v.reason or "静止" in v.reason


def test_diff_pair_accepts_multiple_pairs():
    s = _diff_session()
    for a in (0, 6, 12):
        dark = _chessboard(angle=a)
        assert s.add_frame(dark).ok
        assert s.add_frame(_with_laser(dark)).ok
    assert s.count == 6


def test_diff_undo_restores_pose_reference():
    """撤销某对第 1 张后，该位姿应能重新拍，不被判重复。"""
    s = _diff_session()
    dark = _chessboard()
    assert s.add_frame(dark).ok
    s.remove_last()               # 撤销该对第 1 张
    assert s.count == 0
    assert s.add_frame(dark).ok   # 同姿态可重来


def test_use_diff_ignored_in_camera_mode():
    """use_diff 只对激光模式生效，相机标定仍走原去重逻辑。"""
    s = _make_session(CalibMode.CAMERA, data={"laser_calib.use_diff": True})
    assert s.add_frame(_chessboard()).ok
    assert not s.add_frame(_chessboard()).ok   # 依旧是重复
