"""转轴**棋盘格法**的测试（无需真实设备）。

覆盖三层：

1. 纯几何：板上固定点绕轴旋转 → 共圆 → 轴恢复；
2. 单视角：真实渲染的棋盘图 → ``solveBoardPose`` 位姿正确；
3. 端到端：合成一组"板随转台旋转"的图像 → ``ChessboardAxisCollector``
   恢复轴点/轴向，含**方向符号**（无需角度输入）与自动扫描分流。
"""
import math

import cv2
import numpy as np
import pytest

from triniscan.algorithms.calibration.axis_chessboard import (
    BoardPose,
    ChessboardAxisCollector,
    _solve_pnp_candidates,
    axis_points_from_poses,
    board_reference_point,
    calibrate_axis_from_chessboards,
    direction_sign_from_poses,
)

# --- 场景参数 -------------------------------------------------------------
COLS, ROWS = 11, 8
SQUARE = 0.010
K = np.array([[900.0, 0.0, 480.0],
              [0.0, 900.0, 360.0],
              [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
AXIS = np.array([0.0, 0.0, 1.0])        # 转轴 = 相机 z 轴
AXIS_POINT = np.array([0.0, 0.0, 0.0])
BOARD_OFFSET = np.array([0.060, 0.0, 0.400])   # 板原点离轴 60 mm、距相机 0.4 m

#: 生产代码默认追踪**板中心**，所以这里也把板坐标原点放在网格中心，
#: 让「生成位姿」与「追踪点」是同一个物理点。板尺寸小，平移量只有
#: 几毫米，但正是这几分之一毫米的外推误差会放大成"等价位姿"漂移。
_GRID = [-(COLS - 1) * SQUARE / 2.0, -(ROWS - 1) * SQUARE / 2.0]


def _rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _board_pose(angle_deg):
    """转台转 ``angle_deg`` 时，**板坐标系原点**在相机系中的旋转与平移。

    板坐标系原点是角点网格的第一个角点，其板坐标 = (0,0,0)，
    相机系坐标 = t。板正对相机（法向 +z 指向相机），
    所以旋转矩阵就是纯转台旋转——不能用 ``diag(1,-1,-1)`` 之类翻面，
    那会让 ``solvePnP`` 落到"从背面看"的等价位姿上。
    """
    r = _rot_z(angle_deg)
    t = r @ BOARD_OFFSET
    return r, t


def _render_board(angle_deg):
    """用投影角点绘制一张"拍到棋盘格"的合成图。"""
    r, t = _board_pose(angle_deg)
    objp = np.zeros((COLS * ROWS, 3), np.float64)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    objp *= SQUARE
    # 板坐标原点 = 网格中心（与 ChessboardAxisCollector 默认追踪点一致）
    objp[:, 0] += _GRID[0]
    objp[:, 1] += _GRID[1]

    rvec = cv2.Rodrigues(r)[0]
    proj, _ = cv2.projectPoints(objp, rvec, t, K, DIST)
    pts = proj.reshape(-1, 2)

    img = np.full((720, 960), 235, np.uint8)
    # 棋盘是 (COLS+1)×(ROWS+1) 个方格，内角点位于方格交点；角点检测
    # 需要完整的外圈方格，所以格子索引取 -1 .. COLS-1。
    grid = pts.reshape(ROWS, COLS, 2).astype(np.float64)
    step_x = (grid[0, -1] - grid[0, 0]) / (COLS - 1)
    step_y = (grid[-1, 0] - grid[0, 0]) / (ROWS - 1)

    def _corner(i, j):
        return grid[0, 0] + step_x * j + step_y * i

    for i in range(-1, ROWS):
        for j in range(-1, COLS):
            if (i + j) % 2:
                continue
            quad = np.array([_corner(i, j), _corner(i, j + 1),
                             _corner(i + 1, j + 1), _corner(i + 1, j)],
                            dtype=np.int32)
            cv2.fillConvexPoly(img, quad, 20)
    return img


# ---------------------------------------------------------------------------
# 1. 纯几何
# ---------------------------------------------------------------------------
def test_board_reference_point():
    assert np.allclose(board_reference_point(9, 6, 0.01),
                       [0.04, 0.025, 0.0])
    assert np.allclose(board_reference_point(9, 6, 0.01, use_center=False),
                       np.zeros(3))


def test_axis_points_are_cocircular():
    """同一板上点随转台旋转 → 到轴的距离恒定、沿轴坐标恒定。"""
    poses = []
    for deg in (0.0, 30.0, 60.0, 120.0, 200.0):
        r, t = _board_pose(deg)
        poses.append(BoardPose(rvec=cv2.Rodrigues(r)[0].reshape(3), tvec=t))
    # 这里的位姿是"板原点 = 板心"，所以追踪板心 (0,0,0) 才是同一个点
    ref = board_reference_point(COLS, ROWS, SQUARE, use_center=False)
    pts = axis_points_from_poses(poses, ref)

    assert pts.shape == (5, 3)
    # 追踪点 = 板中心，位姿平移即其相机系坐标，离轴量就是 |offset_xy|
    expected_r = float(np.linalg.norm(BOARD_OFFSET[:2]))
    radius = np.linalg.norm(pts[:, :2], axis=1)
    assert np.allclose(radius, expected_r, atol=1e-9)
    assert np.allclose(pts[:, 2], BOARD_OFFSET[2], atol=1e-9)


def test_direction_sign_follows_rotation_order():
    """正转序列 → +1；反转序列 → -1。"""
    def _poses(sign):
        out = []
        for deg in (0.0, 40.0, 80.0, 120.0):
            r, t = _board_pose(sign * deg)
            out.append(BoardPose(rvec=cv2.Rodrigues(r)[0].reshape(3),
                                 tvec=t))
        return out

    assert direction_sign_from_poses(_poses(+1), AXIS) > 0
    assert direction_sign_from_poses(_poses(-1), AXIS) < 0
    # 帧数不足
    assert direction_sign_from_poses(_poses(+1)[:1], AXIS) == 0.0


# ---------------------------------------------------------------------------
# 2. 单视角位姿
# ---------------------------------------------------------------------------
def test_solve_board_pose_single_frame_is_ambiguous():
    """单帧给出多个候选（重投影都好），无法自行判定哪个对。

    注意：两个解的**板原点** ``t`` 本身并不等于真实值（一个正解 + 一个
    转置解），但它们的**参考点轨迹**在圆度上有明显差别——这正是多帧
    一致性判据的依据。
    """
    cands = _solve_pnp_candidates(_render_board(25.0), K, DIST,
                                  COLS, ROWS, SQUARE, orientation="auto")
    assert len(cands) == 2                    # 长方形棋盘 → 两种排列
    assert {tuple(c.layout) for c in cands} == {(COLS, ROWS), (ROWS, COLS)}
    assert all(c.reproj_rms_px < 1.0 for c in cands)


def test_single_frame_alone_is_not_enough():
    """只用单帧候选直接标定会得到错误半径（说明消歧是必需的）。"""
    from triniscan.algorithms.calibration.axis_chessboard import (
        axis_points_from_poses)

    ref = board_reference_point(COLS, ROWS, SQUARE)
    pts = []
    for deg in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0):
        cands = _solve_pnp_candidates(_render_board(deg), K, DIST,
                                      COLS, ROWS, SQUARE,
                                      orientation="auto")
        pts.append([c for c in cands
                    if tuple(c.layout) == (ROWS, COLS)][0])
    spread = np.linalg.norm(
        axis_points_from_poses(pts, ref)[:, :2], axis=1)
    # 横放的板若按"竖放"解算 → 半径明显偏离真实 60 mm
    assert abs(float(np.mean(spread)) - 0.060) > 0.005


def test_sequence_consistency_picks_correct_layout():
    """多帧一致性 → 选中正确的角点排列，参考点回到真实圆上。"""
    collector = ChessboardAxisCollector(
        K, DIST, pattern_cols=COLS, pattern_rows=ROWS, square_size_m=SQUARE)
    for deg in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0):
        assert collector.add_angle(_render_board(deg))
    assert collector.finalize() == 6

    radius = np.linalg.norm(collector.stack[:, :2], axis=1)
    assert np.allclose(radius, np.linalg.norm(BOARD_OFFSET[:2]), atol=5e-3)


# ---------------------------------------------------------------------------
# 3. 端到端
# ---------------------------------------------------------------------------
def test_collector_recovers_axis_without_angles():
    """不给角度：轴点/轴向仍正确，方向符号由相对旋转判定。"""
    collector = ChessboardAxisCollector(
        K, DIST, pattern_cols=COLS, pattern_rows=ROWS, square_size_m=SQUARE)
    angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0]
    for deg in angles:
        assert collector.add_angle(_render_board(deg), angle_deg=None)

    assert collector.finalize() == len(angles)
    assert collector.count == len(angles)
    assert not collector.has_angles

    result = collector.calibrate(robust=True)
    assert abs(abs(float(result.direction @ AXIS)) - 1.0) < 1e-2
    assert abs(result.radius - np.linalg.norm(BOARD_OFFSET[:2])) < 5e-3
    # 轴上一点到真实轴的距离（沿轴方向任意 → 只看径向）
    p = np.asarray(result.point)
    assert float(np.linalg.norm(p[:2])) < 5e-3
    # 方向符号应与给定的旋转顺序一致（+z）
    assert float(result.direction[2]) > 0.0


def test_collector_with_angles_keeps_sign():
    collector = ChessboardAxisCollector(
        K, DIST, pattern_cols=COLS, pattern_rows=ROWS, square_size_m=SQUARE)
    for deg in (0.0, 60.0, 120.0, 180.0, 240.0):
        collector.add_angle(_render_board(deg), angle_deg=deg)
    collector.finalize()
    assert collector.has_angles
    result = collector.calibrate()
    assert float(result.direction[2]) > 0.0


def test_collector_skips_bad_frames():
    collector = ChessboardAxisCollector(
        K, DIST, pattern_cols=COLS, pattern_rows=ROWS, square_size_m=SQUARE)
    assert not collector.add_angle(np.full((720, 960), 128, np.uint8))
    for deg in (0.0, 90.0, 180.0, 270.0):
        collector.add_angle(_render_board(deg), angle_deg=deg)
    assert collector.finalize() == 4
    assert collector.count == 4
    assert len(collector.errors) == 1


def test_calibrate_axis_from_chessboards_requires_three(tmp_path):
    with pytest.raises(RuntimeError):
        calibrate_axis_from_chessboards(
            [_render_board(0.0), _render_board(90.0)],
            K, DIST, pattern_cols=COLS, pattern_rows=ROWS,
            square_size_m=SQUARE)


def test_calibrate_axis_from_chessboards_writes_file(tmp_path):
    from triniscan.algorithms.calibration import load_axis_calib

    out = tmp_path / "axis.npz"
    imgs = [_render_board(d) for d in (0.0, 60.0, 120.0, 180.0)]
    result = calibrate_axis_from_chessboards(
        imgs, K, DIST, pattern_cols=COLS, pattern_rows=ROWS,
        square_size_m=SQUARE, out_path=str(out))
    assert out.exists()
    loaded = load_axis_calib(str(out))
    assert loaded.num_points == result.num_points == 4
    assert np.allclose(loaded.direction, result.direction)


# ---------------------------------------------------------------------------
# 4. 自动扫描分流（不依赖激光标定）
# ---------------------------------------------------------------------------
class _FakeConfig:
    def __init__(self, root, cam_file, laser_file="", method="chessboard",
                 views=6):
        self.root = root
        self.raw = {"camera": {}, "turntable": {}}
        self._d = {
            "calibration.camera_file": cam_file,
            "calibration.laser_file": laser_file,
            "calibration.axis_file": str(root / "axis.npz"),
            "axis_calib.method": method,
            "axis_calib.num_views": views,
            "axis_calib.min_views": 3,
            "axis_calib.min_radius_mm": 2.0,
            "axis_calib.max_residual_mm": 1.0,
            "axis_calib.chessboard_use_center": True,
            "axis_calib.archive_dir": "arch",
            "camera_calib.pattern_cols": COLS,
            "camera_calib.pattern_rows": ROWS,
            "camera_calib.square_size_m": SQUARE,
            "camera_calib.use_sb": True,
        }

    def get(self, key, default=None):
        return self._d.get(key, default)


class _FakeTurntable:
    def __init__(self):
        self.angle = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def rotate_deg(self, deg):
        self.angle = (self.angle + deg) % 360.0
        return True


class _FakeCamera:
    def __init__(self, table):
        self._t = table

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def grab(self):
        return _render_board(self._t.angle)


def test_sweep_chessboard_needs_no_laser(tmp_path):
    from triniscan.calibration import CameraCalibResult, save_camera_calib
    from triniscan.ui.capture_session import sweep_axis_calibration

    cam = tmp_path / "cam.npz"
    save_camera_calib(str(cam), CameraCalibResult(
        K=K, dist=DIST, image_size=(960, 720), rms=0.1, num_images=10))
    cfg = _FakeConfig(tmp_path, str(cam))       # laser_file 留空
    table = _FakeTurntable()

    result = sweep_axis_calibration(
        cfg, make_camera=lambda: _FakeCamera(table),
        make_turntable=lambda: table)

    assert result.ok, result.message
    assert result.num_views == 6
    assert abs(abs(float(result.direction[2])) - 1.0) < 1e-2
    assert abs(result.radius_m - 0.060) < 5e-3
    assert result.out_path and (tmp_path / "axis.npz").exists()


def test_sweep_sphere_still_requires_laser(tmp_path):
    from triniscan.calibration import CameraCalibResult, save_camera_calib
    from triniscan.ui.capture_session import sweep_axis_calibration

    cam = tmp_path / "cam.npz"
    save_camera_calib(str(cam), CameraCalibResult(
        K=K, dist=DIST, image_size=(960, 720), rms=0.1, num_images=10))
    cfg = _FakeConfig(tmp_path, str(cam), method="sphere")
    result = sweep_axis_calibration(cfg)
    assert not result.ok
    assert "激光平面" in result.message
