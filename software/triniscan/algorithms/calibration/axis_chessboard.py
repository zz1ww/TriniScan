"""转轴标定：**棋盘格法**（不依赖激光、不依赖标记球）。

为什么需要这个模块
------------------
原有转轴取点方法（``peak`` / ``sphere`` / ``depth``）都要求转台上放一个
**标记球**：

- ``peak``  需要激光在**每个角度**都打亮小球——激光只有一条细线，
  球转到背光侧就丢点；
- ``sphere`` / ``depth`` 需要看得清球的圆形轮廓，背景要亮、对比要够。

实测这两个前提都不容易满足。棋盘格法把这两个依赖都去掉。

原理
----
把棋盘格**固定在转台上并离轴放置**（建议 30~60 mm）。转台每转一个角度
拍一张，用 ``solvePnP`` 得到板在相机系下的位姿 ``T_i = (R_i, t_i)``。

板上任意一个**固定点**（默认取角点网格的中心）在板坐标系里坐标不变，
所以在相机系下的坐标为

.. code-block:: text

    P_i = R_i · P_board + t_i        i = 0 … N-1

板随转台做刚体旋转 ⇒ 所有 ``P_i`` 落在**以转轴为轴的同一个圆**上。
拟合该圆即得轴上一点与轴向——直接复用 :func:`calibrate_axis`
（稳健 RANSAC、结果保存、质量判定全都不用重写）。

方向符号（不需要知道转角）
--------------------------
相邻两帧板的相对旋转 ``R_iᵀ · R_{i+1}`` 的旋转向量方向**就是**转轴方向，
其符号就是转台正转方向（相邻步长 &lt; 180° 时成立）。把所有相邻对的旋转
向量相加，与拟合得到的轴向做点积，符号为负就翻转轴向。

因此**手动拍照不需要记录每次转了多少度**，只要"按同一方向依次拍"即可；
自动扫描时另有角度序列可用，会一并传给 :func:`calibrate_axis` 交叉校验。

精度要点
--------
- 板离轴越远，轴点精度越高（半径小于 20 mm 时轴点误差会被放大）；
- 每张图用整板角点估计位姿，远优于单个圆形斑点；
- 视角覆盖建议 &gt;= 6 个；相邻步长别接近 180°（转向量符号会歧义）。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ...common.logging_utils import get_logger
from .axis_calib import AxisCalibResult, calibrate_axis, save_axis_calib
from .camera_calib import _detect_corners, make_object_points

log = get_logger(__name__)

__all__ = [
    "BoardPose",
    "board_reference_point",
    "axis_points_from_poses",
    "solve_board_pose",
    "direction_sign_from_poses",
    "calibrate_axis_from_chessboards",
    "ChessboardAxisCollector",
]


@dataclass
class BoardPose:
    """一个视角下棋盘格在**相机系**中的位姿。"""

    rvec: np.ndarray            # (3,) Rodrigues
    tvec: np.ndarray            # (3,)
    angle_deg: float = 0.0      # 该视角对应的转台角度（可未知）
    reproj_rms_px: float = 0.0  # 该视角 solvePnP 重投影 RMS（像素）
    layout: tuple = (0, 0)      # 求该解时用的内角点排列 (cols, rows)

    @property
    def R(self) -> np.ndarray:
        """旋转矩阵 (3,3)，板坐标系 → 相机坐标系。"""
        import cv2
        return cv2.Rodrigues(np.asarray(self.rvec).reshape(3, 1))[0]


# ---------------------------------------------------------------------------
# 几何核心（纯函数，便于单元测试）
# ---------------------------------------------------------------------------
def board_reference_point(cols: int, rows: int, square_size_m: float,
                          use_center: bool = True) -> np.ndarray:
    """返回板坐标系里用于追踪的**固定参考点**。

    Parameters
    ----------
    cols, rows : 棋盘内角点数
    square_size_m : 方格边长（米）
    use_center : True 取角点网格的几何中心（推荐：离板边最远，受畸变与
        局部遮挡影响最小）；False 取网格第一个角点（原点）。

    Returns
    -------
    (3,) 板坐标系下的点（Z=0）
    """
    if use_center:
        return np.array([(int(cols) - 1) * 0.5 * float(square_size_m),
                         (int(rows) - 1) * 0.5 * float(square_size_m),
                         0.0], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def axis_points_from_poses(poses: Sequence[BoardPose],
                           ref_point: np.ndarray) -> np.ndarray:
    """同一个板上固定点在各视角的相机系坐标（共圆点集）。

    Parameters
    ----------
    poses : 各视角的板位姿（按转台旋转顺序排列）
    ref_point : (3,) 板坐标系下的固定点

    Returns
    -------
    (N, 3) 相机系坐标；``poses`` 为空时返回 (0, 3)
    """
    if not len(poses):
        return np.empty((0, 3), dtype=np.float64)
    p = np.asarray(ref_point, dtype=np.float64).reshape(3)
    out = np.empty((len(poses), 3), dtype=np.float64)
    for i, pose in enumerate(poses):
        t = np.asarray(pose.tvec, dtype=np.float64).reshape(3)
        out[i] = pose.R @ p + t
    return out


def direction_sign_from_poses(poses: Sequence[BoardPose],
                              normal: np.ndarray) -> float:
    """用相邻帧的相对旋转判断转台正方向与 ``normal`` 同向还是反向。

    ``R_rel = R_{i+1} · R_iᵀ`` 是**相机系里**的增量旋转；转台正转时它
    就是绕轴的正向转动（活动坐标系右乘），其旋转向量方向即转轴方向，
    符号即转台正转方向。

    注意不能用 ``R_iᵀ R_{i+1}``：那是转台坐标系里的旋转，它的转轴是
    **轴上方向量的负值**，会把方向符号判反。

    Parameters
    ----------
    poses : 按转台旋转顺序排列的板位姿
    normal : 拟合得到的轴向（任意符号）

    Returns
    -------
    +1.0 表示 ``normal`` 与转台正转方向一致（可直接使用）；
    -1.0 表示需要翻转；
    0.0 表示无法判断（帧数不足，或所有相邻转角都接近 0/180°），
    此时调用方应保持原符号。
    """
    import cv2

    if len(poses) < 2:
        return 0.0
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(n)) < 1e-12:
        return 0.0

    acc = np.zeros(3, dtype=np.float64)
    for a, b in zip(poses[:-1], poses[1:]):
        r_rel = b.R @ a.R.T
        rvec = cv2.Rodrigues(r_rel)[0].reshape(3)
        theta = float(np.linalg.norm(rvec))
        # 接近 0（没转动）或接近 180°（转向量符号歧义）的对不可用
        if 1e-6 < theta < np.pi - 1e-3:
            acc += rvec
    if float(np.linalg.norm(acc)) < 1e-9:
        return 0.0
    return 1.0 if float(acc @ n) >= 0.0 else -1.0


# ---------------------------------------------------------------------------
# 单视角求解
# ---------------------------------------------------------------------------
def _to_gray(image: np.ndarray) -> np.ndarray:
    import cv2
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 1:
        gray = image[..., 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _solve_pnp_candidates(gray: np.ndarray, K: np.ndarray, dist,
                          pattern_cols: int, pattern_rows: int,
                          square_size_m: float,
                          use_sb: bool = True,
                          orientation: str = "landscape"
                          ) -> list[BoardPose]:
    """检测棋盘并返回候选位姿（每种**角点排列**一个）。

    矩形棋盘被写成 ``(cols, rows)`` 还是 ``(rows, cols)``，检测器会给出
    转置排列的角点序列，从而得到**不同的解**；其中最多一个与真实摆放
    一致。单帧无法判断哪个对，只能全序列拍完后用圆度一致性选
    （:func:`select_consistent_poses`）。

    正方形棋盘（cols == rows）只有一种排列。
    """
    import cv2

    if dist is None:
        dist = np.zeros(5)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64)

    cols, rows = int(pattern_cols), int(pattern_rows)
    if cols == rows or orientation == "auto":
        layouts = [(cols, rows)]
        if cols != rows:
            layouts.append((rows, cols))
    elif orientation == "portrait":
        layouts = [(rows, cols)]
    else:                                   # landscape（默认）
        layouts = [(cols, rows)]

    out: list[BoardPose] = []
    seen = set()
    for (c, r) in layouts:
        corners = _detect_corners(gray, (c, r), use_sb=use_sb)
        if corners is None:
            continue
        objp = make_object_points(c, r, square_size_m)
        try:
            ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        except cv2.error:                     # pragma: no cover
            log.debug("solvePnP 失败（layout=%sx%s）", c, r, exc_info=True)
            continue
        if not ok:                            # pragma: no cover
            continue
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err = np.linalg.norm(
            proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
        rms = float(np.sqrt(np.mean(err ** 2)))

        key = (c, r, tuple(np.round(rvec, 6)), tuple(np.round(tvec, 6)))
        if key in seen:                       # 两种排列偶尔解相同
            continue
        seen.add(key)
        out.append(BoardPose(
            rvec=rvec, tvec=tvec, reproj_rms_px=rms, layout=(c, r)))
    return out


def solve_board_pose(gray: np.ndarray, K: np.ndarray, dist,
                     pattern_cols: int, pattern_rows: int,
                     square_size_m: float, use_sb: bool = True
                     ) -> Optional[BoardPose]:
    """单帧位姿（未消歧，取第一个候选）。

    仅用于调试/预览。**不要**用它直接做转轴标定——单帧分不清
    "角点排列转置"的两个解，会静默算出错误半径。正式流程走
    :class:`ChessboardAxisCollector`（多帧一致性消歧）。
    """
    cands = _solve_pnp_candidates(gray, K, dist, pattern_cols,
                                  pattern_rows, square_size_m,
                                  use_sb=use_sb)
    return cands[0] if cands else None


def select_consistent_poses(candidates: Sequence[Sequence[BoardPose]],
                            ref_point: np.ndarray
                            ) -> Optional[list[BoardPose]]:
    """用**全序列一致性**从多帧候选里挑出一组自洽的位姿。

    判据：板随转台刚体旋转 ⇒ 参考点轨迹是一段圆弧（各点到轴距离相等、
    沿轴高度相等）。错解会让部分帧的参考点跳到对称位置，圆度变差。

    做法：对每一帧的所有候选做**组合枚举**（每帧候选数 ≤ 4，帧数一般
    ≤ 20，用逐帧增广+剪枝控制规模），取圆度残差最小的一组。

    Returns
    -------
    list[BoardPose] 或 None
    """
    frames = [list(c) for c in candidates if c]
    if len(frames) < 3:
        return None

    # 候选组合过多时（理论上 4^n），退化为"逐帧就近贪心"
    total = 1
    for f in frames:
        total *= len(f)
        if total > 128:
            break

    if total <= 128:
        best, best_score = None, None
        for combo in itertools.product(*frames):
            score = _circularity_spread(
                axis_points_from_poses(combo, ref_point))
            if best_score is None or score < best_score:
                best_score, best = score, combo
        return list(best) if best is not None else None

    # 退化路径：以第一帧各候选为种子，逐帧取与当前轨迹最相容者
    best_seed, best_score = None, None
    for seed in frames[0]:
        chosen = [seed]
        score = float("inf")
        for f in frames[1:]:
            pts = axis_points_from_poses(chosen, ref_point)
            pick = min(
                f, key=lambda p: _circularity_spread(
                    axis_points_from_poses(chosen + [p], ref_point)))
            chosen.append(pick)
            score = _circularity_spread(
                axis_points_from_poses(chosen, ref_point))
        if best_score is None or score < best_score:
            best_score, best_seed = score, chosen
    return best_seed


def _circularity_spread(pts: np.ndarray) -> float:
    """一串三维点的"圆度"代价：拟合圆后径向残差的 RMS（米）。

    只用于**比较**候选解的优劣，不参与最终标定。
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return float("inf")
    # 先用 SVD 求平面法向，再在平面内最小二乘拟合圆
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    normal = vt[-1]
    u = vt[0]
    v = vt[1]
    xy = np.stack([(pts - center) @ u, (pts - center) @ v], axis=1)
    a = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:                  # pragma: no cover
        return float("inf")
    cx, cy, c = sol
    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0:
        return float("inf")
    radius = float(np.sqrt(r2))
    resid = np.linalg.norm(xy - np.array([cx, cy]), axis=1) - radius
    return float(np.sqrt(np.mean(resid ** 2)))


# ---------------------------------------------------------------------------
# 采集器（与 AxisLiveCollector 接口同构）
# ---------------------------------------------------------------------------
class ChessboardAxisCollector:
    """逐角度喂入"含完整棋盘格"的图像，最后完成转轴标定。

    与 :class:`~.axis_live.AxisLiveCollector` 保持同一套最小接口
    （``add_angle`` / ``count`` / ``stack`` / ``errors`` / ``calibrate`` /
    ``summary``），上层（自动扫描、界面、测试替身）只需按方法切换实现。

    Parameters
    ----------
    K : (3,3) 相机内参
    dist : 畸变系数（None 表示无畸变）
    pattern_cols, pattern_rows : 棋盘内角点数
    square_size_m : 方格边长（米）
    use_sb : 是否优先 SB 检测器
    use_center : 参考点取板中心（默认）还是网格第一个角点
    orientation : 板的安装朝向，决定用哪种**角点排列**解算位姿。
        ``"landscape"``（默认）= 棋盘长边水平；``"portrait"`` = 长边竖直；
        ``"auto"`` = 两种都试并按圆度自动选（**不可靠**，仅调试用）。

        为什么必须指定：矩形棋盘按 ``(cols, rows)`` 与 ``(rows, cols)``
        检测都能成功，得到两个重投影同样完美的位姿，但解出的轴半径相差
        很大（合成验证：真实 60 mm，错解 76 mm）。两种解各自都能拼出
        完美的圆，仅凭图像**无法**区分（板的法向在两种解下同样恒定）。
        因此必须由外部告知板是怎么摆的。
    """

    def __init__(self, K: np.ndarray, dist=None,
                 pattern_cols: int = 11, pattern_rows: int = 8,
                 square_size_m: float = 0.010,
                 use_sb: bool = True, use_center: bool = True,
                 orientation: str = "landscape") -> None:
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = None if dist is None else np.asarray(dist,
                                                         dtype=np.float64)
        self.pattern_cols = int(pattern_cols)
        self.pattern_rows = int(pattern_rows)
        self.square_size_m = float(square_size_m)
        self.use_sb = bool(use_sb)
        self.use_center = bool(use_center)
        self.orientation = str(orientation or "landscape").lower()
        if self.orientation not in ("landscape", "portrait", "auto"):
            raise ValueError(
                f"orientation 必须是 landscape/portrait/auto，"
                f"收到 {orientation!r}")

        # 每帧保存**两种角点排列**的候选位姿，等全序列拍完后统一消歧
        self._candidates: list[list[BoardPose]] = []
        self.poses: list[BoardPose] = []    # 消歧后的有效位姿
        self.angles: list[float] = []       # 未知角度记 nan
        self.errors: list[str] = []
        self.layout = None
        self._ref_point = board_reference_point(
            self.pattern_cols, self.pattern_rows, self.square_size_m,
            use_center=self.use_center)

    # ------------------------------------------------------------------
    def add_angle(self, image: np.ndarray,
                  angle_deg: Optional[float] = None
                  ) -> bool:
        """处理一个角度的图像。

        Parameters
        ----------
        image : 该角度拍摄的图像（BGR 或灰度）
        angle_deg : 转台角度；手动拍照时可为 None（只影响日志与交叉校验，
            方向符号由 :func:`direction_sign_from_poses` 自动判定）

        Returns
        -------
        bool
            本帧是否检出了棋盘格。真正的**位姿消歧**要等所有帧到齐后
            由 :meth:`finalize` 统一做（单帧无法区分两个等价位姿）。
        """
        cands = _solve_pnp_candidates(
            _to_gray(image), self.K, self.dist,
            self.pattern_cols, self.pattern_rows, self.square_size_m,
            use_sb=self.use_sb, orientation=self.orientation)
        if not cands:
            tag = "?" if angle_deg is None else f"{angle_deg:.0f}°"
            self.errors.append(f"{tag}: 未检测到棋盘格")
            return False
        val = float("nan") if angle_deg is None else float(angle_deg)
        for p in cands:
            p.angle_deg = val
        self._candidates.append(cands)
        return True

    # ------------------------------------------------------------------
    def finalize(self) -> int:
        """用全序列一致性消歧，把候选位姿定稿到 :attr:`poses`。

        调用时机：所有帧 ``add_angle`` 完成之后、``calibrate`` 之前。
        重复调用是幂等的（会重新判定）。返回有效视角数。
        """
        self.poses = []
        self.angles = []
        chosen = self._select()
        if chosen is None:
            return 0
        self.poses = list(chosen)
        self.angles = [p.angle_deg for p in self.poses]
        layouts = {tuple(p.layout) for p in self.poses}
        self.layout = next(iter(layouts)) if len(layouts) == 1 else None
        log.info("棋盘法位姿消歧：%d 帧，角点排列 %s，圆度残差 %.2f mm",
                 len(self.poses), self.layout,
                 _circularity_spread(
                     axis_points_from_poses(self.poses,
                                            self._ref_point)) * 1e3)
        return len(self.poses)

    def _select(self):
        """按 :attr:`orientation` 选定位姿；自动模式走圆度一致性。"""
        if self.orientation == "auto":
            return select_consistent_poses(self._candidates,
                                           self._ref_point)

        want = ((self.pattern_cols, self.pattern_rows)
                if self.orientation == "landscape"
                else (self.pattern_rows, self.pattern_cols))
        if want[0] == want[1]:              # 正方形棋盘只有一种排列
            want = (self.pattern_cols, self.pattern_rows)

        chosen = []
        for cands in self._candidates:
            hit = next((c for c in cands if tuple(c.layout) == want), None)
            if hit is not None:
                chosen.append(hit)
        return chosen or None

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.poses)

    @property
    def stack(self) -> np.ndarray:
        """各视角参考点在相机系中的坐标 (N,3)。"""
        self.finalize()
        return axis_points_from_poses(self.poses, self._ref_point)

    @property
    def has_angles(self) -> bool:
        """是否所有视角都有已知转角（自动扫描时为 True）。"""
        return bool(self.angles) and bool(np.all(np.isfinite(self.angles)))

    def save_points(self, path: str) -> None:
        """保存现场点（.npy），便于复查与离线重标。"""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        np.save(path, self.stack)
        log.info("已保存棋盘法转轴点（%d 点）: %s", self.count, path)

    def summary(self) -> str:
        text = f"{self.count} 个有效视角"
        if self.errors:
            text += f"，{len(self.errors)} 个失败"
        return text

    # ------------------------------------------------------------------
    def calibrate(self, out_path: Optional[str] = None,
                  save_points_to: Optional[str] = None,
                  robust: bool = True) -> AxisCalibResult:
        """拟合转轴并统一方向符号。

        Parameters
        ----------
        out_path : 非空则保存标定结果
        save_points_to : 非空则把现场点另存为 .npy
        robust : 是否用 RANSAC 抗跳点
        """
        self.finalize()
        if save_points_to:
            self.save_points(save_points_to)

        # 先不落盘：方向符号要拟合后才能统一，避免写出坏符号的文件
        result = calibrate_axis(
            self.stack,
            out_path=None,
            robust=robust,
            angles_deg=(self.angles if self.has_angles else None))

        # 没有角度序列（手动拍照）时，用相邻帧相对旋转判定正方向
        if not self.has_angles:
            sign = direction_sign_from_poses(self.poses, result.direction)
            if sign < 0:
                result.direction = -np.asarray(result.direction,
                                               dtype=np.float64)
                log.info("棋盘法：按相对旋转翻转了转轴方向符号")
            elif sign == 0.0:
                log.warning(
                    "棋盘法无法从位姿序列判定转轴正方向（帧数不足，或"
                    "相邻转角接近 180°）；保留拟合符号。建议按转台正"
                    "方向依次拍摄 >= 3 张。")

        if out_path:
            save_axis_calib(out_path, result)
        return result


# ---------------------------------------------------------------------------
# 一次性入口（脚本 / 离线标定）
# ---------------------------------------------------------------------------
def calibrate_axis_from_chessboards(
    images: Sequence[np.ndarray],
    K: np.ndarray,
    dist=None,
    pattern_cols: int = 11,
    pattern_rows: int = 8,
    square_size_m: float = 0.010,
    angles_deg: Optional[Sequence[float]] = None,
    out_path: Optional[str] = None,
    use_sb: bool = True,
    use_center: bool = True,
    robust: bool = True,
) -> AxisCalibResult:
    """一组（拍摄顺序 = 转台旋转顺序的）棋盘格图像 → 转轴标定。

    Raises
    ------
    RuntimeError
        有效视角不足 3 个（无法确定一个圆）。
    """
    collector = ChessboardAxisCollector(
        K, dist, pattern_cols=pattern_cols, pattern_rows=pattern_rows,
        square_size_m=square_size_m, use_sb=use_sb,
        use_center=use_center)
    for i, img in enumerate(images):
        angle = (float(angles_deg[i])
                 if angles_deg is not None and i < len(angles_deg) else None)
        collector.add_angle(img, angle)

    collector.finalize()
    if collector.count < 3:
        detail = "；".join(collector.errors[:5])
        raise RuntimeError(
            f"棋盘格有效视角不足（{collector.count} < 3）"
            + (f"：{detail}" if detail else ""))
    return collector.calibrate(out_path=out_path, robust=robust)

