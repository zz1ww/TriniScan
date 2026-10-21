"""交互式标定拍摄会话（与界面无关）。

本模块把"实时拍摄 → 质量检测 → 收集 → 自动标定"的业务逻辑
从 Tk 界面中剥离出来，便于单元测试与复用：

- :class:`CalibMode`          三种标定模式（相机 / 激光 / 转轴）
- :class:`QualityVerdict`     单帧质量判定结果
- :class:`CaptureSession`     会话：收集图像、判定质量、执行标定

界面层（``ui/calibration.py``）只负责显示画面与按钮，所有判定与
计算都在这里完成。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "CalibMode",
    "QualityVerdict",
    "CaptureSession",
    "CalibOutcome",
    "AxisSweepResult",
]


# ---------------------------------------------------------------------------
class CalibMode(str, Enum):
    """标定模式。"""

    CAMERA = "camera"     # 相机内参（纯棋盘格）
    LASER = "laser"       # 激光平面（棋盘格 + 激光线）
    AXIS = "axis"         # 转台转轴（特征点绕轴旋转）

    @property
    def label(self) -> str:
        return {
            CalibMode.CAMERA: "相机内参标定",
            CalibMode.LASER: "激光平面标定",
            CalibMode.AXIS: "转台转轴标定",
        }[self]

    @property
    def min_views(self) -> int:
        """满足标定的最少视角数。"""
        return {
            CalibMode.CAMERA: 6,    # 张正友建议 >= 6
            CalibMode.LASER: 2,     # 代码要求 >= 2，建议 >= 4
            CalibMode.AXIS: 3,      # 拟合圆至少 3 点
        }[self]

    @property
    def recommend_views(self) -> int:
        """推荐视角数（界面提示用）。"""
        return {
            CalibMode.CAMERA: 15,
            CalibMode.LASER: 4,
            CalibMode.AXIS: 12,
        }[self]


# ---------------------------------------------------------------------------
@dataclass
class QualityVerdict:
    """单帧质量判定。"""

    ok: bool                     # 是否可接受并纳入
    reason: str                  # 人类可读的判定说明
    num_corners: int = 0         # 检测到的角点数
    area_ratio: float = 0.0      # 棋盘格占比
    sharpness: float = 0.0       # 清晰度（拉普拉斯方差）
    novelty: float = 0.0         # 与已收集帧的差异（0~1，越大越新）

    @property
    def summary(self) -> str:
        if not self.ok:
            return f"✗ {self.reason}"
        return (f"✓ {self.reason}  角点{self.num_corners} "
                f"占比{self.area_ratio * 100:.0f}% "
                f"清晰{self.sharpness:.0f}")


# ---------------------------------------------------------------------------
@dataclass
class CalibOutcome:
    """一次标定的结果（对界面统一的轻量包装）。"""

    ok: bool
    mode: CalibMode
    message: str
    quality: str = ""            # 例如 "RMS=0.21 px"
    out_path: Optional[str] = None
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
class CaptureSession:
    """一次标定拍摄会话。

    Parameters
    ----------
    mode : 标定模式
    cfg : :class:`~triniscan.core.config.Config`（或兼容对象）
    save_images : 是否把每帧落盘（仅相机模式默认需要；激光/转轴可选）
    """

    #: 清晰度下限（拉普拉斯方差）——低于此判为模糊
    SHARPNESS_MIN = 60.0
    #: 棋盘格面积占比下限——低于此判为过小/过远
    AREA_RATIO_MIN = 0.02
    #: 相邻角点间距下限（像素）——低于此判为过密/过远
    MIN_SIDE_PX = 12.0
    #: 与已收集帧的最小差异——低于此判为重复
    NOVELTY_MIN = 0.02
    #: 棋盘法转轴用的去重阈值（各角度图像本来就相似，只挡"完全没动"）
    NOVELTY_AXIS_MIN = 0.004
    #: 双图差分模式下，同一对内部两张的**最大**差异——超过此判为
    #: "棋盘被碰动了"。一对内部只应有激光线之差，理应是全流程最相似
    #: 的两帧；阈值反着用（越小越合格）。
    DIFF_PAIR_MAX_DIFF = 0.08

    def __init__(self, mode: CalibMode, cfg, save_images: bool = True,
                 on_log: Optional[Callable[[str], None]] = None) -> None:
        self.mode = mode
        self.cfg = cfg
        self.save_images = save_images
        self._on_log = on_log

        cols = int(self._cfg_get("camera_calib.pattern_cols", 9))
        rows = int(self._cfg_get("camera_calib.pattern_rows", 6))
        self.pattern_cols = cols
        self.pattern_rows = rows
        self.square_size_m = float(
            self._cfg_get("camera_calib.square_size_m", 0.020))

        self.images: list[np.ndarray] = []
        self.verdicts: list[QualityVerdict] = []
        # 低分辨率缩略图，用于去重；与 self.images 一一对应
        self._thumbs: list[np.ndarray] = []
        # 双图差分模式：仅存放**每对第 1 张（关激光）**的缩略图。
        # 位姿新颖性只在这些帧之间比较——同一次拍摄的两帧位姿必须
        # 相同，拿它们互相比较会把正确的配对判成"过于相似"。
        self._pose_thumbs: list[np.ndarray] = []
        self.started_at = time.time()

    # ------------------------------------------------------------------
    def _cfg_get(self, key: str, default):
        try:
            return self.cfg.get(key, default)
        except AttributeError:  # pragma: no cover - 兼容 dict
            return default

    def _log(self, msg: str) -> None:
        log.info(msg)
        if self._on_log is not None:
            self._on_log(msg)

    def _is_axis_chessboard(self) -> bool:
        """当前是否为"棋盘法转轴"模式（需要按角度逐张拍摄）。"""
        return (self.mode is CalibMode.AXIS
                and str(self._cfg_get("axis_calib.method", ""))
                == "chessboard")

    def _is_diff_pair(self) -> bool:
        """当前是否为"双图差分"激光标定模式（两帧一对）。

        只有激光标定才有这个概念；其它模式下 ``laser_calib.use_diff``
        即使为真也不生效。
        """
        return (self.mode is CalibMode.LASER
                and bool(self._cfg_get("laser_calib.use_diff", False)))

    def _expects_dark(self) -> bool:
        """差分模式下：下一张是否应为"关激光"帧（即当前对尚未开始）。"""
        return self.count % 2 == 0

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def add_frame(self, image: np.ndarray) -> QualityVerdict:
        """对一帧图像做质量检测，合格则纳入会话。

        返回判定结果（无论是否纳入）。
        """
        verdict = self.evaluate(image)
        self.verdicts.append(verdict)
        if verdict.ok:
            # 先按"收录前"的奇偶状态判断这张是什么角色，再追加
            is_pair_head = self._is_diff_pair() and self._expects_dark()
            self.images.append(image.copy())
            thumb = self._thumbnail(image)
            self._thumbs.append(thumb)
            if is_pair_head:
                self._pose_thumbs.append(thumb)
            if self.save_images:
                self._save_frame(image, len(self.images) - 1)
        return verdict

    def evaluate(self, image: np.ndarray) -> QualityVerdict:
        """仅做质量评价，不改变会话状态（供实时提示用）。"""
        from ..algorithms.calibration import detect_chessboard_quality

        det = detect_chessboard_quality(
            image, self.pattern_cols, self.pattern_rows)

        if not det.found:
            return QualityVerdict(
                ok=False, reason="未检测到棋盘格，请对准并调整角度")

        if det.area_ratio < self.AREA_RATIO_MIN:
            return QualityVerdict(
                ok=False,
                reason=f"棋盘格过小（占比 {det.area_ratio * 100:.1f}%），"
                       f"请靠近一些",
                num_corners=det.num_corners,
                area_ratio=det.area_ratio,
                sharpness=det.sharpness,
            )

        if det.min_side_px < self.MIN_SIDE_PX:
            return QualityVerdict(
                ok=False,
                reason=f"棋盘格过密（间距 {det.min_side_px:.0f}px），"
                       f"请靠近一些",
                num_corners=det.num_corners,
                area_ratio=det.area_ratio,
                sharpness=det.sharpness,
            )

        if det.sharpness < self.SHARPNESS_MIN:
            return QualityVerdict(
                ok=False,
                reason=f"图像模糊（清晰度 {det.sharpness:.0f}），请对焦",
                num_corners=det.num_corners,
                area_ratio=det.area_ratio,
                sharpness=det.sharpness,
            )

        if self._is_diff_pair():
            return self._judge_diff_pair(image, det)
        return self._judge_novelty(image, det)

    def _judge_novelty(self, image: np.ndarray,
                       det) -> QualityVerdict:
        """常规模式的位姿新颖性判据（相机内参 / 单图激光 / 转轴）。"""
        # 棋盘法转轴把每张图当作"转台转过一个角度"：两张**完全一样**的图
        # 说明转台没动，必然无效；但"相似"不代表无效（新参数下用更松的
        # 阈值，避免把正常的小角度变化判为重复）。
        novelty_min = (self.NOVELTY_AXIS_MIN
                       if self._is_axis_chessboard() else self.NOVELTY_MIN)
        novelty = self._novelty(image)
        if novelty < novelty_min:
            return QualityVerdict(
                ok=False,
                reason="与已有图片过于相似，请更换棋盘格姿态",
                num_corners=det.num_corners,
                area_ratio=det.area_ratio,
                sharpness=det.sharpness,
                novelty=novelty,
            )

        reason = "质量合格，已收录"
        if self._is_axis_chessboard():
            reason += f"（第 {self.count + 1} 个角度）"
        return QualityVerdict(
            ok=True,
            reason=reason,
            num_corners=det.num_corners,
            area_ratio=det.area_ratio,
            sharpness=det.sharpness,
            novelty=novelty,
        )

    def _judge_diff_pair(self, image: np.ndarray, det) -> QualityVerdict:
        """双图差分模式的判定：整对之内要求"几乎相同"，对与对之间要求"不同"。

        这是把"相似"用反了地方的修复：差分配对的前提就是除激光外画面
        完全不变，因此

        - **偶数位（对首张，关激光）**：与**其它对的对首张**比较，
          必须拉开差异（棋盘姿态要多样），走常规 ``NOVELTY_MIN``；
        - **奇数位（对该对第 2 张，开激光）**：只与前一张比较，必须
          **足够相似**，差异超限说明拍两张之间棋盘被碰动了，差分会把
          板边缘误判成激光。

        注意"关/开激光"由用户操作决定，这里只按拍摄奇偶位分工，无法
        也从不去判断激光是否真的亮着——那是 ``diff_laser_images``
        在标定阶段的职责。
        """
        base = dict(
            num_corners=det.num_corners,
            area_ratio=det.area_ratio,
            sharpness=det.sharpness,
        )

        if self._expects_dark():
            novelty = self._pose_novelty(image)
            if novelty < self.NOVELTY_MIN:
                return QualityVerdict(
                    ok=False,
                    reason="与已有图片过于相似，请更换棋盘格姿态",
                    novelty=novelty, **base)
            pair_no = self.count // 2 + 1
            return QualityVerdict(
                ok=True,
                reason=f"已收录第 {pair_no} 对的关激光帧，"
                       f"请保持棋盘不动、打开激光后拍第 2 张",
                novelty=novelty, **base)

        # 本对的第 2 张：与前一张比，要求高度相似
        diff = 1.0 - self._similarity(image, self._thumbs[-1])
        if diff > self.DIFF_PAIR_MAX_DIFF:
            return QualityVerdict(
                ok=False,
                reason=f"与本对第 1 张差异过大（{diff * 100:.1f}%，"
                       f"上限 {self.DIFF_PAIR_MAX_DIFF * 100:.0f}%），"
                       f"棋盘格可能被碰动。差分要求除激光外画面完全不变，"
                       f"请保持棋盘静止后重拍第 2 张",
                novelty=1.0 - diff, **base)
        pair_no = (self.count + 1) // 2
        return QualityVerdict(
            ok=True,
            reason=f"第 {pair_no} 对已配对（与关激光帧相差 "
                   f"{diff * 100:.1f}%）",
            novelty=1.0 - diff, **base)

    # ------------------------------------------------------------------
    # 去重与落盘
    # ------------------------------------------------------------------
    @staticmethod
    def _thumbnail(image: np.ndarray, size: int = 32) -> np.ndarray:
        """生成统一大小的灰度缩略图（用于快速比较）。"""
        import cv2

        gray = image if image.ndim == 2 else cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        v = small.astype(np.float32).ravel()
        v -= v.mean()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    @staticmethod
    def _similarity(image: np.ndarray, thumb: np.ndarray) -> float:
        """两张图归一化缩略图的相关度（0~1，越大越像）。"""
        n = float(np.dot(CaptureSession._thumbnail(image), thumb))
        return max(0.0, min(1.0, n))

    def _novelty(self, image: np.ndarray) -> float:
        """与已收集帧的最大差异度（基于归一化缩略图相关度）。"""
        if not self._thumbs:
            return 1.0
        v = self._thumbnail(image)
        sims = [float(np.dot(v, t)) for t in self._thumbs]
        return 1.0 - max(sims)

    def _pose_novelty(self, image: np.ndarray) -> float:
        """与**其它对的对首张**的最大差异度。

        差分模式下位姿多样性只应在"对"这一级讨论：同一对的两帧位姿
        必须一致，因此在整对之内比较恒为 0，会误判。这里只和每对的
        第 1 张比较。
        """
        if not self._pose_thumbs:
            return 1.0
        v = self._thumbnail(image)
        sims = [float(np.dot(v, t)) for t in self._pose_thumbs]
        return 1.0 - max(sims)

    def _output_dir(self) -> str:
        sub = {"camera": "camera", "laser": "laser", "axis": "axis"}[
            self.mode.value]
        base = self._cfg_get("output.calib_dir", "data/calibration")
        root = getattr(self.cfg, "root", ".")
        path = os.path.join(root, base, sub) if not os.path.isabs(base) \
            else os.path.join(base, sub)
        return path

    def _save_frame(self, image: np.ndarray, index: int) -> None:
        import cv2

        try:
            out = self._output_dir()
            os.makedirs(out, exist_ok=True)
            path = os.path.join(out, f"{index:03d}.png")
            cv2.imwrite(path, image)
        except Exception as exc:  # noqa: BLE001
            self._log(f"图像保存失败: {exc}")

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.images)

    @property
    def ready(self) -> bool:
        """是否已收集到足够数量。"""
        return self.count >= self.mode.min_views

    def progress_text(self) -> str:
        return (f"{self.count} / {self.mode.recommend_views} 张"
                f"（最少 {self.mode.min_views}）")

    def clear(self) -> None:
        self.images.clear()
        self.verdicts.clear()
        self._thumbs.clear()
        self._pose_thumbs.clear()
        self.started_at = time.time()

    def remove_last(self) -> bool:
        if not self.images:
            return False
        was_pair_head = (self._is_diff_pair()
                         and (len(self.images) - 1) % 2 == 0)
        self.images.pop()
        self._thumbs.pop()
        if was_pair_head and self._pose_thumbs:
            # 撤销的是某一对的第 1 张：该对的位姿记录也要一并撤掉，
            # 否则下一张同姿态的关激光帧会被误判成"过于相似"
            self._pose_thumbs.pop()
        if self.verdicts:
            self.verdicts.pop()
        return True

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    def run_calibration(self) -> CalibOutcome:
        """用已收集的图像执行标定并保存。

        转轴模式下：

        - ``axis_calib.method == "chessboard"``：用 ``images`` 里**手动拍**
          的棋盘格图直接拟合转轴（不需要激光、不需要标记球）；
        - 其它方法用累积的 3D 特征点（见
          :meth:`calibrate_axis_from_points`），此处返回提示。
        """
        t0 = time.time()
        try:
            if self.mode is CalibMode.CAMERA:
                return self._calibrate_camera(t0)
            if self.mode is CalibMode.LASER:
                return self._calibrate_laser(t0)
            if (self.mode is CalibMode.AXIS
                    and str(self._cfg_get("axis_calib.method", ""))
                    == "chessboard"):
                return self._calibrate_axis_chessboard(t0)
            return CalibOutcome(
                ok=False, mode=self.mode,
                message="转轴标定请使用 calibrate_axis_from_points 接口",
                elapsed_s=time.time() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("标定失败")
            return CalibOutcome(
                ok=False, mode=self.mode, message=f"标定失败: {exc}",
                elapsed_s=time.time() - t0,
            )

    def _calibrate_camera(self, t0: float) -> CalibOutcome:
        from ..algorithms.calibration import calibrate_camera

        out_path = self._cfg_get(
            "calibration.camera_file", None)
        if out_path and not os.path.isabs(out_path):
            out_path = os.path.join(getattr(self.cfg, "root", "."), out_path)

        result = calibrate_camera(
            self.images,
            pattern_cols=self.pattern_cols,
            pattern_rows=self.pattern_rows,
            square_size_m=self.square_size_m,
            out_path=out_path,
            use_sb=bool(self._cfg_get("camera_calib.use_sb", True)),
            auto_prune=bool(self._cfg_get("camera_calib.auto_prune", True)),
            min_images_for_pruning=int(
                self._cfg_get("camera_calib.min_images_for_pruning", 8)),
        )
        msg = (f"相机标定完成：使用 {result.num_images} 张图像，"
               f"重投影 RMS = {result.rms:.3f} px")
        if result.rms > 0.5:
            msg += "（警告：RMS 偏大，建议重拍）"
        return CalibOutcome(
            ok=True, mode=self.mode, message=msg,
            quality=f"RMS={result.rms:.3f} px",
            out_path=out_path, elapsed_s=time.time() - t0,
        )

    def _calibrate_laser(self, t0: float) -> CalibOutcome:
        from ..algorithms.calibration import (
            calibrate_laser_plane,
            load_camera_calib,
        )

        cam_file = self._cfg_get("calibration.camera_file", None)
        if cam_file and not os.path.isabs(cam_file):
            cam_file = os.path.join(
                getattr(self.cfg, "root", "."), cam_file)
        if not cam_file or not os.path.exists(cam_file):
            return CalibOutcome(
                ok=False, mode=self.mode,
                message=f"请先完成相机标定：{cam_file}",
                elapsed_s=time.time() - t0,
            )

        cam = load_camera_calib(cam_file)
        pairs = [{"bgr": img} for img in self.images]

        out_path = self._cfg_get("calibration.laser_file", None)
        if out_path and not os.path.isabs(out_path):
            out_path = os.path.join(getattr(self.cfg, "root", "."), out_path)

        use_diff = bool(self._cfg_get("laser_calib.use_diff", False))
        if use_diff:
            pairs, diff_note = self._build_laser_diff_pairs()
            if not pairs:
                return CalibOutcome(
                    ok=False, mode=self.mode,
                    message=("差分标定无法进行：" + "\n"
                             + diff_note).strip(),
                    elapsed_s=time.time() - t0)
        else:
            diff_note = ""

        result = calibrate_laser_plane(
            pairs, cam.K, cam.dist,
            pattern_cols=self.pattern_cols,
            pattern_rows=self.pattern_rows,
            square_size_m=self.square_size_m,
            out_path=out_path,
            view_reject_mm=self._cfg_get("laser_calib.view_reject_mm", 1.0),
            outlier_mm=self._cfg_get("laser_calib.outlier_mm", 0.5),
            clip_margin_px=self._cfg_get("laser_calib.clip_margin_px", 0.0),
            channel="gray" if use_diff
            else self._cfg_get("laser_calib.channel", "red_laser"),
        )
        msg = (f"激光平面标定完成：{result.num_views} 视角，"
               f"{result.num_points} 点，残差 RMS = "
               f"{result.residual_rms * 1e3:.3f} mm")
        if diff_note:
            msg = diff_note.rstrip() + "\n" + msg
        if result.residual_rms > 5e-4:
            msg += "（警告：残差偏大，建议重拍）"
        return CalibOutcome(
            ok=True, mode=self.mode, message=msg,
            quality=f"RMS={result.residual_rms * 1e3:.3f} mm",
            out_path=out_path, elapsed_s=time.time() - t0,
        )

    def _build_laser_diff_pairs(self):
        """把会话里的成对图像转成差分标定输入。

        约定 ``images`` 按**奇数位（0,2,4...）= 无激光**、
        **偶数位（1,3,5...）= 有激光** 交错存放。

        每对先做 :func:`diff_laser_images` 校验（重复拍摄、板移动、
        光源变化都会被拦下），再从差分图提取中心线，构造
        ``{"bgr": 暗图, "centerline": 线}``。

        Returns
        -------
        (pairs, note) : pairs 为可用输入列表；note 为过程说明。
            没有任何可用对时 pairs 为空。
        """
        from ..algorithms.calibration import (diff_laser_images,
                                              extract_centerline_from_diff)

        threshold = float(self._cfg_get("laser_calib.diff_threshold", 60.0))
        use_sb = bool(self._cfg_get("camera_calib.use_sb", True))

        lines = []
        pairs = []
        n_pairs = self.count // 2
        for k in range(n_pairs):
            dark = self.images[2 * k]
            lit = self.images[2 * k + 1]
            res = diff_laser_images(
                dark, lit, threshold=threshold,
                pattern_cols=self.pattern_cols,
                pattern_rows=self.pattern_rows, use_sb=use_sb)
            if not res.ok:
                lines.append(f"  第 {k + 1} 对废弃：{res.reason}")
                continue
            line = extract_centerline_from_diff(
                res.diff, threshold=threshold)
            if len(line) == 0:
                lines.append(
                    f"  第 {k + 1} 对废弃：差分后未提取到激光线"
                    f"（差分亮区 {res.bright_ratio * 100:.2f}%）")
                continue
            pairs.append({"bgr": dark, "centerline": line})
            lines.append(f"  第 {k + 1} 对收录：激光线 {len(line)} 列")

        if self.count % 2:
            lines.append(
                f"  警告：共 {self.count} 张，最后 1 张落单已忽略"
                "（应为成对的「无激光 → 有激光」）")

        head = (f"差分模式：{self.count} 张 → {n_pairs} 对，"
                f"可用 {len(pairs)} 对")
        note = "\n".join([head] + lines)
        for ln in [head] + lines:
            self._log(ln)
        return pairs, note

    def _calibrate_axis_chessboard(self, t0: float) -> CalibOutcome:
        """棋盘格法（手动拍照）：用已收集的棋盘图拟合转轴。

        图像按**拍摄顺序**视为转台旋转顺序；不要求记录每张的转角——
        转轴方向的正负由相邻帧板的相对旋转自动判定。
        """
        from ..algorithms.calibration import (
            ChessboardAxisCollector,
            load_camera_calib,
        )

        cam_file = self._cfg_get("calibration.camera_file", None)
        if cam_file and not os.path.isabs(cam_file):
            cam_file = os.path.join(
                getattr(self.cfg, "root", "."), cam_file)
        if not cam_file or not os.path.exists(cam_file):
            return CalibOutcome(
                ok=False, mode=self.mode,
                message=f"请先完成相机标定：{cam_file}",
                elapsed_s=time.time() - t0)

        cam = load_camera_calib(cam_file)
        collector = ChessboardAxisCollector(
            cam.K, cam.dist,
            pattern_cols=self.pattern_cols,
            pattern_rows=self.pattern_rows,
            square_size_m=self.square_size_m,
            use_sb=bool(self._cfg_get("camera_calib.use_sb", True)),
            use_center=bool(
                self._cfg_get("axis_calib.chessboard_use_center", True)),
            orientation=str(self._cfg_get(
                "axis_calib.chessboard_orientation", "landscape")),
        )
        for img in self.images:
            collector.add_angle(img, angle_deg=None)
        collector.finalize()

        if collector.count < CalibMode.AXIS.min_views:
            detail = "；".join(collector.errors[:5])
            return CalibOutcome(
                ok=False, mode=self.mode,
                message=(f"有效棋盘视角不足（{collector.count} < "
                         f"{CalibMode.AXIS.min_views}）：{detail}"),
                elapsed_s=time.time() - t0)

        out_path = self._cfg_get("calibration.axis_file", None)
        if out_path and not os.path.isabs(out_path):
            out_path = os.path.join(
                getattr(self.cfg, "root", "."), out_path)

        result = collector.calibrate(out_path=out_path)
        rms = float(result.residual_rms)
        msg = (f"转轴标定完成（棋盘法）：{collector.count} 个视角，"
               f"半径 {result.radius * 1e3:.1f} mm，"
               f"残差 RMS = {rms * 1e3:.3f} mm")
        if collector.errors:
            msg += f"；{len(collector.errors)} 张未检出棋盘格"
        if rms > 1e-3:
            msg += "（警告：残差偏大，建议重拍）"
        return CalibOutcome(
            ok=True, mode=self.mode, message=msg,
            quality=f"RMS={rms * 1e3:.3f} mm",
            out_path=out_path, elapsed_s=time.time() - t0,
        )

    def calibrate_axis_from_points(
            self, feature_points: np.ndarray) -> CalibOutcome:
        """由绕轴旋转的特征点三维坐标完成转轴标定。"""
        from ..algorithms.calibration import calibrate_axis

        t0 = time.time()
        out_path = self._cfg_get("calibration.axis_file", None)
        if out_path and not os.path.isabs(out_path):
            out_path = os.path.join(getattr(self.cfg, "root", "."), out_path)
        try:
            result = calibrate_axis(feature_points, out_path=out_path)
        except Exception as exc:  # noqa: BLE001
            log.exception("转轴标定失败")
            return CalibOutcome(
                ok=False, mode=self.mode, message=f"转轴标定失败: {exc}",
                elapsed_s=time.time() - t0,
            )
        msg = (f"转轴标定完成：{result.num_points} 点，"
               f"半径 {result.radius * 1e3:.1f} mm，"
               f"残差 RMS = {result.residual_rms * 1e3:.3f} mm")
        return CalibOutcome(
            ok=True, mode=self.mode, message=msg,
            quality=f"RMS={result.residual_rms * 1e3:.3f} mm",
            out_path=out_path, elapsed_s=time.time() - t0,
        )


# ---------------------------------------------------------------------------
@dataclass
class AxisSweepResult:
    """转轴自动扫描的结果。"""

    ok: bool
    message: str
    num_views: int = 0
    num_failed: int = 0
    radius_m: float = 0.0
    direction: Optional[np.ndarray] = None
    point: Optional[np.ndarray] = None
    residual_rms: float = 0.0
    points_path: Optional[str] = None
    out_path: Optional[str] = None
    elapsed_s: float = 0.0

    @property
    def quality(self) -> str:
        return f"RMS={self.residual_rms * 1e3:.3f} mm"


def sweep_axis_calibration(
    cfg,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    make_camera=None,
    make_turntable=None,
    save_images: bool = False,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> AxisSweepResult:
    """**全自动**转轴标定：转台转 N 个角度，逐个拍照并提取特征。

    取点方法由 ``axis_calib.method`` 决定，**只影响采集器，不影响后续
    拟合与质检**：

    ===========  ==========================================================
    method       采集什么
    ===========  ==========================================================
    depth        标记球的激光深度
    sphere       标记球的圆心像素 + 像素半径
    chessboard   棋盘格板位姿（**不需要激光标定、不需要标记球**）
    ===========  ==========================================================

    因为 chessboard 法不需要激光平面，此时不再要求
    ``calibration.laser_file`` 存在。

    Parameters
    ----------
    cfg : 配置对象
    on_progress : ``(index, total, message)`` 回调，用于界面进度提示
    make_camera : 工厂 ``() -> Camera``；默认用 :class:`~triniscan.camera.Camera`
    make_turntable : 工厂 ``() -> Turntable``；默认用转台控制器
    save_images : 是否把每个角度的图像落盘（默认否）
    stop_flag : ``() -> bool``，返回 True 时提前中止

    Returns
    -------
    AxisSweepResult
    """
    import cv2

    from ..algorithms.calibration import load_camera_calib

    t0 = time.time()

    def _get(key: str, default):
        try:
            return cfg.get(key, default)
        except AttributeError:
            return default

    method = str(_get("axis_calib.method", "peak"))
    # 棋盘法只依赖相机内参：板位姿由整块棋盘给出，与激光无关。
    needs_laser = method != "chessboard"

    def _log_progress(i: int, total: int, msg: str) -> None:
        log.info("[%d/%d] %s", i, total, msg)
        if on_progress is not None:
            on_progress(i, total, msg)

    # --- 前置：相机内参必需；激光平面仅非棋盘法需要 ---
    def _resolve(key: str) -> str:
        try:
            path = cfg.get(key, None)
        except AttributeError:
            path = None
        if not path:
            return ""
        if not os.path.isabs(path):
            path = os.path.join(getattr(cfg, "root", "."), path)
        return path

    cam_file = _resolve("calibration.camera_file")
    if not cam_file or not os.path.exists(cam_file):
        return AxisSweepResult(
            ok=False, message=f"请先完成相机标定：{cam_file}")

    from ..algorithms.calibration import (AxisLiveCollector,
                                          ChessboardAxisCollector,
                                          board_pattern)

    cam = load_camera_calib(cam_file)

    # --- 参数 ---
    radius = float(_get("axis_calib.sphere_radius_m", 0.020))
    num_views = int(_get("axis_calib.num_views", 12))
    step = 360.0 / max(num_views, 1)
    save_name = "标记球" if needs_laser else "棋盘格"

    if needs_laser:
        laser_file = _resolve("calibration.laser_file")
        if not laser_file or not os.path.exists(laser_file):
            return AxisSweepResult(
                ok=False, message=f"请先完成激光平面标定：{laser_file}")
        from ..algorithms.calibration import load_laser_calib
        laser = load_laser_calib(laser_file)
        collector = AxisLiveCollector(
            cam.K, laser.plane, sphere_radius_m=radius,
            method=method, dist=cam.dist,
            algo_cfg={
                "spot_threshold": float(
                    _get("axis_calib.spot_threshold", 60.0)),
                "spot_win_px": int(_get("axis_calib.spot_win_px", 25)),
                "spot_channel": str(_get("axis_calib.spot_channel",
                                         "red_laser")),
                "min_radius_px": float(
                    _get("axis_calib.min_radius_px", 8.0)),
                "max_radius_px": float(
                    _get("axis_calib.max_radius_px", 240.0)),
                "vertical_hint": np.array(
                    [float(x) for x in _get("axis_calib.vertical_hint",
                                            [0.0, 0.0, 1.0])]),
            },
        )
    else:
        cols, rows, square = board_pattern(cfg)
        orient = str(_get("axis_calib.chessboard_orientation", "landscape"))
        total_hint = ("长边水平" if orient == "landscape" else "长边竖直")
        collector = ChessboardAxisCollector(
            cam.K, cam.dist,
            pattern_cols=cols, pattern_rows=rows, square_size_m=square,
            use_sb=bool(_get("camera_calib.use_sb", True)),
            use_center=bool(_get("axis_calib.chessboard_use_center", True)),
            orientation=orient,
        )
        _log_progress(0, num_views,
                      f"棋盘法：{cols}×{rows} 内角点，方格 "
                      f"{square * 1e3:.1f} mm，{total_hint}")

    # --- 打开设备 ---
    if make_camera is None:
        from ..hardware.camera import Camera as _Camera
        make_camera = lambda: _Camera(cfg.raw["camera"])
    if make_turntable is None:
        from ..hardware.turntable import Turntable as _Turntable
        make_turntable = lambda: _Turntable(cfg.raw["turntable"])

    archive_dir = _get("axis_calib.archive_dir",
                       "data/calibration/axis")
    if not os.path.isabs(archive_dir):
        archive_dir = os.path.join(getattr(cfg, "root", "."), archive_dir)
    out_path = _resolve("calibration.axis_file")

    try:
        with make_camera() as camera, make_turntable() as table:
            for i in range(num_views):
                if stop_flag is not None and stop_flag():
                    _log_progress(i, num_views, "已中止")
                    break
                angle = i * step
                if i > 0:
                    table.rotate_deg(step)
                img = camera.grab()
                if img is None:
                    collector.errors.append(f"{angle:.0f}°: 取像失败")
                    _log_progress(i + 1, num_views,
                                  f"{angle:.0f}° 取像失败（已跳过）")
                    continue
                if save_images:
                    try:
                        os.makedirs(archive_dir, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(archive_dir, f"{i:03d}.png"), img)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("图像保存失败: %s", exc)
                # 两种采集器的 add_angle 语义不同：AxisLive 返回点坐标
                # （数组，真值有歧义），Chessboard 返回 bool。统一以"失败
                # 列表是否增长"判断本帧是否收录成功。
                n_err = len(collector.errors)
                collector.add_angle(img, angle_deg=angle)
                if len(collector.errors) > n_err:
                    _log_progress(i + 1, num_views,
                                  f"{angle:.0f}° 未找到{save_name}（已跳过）")
                else:
                    _log_progress(i + 1, num_views,
                                  f"{angle:.0f}° 已收录")
    except Exception as exc:  # noqa: BLE001
        log.exception("转轴扫描失败")
        return AxisSweepResult(
            ok=False, message=f"设备或采集失败：{exc}",
            num_views=collector.count,
            elapsed_s=time.time() - t0)

    # 棋盘法：全序列到齐后统一消歧（两种角点排列都试过，这里按朝向定稿）
    if hasattr(collector, "finalize"):
        collector.finalize()

    hint = ("请确保每张图都能完整拍到棋盘格（板固定在转台上并离轴放置）"
            if not needs_laser else
            "请检查标记球、环境光与取点方法")
    if collector.count < 3:
        return AxisSweepResult(
            ok=False,
            message=(f"有效视角不足（{collector.count} < 3）。{hint}。"),
            num_views=collector.count, num_failed=len(collector.errors),
            elapsed_s=time.time() - t0)

    min_views = int(_get("axis_calib.min_views", 6))
    if collector.count < min_views:
        return AxisSweepResult(
            ok=False,
            message=(f"有效视角不足（{collector.count} < {min_views}）。"
                     f"转轴标定不可靠，请重拍：{hint}。"),
            num_views=collector.count, num_failed=len(collector.errors),
            elapsed_s=time.time() - t0)

    # --- 标定（复用原有 calibrate_axis）---
    points_path = os.path.join(archive_dir, "axis_points.npy")
    try:
        result = collector.calibrate(out_path=out_path or None,
                                     save_points_to=points_path)
    except Exception as exc:  # noqa: BLE001
        log.exception("转轴标定失败")
        return AxisSweepResult(
            ok=False, message=f"拟合失败：{exc}",
            num_views=collector.count, num_failed=len(collector.errors),
            elapsed_s=time.time() - t0)

    # --- 质检：不合格**不写入**文件，避免坏标定被静默使用 ---
    rms = float(result.residual_rms)
    radius_mm = float(result.radius) * 1e3
    max_res_mm = float(_get("axis_calib.max_residual_mm", 1.0))
    min_rad_mm = float(_get("axis_calib.min_radius_mm", 2.0))
    problems = []
    if rms * 1e3 > max_res_mm:
        problems.append(
            f"残差 RMS {rms * 1e3:.3f}mm > {max_res_mm:.3f}mm")
    if abs(radius_mm) < min_rad_mm:
        problems.append(
            f"拟合圆半径 {radius_mm:.2f}mm < {min_rad_mm:.2f}mm"
            f"（特征点几乎重合，说明取点失败）")
    if problems:
        # 不要留下坏的结果文件
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        return AxisSweepResult(
            ok=False,
            message=("转轴标定质检未通过，已废弃："
                     + "；".join(problems)
                     + f"。{hint}后重拍。"),
            num_views=collector.count, num_failed=len(collector.errors),
            elapsed_s=time.time() - t0)

    msg = (f"转轴标定完成：{collector.count} 个视角，"
           f"半径 {result.radius * 1e3:.1f} mm，"
           f"残差 RMS {rms * 1e3:.3f} mm")
    if rms > 1e-3:
        msg += "（警告：残差偏大，建议重拍）"
    if collector.errors:
        msg += f"；{len(collector.errors)} 个角度被跳过"

    return AxisSweepResult(
        ok=True, message=msg,
        num_views=collector.count, num_failed=len(collector.errors),
        radius_m=float(result.radius),
        direction=result.direction, point=result.point,
        residual_rms=rms,
        points_path=points_path,
        out_path=out_path or None,
        elapsed_s=time.time() - t0,
    )
