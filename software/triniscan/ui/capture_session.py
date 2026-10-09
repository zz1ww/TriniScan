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
        self._thumbs: list[np.ndarray] = []   # 低分辨率缩略图，用于去重
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
            self.images.append(image.copy())
            self._thumbs.append(self._thumbnail(image))
            if self.save_images:
                self._save_frame(image, len(self.images) - 1)
        return verdict

    def evaluate(self, image: np.ndarray) -> QualityVerdict:
        """仅做质量评价，不改变会话状态（供实时提示用）。"""
        from ..calibration import detect_chessboard_quality

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

        novelty = self._novelty(image)
        if novelty < self.NOVELTY_MIN:
            return QualityVerdict(
                ok=False,
                reason="与已有图片过于相似，请更换棋盘格姿态",
                num_corners=det.num_corners,
                area_ratio=det.area_ratio,
                sharpness=det.sharpness,
                novelty=novelty,
            )

        return QualityVerdict(
            ok=True,
            reason="质量合格，已收录",
            num_corners=det.num_corners,
            area_ratio=det.area_ratio,
            sharpness=det.sharpness,
            novelty=novelty,
        )

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

    def _novelty(self, image: np.ndarray) -> float:
        """与已收集帧的最大差异度（基于归一化缩略图相关度）。"""
        if not self._thumbs:
            return 1.0
        v = self._thumbnail(image)
        sims = [float(np.dot(v, t)) for t in self._thumbs]
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
        self.started_at = time.time()

    def remove_last(self) -> bool:
        if not self.images:
            return False
        self.images.pop()
        self._thumbs.pop()
        if self.verdicts:
            self.verdicts.pop()
        return True

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    def run_calibration(self) -> CalibOutcome:
        """用已收集的图像执行标定并保存。

        转轴模式用 ``images`` 中累积的 3D 特征点（见
        :meth:`add_axis_points`）而非图像，故此处仅处理相机/激光。
        """
        t0 = time.time()
        try:
            if self.mode is CalibMode.CAMERA:
                return self._calibrate_camera(t0)
            if self.mode is CalibMode.LASER:
                return self._calibrate_laser(t0)
            return CalibOutcome(
                ok=False, mode=self.mode,
                message="转轴标定请使用 add_axis_points 接口",
                elapsed_s=time.time() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("标定失败")
            return CalibOutcome(
                ok=False, mode=self.mode, message=f"标定失败: {exc}",
                elapsed_s=time.time() - t0,
            )

    def _calibrate_camera(self, t0: float) -> CalibOutcome:
        from ..calibration import calibrate_camera

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
        from ..calibration import (
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
        pairs = [{"gray": img} for img in self.images]

        out_path = self._cfg_get("calibration.laser_file", None)
        if out_path and not os.path.isabs(out_path):
            out_path = os.path.join(getattr(self.cfg, "root", "."), out_path)

        result = calibrate_laser_plane(
            pairs, cam.K, cam.dist,
            pattern_cols=self.pattern_cols,
            pattern_rows=self.pattern_rows,
            square_size_m=self.square_size_m,
            out_path=out_path,
        )
        msg = (f"激光平面标定完成：{result.num_views} 视角，"
               f"{result.num_points} 点，残差 RMS = "
               f"{result.residual_rms * 1e3:.3f} mm")
        if result.residual_rms > 5e-4:
            msg += "（警告：残差偏大，建议重拍）"
        return CalibOutcome(
            ok=True, mode=self.mode, message=msg,
            quality=f"RMS={result.residual_rms * 1e3:.3f} mm",
            out_path=out_path, elapsed_s=time.time() - t0,
        )

    def calibrate_axis_from_points(
            self, feature_points: np.ndarray) -> CalibOutcome:
        """由绕轴旋转的特征点三维坐标完成转轴标定。"""
        from ..calibration import calibrate_axis

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
    """**全自动**转轴标定：转台转 N 个角度，逐个拍照并提取标记球。

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

    from ..calibration import AxisLiveCollector, load_camera_calib
    from ..calibration import load_laser_calib

    t0 = time.time()

    def _log_progress(i: int, total: int, msg: str) -> None:
        log.info("[%d/%d] %s", i, total, msg)
        if on_progress is not None:
            on_progress(i, total, msg)

    # --- 前置：两个标定文件必须存在 ---
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
    laser_file = _resolve("calibration.laser_file")
    if not cam_file or not os.path.exists(cam_file):
        return AxisSweepResult(
            ok=False, message=f"请先完成相机标定：{cam_file}")
    if not laser_file or not os.path.exists(laser_file):
        return AxisSweepResult(
            ok=False, message=f"请先完成激光平面标定：{laser_file}")

    cam = load_camera_calib(cam_file)
    laser = load_laser_calib(laser_file)

    # --- 参数 ---
    def _get(key: str, default):
        try:
            return cfg.get(key, default)
        except AttributeError:
            return default

    radius = float(_get("axis_calib.sphere_radius_m", 0.020))
    method = str(_get("axis_calib.method", "peak"))
    num_views = int(_get("axis_calib.num_views", 12))
    step = 360.0 / max(num_views, 1)

    collector = AxisLiveCollector(
        cam.K, laser.plane, sphere_radius_m=radius,
        method=method, dist=cam.dist)
    collector._algo_cfg = {
        "spot_threshold": float(_get("axis_calib.spot_threshold", 60.0)),
        "spot_win_px": int(_get("axis_calib.spot_win_px", 25)),
        "min_radius_px": float(_get("axis_calib.min_radius_px", 8.0)),
        "max_radius_px": float(_get("axis_calib.max_radius_px", 240.0)),
    }

    # --- 打开设备 ---
    if make_camera is None:
        from ..camera import Camera as _Camera
        make_camera = lambda: _Camera(cfg.raw["camera"])
    if make_turntable is None:
        from ..turntable.controller import Turntable as _Turntable
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
                    continue
                if save_images:
                    try:
                        os.makedirs(archive_dir, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(archive_dir, f"{i:03d}.png"), img)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("图像保存失败: %s", exc)
                point = collector.add_angle(img, angle_deg=angle)
                if point is None:
                    _log_progress(i + 1, num_views,
                                  f"{angle:.0f}° 未找到标记球（已跳过）")
                else:
                    _log_progress(i + 1, num_views,
                                  f"{angle:.0f}° 已收录")
    except Exception as exc:  # noqa: BLE001
        log.exception("转轴扫描失败")
        return AxisSweepResult(
            ok=False, message=f"设备或采集失败：{exc}",
            num_views=collector.count,
            elapsed_s=time.time() - t0)

    if collector.count < 3:
        return AxisSweepResult(
            ok=False,
            message=(f"有效视角不足（{collector.count} < 3）。"
                     f"请检查标记球、环境光与取点方法。"),
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

    rms = float(result.residual_rms)
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
