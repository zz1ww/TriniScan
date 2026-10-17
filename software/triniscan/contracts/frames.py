"""数据契约：跨层传递的数据结构。

问题
----
重构前跨层传的是**裸 numpy 数组和裸 dict**：

- ``ViewCloud(angle_deg, points)`` 在 ``reconstruction/register.py``
- ``views`` 是 ``list[dict]``，键名 ``index/angle_deg/image/raw`` 靠约定
- ``pcd`` 是 Open3D 对象，看不出是相机系还是世界系

**坐标系靠人记**：``filter_by_depth`` 要相机系，``filter_by_bounds`` 要
世界系，两个函数签名一模一样（都是 ``(N,3)`` 数组），写错了不报错，
只会静默删错点。

方案
----
用 dataclass 包一层，**让坐标系成为类型的一部分**：

- :class:`ViewFrame` —— 一个视角（图像 + 角度 + 索引）
- :class:`Cloud` —— 点云（带坐标系数值 + 来源信息）
- :class:`CalibrationBundle` —— 三类标定的集合 + 指纹

注意：这里**不引入 Open3D 依赖**（contracts 保持轻量），
Open3D 对象由 algorithms 层负责转换。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["ViewFrame", "Cloud", "CalibrationBundle"]


# ---------------------------------------------------------------------------
@dataclass
class ViewFrame:
    """一个视角的采集数据。

    替代重构前那个"键名靠约定"的 dict：
    ``{"index": i, "angle_deg": ..., "image": path, "raw": ndarray}``
    """

    index: int
    angle_deg: float
    raw: np.ndarray                                  # BGR 或灰度原图
    image_path: Optional[str] = None

    @property
    def is_mono(self) -> bool:
        """相机是否输出单通道（黑白）。"""
        return self.raw.ndim == 2

    @classmethod
    def from_dict(cls, d: dict) -> "ViewFrame":
        """兼容旧的 dict 形式（采集模块仍返回 dict 时使用）。"""
        return cls(index=int(d.get("index", 0)),
                   angle_deg=float(d.get("angle_deg", 0.0)),
                   raw=d.get("raw"),
                   image_path=d.get("image"))

    def to_dict(self) -> dict:
        """转回旧 dict（兼容仍按 dict 消费的旧代码路径）。"""
        return {"index": self.index, "angle_deg": self.angle_deg,
                "image": self.image_path, "raw": self.raw}


# ---------------------------------------------------------------------------
@dataclass
class Cloud:
    """一束点云（N,3）+ 坐标系标记。

    Attributes
    ----------
    points : (N, 3) float64
    frame : ``"camera"`` 或 ``"world"``。**由构造方负责正确标注**，
        下游可以用 :meth:`require` 断言，避免"把相机系点当世界系过滤"。
    source : 来源说明（如 ``"view_0007"``），便于追踪异常点。
    """

    points: np.ndarray
    frame: str = "world"
    source: str = ""

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float64)
        if pts.ndim == 1 and pts.size == 0:
            pts = pts.reshape(0, 3)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"点云形状必须是 (N,3)，收到 {pts.shape}")
        if self.frame not in ("camera", "world"):
            raise ValueError(f"frame 必须是 'camera' 或 'world'，收到 {self.frame!r}")
        self.points = pts

    def __len__(self) -> int:
        return len(self.points)

    @property
    def camera_frame(self) -> "Cloud":
        return Cloud(self.points, "camera", self.source)

    @property
    def world_frame(self) -> "Cloud":
        return Cloud(self.points, "world", self.source)

    def require(self, frame: str) -> "Cloud":
        """断言坐标系——写错坐标系时**立即报错**而不是静默删错点。"""
        if self.frame != frame:
            raise ValueError(
                f"点云坐标系为 '{self.frame}'，但此处需要 '{frame}'"
                f"（来源: {self.source or '未标注'}）")
        return self

    def subset(self, mask: np.ndarray) -> "Cloud":
        return Cloud(self.points[np.asarray(mask, bool)], self.frame,
                     self.source)

    # 兼容旧代码：有些地方直接当数组用
    def to_array(self) -> np.ndarray:
        return self.points


# ---------------------------------------------------------------------------
@dataclass
class CalibrationBundle:
    """三类标定的集合（相机内参 + 激光平面 + 转轴）。

    与重构前分别传 ``cam_calib / laser_calib / axis`` 三个参数相比，
    一个对象带来的好处：
    - **指纹**：:meth:`fingerprint` 把标定数值 + 关键参数算成 hash，
      写进扫描报告。没有它，你会把"换了标定"误判成"算法不稳定"。
    - 传递统一：插件从 ``ctx.calib`` 一处取，不必层层传参。
    """

    camera: Any = None          # CameraCalibResult
    laser: Any = None           # LaserCalibResult
    axis: Any = None            # AxisCalibResult
    reports: list[dict] = field(default_factory=list)
    axis_mode: str = "calibrated"

    # ------------------------------------------------------------------
    @property
    def K(self) -> Optional[np.ndarray]:
        return None if self.camera is None else self.camera.K

    @property
    def dist(self) -> Optional[np.ndarray]:
        return None if self.camera is None else self.camera.dist

    @property
    def plane(self) -> Optional[np.ndarray]:
        return None if self.laser is None else self.laser.plane

    @property
    def axis_point(self) -> np.ndarray:
        return np.asarray(self.axis.point, dtype=np.float64)

    @property
    def axis_dir(self) -> np.ndarray:
        d = np.asarray(self.axis.direction, dtype=np.float64)
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else d

    # ------------------------------------------------------------------
    def fingerprint(self) -> str:
        """标定指纹（12 位十六进制）。

        把相机 K/dist、激光平面、轴点/轴向量 + 转轴模式一起哈希。
        **同一指纹下两次扫描的差异才是算法差异。**
        """
        h = hashlib.sha256()
        for arr in (self.K, self.dist, self.plane,
                    None if self.axis is None else self.axis.point,
                    None if self.axis is None else self.axis.direction):
            if arr is None:
                h.update(b"\x00")
            else:
                h.update(np.round(np.asarray(arr, float), 9).tobytes())
        h.update(self.axis_mode.encode("utf-8"))
        return h.hexdigest()[:12]
