"""相机采集：打开、锁定参数、抓图、存图。

关键
----
线结构光测量对曝光极其敏感，必须**锁定**曝光/增益/白平衡，否则
激光亮度漂移会直接污染中心线提取与三角测量。UVC 相机在 Windows 上
锁定参数的可靠做法：
- 用 ``CAP_DSHOW`` 后端；
- 关闭自动曝光/自动白平衡；
- 打开后丢弃若干帧，等曝光收敛。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

from ..common.io_utils import ensure_parent
from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["CameraConfig", "Camera"]


@dataclass
class CameraConfig:
    """相机配置。"""

    index: int = 0
    backend: str = "dshow"          # "dshow" | "any"
    width: int = 1920
    height: int = 1080
    auto_exposure: bool = False
    exposure: float = -6.0
    gain: float = 0.0
    auto_wb: bool = False
    discard_frames: int = 3

    @classmethod
    def from_config(cls, cfg) -> "CameraConfig":
        """从通用配置对象构造。"""
        get = _make_getter(cfg)
        return cls(
            index=int(get("index", 0)),
            backend=str(get("backend", "dshow")),
            width=int(get("width", 1920)),
            height=int(get("height", 1080)),
            auto_exposure=bool(get("auto_exposure", False)),
            exposure=float(get("exposure", -6.0)),
            gain=float(get("gain", 0.0)),
            auto_wb=bool(get("auto_wb", False)),
            discard_frames=int(get("discard_frames", 3)),
        )


class Camera:
    """USB/UVC 相机封装（上下文管理器）。"""

    def __init__(self, config: Union[CameraConfig, dict, object]) -> None:
        if isinstance(config, CameraConfig):
            self.cfg = config
        else:
            self.cfg = CameraConfig.from_config(config)
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> "Camera":
        """打开相机并锁定参数。"""
        backend = cv2.CAP_DSHOW if self.cfg.backend == "dshow" \
            else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.cfg.index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"无法打开相机 index={self.cfg.index} "
                f"(backend={self.cfg.backend})"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._lock_params()
        self._warmup()

        real_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("相机已打开: %dx%d (index=%d)", real_w, real_h,
                 self.cfg.index)
        return self

    def _lock_params(self) -> None:
        assert self._cap is not None
        if self.cfg.auto_exposure:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.cfg.exposure)
        self._cap.set(cv2.CAP_PROP_GAIN, self.cfg.gain)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 1 if self.cfg.auto_wb else 0)

    def _warmup(self) -> None:
        """丢弃前若干帧，使曝光稳定。"""
        n = max(1, self.cfg.discard_frames)
        for _ in range(n):
            if self._cap is not None:
                self._cap.read()

    # ------------------------------------------------------------------
    def grab(self) -> np.ndarray:
        """抓取一帧稳定图像（丢弃前几帧）。"""
        if not self.is_open:
            raise RuntimeError("相机未打开")
        frame = None
        for _ in range(max(1, self.cfg.discard_frames)):
            ok, f = self._cap.read()
            if not ok:
                raise RuntimeError("相机读取失败")
            frame = f
        return frame

    def grab_and_save(self, path: str) -> np.ndarray:
        """抓图并保存到磁盘。"""
        frame = self.grab()
        ensure_parent(path)
        if not cv2.imwrite(path, frame):
            raise IOError(f"无法写入图像: {path}")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.debug("相机已关闭")

    # ------------------------------------------------------------------
    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


def _make_getter(cfg):
    """把 Config / dict / dataclass 统一成 getter。"""
    if hasattr(cfg, "get"):
        return cfg.get
    if isinstance(cfg, dict):
        return lambda k, d=None: cfg.get(k, d)
    return lambda k, d=None: getattr(cfg, k, d)
