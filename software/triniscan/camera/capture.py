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
    """相机配置。

    Notes
    -----
    ``backend`` 取值：

    - ``"msmf"``（默认）：Windows Media Foundation。与 ``dshow`` 不同，
      它允许相机输出**压缩流（MJPG）**，从而突破 USB 带宽限制。
      实测同一台 1080p 相机：``dshow`` 强制 YUY2 未压缩只有约 5 FPS，
      ``msmf`` 可达约 30 FPS。
    - ``"dshow"``：DirectShow。锁定曝光/增益等参数最可靠，但通常
      只能跑未压缩 YUY2，高分辨率下帧率低。若 ``msmf`` 下无法锁定
      曝光，可切回此项。
    - ``"any"``：交给 OpenCV 自动选择。
    """

    index: int = 0
    backend: str = "msmf"           # "msmf" | "dshow" | "any"
    width: int = 1920
    height: int = 1080
    fourcc: str = "MJPG"            # "MJPG" | "YUY2" | ""(不改)
    fps: float = 30.0               # 目标帧率（0 表示不设置）
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
            backend=str(get("backend", "msmf")).lower(),
            width=int(get("width", 1920)),
            height=int(get("height", 1080)),
            fourcc=str(get("fourcc", "MJPG") or ""),
            fps=float(get("fps", 30.0) or 0.0),
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
        backend = _resolve_backend(self.cfg.backend)
        self._cap = cv2.VideoCapture(self.cfg.index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"无法打开相机 index={self.cfg.index} "
                f"(backend={self.cfg.backend})"
            )

        # 顺序很重要：先设 FOURCC（像素格式/压缩），再设分辨率与帧率。
        # 若只设分辨率，DSHOW 下相机会默认输出未压缩 YUY2，
        # 1080p 时受 USB 带宽限制只能跑到约 5 FPS。
        self._apply_format()
        self._lock_params()
        self._warmup()

        real_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = self._cap.get(cv2.CAP_PROP_FPS)
        log.info("相机已打开: %dx%d @ %.1f fps (index=%d, backend=%s, "
                 "fourcc=%s)", real_w, real_h, real_fps,
                 self.cfg.index, self.cfg.backend,
                 self.cfg.fourcc or "默认")
        return self

    def _apply_format(self) -> None:
        """设置像素格式 / 分辨率 / 帧率（顺序敏感）。"""
        assert self._cap is not None
        if self.cfg.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC,
                          cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        if self.cfg.width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        if self.cfg.height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if self.cfg.fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # 限制驱动内部缓冲，避免读到陈旧帧（降低延迟）
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - 部分后端不支持
            pass

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
    def read(self) -> np.ndarray:
        """读取单帧（不做多次丢弃），用于实时取景。

        实时预览要的是"最新帧"，逐帧丢弃只会降低有效帧率；
        真正需要"稳定帧"的抓图请用 :meth:`grab`。
        """
        if not self.is_open:
            raise RuntimeError("相机未打开")
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("相机读取失败")
        return frame

    def grab(self) -> np.ndarray:
        """抓取一帧稳定图像（丢弃前几帧，供正式采集/标定用）。"""
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


def _resolve_backend(name: str) -> int:
    """把后端名解析成 OpenCV 常量。"""
    table = {
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "any": cv2.CAP_ANY,
    }
    return table.get(str(name).lower(), cv2.CAP_ANY)


def _make_getter(cfg):
    """把 Config / dict / dataclass 统一成 getter。"""
    if hasattr(cfg, "get"):
        return cfg.get
    if isinstance(cfg, dict):
        return lambda k, d=None: cfg.get(k, d)
    return lambda k, d=None: getattr(cfg, k, d)
