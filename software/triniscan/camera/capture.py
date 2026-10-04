"""相机采集：打开、锁定参数、抓图、存图。"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


class Camera:
    """UVC/USB 相机封装。

    关键：必须锁定曝光/增益/白平衡，保证激光提取稳定。
    """

    def __init__(self, cfg) -> None:
        self.index: int = cfg.get("index", 0)
        self.backend: str = cfg.get("backend", "dshow")
        self.width: int = cfg.get("width", 1920)
        self.height: int = cfg.get("height", 1080)
        self.auto_exposure: bool = cfg.get("auto_exposure", False)
        self.exposure: float = cfg.get("exposure", -6)
        self.gain: float = cfg.get("gain", 0)
        self.auto_wb: bool = cfg.get("auto_wb", False)
        self.discard_frames: int = cfg.get("discard_frames", 3)

        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    def open(self) -> None:
        backend = cv2.CAP_DSHOW if self.backend == "dshow" \
            else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开相机 index={self.index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._lock_params()
        self._warmup()

    def _lock_params(self) -> None:
        assert self._cap is not None
        if self.auto_exposure:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        self._cap.set(cv2.CAP_PROP_GAIN, self.gain)
        self._cap.set(cv2.CAP_PROP_AUTO_WB,
                      1 if self.auto_wb else 0)

    def _warmup(self) -> None:
        """丢弃前若干帧，等曝光稳定。"""
        for _ in range(max(1, self.discard_frames)):
            self._cap.read()

    # ------------------------------------------------------------------
    def grab(self) -> np.ndarray:
        """抓取一帧并丢弃若干帧保证稳定。"""
        assert self._cap is not None, "相机未打开"
        frame = None
        for _ in range(max(1, self.discard_frames)):
            ret, f = self._cap.read()
            if not ret:
                raise RuntimeError("相机读取失败")
            frame = f
        return frame

    def grab_and_save(self, path: str) -> np.ndarray:
        frame = self.grab()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cv2.imwrite(path, frame)
        return frame

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    # 简单自测：打开相机并预览
    from triniscan.core.config import Config

    cam = Camera(Config.load().camera)
    cam.open()
    print("相机已打开，按 q 退出预览")
    while True:
        frame = cam.grab()
        cv2.imshow("TriniScan preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.close()
    cv2.destroyAllWindows()
