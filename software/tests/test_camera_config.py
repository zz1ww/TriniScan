"""相机配置与后端解析的测试（不依赖真实硬件）。

覆盖要点
--------
- ``CameraConfig.from_config`` 读取新增的 ``fourcc`` / ``fps`` 字段；
- 后端名到 OpenCV 常量的映射（msmf / dshow / any）；
- 默认后端为 msmf（帧率关键），默认像素格式 MJPG。
"""
import cv2
import pytest

from triniscan.camera import Camera, CameraConfig
from triniscan.camera.capture import _resolve_backend


def test_defaults_favor_high_fps():
    """默认应选 msmf + MJPG，避免 DSHOW/YUY2 的低帧率陷阱。"""
    cfg = CameraConfig()
    assert cfg.backend == "msmf"
    assert cfg.fourcc == "MJPG"
    assert cfg.fps == 30.0


def test_from_config_reads_new_fields():
    raw = {
        "index": 1, "backend": "msmf", "width": 1920, "height": 1080,
        "fourcc": "MJPG", "fps": 30, "auto_exposure": False,
        "exposure": -6, "gain": 0, "auto_wb": False, "discard_frames": 3,
    }
    cfg = CameraConfig.from_config(raw)
    assert cfg.index == 1
    assert cfg.backend == "msmf"
    assert cfg.fourcc == "MJPG"
    assert cfg.fps == 30.0


def test_from_config_defaults_when_missing():
    """旧配置缺 fourcc/fps 时也要给出合理默认。"""
    cfg = CameraConfig.from_config({"index": 0})
    assert cfg.fourcc == "MJPG"
    assert cfg.fps == 30.0
    assert cfg.backend == "msmf"


def test_backend_normalised_to_lowercase():
    cfg = CameraConfig.from_config({"backend": "DSHOW"})
    assert cfg.backend == "dshow"


def test_fps_zero_means_unset():
    cfg = CameraConfig.from_config({"fps": 0})
    assert cfg.fps == 0.0


@pytest.mark.parametrize("name,expected", [
    ("msmf", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
    ("dshow", getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)),
    ("any", cv2.CAP_ANY),
    ("unknown", cv2.CAP_ANY),
])
def test_resolve_backend(name, expected):
    assert _resolve_backend(name) == expected


def test_resolve_backend_case_insensitive():
    assert _resolve_backend("MSMF") == getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)


def test_camera_accepts_config_object():
    cam = Camera(CameraConfig())
    assert cam.cfg.backend == "msmf"
    assert not cam.is_open


def test_camera_accepts_plain_dict():
    cam = Camera({"index": 1, "backend": "msmf", "fourcc": "MJPG"})
    assert cam.cfg.index == 1
    assert cam.cfg.fourcc == "MJPG"
