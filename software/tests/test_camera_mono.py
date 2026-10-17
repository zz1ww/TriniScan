"""相机黑白（单色）适配测试。"""
import numpy as np

from triniscan.camera.capture import CameraConfig, _fourcc_to_str


def test_mono_default_false():
    cfg = CameraConfig()
    assert cfg.mono is False
    assert cfg.fourcc_mono == "Y800"


def test_from_config_reads_mono():
    cfg = CameraConfig.from_config({
        "mono": True,
        "fourcc_mono": "GREY",
        "width": 1280,
        "height": 720,
    })
    assert cfg.mono is True
    assert cfg.fourcc_mono == "GREY"
    assert (cfg.width, cfg.height) == (1280, 720)


def test_from_config_defaults_when_missing():
    cfg = CameraConfig.from_config({})
    assert cfg.mono is False
    assert cfg.fourcc_mono == "Y800"


def test_from_config_object_getter():
    class _C:
        def get(self, k, d=None):
            return {"mono": True}.get(k, d)

    cfg = CameraConfig.from_config(_C())
    assert cfg.mono is True


def test_fourcc_to_str_roundtrip():
    import cv2
    val = cv2.VideoWriter_fourcc(*"Y800")
    assert _fourcc_to_str(val) == "Y800"
    assert _fourcc_to_str(0) == ""


def test_fourcc_to_str_garbage_is_empty():
    # 不可打印值不应返回四字符码
    assert _fourcc_to_str(0x00000001) == ""


def test_fourcc_to_str_mjpg():
    import cv2
    assert _fourcc_to_str(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"
