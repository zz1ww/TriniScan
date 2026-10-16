"""一键测量工具（``triniscan.tools.measure``）的单元测试。

覆盖要点
--------
- CLI 参数解析（--check / --replay / --no-report）；
- ``--check`` 在标定文件齐全时返回 0，缺失时返回 1；
- ``_write_report`` 原子写出 JSON + 文本两版报告；
- ``run_replay`` 在无有效图像时报错（不静默成功）。
"""
from __future__ import annotations

import json
import os

import pytest

from triniscan.tools import measure as M


def _cfg_with(tmp_path, overrides=None):
    """构造一个最小配置对象，resolve() 指向 tmp_path。"""
    from triniscan.core.config import Config

    data = {
        "output": {
            "raw_dir": str(tmp_path / "raw"),
            "pointcloud_dir": str(tmp_path / "pc"),
            "mesh_dir": str(tmp_path / "mesh"),
            "result_dir": str(tmp_path / "res"),
        },
        "calibration": {
            "camera_file": str(tmp_path / "cam.npz"),
            "laser_file": str(tmp_path / "laser.npz"),
            "axis_file": str(tmp_path / "axis.npz"),
            "axis_mode": "calibrated",
        },
        "scan": {"num_views": 360, "angle_step_deg": 1.0},
        "_meta": {"config_path": str(tmp_path / "default.yaml")},
    }
    if overrides:
        for k, v in overrides.items():
            data.setdefault(k, {}).update(v)

    obj = Config(data, root=str(tmp_path))
    return obj


def test_parser_defaults():
    args = M.build_parser().parse_args([])
    assert args.replay is None
    assert args.check is False
    assert args.no_report is False


def test_parser_replay_optional_dir():
    args = M.build_parser().parse_args(["--replay"])
    assert args.replay == ""          # 省略 DIR → 空串（用默认目录）
    args = M.build_parser().parse_args(["--replay", "/tmp/x"])
    assert args.replay == "/tmp/x"


def test_check_missing_files_returns_1(tmp_path):
    cfg = _cfg_with(tmp_path)
    assert M._check(cfg) == 1


def test_check_present_files_returns_0(tmp_path):
    cfg = _cfg_with(tmp_path)
    for key in ("calibration.camera_file", "calibration.laser_file",
                "calibration.axis_file"):
        path = cfg.resolve(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"x")
    assert M._check(cfg) == 0


def test_write_report_creates_both_formats(tmp_path):
    cfg = _cfg_with(tmp_path)
    result = {
        "volume_cm3": 125.5, "volume_m3": 0.0001255,
        "num_views": 360, "num_points": 20000,
        "mesh_watertight": True, "close_fallback": "convex_hull",
        "cross_check_cm3": 128.0, "cross_check_diff": 0.02,
        "hull_volume_cm3": 121.0,
        "mesh_path": "m.obj", "points_path": "c.ply",
        "mode": "replay",
    }
    M._write_report(cfg, result)

    res_dir = cfg.resolve("output.result_dir")
    json_path = os.path.join(res_dir, "measure.json")
    txt_path = os.path.join(res_dir, "measure.txt")
    assert os.path.exists(json_path)
    assert os.path.exists(txt_path)
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["volume_cm3"] == 125.5
    assert "created_at" in payload
    text = open(txt_path, encoding="utf-8").read()
    assert "TriniScan 测量报告" in text
    assert "凸包体积" in text


def test_load_images_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        M._load_images(str(tmp_path))


def test_run_replay_rejects_empty_dir(tmp_path):
    cfg = _cfg_with(tmp_path)
    cv2 = pytest.importorskip("cv2")
    os.makedirs(cfg.resolve("output.raw_dir"), exist_ok=True)
    with pytest.raises(FileNotFoundError):
        M.run_replay(cfg, save_report=False)
