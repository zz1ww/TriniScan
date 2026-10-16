"""轴向裁剪 / 主体区间自动识别的单元测试。

覆盖要点
--------
- ``_dominant_band`` 能从"主体 + 上下孤立小簇"中正确挑出主体区间；
- ``ScanPipeline._clip_axial_band`` 在给出门限时按高度/半径裁剪；
- ``auto_trim`` 能自动剔除物体上下方的背景环带；
- 关掉轴向裁剪时不改变点云。
"""
from __future__ import annotations

import numpy as np
import pytest

from triniscan.core.pipeline import _dominant_band


def test_dominant_band_picks_main_body():
    """主体在 -20~30mm，上下各有一小簇，应只挑主体。"""
    rng = np.random.default_rng(0)
    body = rng.uniform(-0.020, 0.030, 5000)
    top = rng.uniform(0.070, 0.075, 300)
    bottom = rng.uniform(-0.050, -0.045, 300)
    along = np.concatenate([body, top, bottom])
    band = _dominant_band(along)
    assert band is not None
    lo, hi = band
    assert -0.025 <= lo <= -0.015
    assert 0.025 <= hi <= 0.035


def test_dominant_band_handles_uniform():
    """均匀分布无孤立簇时，应覆盖大致全区间。"""
    rng = np.random.default_rng(1)
    along = rng.uniform(0.0, 0.05, 4000)
    band = _dominant_band(along)
    assert band is not None
    lo, hi = band
    assert lo < 0.01 and hi > 0.04


def test_dominant_band_too_few_points():
    assert _dominant_band(np.array([1.0, 2.0])) is None


def test_dominant_band_constant_values():
    assert _dominant_band(np.ones(100)) is None


# ---------------------------------------------------------------------------
def _make_pipeline(axis_dir=(0.0, 0.0, 1.0), axis_point=(0.0, 0.0, 0.0)):
    """构造一个只带轴标定的 ScanPipeline（不连硬件）。"""
    from triniscan.core.pipeline import ScanPipeline

    class _Axis:
        direction = np.asarray(axis_dir, dtype=float)
        point = np.asarray(axis_point, dtype=float)
        radius = 0.03
        residual_rms = 0.0005

    class _Cfg:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            node = self._data
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return node

    pipe = ScanPipeline.__new__(ScanPipeline)
    pipe.cfg = _Cfg({"reconstruction": {"axial": {}}})
    pipe._axis_calib = _Axis()
    pipe._cam_calib = None
    pipe._laser_calib = None
    pipe._calib_reports = []
    pipe._last_hull_volume_cm3 = None
    return pipe


def _cloud(points):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return pcd


def _open3d():
    return pytest.importorskip("open3d")


def test_clip_axial_band_height_limits():
    _open3d()
    pipe = _make_pipeline()
    pipe.cfg._data["reconstruction"]["axial"] = {
        "height_min_m": -0.01, "height_max_m": 0.01}
    body = np.random.default_rng(2).uniform(-0.009, 0.009, (500, 3))
    far = np.array([[0.0, 0.0, 0.05]] * 100)
    pcd = _cloud(np.vstack([body, far]))
    out, info = pipe._clip_axial_band(pcd)
    assert info["applied"]
    assert info["removed"] == 100
    assert len(np.asarray(out.points)) == 500


def test_clip_axial_band_radius_limit():
    _open3d()
    pipe = _make_pipeline()
    pipe.cfg._data["reconstruction"]["axial"] = {"radius_max_m": 0.05}
    near = np.zeros((200, 3))
    near[:, 0] = np.linspace(0.0, 0.04, 200)     # 半径 <= 40mm
    far = np.zeros((50, 3))
    far[:, 0] = 0.20                              # 半径 200mm
    pcd = _cloud(np.vstack([near, far]))
    out, info = pipe._clip_axial_band(pcd)
    assert info["removed"] == 50
    assert len(np.asarray(out.points)) == 200


def test_clip_axial_band_auto_trim():
    _open3d()
    pipe = _make_pipeline()
    pipe.cfg._data["reconstruction"]["axial"] = {"auto_trim": True}
    rng = np.random.default_rng(3)
    body = np.zeros((4000, 3))
    body[:, 2] = rng.uniform(-0.02, 0.03, 4000)
    top = np.zeros((400, 3))
    top[:, 2] = rng.uniform(0.07, 0.075, 400)
    pcd = _cloud(np.vstack([body, top]))
    out, info = pipe._clip_axial_band(pcd)
    assert info["applied"]
    assert info["removed"] >= 380        # 顶部背景簇基本被剔除


def test_clip_axial_band_disabled():
    _open3d()
    pipe = _make_pipeline()
    pipe.cfg._data["reconstruction"]["axial"] = {}
    pcd = _cloud(np.random.default_rng(4).uniform(-1, 1, (100, 3)))
    out, info = pipe._clip_axial_band(pcd)
    assert not info["applied"]
    assert len(np.asarray(out.points)) == 100
