"""过滤插件与滤镜链（FilterChain）的测试。

覆盖
----
1. 注册表：6 个过滤器全部注册、参数声明可读；
2. **行为等价**：旧式扁平键生成的链 == 旧私有方法逐级调用的结果；
3. 掩膜语义：每层掩膜长度 == 原始点数（不是当前子集）；
4. 顺序追踪：``depends_on_current`` 标记正确；
5. 幂等与空输入：不崩溃、不误删；
6. 报告：能够读出"被删掉的第二大簇有多大"（少一面问题的诊断）；
7. dry_run：只报告不修改点云。
"""
from __future__ import annotations

import numpy as np
import pytest

open3d = pytest.importorskip("open3d")

from triniscan.contracts import Cloud, FILTERS, MaskStack, RunContext
from triniscan.algorithms.filtering import (AxialTrimFilter, FilterChain,
                                            FilterPlugin, TurntableFilter)
from triniscan.algorithms.filtering.axial import dominant_band


# ---------------------------------------------------------------------------
class _Cfg:
    """最小可用的配置对象（点号键 + get）。"""

    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        # 支持层级 fallback：先精确匹配，再按前缀匹配
        for k, v in self.data.items():
            if k == key:
                return v
        return default


class _Axis:
    point = np.zeros(3)
    direction = np.array([0.0, 0.0, 1.0])


def _ctx(**cfg):
    cfg.setdefault("reconstruction.depth_min_m", 0.0)
    cfg.setdefault("reconstruction.depth_max_m", 0.0)
    return RunContext(cfg=_Cfg(cfg), axis_point=_Axis.point,
                      axis_dir=_Axis.direction)


@pytest.fixture()
def sample_cloud():
    rng = np.random.default_rng(7)
    n = 4000
    th = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0, 0.05, n)
    table = np.c_[r * np.cos(th), r * np.sin(th), np.zeros(n)]

    m = 2000
    th2 = rng.uniform(0, 2 * np.pi, m)
    h = rng.uniform(0.003, 0.05, m)
    body = np.c_[0.01 * np.cos(th2), 0.01 * np.sin(th2), h]

    far = np.c_[rng.uniform(0.3, 0.4, 150), rng.uniform(-0.1, 0.1, 150),
                rng.uniform(0, 0.05, 150)]
    pts = np.vstack([table, body, far])
    return Cloud(pts, "world", "test")


# ---------------------------------------------------------------------------
def test_all_filters_registered():
    from triniscan.algorithms import load_all
    load_all()
    names = set(FILTERS.names())
    assert {"depth", "bounds", "cluster", "statistical",
            "turntable", "axial"} <= names


@pytest.mark.parametrize("name", ["depth", "bounds", "cluster",
                                  "statistical", "turntable", "axial"])
def test_params_are_declared_and_readable(name):
    plugin = FILTERS.get(name)()
    params = plugin.read_params(_Cfg({}))
    # 每个声明都必须被读到，且 short 名唯一
    assert len(params) == len(plugin.params)
    assert all(spec.key in params for spec in plugin.params)
    keys = [spec.key for spec in plugin.params]
    assert len(keys) == len(set(keys)), f"{name} 存在重复 short 名"


def test_mask_length_is_original(sample_cloud):
    """每层掩膜必须是**原始点数**长度，而不是上一层子集。"""
    cfg = _Cfg({"reconstruction.cluster_filter": True,
                "reconstruction.cluster_eps_m": 0.002,
                "reconstruction.cluster_min_points": 10,
                "reconstruction.cluster_keep_top": 1,
                "reconstruction.axial.auto_trim": True,
                "reconstruction.remove_turntable": True})
    ctx = RunContext(cfg=cfg, axis_point=_Axis.point, axis_dir=_Axis.direction)
    chain = FilterChain.from_config(cfg)
    out, masks = chain.apply(sample_cloud, ctx)

    assert masks.n == len(sample_cloud)
    for layer in masks.layers.values():
        assert layer.mask.shape == (len(sample_cloud),)
    assert len(out) == masks.n_keep


def test_legacy_config_matches_private_methods(sample_cloud):
    """旧式扁平配置的链结果，应与旧的逐级私有方法**逐点一致**。"""
    from triniscan.core.config import Config
    from triniscan.core.pipeline import ScanPipeline

    cfg = Config.load("config/default.yaml")
    cfg.raw["reconstruction"].pop("filter_chain", None)   # 强制走旧路径

    pipe = ScanPipeline(cfg)
    pipe._axis_calib = _Axis()
    pipe._axis_calib.point = _Axis.point
    pipe._axis_calib.direction = _Axis.direction
    pipe._cam_calib = type("C", (), {"K": np.eye(3),
                                     "dist": np.zeros(5)})()
    pipe._laser_calib = type("L", (), {"plane": np.array([0, 0, 1, 0.])})()

    import open3d
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(sample_cloud.points)
    pcd, _ = pipe._filter_bounds(pcd)
    pcd, _ = pipe._filter_clusters(pcd)
    pcd, _ = pipe._remove_turntable(pcd)
    pcd, _ = pipe._clip_axial_band(pcd)
    expected = np.asarray(pcd.points)

    ctx = pipe.make_context()
    out, masks = FilterChain.from_config(cfg).apply(sample_cloud, ctx)

    np.testing.assert_allclose(out.points, expected, atol=1e-12)
    assert masks.n_keep == len(expected)


def test_cluster_reports_detached_clusters(sample_cloud):
    """连通域层必须报告"被删掉的第二大簇有多大"（少一面的关键线索）。"""
    cfg = _Cfg({"reconstruction.cluster_filter": True,
                "reconstruction.cluster_eps_m": 0.002,
                "reconstruction.cluster_min_points": 10,
                "reconstruction.cluster_keep_top": 1})
    ctx = RunContext(cfg=cfg, axis_point=_Axis.point, axis_dir=_Axis.direction)
    _, masks = FilterChain([FILTERS.get("cluster")()]).apply(sample_cloud, ctx)
    layer = masks.layers["cluster"]
    sizes = layer.extra.get("cluster_sizes") or []
    assert sizes, "缺少 cluster_sizes 诊断"
    assert sizes == sorted(sizes, reverse=True)


def test_depends_on_current_flags():
    """依赖整体分布的过滤器必须被标记，提醒不能随意换序。"""
    assert TurntableFilter.depends_on_current is True
    assert AxialTrimFilter.depends_on_current is True
    assert FILTERS.get("depth")().depends_on_current is False
    assert FILTERS.get("bounds")().depends_on_current is False
    assert FILTERS.get("cluster")().depends_on_current is False


def test_chain_skip_when_all_removed():
    """极端情况：第一层把点删光，后续层要跳过而不是崩溃。"""
    pts = np.zeros((50, 3))
    pts[:, 2] = 5.0                      # 全部远于 depth_max
    cloud = Cloud(pts, "world", "far")
    cfg = _Cfg({"reconstruction.depth_min_m": 0.0,
                "reconstruction.depth_max_m": 0.3,
                "reconstruction.cluster_filter": True})
    ctx = RunContext(cfg=cfg, axis_point=_Axis.point, axis_dir=_Axis.direction)
    chain = FilterChain([FILTERS.get("depth")(),
                         FILTERS.get("cluster")()])
    out, masks = chain.apply(cloud, ctx)
    assert len(out) == 0
    assert masks.n_keep == 0
    assert any("过滤掉" in w for r in ctx.report.stages
               for w in r.warnings)


def test_chain_disabled_item_is_skipped():
    cfg = _Cfg({"reconstruction.filter_chain": [
        {"name": "depth", "enable": False},
        {"name": "cluster"},
    ]})
    names = FilterChain.from_config(cfg).names()
    assert names == ["cluster"]


def test_inline_override_wins_over_config(sample_cloud):
    """链项里就地写的参数应覆盖全局配置。"""
    cfg = _Cfg({"reconstruction.cluster_eps_m": 0.002,
                "reconstruction.filter_chain": [
                    {"name": "cluster", "eps_m": 0.02}]})
    chain = FilterChain.from_config(cfg)
    params = chain.plugins[0].read_params(cfg)
    assert params["eps_m"] == 0.02


def test_dry_run_keeps_original_cloud(sample_cloud):
    cfg = _Cfg({"reconstruction.cluster_filter": True,
                "reconstruction.cluster_eps_m": 0.002,
                "reconstruction.cluster_min_points": 10,
                "reconstruction.cluster_keep_top": 1})
    ctx = RunContext(cfg=cfg, axis_point=_Axis.point, axis_dir=_Axis.direction)
    chain = FilterChain.from_config(cfg, dry_run=True)
    out, masks = chain.apply(sample_cloud, ctx)
    assert len(out) == len(sample_cloud)
    assert masks.n_keep < len(sample_cloud)     # 但报告反映真实删除量


def test_dominant_band_ignores_isolated_bands():
    rng = np.random.default_rng(3)
    body = rng.uniform(-0.02, 0.03, 4000)
    above = rng.uniform(0.08, 0.085, 200)
    below = rng.uniform(-0.06, -0.055, 200)
    band = dominant_band(np.concatenate([body, above, below]))
    assert band is not None
    lo, hi = band
    assert lo >= -0.03 and hi <= 0.04


def test_filter_plugin_is_plugin_protocol():
    from triniscan.contracts import Plugin
    plugin = FILTERS.get("depth")()
    assert isinstance(plugin, Plugin)
    assert isinstance(plugin, FilterPlugin)
