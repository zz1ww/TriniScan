"""算法参数声明（AlgorithmSpec）与结果面板的测试。

覆盖
----
1. **默认值单一来源**：spec 默认值 == ``default.yaml`` 的值
   （不一致说明 yaml 与代码漂移，是"配置不生效"最常见的来源）；
2. spec 的读取 / 覆盖 / 校验 / 越界报错；
3. 每个域都有声明（可被 UI 枚举）；
4. ``quality_thresholds`` 与旧实现等价；
5. ``ResultPanel`` 的过滤诊断表能正确渲染（含标红判定）。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from triniscan.algorithms import describe, load_all, spec_table
from triniscan.contracts import (CLOSERS, EXTRACTORS, MESHERS, REGISTRARS,
                                 VOLUMETERS, AlgorithmSpec, ConfigError,
                                 ParamSpec)
from triniscan.core.config import Config


class _Cfg:
    """最小配置对象（测试用）。"""

    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)


# 迷你 pytest 不调用 setup_module，因此在模块级显式加载一次
load_all()


# ---------------------------------------------------------------------------
# AlgorithmSpec 基本行为
# ---------------------------------------------------------------------------
def test_read_fills_defaults():
    spec = AlgorithmSpec("t", "测试", lambda: None, params=(
        ParamSpec("a.b", int, 7, low=1, high=9, short="b"),
        ParamSpec("a.c", bool, True, short="c"),
    ))
    assert spec.read(_Cfg()) == {"b": 7, "c": True}


def test_read_prefers_config():
    spec = AlgorithmSpec("t", "测试", lambda: None, params=(
        ParamSpec("a.b", int, 7, low=1, high=9, short="b"),
    ))
    assert spec.read(_Cfg({"a.b": 3})) == {"b": 3}


def test_read_raises_on_out_of_range():
    spec = AlgorithmSpec("t", "测试", lambda: None, params=(
        ParamSpec("a.b", int, 7, low=1, high=9, short="b"),
    ))
    with pytest.raises(ConfigError):
        spec.read(_Cfg({"a.b": 100}))


def test_read_accepts_string_numbers():
    """yaml 里写成字符串的数字也应被正确转换。"""
    spec = AlgorithmSpec("t", "测试", lambda: None, params=(
        ParamSpec("a.b", float, 1.0, low=0.0, high=2.0, short="b"),
    ))
    assert spec.read(_Cfg({"a.b": "0.25"})) == {"b": 0.25}


def test_overrides_win_and_are_coerced():
    spec = AlgorithmSpec("t", "测试", lambda: None, params=(
        ParamSpec("a.b", int, 7, low=1, high=9, short="b"),
    ))
    assert spec.read(_Cfg({"a.b": 3}), overrides={"b": "5"}) == {"b": 5}
    with pytest.raises(ConfigError):
        spec.read(_Cfg(), overrides={"b": 99})


def test_validate_returns_errors_without_raising():
    spec = AlgorithmSpec("t", "测试", lambda: None, params=(
        ParamSpec("a.b", int, 7, low=1, high=9, short="b"),
    ))
    assert spec.validate(_Cfg({"a.b": 3})) == []
    assert spec.validate(_Cfg({"a.b": 100}))


def test_keys_distinguish_short_and_global():
    spec = AlgorithmSpec("t", "测试", lambda: None, params=(
        ParamSpec("reconstruction.poisson_depth", int, 9, short="depth"),
    ))
    assert spec.keys() == ["depth"]
    assert spec.config_keys() == ["reconstruction.poisson_depth"]


# ---------------------------------------------------------------------------
# 默认值单一来源（最重要）
# ---------------------------------------------------------------------------
_SPEC_REGISTRIES = {
    ("meshing", "poisson"): MESHERS,
    ("meshing", "ball_pivoting"): MESHERS,
    ("closing", "default"): CLOSERS,
    ("volumetry", "divergence"): VOLUMETERS,
    ("volumetry", "voxel"): VOLUMETERS,
    ("registration", "sequential"): REGISTRARS,
    ("extraction", "centerline"): EXTRACTORS,
}


def _default_config() -> Config:
    """加载随包发布的默认配置（相对 ``software/`` 定位，不依赖 CWD）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "config", "default.yaml")
    return Config.load(path)


@pytest.mark.parametrize("domain,spec_name",
                         sorted(_SPEC_REGISTRIES))
def test_spec_defaults_match_yaml(domain, spec_name):
    """spec 默认值必须与 default.yaml 一致，否则"改配置不生效"。"""
    registry = _SPEC_REGISTRIES[(domain, spec_name)]
    spec = registry.spec(spec_name)
    cfg = _default_config()
    mismatches = []
    for param in spec.params:
        marker = object()
        declared = cfg.get(param.name, marker)
        if declared is marker:
            continue                    # yaml 未列出的可选键
        if declared == param.default:
            continue
        mismatches.append(f"{param.name}: yaml={declared!r} "
                          f"spec={param.default!r}")
    assert not mismatches, f"{domain}.{spec_name} 默认值漂移: {mismatches}"


# ---------------------------------------------------------------------------
# 注册表覆盖
# ---------------------------------------------------------------------------
def test_every_domain_has_declarations():
    table = spec_table()
    missing = [d for d in ("extraction", "registration", "meshing",
                           "closing", "volumetry")
               if not table[d]]
    assert not missing, f"以下域没有参数声明: {missing}"


def test_filters_are_listed_as_declared():
    """过滤器用 PluginBase.params 声明参数，describe 也应视作已声明。"""
    text = describe()
    assert "depth" in text
    assert "depth(无声明)" not in text


def test_spec_table_shape():
    entry = spec_table()["meshing"][0]
    assert {"name", "label", "help", "tags", "params"} <= set(entry)
    if entry["params"]:
        assert {"key", "short", "type", "default"} <= set(entry["params"][0])


# ---------------------------------------------------------------------------
# 标定质量阈值
# ---------------------------------------------------------------------------
def test_thresholds_from_yaml():
    from triniscan.algorithms.calibration.specs import quality_thresholds
    th = quality_thresholds(_default_config())
    assert th["camera"]["good"] == 0.3
    assert th["axis"]["warn"] == 0.3


def test_empty_config_uses_module_defaults():
    from triniscan.algorithms.calibration.specs import quality_thresholds
    assert quality_thresholds(_Cfg()) == {}


# ---------------------------------------------------------------------------
# ResultPanel 过滤诊断
# ---------------------------------------------------------------------------
def _make_panel():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    from triniscan.ui.panels import ResultPanel
    return root, ResultPanel(root)


def test_panel_renders_layers_and_marks_warn():
    root, panel = _make_panel()
    try:
        panel.set_filter_layers([
            {"name": "cluster", "removed": 1200, "removed_ratio": 0.4,
             "applied": True, "depends_on_previous": False,
             "reason": "保留最大 1 簇"},
            {"name": "turntable", "removed": 0, "removed_ratio": 0.0,
             "applied": False, "depends_on_previous": True,
             "reason": "无法估计 h_table"},
        ])
        rows = panel._filter_tree.get_children()
        assert len(rows) == 2
        assert "warn" in panel._filter_tree.item(rows[0], "tags")
        assert "skip" in panel._filter_tree.item(rows[1], "tags")
    finally:
        root.destroy()


def test_panel_clears_when_no_layers():
    root, panel = _make_panel()
    try:
        panel.set_filter_layers([{"name": "a", "removed": 1,
                                  "removed_ratio": 0.1, "applied": True}])
        panel.set_filter_layers([])
        assert panel._filter_tree.get_children() == ()
    finally:
        root.destroy()


def test_panel_update_result_shows_filter_stats():
    root, panel = _make_panel()
    try:
        result = SimpleNamespace(
            volume_cm3=12.5, cross_check_cm3=None, elapsed_s=3.0,
            num_views=360, num_points=100000, mesh_watertight=True,
            mesh_path=None, points_before_filter=100000,
            points_after_filter=40000, calib_fingerprint="abc123",
            filter_layers=[{"name": "cluster", "removed": 60000,
                            "removed_ratio": 0.6, "applied": True,
                            "depends_on_previous": False,
                            "reason": "只保留最大簇"}],
        )
        panel.update_result(result)
        assert "40.0%" in panel._labels["过滤后点数"].cget("text")
        assert panel._labels["标定指纹"].cget("text") == "abc123"
    finally:
        root.destroy()
