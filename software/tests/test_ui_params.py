"""参数面板 / 算法选择 / 过滤链编辑器的测试（P2、P3）。"""
from __future__ import annotations

import os

import pytest

from triniscan.core.config import Config
from triniscan.core.config_store import ConfigStore
from triniscan.ui.field_spec import FieldSpec


def _cfg() -> Config:
    path = os.path.join(os.path.dirname(__file__), "..", "config",
                        "default.yaml")
    return Config.load(os.path.abspath(path))


@pytest.fixture
def root():
    import tkinter as tk
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("无显示环境，跳过界面测试")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def store():
    return ConfigStore(_cfg())


# ---------------------------------------------------------------------------
# ParamGroup：控件类型由声明决定
# ---------------------------------------------------------------------------
def test_param_group_bool_uses_checkbutton(root, store):
    from triniscan.ui.params import ParamGroup

    fs = FieldSpec(key="volume.cross_check", label="交叉校验",
                   group="", kind="bool")
    g = ParamGroup(root)
    g.add_row(fs, True)
    root.update_idletasks()
    assert str(g.row(fs.key).widgets[0].winfo_class()) == "TCheckbutton"


def test_param_group_choice_uses_combobox(root, store):
    from triniscan.ui.params import ParamGroup

    fs = FieldSpec(key="extraction.method", label="方法", group="", kind="str",
                   choices=["a", "b"])
    g = ParamGroup(root)
    g.add_row(fs, "a")
    root.update_idletasks()
    assert str(g.row(fs.key).widgets[0].winfo_class()) == "TCombobox"


def test_param_group_int_range_uses_spinbox(root, store):
    from triniscan.ui.params import ParamGroup

    fs = FieldSpec(key="reconstruction.poisson_depth", label="深度",
                   group="", kind="int", low=4, high=14)
    g = ParamGroup(root)
    g.add_row(fs, 9)
    root.update_idletasks()
    assert str(g.row(fs.key).widgets[0].winfo_class()) == "TSpinbox"


def test_param_group_float_range_uses_scale(root, store):
    from triniscan.ui.params import ParamGroup

    fs = FieldSpec(key="camera.exposure", label="曝光", group="",
                   kind="float", low=-13, high=0)
    g = ParamGroup(root)
    g.add_row(fs, -8.0)
    root.update_idletasks()
    assert str(g.row(fs.key).widgets[0].winfo_class()) == "TScale"


def test_param_group_structural_is_not_text_input(root, store):
    """结构型配置不得出现 Entry（否则会重演静默损坏）。"""
    from triniscan.ui.params import ParamGroup

    fs = FieldSpec(key="reconstruction.axial", label="轴向", group="",
                   kind="dict")
    g = ParamGroup(root)
    g.add_row(fs, {"auto_trim": True})
    root.update_idletasks()
    cls = str(g.row(fs.key).widgets[0].winfo_class())
    assert cls == "TLabel", cls


def test_param_group_validates_and_reports_error(root, store):
    from triniscan.ui.params import ParamGroup

    got = []
    fs = FieldSpec(key="reconstruction.poisson_depth", label="深度",
                   group="", kind="int", low=4, high=14)
    g = ParamGroup(root, on_change=lambda k, v: got.append((k, v)))
    g.add_row(fs, 9)
    root.update_idletasks()

    g._commit_text(fs.key, "99")
    assert got == [], "越界值不应提交"
    assert fs.key in g.errors()

    g._commit_text(fs.key, "9")
    assert got[-1] == (fs.key, 9)
    assert g.errors() == {}


def test_param_group_bool_toggles(root, store):
    from triniscan.ui.params import ParamGroup

    got = []
    fs = FieldSpec(key="volume.cross_check", label="交叉校验",
                   group="", kind="bool")
    g = ParamGroup(root, on_change=lambda k, v: got.append((k, v)))
    g.add_row(fs, True)
    root.update_idletasks()
    widget = g.row(fs.key).widgets[0]
    widget.invoke()
    assert got == [(fs.key, False)]


def test_param_group_choice_commits_name(root, store):
    from triniscan.ui.params import ParamGroup

    got = []
    fs = FieldSpec(key="extraction.channel", label="通道", group="",
                   kind="str", choices=["gray", "red_laser"])
    g = ParamGroup(root, on_change=lambda k, v: got.append((k, v)))
    g.add_row(fs, "gray")
    root.update_idletasks()
    g._commit_text(fs.key, "red_laser")
    assert got == [(fs.key, "red_laser")]


# ---------------------------------------------------------------------------
# AlgorithmPanel
# ---------------------------------------------------------------------------
def test_algorithm_panel_lists_methods_from_declaration(root, store):
    from triniscan.ui.algorithm_panel import AlgorithmPanel

    p = AlgorithmPanel(root, "EXTRACTORS", store,
                       method_key="extraction.method")
    p.refresh()
    root.update_idletasks()

    assert p.selector.value == store.cfg.get("extraction.method")
    values = list(p.selector.combo.cget("values"))
    assert any(v.startswith("row_peak") for v in values)
    assert any(v.startswith("steger") for v in values)


def test_algorithm_panel_shows_only_current_method_params(root, store):
    from triniscan.ui.algorithm_panel import AlgorithmPanel

    p = AlgorithmPanel(root, "EXTRACTORS", store,
                       method_key="extraction.method")
    p.refresh()
    root.update_idletasks()
    keys = [k.rsplit(".", 1)[-1] for k in p.keys()]
    assert "threshold" in keys
    assert "method" not in keys, "方法本身由下拉框承担，不应重复成参数行"


def test_algorithm_panel_switch_writes_store(root, store):
    from triniscan.ui.algorithm_panel import AlgorithmPanel

    p = AlgorithmPanel(root, "EXTRACTORS", store,
                       method_key="extraction.method")
    p.refresh()
    root.update_idletasks()

    p.selector._var.set("steger — 最精确但最慢")
    p.selector._on_selected(None)
    root.update_idletasks()
    assert store.cfg.get("extraction.method") == "steger"
    assert store.dirty


# ---------------------------------------------------------------------------
# FilterChainEditor
# ---------------------------------------------------------------------------
def test_filter_chain_lists_current_chain(root, store):
    from triniscan.ui.filter_chain_ui import FilterChainEditor

    ed = FilterChainEditor(root, store)
    root.update_idletasks()
    names = [r.name for r in ed._rows]
    assert names == [i["name"] for i in
                     store.cfg.get("reconstruction.filter_chain")]


def test_filter_chain_toggle_writes_real_list(root, store):
    from triniscan.ui.filter_chain_ui import FilterChainEditor

    ed = FilterChainEditor(root, store)
    root.update_idletasks()
    row = ed._rows[0]
    row.enabled.set(False)
    row._on_enable()
    root.update_idletasks()

    chain = store.cfg.get("reconstruction.filter_chain")
    assert isinstance(chain, list), "必须写回 list，不能是字符串"
    assert chain[0]["enable"] is False


def test_filter_chain_move_reorders(root, store):
    from triniscan.ui.filter_chain_ui import FilterChainEditor

    ed = FilterChainEditor(root, store)
    root.update_idletasks()
    before = [r.name for r in ed._rows]
    ed.move(1, -1)                       # cluster 上移，越过 bounds
    root.update_idletasks()
    after = [r.name for r in ed._rows]
    assert after[0] == before[1]
    assert isinstance(store.cfg.get("reconstruction.filter_chain"), list)


def test_filter_chain_rejects_dependent_move(root, store, monkeypatch):
    """turntable/axial 不能被移到普通过滤器之前。"""
    from triniscan.ui import filter_chain_ui as fc

    warned = []
    monkeypatch.setattr(fc.FilterChainEditor, "_complain",
                        lambda self, msg: warned.append(msg))

    ed = fc.FilterChainEditor(root, store)
    root.update_idletasks()
    names = [r.name for r in ed._rows]
    idx = names.index("turntable")
    ed.move(idx, -1)                     # 试图越过 cluster
    root.update_idletasks()

    assert warned, "应提示顺序不允许"
    assert [r.name for r in ed._rows] == names, "顺序不应改变"


def test_filter_chain_add_and_remove(root, store):
    from triniscan.ui.filter_chain_ui import FilterChainEditor

    ed = FilterChainEditor(root, store)
    root.update_idletasks()
    n0 = len(ed._rows)

    ed.remove(0)
    root.update_idletasks()
    assert len(ed._rows) == n0 - 1

    ed._add_var.set("statistical")
    ed._on_add()
    root.update_idletasks()
    assert len(ed._rows) == n0
    assert "statistical" in [r.name for r in ed._rows]
    assert isinstance(store.cfg.get("reconstruction.filter_chain"), list)


def test_filter_chain_params_written_as_override(root, store):
    """改某行的参数应写入该行的覆盖项，而不是全局配置。"""
    from triniscan.ui.filter_chain_ui import FilterChainEditor

    ed = FilterChainEditor(root, store)
    root.update_idletasks()
    row = ed._rows[0]
    row._toggle_params()
    root.update_idletasks()

    assert row._param_group is not None
    key = row._param_group.keys[0]
    row._on_param_change(key, 0.001)
    root.update_idletasks()

    chain = store.cfg.get("reconstruction.filter_chain")
    assert chain[0].get(key.rsplit(".", 1)[-1]) == 0.001
