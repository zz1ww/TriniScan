"""界面层两个新模块的测试：``field_spec`` 与 ``algorithm_list``。

为什么要有这组
--------------
重构前"配置项在界面上长什么样"散在三处，各有各的规则：

- ``ConfigPanel`` 靠 ``isinstance`` 猜控件类型，**不校验范围**；
- ``_FIELD_CHOICES`` 手写候选值，与算法声明各写一份；
- 标定页手写 ``"peak" / "sphere"``，其中 ``peak`` 根本不在声明里
  （声明只有 ``depth`` / ``sphere``）——选中后运行才崩。

这两组测试锁住：**界面看到的候选值/范围，必须等于算法声明里的**。
"""
from __future__ import annotations

import os

import pytest

from triniscan.core.config import Config
from triniscan.ui import algorithm_list as al
from triniscan.ui import field_spec as fs_mod
from triniscan.ui.field_spec import FieldSpec, all_fields, field_groups


def _cfg() -> Config:
    path = os.path.join(os.path.dirname(__file__), "..", "config",
                        "default.yaml")
    return Config.load(os.path.abspath(path))


# ---------------------------------------------------------------------------
# algorithm_list：候选值来自注册表
# ---------------------------------------------------------------------------
def test_entries_nonempty_for_all_domains():
    for domain in al.REGISTRY_NAMES:
        assert al.entries(domain), f"{domain} 没有实现"


def test_entries_have_labels():
    """每个实现都应有中文名，界面才能显示得可读。"""
    for domain in al.REGISTRY_NAMES:
        for e in al.entries(domain):
            assert e.name, domain
            assert e.label, f"{domain}/{e.name} 缺 label"


def test_choice_matches_algorithm_declaration():
    """``extraction.method`` 的候选值必须与声明完全一致。"""
    got = al.choices_for_key("extraction.method")
    assert got == ["row_peak", "row_centroid", "steger", "gray_centroid",
                   "parabola", "peak"]


def test_axis_method_has_no_illegal_peak_only():
    """回归锁：标定页曾硬编码 peak，而声明里没有它。

    声明新增了 chessboard（棋盘格法）后，这里只锁"不含 peak"这件事，
    避免每次加方法都要改测试。
    """
    got = al.choices_for_key("axis_calib.method")
    assert got is not None
    assert set(got) == {"depth", "sphere", "chessboard"}
    assert "peak" not in got


def test_unknown_key_returns_none():
    assert al.choices_for_key("totally.unknown_key") is None


def test_entry_of_unknown_is_none():
    assert al.entry_of("MESHERS", "no_such_mesher") is None


def test_summary_mentions_all_domains():
    text = al.all_domains_summary()
    for title in al.DOMAIN_TITLES.values():
        assert title in text


# ---------------------------------------------------------------------------
# field_spec：类型 / 范围 / 校验
# ---------------------------------------------------------------------------
def test_fields_cover_every_config_key():
    cfg = _cfg()
    fields = all_fields(cfg)
    expected = {
        f"{g}.{k}" for g, p in cfg.raw.items()
        if not str(g).startswith("_") and isinstance(p, dict)
        for k in p
    }
    assert set(fields) == expected


def test_algorithm_declared_fields_carry_range():
    """算法声明过的参数应带上范围（旧界面完全没有校验）。"""
    fields = all_fields(_cfg())
    f = fields["reconstruction.poisson_depth"]
    assert f.from_algorithm
    assert f.low == 4 and f.high == 14


def test_coerce_int_and_bool():
    cfg = _cfg()
    fields = all_fields(cfg)
    assert fields["scan.num_views"].coerce("72") == 72
    assert isinstance(fields["scan.num_views"].coerce("72"), int)
    assert fields["volume.cross_check"].coerce("否") is False
    assert fields["volume.cross_check"].coerce("是") is True


def test_coerce_choice_keeps_original_type():
    """候选值里 ``180`` 是 int，选出来也应是 int 而不是 "180"。"""
    fields = all_fields(_cfg())
    f = fields.get("scan.num_views")
    if f.choices:
        picked = f.coerce(f.choices[0])
        assert isinstance(picked, type(f.choices[0]))


def test_validate_out_of_range():
    fields = all_fields(_cfg())
    f = fields["reconstruction.poisson_depth"]
    assert f.validate(9) == []
    errors = f.validate(99)
    assert errors and "99" in errors[0]


def test_validate_choice_rejects_unknown():
    fields = all_fields(_cfg())
    errors = fields["extraction.method"].validate("not_a_method")
    assert errors


def test_bad_text_is_not_silently_accepted():
    """输入 "abc" 到整数项：不应变成某个合法 int。"""
    fields = all_fields(_cfg())
    raw = fields["scan.num_views"].coerce("abc")
    assert not isinstance(raw, int)


# ---------------------------------------------------------------------------
def test_field_groups_preserve_all_fields():
    fields = all_fields(_cfg())
    grouped = field_groups(fields)
    assert sum(len(v) for v in grouped.values()) == len(fields)


def test_to_text_bool():
    f = FieldSpec(key="x.y", label="t", group="g", kind="bool")
    assert f.to_text(True) == "是"
    assert f.to_text(False) == "否"
    assert f.to_text(None) == ""


# ---------------------------------------------------------------------------
# FieldTable：越界值在输入处拦截（旧版是静默放行）
# ---------------------------------------------------------------------------
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


def test_field_table_rejects_out_of_range(root):
    """``poisson_depth=99`` 不应被提交，而应在界面上标红。"""
    from triniscan.ui.panels import FieldTable

    cfg = _cfg()
    got, errs = [], []
    table = FieldTable(root, cfg,
                       on_change=lambda k, v: got.append((k, v)),
                       on_error=lambda k, m: errs.append((k, m)))
    root.update_idletasks()

    table._vars["reconstruction.poisson_depth"].set("99")
    table._commit("reconstruction.poisson_depth")
    assert got == [], "越界值不应被提交"
    assert errs and errs[-1][0] == "reconstruction.poisson_depth"
    assert "reconstruction.poisson_depth" in table.errors()

    table._vars["reconstruction.poisson_depth"].set("9")
    table._commit("reconstruction.poisson_depth")
    assert got[-1] == ("reconstruction.poisson_depth", 9)
    assert table.errors() == {}


def test_field_table_offers_algorithm_choices(root):
    """算法选择项必须是下拉框，且候选值来自声明（不是手写清单）。"""
    from triniscan.ui.panels import FieldTable

    table = FieldTable(root, _cfg())
    root.update_idletasks()
    fs = table.fields["extraction.method"]
    assert fs.choices == ["row_peak", "row_centroid", "steger",
                          "gray_centroid", "parabola", "peak"]
    widget = table._widgets["extraction.method"]
    assert str(widget.winfo_class()) == "TCombobox"


# ---------------------------------------------------------------------------
# 结构型配置（dict / list）：不得被文本化
#
# 回归背景：``reconstruction.axial`` 是 dict、``reconstruction.filter_chain``
# 是 list，旧版 ``FieldTable`` 把它们渲染成单行 Entry（内容为完整 repr）。
# 用户点一下再失焦即触发 ``_commit`` → ``coerce`` 转成 str 写回 store，
# 于是 ``AxialTrimFilter.read_params`` 读不到子键，``auto_trim`` 从 True
# 静默变成 False——没有任何报错。这组测试锁住该行为。
# ---------------------------------------------------------------------------
STRUCTURAL_KEYS = ("reconstruction.filter_chain", "reconstruction.axial",
                   "calibration.z_axis_point", "calibration.z_axis_dir")


def test_structural_fields_are_recognized():
    fields = all_fields(_cfg())
    for key in STRUCTURAL_KEYS:
        assert key in fields, key
        assert fields[key].is_structural, f"{key} 应判定为结构型"
    assert fields["reconstruction.axial"].kind == "dict"
    assert fields["reconstruction.filter_chain"].kind == "list"


def test_structural_fields_reject_text():
    """把字符串喂给结构型配置应校验失败（提示改用专用编辑器）。"""
    fields = all_fields(_cfg())
    for key in STRUCTURAL_KEYS:
        fs = fields[key]
        assert fs.validate("{'auto_trim': True}"), key
        assert fs.validate("[1, 2]"), key


def test_structural_to_text_is_summary_not_repr():
    """显示文本只是类型摘要，不能包含完整结构（避免诱导用户编辑）。"""
    cfg = _cfg()
    fields = all_fields(cfg)
    text = fields["reconstruction.axial"].to_text(
        cfg.get("reconstruction.axial"))
    assert "auto_trim" not in text
    assert "dict" in text


def test_coerce_structural_is_identity():
    """coerce 对结构型必须原样返回（不做任何文本转换）。"""
    cfg = _cfg()
    fields = all_fields(cfg)
    for key in STRUCTURAL_KEYS:
        value = cfg.get(key)
        assert fields[key].coerce(value) is value, key


def test_field_table_uses_label_for_structural(root):
    """结构型字段在界面上必须是只读 Label，不是 Entry/Combobox。"""
    from triniscan.ui.panels import FieldTable

    table = FieldTable(root, _cfg())
    root.update_idletasks()
    for key in STRUCTURAL_KEYS:
        cls = str(table._widgets[key].winfo_class())
        assert cls == "TLabel", f"{key} 用了 {cls}，应只读显示"


def test_field_table_never_commits_structural(root):
    """即使外部误触发 _commit，也不能改动结构型配置。"""
    from triniscan.ui.panels import FieldTable

    cfg = _cfg()
    before = {k: cfg.get(k) for k in STRUCTURAL_KEYS}
    got = []
    table = FieldTable(root, cfg, on_change=lambda k, v: got.append((k, v)))
    root.update_idletasks()

    for key in STRUCTURAL_KEYS:
        table._commit(key)

    assert got == [], "结构型配置不应触发提交回调"
    for key in STRUCTURAL_KEYS:
        assert cfg.get(key) == before[key], key
