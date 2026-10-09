"""配置中文标签（ui/labels.py）与 ConfigPanel 的测试。"""
import tkinter as tk

import pytest

from triniscan.ui.labels import (
    FIELD_HINTS,
    FIELD_LABELS,
    GROUP_LABELS,
    field_hint,
    field_label,
    group_label,
)


# ---------------------------------------------------------------------------
# 纯映射函数
# ---------------------------------------------------------------------------
def test_group_label_known():
    assert group_label("camera") == "相机"
    assert group_label("turntable") == "转台"
    assert group_label("volume") == "体积计算"


def test_group_label_fallback():
    """未登记的分组原样返回，不应报错。"""
    assert group_label("totally_unknown") == "totally_unknown"


def test_field_label_known():
    assert field_label("scan.angle_step_deg") == "每视角转角"
    assert field_label("turntable.port") == "串口号"
    assert field_label("axis_calib.sphere_radius_m") == "标记球半径"


def test_field_label_fallback_strips_group():
    """未登记的字段应返回去掉分组前缀的英文键。"""
    assert field_label("scan.brand_new_field") == "brand_new_field"


def test_field_hint():
    assert field_hint("camera.auto_exposure")
    assert field_hint("nonexistent.key") == ""


def test_all_labels_have_group_prefix():
    """映射表里的键都应是 "分组.字段" 形式，防止写错。"""
    for key in list(FIELD_LABELS) + list(FIELD_HINTS):
        assert "." in key, f"键缺少分组前缀: {key}"
        group = key.split(".", 1)[0]
        assert group in GROUP_LABELS, f"分组未登记: {group} ({key})"


# ---------------------------------------------------------------------------
# 与真实配置对齐：默认配置里的每个键都应有中文
# ---------------------------------------------------------------------------
def test_every_config_key_has_chinese_label():
    from triniscan.core.config import Config

    cfg = Config.load()
    missing = []
    for group, params in cfg.raw.items():
        if group.startswith("_") or not isinstance(params, dict):
            continue
        for key in params:
            key_path = f"{group}.{key}"
            if key_path not in FIELD_LABELS:
                missing.append(key_path)
    assert not missing, f"以下配置项缺少中文标签: {missing}"


# ---------------------------------------------------------------------------
# ConfigPanel 行为
# ---------------------------------------------------------------------------
@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("无显示环境，跳过界面测试")
    r.withdraw()
    yield r
    r.destroy()


def test_panel_builds_and_labels_are_chinese(root):
    from triniscan.core.config import Config
    from triniscan.ui.panels import ConfigPanel

    panel = ConfigPanel(root, Config.load())
    root.update_idletasks()
    # 布尔项用中文显示
    assert panel._vars["volume.cross_check"].get() in ("是", "否")
    assert panel._vars["camera.auto_exposure"].get() in ("是", "否")
    # 数值项仍是数字文本
    assert panel._vars["scan.num_views"].get() == "180"


def test_panel_bool_roundtrip_keeps_python_bool(root):
    from triniscan.core.config import Config
    from triniscan.ui.panels import ConfigPanel

    got = []
    panel = ConfigPanel(root, Config.load(),
                        on_change=lambda k, v: got.append((k, v)))
    root.update_idletasks()

    panel._vars["volume.cross_check"].set("否")
    panel._commit("volume.cross_check")
    assert got[-1] == ("volume.cross_check", False)
    assert isinstance(got[-1][1], bool)

    panel._vars["volume.cross_check"].set("是")
    panel._commit("volume.cross_check")
    assert got[-1] == ("volume.cross_check", True)


def test_panel_numeric_commit(root):
    from triniscan.core.config import Config
    from triniscan.ui.panels import ConfigPanel

    got = []
    panel = ConfigPanel(root, Config.load(),
                        on_change=lambda k, v: got.append((k, v)))
    root.update_idletasks()

    panel._vars["scan.angle_step_deg"].set("2.5")
    panel._commit("scan.angle_step_deg")
    assert got[-1] == ("scan.angle_step_deg", 2.5)

    panel._vars["scan.num_views"].set("120")
    panel._commit("scan.num_views")
    assert got[-1] == ("scan.num_views", 120)
    assert isinstance(got[-1][1], int)


def test_panel_refresh_updates_bool_text(root):
    from triniscan.core.config import Config
    from triniscan.ui.panels import ConfigPanel

    cfg = Config.load()
    panel = ConfigPanel(root, cfg)
    root.update_idletasks()

    cfg.set("volume.cross_check", False)
    panel.refresh(cfg)
    assert panel._vars["volume.cross_check"].get() == "否"

    cfg.set("volume.cross_check", True)
    panel.refresh(cfg)
    assert panel._vars["volume.cross_check"].get() == "是"
