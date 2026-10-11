"""界面 ↔ 配置文件的联动测试（需图形环境，无则跳过）。

覆盖要点
--------
- 主窗口与配置面板、标定面板共享**同一份**配置（曾经的 bug）；
- 界面改参数 → 内存生效 → 保存 → 落盘；
- 改参数后，测量 / 标定读到的就是新值。
"""
import os
import shutil

import pytest
import tkinter as tk
import yaml

from triniscan.core.config import Config


@pytest.fixture(scope="module")
def _tk_root():
    """整个模块共用一个 Tk 根，避免频繁创建/销毁导致的环境异常。"""
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("无显示环境，跳过界面测试")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def window(_tk_root, tmp_path):
    """构造 MainWindow，配置指向临时副本（避免污染真实文件）。"""
    from triniscan.ui.app import MainWindow

    src = os.path.join(os.path.dirname(__file__), "..", "config",
                       "default.yaml")
    dst = str(tmp_path / "default.yaml")
    shutil.copy(src, dst)
    app = MainWindow(_tk_root, Config.load(dst))
    _tk_root.update()
    app._tmp_config = dst
    try:
        yield app
    finally:
        app.destroy()


# ---------------------------------------------------------------------------
def test_panels_share_same_store(window):
    """参数面板与标定面板必须与主窗口共用同一个 store。"""
    assert window.calib_panel.store is window.store
    assert window.calib_panel.cfg is window.store.cfg


def test_edit_then_save_persists(window):
    window.config_panel._vars["camera.index"].set("1")
    window.config_panel._commit("camera.index")
    window.master.update()

    assert window.cfg.get("camera.index") == 1
    assert window.store.dirty is True

    ok = window.save_config_to_file(quiet=True)
    assert ok
    assert window.store.dirty is False

    on_disk = yaml.safe_load(open(window._tmp_config, encoding="utf-8"))
    assert on_disk["camera"]["index"] == 1


def test_edit_reflected_in_measurement_config(window):
    """改参数后，测量流程取到的 cfg 必须是新值。"""
    window.config_panel._vars["scan.num_views"].set("72")
    window.config_panel._commit("scan.num_views")
    window.config_panel._vars["camera.width"].set("1280")
    window.config_panel._commit("camera.width")
    window.master.update()

    cfg = window.cfg                    # 与 _on_run 取的是同一份
    assert cfg.get("scan.num_views") == 72
    assert cfg.get("camera.width") == 1280
    assert cfg.raw["camera"]["width"] == 1280


def test_calib_panel_edit_goes_to_store(window):
    """标定面板改转轴参数也应进入 store，并同步到参数面板。"""
    window.calib_panel._sphere_radius_mm.set("21.5")
    window.calib_panel._axis_views.set("16")
    assert window.calib_panel._read_axis_params() is True

    assert window.cfg.get("axis_calib.sphere_radius_m") == \
        pytest.approx(0.0215)
    assert window.cfg.get("axis_calib.num_views") == 16
    assert window.store.dirty is True
    # 参数面板显示已同步
    assert window.config_panel._vars[
        "axis_calib.sphere_radius_m"].get() == "0.0215"


def test_boolean_survives_save(window):
    window.config_panel._vars["volume.cross_check"].set("否")
    window.config_panel._commit("volume.cross_check")
    window.save_config_to_file(quiet=True)
    on_disk = yaml.safe_load(open(window._tmp_config, encoding="utf-8"))
    assert on_disk["volume"]["cross_check"] is False


def test_save_as_switches_path(window, tmp_path):
    new_path = str(tmp_path / "saved.yaml")
    window.config_panel._vars["scan.num_views"].set("90")
    window.config_panel._commit("scan.num_views")
    assert window.save_config_to_file(new_path, quiet=True)
    assert window.store.path == os.path.abspath(new_path)
    assert window.store.dirty is False
    assert yaml.safe_load(open(new_path, encoding="utf-8"))[
        "scan"]["num_views"] == 90


def test_calib_on_config_changed_keeps_user_input(window):
    """用户在标定输入框打字时，外部配置刷新不应覆盖其输入。"""
    window.calib_panel._sphere_radius_mm.set("19.7")   # 用户手动输入
    window.calib_panel.on_config_changed()             # 外部触发刷新
    assert window.calib_panel._sphere_radius_mm.get() == "19.7"
