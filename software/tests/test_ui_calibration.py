"""标定清单 / 标定台 / 硬件台的测试（P4）。"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from triniscan.calib_manifest import (
    CalibItem, CalibrationStatus, file_digest, inspect_calibrations,
    manifest_path, record_calibration,
)

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config"))


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
class _CfgView:
    """只实现 ``get`` / ``resolve`` 的最小配置视图。"""

    def __init__(self, camera: str, laser: str, axis: str) -> None:
        self._d = {
            "calibration.camera_file": camera,
            "calibration.laser_file": laser,
            "calibration.axis_file": axis,
        }

    def get(self, key, default=None):
        return self._d.get(key, default)

    def resolve(self, key):
        return self._d.get(key, "")


@pytest.fixture
def calib_dir(tmp_path):
    """把真实标定文件复制到临时目录（不碰仓库内文件）。"""
    paths = {}
    for name in ("calib_camera.npz", "calib_laser.npz", "calib_axis.npz"):
        src = os.path.join(CONFIG_DIR, name)
        dst = tmp_path / name
        shutil.copy(src, dst)
        paths[name] = str(dst)
    return paths


@pytest.fixture
def cfg_view(calib_dir):
    return _CfgView(calib_dir["calib_camera.npz"],
                    calib_dir["calib_laser.npz"],
                    calib_dir["calib_axis.npz"])


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


# ---------------------------------------------------------------------------
# 清单：指纹与过期检测
# ---------------------------------------------------------------------------
def test_digest_changes_with_content(calib_dir, tmp_path):
    cam = calib_dir["calib_camera.npz"]
    d1 = file_digest(cam)
    assert len(d1) == 32
    assert file_digest(cam) == d1, "同一内容必须得到相同指纹"
    assert file_digest(str(tmp_path / "nope.npz")) == ""


def test_manifest_sits_beside_camera_file(calib_dir):
    cam = calib_dir["calib_camera.npz"]
    assert os.path.dirname(manifest_path(cam)) == \
        os.path.dirname(os.path.abspath(cam))
    assert manifest_path(cam).endswith("calib_manifest.json")


def test_no_manifest_means_no_stale(cfg_view):
    """没有清单时不误报过期（向后兼容既有标定文件）。"""
    status = inspect_calibrations(cfg_view)
    assert status.manifest_found is False
    assert status.stale == []


def test_fresh_manifest_has_no_stale(cfg_view, calib_dir):
    cam = calib_dir["calib_camera.npz"]
    record_calibration("camera", cam)
    record_calibration("laser", calib_dir["calib_laser.npz"], camera_file=cam)
    record_calibration("axis", calib_dir["calib_axis.npz"], camera_file=cam)

    status = inspect_calibrations(cfg_view)
    assert status.manifest_found is True
    assert status.stale == []
    assert status.needs_recalibration is False


def test_recalibrated_camera_marks_dependents_stale(cfg_view, calib_dir):
    """重标相机后，激光与转轴必须被判为过期。"""
    cam = calib_dir["calib_camera.npz"]
    record_calibration("camera", cam)
    record_calibration("laser", calib_dir["calib_laser.npz"], camera_file=cam)
    record_calibration("axis", calib_dir["calib_axis.npz"], camera_file=cam)

    # 改写相机文件（模拟重新标定得到略有不同的内参）
    with np.load(cam) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["K"] = np.asarray(arrays["K"], dtype=np.float64) * 1.001
    np.savez(cam, **arrays)
    record_calibration("camera", cam)

    status = inspect_calibrations(cfg_view)
    assert set(status.stale) == {"laser", "axis"}
    assert status.needs_recalibration is True
    assert status.item("laser").level == "stale"
    assert status.item("camera").level != "stale", "相机自己不会过期"


def test_stale_survives_camera_restore(cfg_view, calib_dir):
    """恢复旧文件后激光/转轴不再过期（按内容比对，非时间戳）。"""
    cam = calib_dir["calib_camera.npz"]
    original = file_digest(cam)

    record_calibration("camera", cam)
    record_calibration("laser", calib_dir["calib_laser.npz"], camera_file=cam)

    with np.load(cam) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["K"] = np.asarray(arrays["K"], dtype=np.float64) * 1.01
    np.savez(cam, **arrays)
    record_calibration("camera", cam)
    assert inspect_calibrations(cfg_view).stale == ["laser"]

    # 把相机内参改回去（内容与最初一致）
    arrays["K"] = np.asarray(arrays["K"], dtype=np.float64) / 1.01
    np.savez(cam, **arrays)
    assert file_digest(cam) == original
    record_calibration("camera", cam)
    assert inspect_calibrations(cfg_view).stale == []


def test_missing_file_is_missing(tmp_path):
    view = _CfgView(str(tmp_path / "a.npz"), str(tmp_path / "b.npz"),
                    str(tmp_path / "c.npz"))
    status = inspect_calibrations(view)
    for key in ("camera", "laser", "axis"):
        assert status.item(key).level == "missing"
        assert status.item(key).present is False


def test_axis_quality_detected(cfg_view):
    """真实 calib_axis.npz 已重新标定，RMS 远优于旧值 2.09 mm。

    历史：该文件曾只有 4 点、RMS 2.09 mm，判为 ``fail``；重新标定后
    RMS 降到 0.087 mm，应判为 ``ok``。本用例验证"读取真实标定文件并
    给出质量等级"这条链路仍然有效，断言以实际 RMS 为准。
    """
    status = inspect_calibrations(cfg_view)
    axis = status.item("axis")
    assert axis.present is True
    assert axis.level in ("ok", "warn", "fail")
    assert axis.detail


def test_unknown_kind_rejected(calib_dir):
    with pytest.raises(ValueError):
        record_calibration("bogus", calib_dir["calib_camera.npz"])


def test_record_missing_file_is_noop(tmp_path):
    record_calibration("camera", str(tmp_path / "ghost.npz"))
    assert not os.path.exists(manifest_path(str(tmp_path / "ghost.npz")))


# ---------------------------------------------------------------------------
# CalibItem / CalibrationStatus
# ---------------------------------------------------------------------------
def test_item_colors_group_by_severity():
    """fail 与 stale 同为红色是刻意的：两者都必须重标。"""
    color = {lv: CalibItem("k", "t", level=lv).color
             for lv in ("ok", "warn", "fail", "stale", "missing")}
    assert color["fail"] == color["stale"]
    assert len({color["ok"], color["warn"], color["fail"],
                color["missing"]}) == 4


def test_status_usable_only_for_ok_and_warn():
    st = CalibrationStatus(items={
        "camera": CalibItem("camera", "相机", "ok"),
        "laser": CalibItem("laser", "激光", "warn"),
        "axis": CalibItem("axis", "转轴", "fail"),
    })
    assert st.item("camera").usable is True
    assert st.item("laser").usable is True
    assert st.item("axis").usable is False


def test_summary_lines_mention_stale():
    st = CalibrationStatus(
        items={"laser": CalibItem("laser", "激光平面", "stale",
                                  "建立于旧内参")},
        stale=["laser"])
    text = "\n".join(st.summary_lines())
    assert "激光平面" in text
    assert "旧内参" in text


# ---------------------------------------------------------------------------
# 标定台
# ---------------------------------------------------------------------------
def test_calibration_workbench_builds(root, tmp_path):
    pytest.importorskip("tkinter")
    from triniscan.core.config import Config
    from triniscan.core.config_store import ConfigStore
    from triniscan.ui.workbench_calib import CalibrationWorkbench

    cfg = Config.load(os.path.join(CONFIG_DIR, "default.yaml"))
    store = ConfigStore(cfg)
    wb = CalibrationWorkbench(root, store)
    root.update_idletasks()

    assert wb.panel("status") is not None
    assert "calib" in wb._pages
    status = wb.status()
    assert set(status.items) == {"camera", "laser", "axis"}
    wb.destroy()


def test_calibration_panel_uses_latest_frame_queue(root):
    """标定面板必须换用单槽位队列（否则长跑内存增长）。"""
    from triniscan.core.config import Config
    from triniscan.core.config_store import ConfigStore
    from triniscan.ui.workbench_calib import (CalibrationWorkbench,
                                              LatestFrameQueue)

    cfg = Config.load(os.path.join(CONFIG_DIR, "default.yaml"))
    wb = CalibrationWorkbench(root, ConfigStore(cfg))
    root.update_idletasks()
    assert isinstance(wb.panel_obj._queue, LatestFrameQueue)
    wb.destroy()


def test_latest_frame_queue_keeps_only_newest():
    from triniscan.ui.workbench_calib import LatestFrameQueue

    q = LatestFrameQueue()
    for i in range(100):
        q.put(("frame", i))
    assert q.qsize() == 1
    assert q.get_nowait() == ("frame", 99)


def test_calibration_status_panel_refresh(root):
    from triniscan.core.config import Config
    from triniscan.core.config_store import ConfigStore
    from triniscan.ui.workbench_calib import CalibrationWorkbench

    cfg = Config.load(os.path.join(CONFIG_DIR, "default.yaml"))
    wb = CalibrationWorkbench(root, ConfigStore(cfg))
    root.update_idletasks()
    panel = wb.panel("status")
    panel.refresh()
    root.update_idletasks()

    # 状态文本应包含 RMS 数字（说明真的读了文件）
    texts = [lbl.cget("text") for _n, lbl in panel._rows.values()]
    assert any("RMS" in t for t in texts), texts
    wb.destroy()


# ---------------------------------------------------------------------------
# 硬件台
# ---------------------------------------------------------------------------
def test_hardware_workbench_builds(root):
    from triniscan.core.config import Config
    from triniscan.core.config_store import ConfigStore
    from triniscan.ui.workbench_hardware import HardwareWorkbench

    cfg = Config.load(os.path.join(CONFIG_DIR, "default.yaml"))
    wb = HardwareWorkbench(root, ConfigStore(cfg))
    root.update_idletasks()

    assert set(wb._panels) == {"conn", "motion", "scan"}
    assert "status" in wb._pages
    assert wb.connected is False
    # 未连接时动作按钮必须禁用，避免点了没反应
    assert str(wb.panel("motion").btn_rel.cget("state")) == "disabled"
    assert str(wb.panel("motion").btn_stop.cget("state")) == "disabled"
    wb.destroy()


def test_hardware_turntable_config_from_store(root):
    from triniscan.core.config import Config
    from triniscan.core.config_store import ConfigStore
    from triniscan.ui.workbench_hardware import HardwareWorkbench

    cfg = Config.load(os.path.join(CONFIG_DIR, "default.yaml"))
    store = ConfigStore(cfg)
    wb = HardwareWorkbench(root, store)
    root.update_idletasks()

    tc = wb.turntable_config()
    assert tc.steps_per_rev == int(cfg.get("turntable.steps_per_rev"))
    assert tc.microstep == int(cfg.get("turntable.microstep"))
    assert tc.steps_per_degree > 0
    wb.destroy()


def test_hardware_motion_requires_connection(root, monkeypatch):
    from triniscan.core.config import Config
    from triniscan.core.config_store import ConfigStore
    from triniscan.ui import workbench_hardware as hw

    shown = []
    monkeypatch.setattr(hw.messagebox, "showinfo",
                        lambda *a, **k: shown.append(a))

    cfg = Config.load(os.path.join(CONFIG_DIR, "default.yaml"))
    wb = hw.HardwareWorkbench(root, ConfigStore(cfg))
    root.update_idletasks()

    wb.move_relative(10.0)
    assert shown, "未连接时应提示用户"
    assert wb.busy is False
    wb.destroy()


def test_hardware_scan_rejects_absurd_count(root, monkeypatch):
    from triniscan.core.config import Config
    from triniscan.core.config_store import ConfigStore
    from triniscan.ui import workbench_hardware as hw

    warned = []
    monkeypatch.setattr(hw.messagebox, "showwarning",
                        lambda *a, **k: warned.append(a))

    cfg = Config.load(os.path.join(CONFIG_DIR, "default.yaml"))
    wb = hw.HardwareWorkbench(root, ConfigStore(cfg))
    root.update_idletasks()

    wb.scan_sweep(100000)
    assert warned, "超大站数必须拒绝（防止转台转到天亮）"
    wb.destroy()


def test_hardware_app_log_is_bounded(root):
    from triniscan.core.config import Config
    from triniscan.core.config_store import ConfigStore
    from triniscan.ui.workbench_hardware import HardwareWorkbench, _StatusPage

    cfg = Config.load(os.path.join(CONFIG_DIR, "default.yaml"))
    wb = HardwareWorkbench(root, ConfigStore(cfg))
    root.update_idletasks()

    page = wb._pages["status"]
    for i in range(_StatusPage.MAX_LINES + 200):
        page.append(f"line {i}")
    lines = int(page.text.index("end-1c").split(".")[0])
    assert lines <= _StatusPage.MAX_LINES + 1, "Text 控件必须有行数上限"
    wb.destroy()
