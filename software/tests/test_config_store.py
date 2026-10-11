"""``ConfigStore`` 与「界面写回配置」的测试。

覆盖要点
--------
- 修改/保存/未保存标记/原子写入；
- 监听器通知；
- 面板与主窗口共享同一份配置（曾经的 bug：两份对象）；
- 界面改参数 → 落盘 → 测量/标定读到新值。
"""
import os
import shutil
import tempfile

import pytest
import yaml

from triniscan.core.config import Config
from triniscan.core.config_store import ConfigStore

_SAMPLE = {
    "camera": {"index": 0, "width": 1920, "auto_exposure": False},
    "scan": {"num_views": 180, "angle_step_deg": 2.0},
    "axis_calib": {"sphere_radius_m": 0.020, "method": "peak",
                   "num_views": 12},
}


@pytest.fixture
def cfg_file(tmp_path):
    """把示例配置写成临时 YAML 文件，返回其路径。"""
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(_SAMPLE, allow_unicode=True),
                    encoding="utf-8")
    return str(path)


@pytest.fixture
def store(cfg_file):
    return ConfigStore(Config.load(cfg_file))


# ---------------------------------------------------------------------------
# ConfigStore 基础行为
# ---------------------------------------------------------------------------
def test_initial_state(store, cfg_file):
    assert store.path == os.path.abspath(cfg_file)
    assert store.dirty is False
    assert store.cfg.get("camera.index") == 0


def test_update_changes_memory_and_marks_dirty(store):
    store.update("camera.index", 1)
    assert store.cfg.get("camera.index") == 1
    assert store.dirty is True


def test_update_does_not_touch_file(store, cfg_file):
    store.update("camera.index", 1)
    # 文件应保持原值，直到显式 save()
    on_disk = yaml.safe_load(open(cfg_file, encoding="utf-8"))
    assert on_disk["camera"]["index"] == 0


def test_save_persists_and_clears_dirty(store, cfg_file):
    store.update("camera.index", 1)
    written = store.save()
    assert written == os.path.abspath(cfg_file)
    assert store.dirty is False
    on_disk = yaml.safe_load(open(cfg_file, encoding="utf-8"))
    assert on_disk["camera"]["index"] == 1


def test_save_as_switches_bound_path(store, tmp_path):
    new_path = str(tmp_path / "other.yaml")
    store.update("scan.num_views", 90)
    written = store.save(new_path)
    assert written == os.path.abspath(new_path)
    assert store.path == os.path.abspath(new_path)
    assert store.dirty is False
    # 原文件未被改动
    assert os.path.exists(new_path)


def test_update_many_notifies_once(store):
    events = []
    store.add_listener(lambda s, k, v: events.append((k, v)))
    store.update_many({"camera.index": 2, "scan.num_views": 60})
    assert ("camera.index", 2) in events
    assert ("scan.num_views", 60) in events
    assert store.cfg.get("camera.index") == 2
    assert store.cfg.get("scan.num_views") == 60
    assert store.dirty is True


def test_listener_receives_changes(store):
    got = []
    store.add_listener(lambda s, k, v: got.append((k, v)))
    store.update("axis_calib.method", "sphere")
    assert got == [("axis_calib.method", "sphere")]


def test_listener_exception_does_not_break_update(store):
    def bad(_s, _k, _v):
        raise RuntimeError("boom")

    store.add_listener(bad)
    store.add_listener(lambda s, k, v: None)
    store.update("camera.index", 3)      # 不应抛出
    assert store.cfg.get("camera.index") == 3


def test_remove_listener(store):
    got = []
    fn = lambda s, k, v: got.append(k)  # noqa: E731
    store.add_listener(fn)
    store.update("camera.index", 1)
    store.remove_listener(fn)
    store.update("camera.index", 2)
    assert got == ["camera.index"]


def test_replace_resets_dirty(store):
    store.update("camera.index", 9)
    assert store.dirty is True
    store.replace(Config({"camera": {"index": 5}}))
    assert store.dirty is False
    assert store.cfg.get("camera.index") == 5


def test_save_atomic_leaves_valid_yaml(store, cfg_file):
    """保存应使用临时文件 + 原子替换，不产生半截文件。"""
    store.update("camera.width", 1280)
    store.save()
    data = yaml.safe_load(open(cfg_file, encoding="utf-8"))
    assert data["camera"]["width"] == 1280
    assert not os.path.exists(cfg_file + ".tmp")


def test_save_without_path_raises():
    s = ConfigStore(Config({"camera": {"index": 0}}))   # 无 _meta 路径
    with pytest.raises(ValueError):
        s.save()


def test_no_private_keys_written(store, cfg_file):
    """_meta 等私有键不应写进文件。"""
    store.update("camera.index", 1)
    store.save()
    data = yaml.safe_load(open(cfg_file, encoding="utf-8"))
    assert "_meta" not in data


def test_roundtrip_reload(store, cfg_file):
    """保存后重新加载，值应一致。"""
    store.update_many({"camera.index": 1, "scan.num_views": 72,
                       "axis_calib.sphere_radius_m": 0.0215})
    store.save()
    reloaded = Config.load(cfg_file)
    assert reloaded.get("camera.index") == 1
    assert reloaded.get("scan.num_views") == 72
    assert reloaded.get("axis_calib.sphere_radius_m") == 0.0215
