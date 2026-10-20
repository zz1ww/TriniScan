"""标定清单：记录"标定在什么条件下产生"，并在载入时校验是否过期。

问题
----
标定存在**严格的生成依赖**：

.. code-block:: text

    camera (内参 K, dist)
      ├── laser (平面方程)      像素→射线依赖 K/dist
      └── axis  (转轴位姿)      点云坐标系依赖 K + laser

旧实现只检查"文件是否存在 + 残差是否达标"。于是重新标定相机内参后，
激光平面与转轴仍按旧内参解算，点云会整体形变或错位，而界面照旧显示
"标定合格"——错误被静默接受。

做法
----
不修改既有 ``.npz`` 的写入方（那属于用户未提交的工作），而是在
:func:`triniscan.common.io_utils.save_calibration` 之外**旁挂一份清单**
``config/calib_manifest.json``：

.. code-block:: json

    {
      "format_version": 1,
      "camera": {"file": "calib_camera.npz", "md5": "...", "at": "..."},
      "laser":  {"file": "calib_laser.npz",  "md5": "...", "camera_md5": "..."}
    }

``camera_md5`` 是**生成激光/转轴标定时相机文件的指纹**。载入时比对：

- 一致 → 有效；
- 不一致 → ``stale``（已过期，建议重标）；
- 无记录 → ``ok``（向后兼容：老文件没有清单，不误报）。

清单缺失不影响测量，只是失去过期检测能力。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .common.io_utils import load_calibration
from .common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "MANIFEST_NAME",
    "CalibItem",
    "CalibrationStatus",
    "file_digest",
    "inspect_calibrations",
    "record_calibration",
    "manifest_path",
]

MANIFEST_NAME = "calib_manifest.json"
FORMAT_VERSION = 1

#: 三项标定的固定顺序与中文名
ITEMS: tuple[tuple[str, str], ...] = (
    ("camera", "相机内参"),
    ("laser", "激光平面"),
    ("axis", "转台转轴"),
)

#: 颜色（与 UI 主题一致）
_COLORS = {
    "ok": "#27ae60",
    "warn": "#e67e22",
    "fail": "#c0392b",
    "stale": "#c0392b",
    "missing": "#95a5a6",
}


# ---------------------------------------------------------------------------
def file_digest(path: str) -> str:
    """文件内容的 MD5（不存在返回空串）。

    用内容而非修改时间：复制、备份恢复、编辑器重存都不会改变指纹。
    """
    if not os.path.exists(path):
        return ""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(calib_file: str) -> str:
    """清单与相机标定同目录（便于整体拷贝/备份）。"""
    return os.path.join(os.path.dirname(os.path.abspath(calib_file)),
                        MANIFEST_NAME)


def _load_manifest(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        log.warning("标定清单解析失败: %s", path)
        return {}


def _save_manifest(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
@dataclass
class CalibItem:
    """单项标定的状态。"""

    key: str
    title: str
    level: str = "missing"          # ok / warn / fail / stale / missing
    detail: str = ""

    @property
    def color(self) -> str:
        return _COLORS.get(self.level, _COLORS["missing"])

    @property
    def text(self) -> str:
        return self.detail or "未标定"

    @property
    def present(self) -> bool:
        return self.level != "missing"

    @property
    def usable(self) -> bool:
        return self.level in ("ok", "warn")


@dataclass
class CalibrationStatus:
    """三项标定的整体状态。"""

    items: dict[str, CalibItem] = field(default_factory=dict)
    stale: list[str] = field(default_factory=list)
    manifest_found: bool = False

    def item(self, key: str) -> CalibItem:
        return self.items.get(key) or CalibItem(
            key=key, title=key, level="missing", detail="未配置")

    def stale_items(self) -> list[CalibItem]:
        return [self.item(k) for k in self.stale]

    @property
    def needs_recalibration(self) -> bool:
        return bool(self.stale)

    @property
    def all_ok(self) -> bool:
        return all(i.usable for i in self.items.values())

    # ------------------------------------------------------------------
    def summary_lines(self) -> list[str]:
        """给只读文本框用的多行摘要。"""
        lines: list[str] = []
        for key, title in ITEMS:
            item = self.item(key)
            lines.append(f"{title:<6}{'[' + item.level + ']':<9}{item.text}")
        if self.stale:
            lines.append("")
            lines.append("⚠ 以下标定建立在旧内参之上：")
            lines.extend(f"  · {self.item(k).title}" for k in self.stale)
        if not self.manifest_found:
            lines.append("")
            lines.append("（无标定清单：不再检测过期，仅看残差）")
        return lines


# ---------------------------------------------------------------------------
def inspect_calibrations(cfg, *, thresholds: Optional[dict] = None) -> CalibrationStatus:
    """检查三项标定文件：是否存在、残差是否合格、是否已过期。

    Parameters
    ----------
    cfg : 配置对象（支持 ``get`` / ``resolve``）
    thresholds : 覆盖 :data:`DEFAULT_THRESHOLDS`
    """
    from .algorithms.calibration.quality import (assess_camera, assess_axis,
                                                  assess_laser)

    status = CalibrationStatus()

    def _path(key: str) -> str:
        raw = str(cfg.get(f"calibration.{key}_file", "") or "")
        try:
            return cfg.resolve(f"calibration.{key}_file")
        except Exception:                   # noqa: BLE001
            return raw

    camera_path = _path("camera")
    manifest = _load_manifest(manifest_path(camera_path)) if camera_path else {}
    status.manifest_found = bool(manifest)

    camera_digest = file_digest(camera_path)
    recorded_camera = str(
        (manifest.get("camera") or {}).get("md5", "") or "")

    for key, title in ITEMS:
        path = _path(key)
        item = CalibItem(key=key, title=title)

        if not path or not os.path.exists(path):
            item.level, item.detail = "missing", "文件不存在"
            status.items[key] = item
            continue

        # 过期检测：激光/转轴必须基于当前相机内参
        entry = manifest.get(key) or {}
        born_with = str(entry.get("camera_md5", "") or "")
        if key != "camera" and born_with and recorded_camera \
                and born_with != recorded_camera:
            item.level = "stale"
            item.detail = "建立于旧内参，需重标"
            status.stale.append(key)
            status.items[key] = item
            continue

        try:
            item2 = _assess(key, path, assess_camera, assess_laser,
                            assess_axis, thresholds)
        except Exception as exc:            # noqa: BLE001
            item.level, item.detail = "fail", f"读取失败: {exc}"
            status.items[key] = item
            continue

        status.items[key] = item2

    return status


def _assess(key: str, path: str, assess_camera, assess_laser,
            assess_axis, thresholds: Optional[dict]) -> CalibItem:
    """读取文件并套用统一阈值，返回单项状态。"""
    from .algorithms.calibration import (load_axis_calib, load_camera_calib,
                                          load_laser_calib)

    title = dict(ITEMS)[key]
    item = CalibItem(key=key, title=title)

    if key == "camera":
        result = load_camera_calib(path)
        report = assess_camera(result, thresholds=thresholds)
        size = getattr(result, "image_size", None) or (0, 0)
        detail = (f"RMS {result.rms:.3f} px  "
                  f"{int(size[0])}×{int(size[1])}")
    elif key == "laser":
        result = load_laser_calib(path)
        report = assess_laser(result, thresholds=thresholds)
        detail = f"RMS {result.residual_rms * 1e3:.3f} mm"
        n = getattr(result, "num_points", None)
        if n:
            detail += f"  {n} 点"
    else:
        result = load_axis_calib(path)
        report = assess_axis(result, thresholds=thresholds)
        detail = f"RMS {result.residual_rms * 1e3:.3f} mm"
        n = getattr(result, "num_points", None)
        if n is not None:
            detail += f"  {n} 点"

    level = getattr(getattr(report, "level", None), "value", "warn")
    item.level = level
    item.detail = detail
    return item


# ---------------------------------------------------------------------------
def record_calibration(kind: str, calib_file: str, *,
                       camera_file: Optional[str] = None) -> None:
    """标定成功后登记指纹，供后续过期检测使用。

    Parameters
    ----------
    kind : ``"camera"`` / ``"laser"`` / ``"axis"``
    calib_file : 刚写入的标定文件
    camera_file : 生成该标定时所用的相机标定文件（``camera`` 自身可省略）

    Notes
    -----
    调用失败**不应中断标定流程**——清单只是附加信息，写不进去最坏
    退化成"不检测过期"，与旧行为一致。
    """
    if kind not in dict(ITEMS):
        raise ValueError(f"未知标定类型: {kind}")
    if not os.path.exists(calib_file):
        log.warning("标定文件不存在，跳过登记: %s", calib_file)
        return

    anchor = camera_file or calib_file
    path = manifest_path(anchor)
    data = _load_manifest(path)
    data.setdefault("format_version", FORMAT_VERSION)

    entry: dict[str, Any] = {
        "file": os.path.basename(calib_file),
        "md5": file_digest(calib_file),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if kind != "camera":
        entry["camera_md5"] = file_digest(
            camera_file or str(data.get("_camera_file", "") or ""))
    else:
        data["_camera_file"] = os.path.abspath(calib_file)

    data[kind] = entry
    try:
        _save_manifest(path, data)
        log.info("已登记标定清单: %s → %s", kind, path)
    except OSError as exc:
        log.warning("写入标定清单失败（不影响测量）: %s", exc)
