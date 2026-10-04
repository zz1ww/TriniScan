"""IO 工具：路径处理、图像读写、标定文件序列化。

标定文件统一使用 ``.npz``，并在保存时写入元数据（版本、时间、参数），
便于追溯哪次标定产生的参数。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Optional

import cv2
import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)

CALIB_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    """确保目录存在，返回该目录。"""
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent(path: str) -> str:
    """确保文件所在目录存在。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 图像
# ---------------------------------------------------------------------------
def read_image(path: str, flags: int = cv2.IMREAD_COLOR,
               must_exist: bool = True) -> Optional[np.ndarray]:
    """读取图像，失败时按 ``must_exist`` 决定抛错或返回 None。"""
    img = cv2.imread(path, flags)
    if img is None and must_exist:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def write_image(path: str, image: np.ndarray) -> None:
    """写入图像（自动建目录）。"""
    ensure_parent(path)
    if not cv2.imwrite(path, image):
        raise IOError(f"无法写入图像: {path}")


# ---------------------------------------------------------------------------
# 标定文件
# ---------------------------------------------------------------------------
def save_calibration(path: str, arrays: dict[str, np.ndarray],
                     meta: Optional[dict[str, Any]] = None) -> None:
    """保存标定数据为 ``.npz``。

    Parameters
    ----------
    path : 输出路径
    arrays : 数值数组字典（如 ``{"K": ..., "dist": ...}``）
    meta : 附加元数据（会序列化为 JSON 字符串存入 ``_meta``）
    """
    ensure_parent(path)
    payload: dict[str, Any] = {k: np.asarray(v) for k, v in arrays.items()}

    meta_full = {
        "format_version": CALIB_FORMAT_VERSION,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if meta:
        meta_full.update(meta)
    payload["_meta"] = np.array(json.dumps(meta_full, ensure_ascii=False))

    np.savez(path, **payload)
    log.info("已保存标定文件: %s", path)


def load_calibration(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """加载标定文件，返回 ``(arrays, meta)``。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"标定文件不存在: {path}")

    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "_meta"}
        meta: dict[str, Any] = {}
        if "_meta" in data.files:
            try:
                meta = json.loads(str(data["_meta"]))
            except (json.JSONDecodeError, TypeError):
                log.warning("标定文件元数据解析失败: %s", path)
    return arrays, meta
