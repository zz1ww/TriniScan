"""统一日志工具。

用法::

    from triniscan.common import get_logger
    log = get_logger(__name__)
    log.info("hello")

设计要点
--------
- 全局只配置一次 handler（避免重复输出）。
- 支持通过 ``TRINISCAN_LOG_LEVEL`` 环境变量或 :func:`set_level` 调整。
- 输出带时间戳、模块名、级别，便于排查现场问题。
"""
from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATE_FMT = "%H:%M:%S"

_configured = False


def _configure_root() -> None:
    """配置根 logger（仅一次）。"""
    global _configured
    if _configured:
        return

    level_name = os.environ.get("TRINISCAN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATE_FMT))

    root = logging.getLogger("triniscan")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取带 ``triniscan.`` 前缀的 logger。"""
    _configure_root()
    if not name.startswith("triniscan"):
        name = f"triniscan.{name}"
    return logging.getLogger(name)


def set_level(level: str | int) -> None:
    """运行期调整日志级别。"""
    _configure_root()
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("triniscan").setLevel(level)
