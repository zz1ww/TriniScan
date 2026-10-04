"""通用基础层：几何、日志、IO。

本层不依赖任何业务模块，供上层（标定/提取/重建/体积）复用。
"""
from .logging_utils import get_logger, set_level
from . import geometry, io_utils

__all__ = ["get_logger", "set_level", "geometry", "io_utils"]
