"""【兼容层】旧的 ``triniscan.camera`` 路径。

实现已迁移到 :mod:`triniscan.hardware.camera`。
"""
from __future__ import annotations

from ..hardware.camera import *                 # noqa: F401,F403
from ..hardware.camera import __all__           # noqa: F401
