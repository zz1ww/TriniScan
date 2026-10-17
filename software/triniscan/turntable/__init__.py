"""【兼容层】旧的 ``triniscan.turntable`` 路径。

实现已迁移到 :mod:`triniscan.hardware.turntable`。
"""
from __future__ import annotations

from ..hardware.turntable import *              # noqa: F401,F403
from ..hardware.turntable import __all__        # noqa: F401
