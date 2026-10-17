"""【兼容层】旧的 ``triniscan.extraction`` 路径。

实现已迁移到 :mod:`triniscan.algorithms.extraction`。
"""
from __future__ import annotations

from ..algorithms.extraction import *           # noqa: F401,F403
from ..algorithms.extraction import __all__     # noqa: F401
