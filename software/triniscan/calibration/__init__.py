"""【兼容层】旧的 ``triniscan.calibration`` 路径。

实现已迁移到 :mod:`triniscan.algorithms.calibration`。
本模块只做转发，保证旧代码 ``from triniscan.calibration import ...``
仍然可用。**请勿在此添加新逻辑。**
"""
from __future__ import annotations

from ..algorithms.calibration import *          # noqa: F401,F403
from ..algorithms.calibration import __all__    # noqa: F401
