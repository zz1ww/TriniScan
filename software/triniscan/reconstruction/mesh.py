"""【兼容层】旧路径 ``triniscan.reconstruction.mesh`` → ``:mod:`triniscan.algorithms.meshing.mesh``。

用 import * **加上显式的私有名转发**，保证 from ... import _xxx
这类旧测试/旧代码仍然可用。
"""
from __future__ import annotations

import importlib as _importlib

_target = _importlib.import_module("triniscan.algorithms.meshing.mesh")
globals().update({k: v for k, v in vars(_target).items()
                  if not k.startswith("__")})

__all__ = getattr(_target, "__all__", [])
