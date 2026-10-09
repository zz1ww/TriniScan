"""核心模块：配置、主控流程。

注意
----
``Config`` 与 ``ScanPipeline`` 采用**惰性导入**：``core.pipeline`` 会
连带导入 ``volume`` -> ``trimesh``（导入耗时约 2.6s），而打开图形界面
时只需要 ``Config``。若在此处直接 ``from .pipeline import ...``，
则任何 ``import triniscan.core.config`` 都会被迫付出这段开销，
导致界面启动明显变慢。

因此这里改用模块级 ``__getattr__``（PEP 562）按需导入，
保持 ``from triniscan.core import Config / ScanPipeline`` 的用法不变。
"""
from __future__ import annotations

from .config import Config, find_project_root

__all__ = ["Config", "find_project_root", "ScanPipeline", "ScanResult"]


def __getattr__(name: str):
    if name in ("ScanPipeline", "ScanResult"):
        from .pipeline import ScanPipeline, ScanResult
        return {"ScanPipeline": ScanPipeline,
                "ScanResult": ScanResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
