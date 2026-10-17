"""多视角配准：把各视角点云对齐并融合。"""
from .register import (RegisterResult, ViewCloud,  # noqa: F401
                       register_views, to_open3d)

__all__ = ["ViewCloud", "register_views", "to_open3d", "RegisterResult"]
