"""配置加载与访问。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "default.yaml"
)


class Config:
    """轻量配置对象，支持属性/字典访问与点号路径查询。"""

    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        path = path or DEFAULT_CONFIG_PATH
        path = os.path.abspath(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data or {})

    def get(self, key_path: str, default: Any = None) -> Any:
        """按点号路径取值，如 get('camera.width')。"""
        node: Any = self._data
        for part in key_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            val = self._data[name]
            return Config(val) if isinstance(val, dict) else val
        raise AttributeError(name)

    @property
    def raw(self) -> dict:
        return self._data


def find_project_root(start: str | None = None) -> str:
    """向上查找含 README.md 与 software 目录的项目根。"""
    cur = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isdir(os.path.join(cur, "software")) and \
           os.path.isfile(os.path.join(cur, "README.md")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start or os.getcwd())
        cur = parent
