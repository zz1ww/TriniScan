"""配置加载与访问。

特性
----
- YAML 加载，支持点号路径查询 ``cfg.get("camera.width")``；
- 属性访问 ``cfg.camera.width``；
- 深度合并覆盖（便于命令行覆盖个别参数）；
- 项目根自动发现，使相对路径可解析。
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterator, Mapping

import yaml

__all__ = ["Config", "find_project_root", "DEFAULT_CONFIG_REL"]

DEFAULT_CONFIG_REL = os.path.join("software", "config", "default.yaml")


class Config(Mapping):
    """轻量、只读、可点号访问的配置对象。"""

    def __init__(self, data: dict | None = None, root: str | None = None):
        self._data: dict = copy.deepcopy(data) if data else {}
        self._root: str | None = root

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | None = None,
             root: str | None = None,
             overrides: dict | None = None) -> "Config":
        """从 YAML 文件加载配置。

        Parameters
        ----------
        path : 配置文件路径；None 时用项目根下的默认配置。
        root : 项目根；None 时自动发现。
        overrides : 需覆盖的嵌套字典。
        """
        root = root or find_project_root()
        if path is None:
            path = os.path.join(root, DEFAULT_CONFIG_REL)
        path = os.path.abspath(path)

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if overrides:
            data = _deep_merge(data, overrides)

        cfg = cls(data, root=root)
        cfg._data.setdefault("_meta", {})["config_path"] = path
        return cfg

    # ------------------------------------------------------------------
    @property
    def root(self) -> str:
        return self._root or find_project_root()

    def get(self, key_path: str, default: Any = None) -> Any:
        """按点号路径取值，如 ``get("camera.width")``。"""
        node: Any = self._data
        for part in key_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def resolve(self, key_path: str, default: str | None = None) -> str:
        """取路径型配置并解析为绝对路径（相对项目根）。"""
        value = self.get(key_path, default)
        if value is None:
            raise KeyError(f"缺少路径配置: {key_path}")
        return value if os.path.isabs(value) \
            else os.path.join(self.root, value)

    def with_overrides(self, overrides: dict) -> "Config":
        """返回带覆盖的新配置。"""
        return Config(_deep_merge(self._data, overrides), root=self._root)

    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        return Config(val, self._root) if isinstance(val, dict) else val

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            val = self._data[name]
            return Config(val, self._root) if isinstance(val, dict) else val
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"Config(keys={list(self._data.keys())}, root={self._root!r})"

    @property
    def raw(self) -> dict:
        """底层字典（只读用途）。"""
        return self._data


# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典（override 优先）。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (key in result and isinstance(result[key], dict)
                and isinstance(value, dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def find_project_root(start: str | None = None,
                      markers: tuple[str, ...] = ("README.md", "software")
                      ) -> str:
    """向上查找同时含指定标记的项目根目录。"""
    cur = os.path.abspath(start or os.getcwd())
    while True:
        if all(os.path.exists(os.path.join(cur, m)) for m in markers):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start or os.getcwd())
        cur = parent
