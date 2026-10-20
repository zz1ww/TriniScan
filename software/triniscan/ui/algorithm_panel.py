"""算法选择卡：先选实现，再只显示该实现的参数。

为什么单独做
------------
旧界面里"选算法"只是参数表中间一个普通下拉框，和几十个数值参数混在
一起；而且**所有算法的参数同时显示**（``steger_sigma`` 在 ``row_peak``
下也出现），用户无法判断哪些参数真正生效。

本模块把它拆成两块：

1. 顶部：算法下拉（候选值来自注册表 ``AlgorithmSpec``，不是手写清单）
2. 下方：**只显示当前算法声明的参数**，切换算法时重建

结构型参数（如过滤器链）由调用方通过 ``editor_for`` 注入专用编辑器。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from ..common.logging_utils import get_logger
from .algorithm_list import entries, spec_for
from .field_spec import FieldSpec
from .params import ParamGroup, bind_tooltip, param_to_field_spec

log = get_logger(__name__)

__all__ = ["AlgorithmPanel", "AlgorithmSelector"]


class _ChoiceSelector(ttk.Frame):
    """按参数声明的 ``choices`` 列出"方法"的下拉框。

    用于注册表按**算法域**登记、具体方法由参数选择的情形
    （如 ``EXTRACTORS`` 只有 ``centerline``，方法由
    ``extraction.method`` 决定）。接口与 :class:`AlgorithmSelector`
    保持一致，供 :class:`AlgorithmPanel` 透明替换。
    """

    def __init__(self, master: tk.Misc, param,
                 on_change: Optional[Callable[[str], None]] = None,
                 **kw) -> None:
        super().__init__(master, **kw)
        self._param = param
        self._on_change = on_change
        self._choices = [str(c) for c in (getattr(param, "choices", ()) or ())]
        self._labels = _option_labels(param, self._choices)

        self._var = tk.StringVar()
        self.combo = ttk.Combobox(
            self, textvariable=self._var, state="readonly",
            values=[self._labels.get(c, c) for c in self._choices],
            width=30)
        self.combo.pack(fill="x")
        self.combo.bind("<<ComboboxSelected>>", self._on_selected)

        self.help_label = ttk.Label(
            self, text=getattr(param, "help", "") or "",
            foreground="#7f8c8d", wraplength=300, justify="left")
        self.help_label.pack(fill="x", pady=(2, 0))

    @property
    def value(self) -> str:
        label = self._var.get()
        for name, text in self._labels.items():
            if text == label:
                return name
        return label

    def set(self, name: str) -> None:
        name = str(name)
        self._var.set(self._labels.get(name, name))

    def _on_selected(self, _event: tk.Event) -> None:
        if self._on_change is not None:
            self._on_change(self.value)


def _option_labels(param, choices: list[str]) -> dict[str, str]:
    """从参数声明的 ``help`` 里提取每个选项的一句话说明。

    ``ParamSpec`` 没有独立的"选项标签"字段，但约定 ``help`` 用
    ``"name=说明；name2=说明2"`` 的形式描述各选项。解析失败时
    显示原始名字，不影响使用。
    """
    help_text = getattr(param, "help", "") or ""
    out: dict[str, str] = {}
    for part in help_text.replace("；", ";").split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        name, _, desc = part.partition("=")
        name = name.strip()
        if name in choices:
            out[name] = f"{name} — {desc.strip()}"
    return out


class AlgorithmPanel(ttk.Frame):
    """算法选择 + 该算法的参数。

    Parameters
    ----------
    master : 父控件
    domain : 注册表名
    store : :class:`ConfigStore`
    method_key : 保存算法名的配置路径（如 ``"extraction.method"``）
    on_change : 参数变化回调 ``fn(key_path, value)``
    on_preview : 实时预览回调
    editor_for : 结构型参数的专用编辑器工厂
    extra_keys : 除算法声明外还要展示的配置键（如通用参数）
    """

    def __init__(self, master: tk.Misc, domain: str, store,
                 method_key: str,
                 on_change: Optional[Callable[[str, Any], None]] = None,
                 on_preview: Optional[Callable[[str, Any], None]] = None,
                 editor_for: Optional[Callable[[FieldSpec],
                                               Optional[tk.Widget]]] = None,
                 extra_keys: Optional[list[str]] = None,
                 **kw) -> None:
        super().__init__(master, **kw)
        self.domain = domain
        self.store = store
        self.method_key = method_key
        self._on_change = on_change
        self._on_preview = on_preview
        self._editor_for = editor_for
        self._extra_keys = list(extra_keys or [])

        ttk.Label(self, text="算法实现", style="Key.TLabel").pack(
            anchor="w")
        self.selector = self._make_selector()
        self.selector.pack(fill="x", pady=(2, 6))

        self._body = ttk.Frame(self)
        self._body.pack(fill="x")

        self._group: Optional[ParamGroup] = None
        self._built_for: Optional[str] = None
        self._extra_labels: dict[str, ttk.Label] = {}

    # ------------------------------------------------------------------
    def _make_selector(self) -> "AlgorithmSelector":
        """构造算法下拉框。

        两种情形：

        - **注册表按实现划分**（``FILTERS`` 等）：下拉列出每个实现，
          值即注册名；
        - **注册表按算法域划分**（``EXTRACTORS`` 只有 ``centerline``，
          具体方法由 ``extraction.method`` 选择）：下拉列出该域
          ``method`` 参数的 ``choices``，值写入 ``method_key``。
        """
        names = [e.name for e in entries(self.domain)]
        method_param = self._method_param()
        if method_param is not None and len(names) <= 1:
            return _ChoiceSelector(
                self, method_param, on_change=self._on_method_changed)
        return AlgorithmSelector(self, self.domain,
                                 on_change=self._on_method_changed)

    def _method_param(self):
        """取 ``method_key`` 在算法声明中对应的 ``ParamSpec``。"""
        short = self.method_key.rsplit(".", 1)[-1]
        for e in entries(self.domain):
            spec = spec_for(self.domain, e.name)
            for param in getattr(spec, "params", ()) or ():
                if getattr(param, "key", "") == short:
                    return param
        return None

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """从 store 重新读取并重建（配置整体替换后调用）。"""
        current = self._cfg_get(self.method_key, "")
        self.selector.set(current)
        self._rebuild(force=True)

    def set_value(self, key: str, value: Any) -> None:
        if self._group is not None:
            self._group.set_value(key, value)

    # ------------------------------------------------------------------
    def _cfg_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.store.cfg.get(key, default)
        except Exception:                   # noqa: BLE001
            return default

    def _on_method_changed(self, name: str) -> None:
        if name != self._cfg_get(self.method_key):
            if self._on_change is not None:
                self._on_change(self.method_key, name)
            else:
                self.store.update(self.method_key, name)
        self._rebuild(force=True)

    # ------------------------------------------------------------------
    def _rebuild(self, force: bool = False) -> None:
        name = self.selector.value
        if not force and name == self._built_for:
            return
        self._built_for = name

        for child in self._body.winfo_children():
            child.destroy()
        self._group = None
        self._extra_labels.clear()

        spec = self._spec_for(name)
        if spec is None:
            ttk.Label(self._body, text=f"（{name} 未声明参数）",
                      foreground="#7f8c8d").pack(anchor="w")
            return

        group = ParamGroup(self._body, on_change=self._on_change,
                           on_preview=self._on_preview,
                           editor_for=self._editor_for)
        group.pack(fill="x")

        for param in getattr(spec, "params", ()) or ():
            key_path = param.name
            if key_path == self.method_key:
                continue            # 方法本身已由上方下拉框承担
            fs = param_to_field_spec(param, key_path)
            group.add_row(fs, self._cfg_get(key_path, param.default))

        self._group = group
        self._add_extras()

    def _spec_for(self, name: str):
        """取当前选择对应的参数声明。

        - 方法型选择器（``_ChoiceSelector``）：注册表里只有一个域，
          直接取该域的 ``AlgorithmSpec``；
        - 实现型选择器：按注册名取。
        """
        if isinstance(self.selector, _ChoiceSelector):
            names = [e.name for e in entries(self.domain)]
            return spec_for(self.domain, names[0]) if names else None
        return spec_for(self.domain, name)

    def _add_extras(self) -> None:
        """追加算法声明之外但仍需可见的键（只读提示，避免重复编辑）。"""
        if not self._extra_keys:
            return
        for key in self._extra_keys:
            if self._group is not None and key in self._group.keys:
                continue
            value = self._cfg_get(key)
            lbl = ttk.Label(
                self._body, text=f"{key} = {value!r}",
                foreground="#7f8c8d")
            lbl.pack(anchor="w", pady=(4, 0))
            bind_tooltip(lbl, "该键未在当前算法中声明")
            self._extra_labels[key] = lbl

    # ------------------------------------------------------------------
    @property
    def group(self) -> Optional[ParamGroup]:
        """当前算法的参数组（测试用）。"""
        return self._group

    def keys(self) -> list[str]:
        return self._group.keys if self._group is not None else []
