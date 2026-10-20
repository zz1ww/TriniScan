"""过滤链编辑器：``reconstruction.filter_chain`` 的可视化有序列表。

为什么必须单独做
----------------
``filter_chain`` 是 ``list[dict]``。旧界面把它当成普通配置项渲染成单行
Entry，内容为完整 ``repr``；用户点一下即失焦，``coerce`` 把整个列表变成
字符串写回 store——``FilterChain.from_config`` 随后读不到任何项，所有
过滤器静默消失。

本模块提供专业的有序列表编辑：

.. code-block:: text

    ☑ 1. depth      深度门限            [参数…]  ↑  ↓  ✕
    ☑ 2. cluster    连通域去噪          [参数…]  ↑  ↓  ✕
    ☐ 3. bounds     物理边界（未启用）   [参数…]  ↑  ↓  ✕
    ☑ 4. turntable  去转盘              [参数…]  ↑  ↓  ✕
    ☑ 5. axial      轴向裁剪            [参数…]  ↑  ↓  ✕
                                   [+ 添加过滤器 ▼]

约束
----
``PluginBase.depends_on_current`` 为 ``True`` 的过滤器（``turntable`` /
``axial``）依赖"当前点云的整体分布"（估台面高度、看高度直方图）。它们
被移到这类过滤器之前时，输入是尚未清理的原始点云，估计结果会失真。
因此移动时**直接拒绝**并给出说明，而不是静默允许。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from ..common.logging_utils import get_logger
from ..contracts import FILTERS
from .algorithm_list import entries
from .field_spec import FieldSpec
from .params import ParamGroup, bind_tooltip

log = get_logger(__name__)

__all__ = ["FilterChainEditor"]

CHAIN_KEY = "reconstruction.filter_chain"


def _plugin_class(name: str):
    """取过滤器的插件类（未注册返回 ``None``）。

    注意：``FILTERS`` 登记的是**插件类**（``PluginBase`` 子类），不是
    ``AlgorithmSpec``——参数声明、中文名、顺序依赖都在类属性上。
    """
    try:
        if name not in set(FILTERS.names()):
            return None
        return FILTERS.get(name)
    except Exception:                       # noqa: BLE001
        log.exception("读取过滤器插件失败: %s", name)
        return None


def _params_of(name: str) -> tuple:
    """取过滤器的 ``ParamSpec`` 元组。"""
    cls = _plugin_class(name)
    return tuple(getattr(cls, "params", ()) or ())


def _label_of(name: str) -> str:
    cls = _plugin_class(name)
    return getattr(cls, "label", "") or name


def _depends_on_current(name: str) -> bool:
    return bool(getattr(_plugin_class(name), "depends_on_current", False))


class _FilterRow(ttk.Frame):
    """过滤链中的一行：启用 + 名称 + 参数 + 排序按钮。"""

    def __init__(self, master: tk.Misc, editor: "FilterChainEditor",
                 index: int, item: dict) -> None:
        super().__init__(master)
        self.editor = editor
        self.index = index
        self.item = dict(item)

        self.name = str(self.item.get("name", ""))
        self._params_open = False

        self.columnconfigure(1, weight=1)

        self.enabled = tk.BooleanVar(
            value=bool(self.item.get("enable", True)))
        chk = ttk.Checkbutton(self, variable=self.enabled,
                              command=self._on_enable)
        chk.grid(row=0, column=0, sticky="w")

        label = ttk.Label(self, text=f"{index + 1}. {self.name}")
        label.grid(row=0, column=1, sticky="w", padx=(2, 6))
        bind_tooltip(label, _label_of(self.name))

        self.params_btn = ttk.Button(self, text="参数…", width=7,
                                     command=self._toggle_params)
        self.params_btn.grid(row=0, column=2, padx=2)
        if not self._has_params():
            self.params_btn.state(["disabled"])

        ttk.Button(self, text="↑", width=3,
                   command=lambda: editor.move(index, -1)).grid(
            row=0, column=3)
        ttk.Button(self, text="↓", width=3,
                   command=lambda: editor.move(index, +1)).grid(
            row=0, column=4)
        ttk.Button(self, text="✕", width=3,
                   command=lambda: editor.remove(index)).grid(
            row=0, column=5, padx=(2, 0))

        if not self.enabled.get():
            label.configure(foreground="#9aa0a6")

        self._params_frame: Optional[ttk.Frame] = None
        self._param_group: Optional[ParamGroup] = None

    # ------------------------------------------------------------------
    def _has_params(self) -> bool:
        return bool(_params_of(self.name))

    def _on_enable(self) -> None:
        self.item["enable"] = bool(self.enabled.get())
        self.editor.commit()

    # ------------------------------------------------------------------
    def _toggle_params(self) -> None:
        if self._params_frame is not None:
            self._params_frame.destroy()
            self._params_frame = None
            self._param_group = None
            self.params_btn.configure(text="参数…")
            return

        params = _params_of(self.name)
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, columnspan=6, sticky="ew",
                   padx=(20, 0), pady=(2, 2))
        group = ParamGroup(frame, on_change=self._on_param_change)
        group.pack(fill="x")

        cfg = self.editor.cfg
        for param in params:
            # 参数值优先取本行覆盖项，其次取全局配置
            short = getattr(param, "short", "") or param.key
            value = self.item.get(short)
            if value is None:
                value = cfg.get(param.name, param.default)
            kind = {bool: "bool", int: "int", float: "float",
                    str: "str"}.get(getattr(param, "type", str), "str")
            fs = FieldSpec(
                key=param.name,
                label=getattr(param, "label", "") or short,
                group="",
                hint=getattr(param, "help", "") or "",
                kind=kind,
                choices=(list(param.choices)
                         if getattr(param, "choices", None) else None),
                low=getattr(param, "low", None),
                high=getattr(param, "high", None),
                unit=getattr(param, "unit", "") or "",
                from_algorithm=True,
            )
            group.add_row(fs, value)

        self._params_frame = frame
        self._param_group = group
        self.params_btn.configure(text="收起")

    def _on_param_change(self, key_path: str, value: Any) -> None:
        """把改动写进本行的覆盖项（``item[short]``），并提交整条链。"""
        short = key_path.rsplit(".", 1)[-1]
        self.item[short] = value
        self.editor.commit()


class FilterChainEditor(ttk.Frame):
    """``filter_chain`` 的有序列表编辑器。

    Parameters
    ----------
    master : 父控件
    store : :class:`ConfigStore`
    key : 配置路径（默认 ``reconstruction.filter_chain``）
    on_change : 提交后的回调 ``fn(chain)``

    Notes
    -----
    每次结构变化（增删/排序/启用/参数）都会调用
    ``store.update(key, list[dict])``，写入的是**真正的列表**，
    不是字符串。
    """

    def __init__(self, master: tk.Misc, store, *,
                 key: str = CHAIN_KEY,
                 on_change: Optional[Callable[[list], None]] = None,
                 **kw) -> None:
        super().__init__(master, **kw)
        self.store = store
        self.key = key
        self._on_change = on_change
        self._rows: list[_FilterRow] = []

        self._list_frame = ttk.Frame(self)
        self._list_frame.pack(fill="x")

        add_bar = ttk.Frame(self)
        add_bar.pack(fill="x", pady=(4, 0))
        ttk.Label(add_bar, text="添加:", foreground="#7f8c8d").pack(
            side="left")
        self._add_var = tk.StringVar()
        self._add_combo = ttk.Combobox(
            add_bar, textvariable=self._add_var, state="readonly",
            width=24, values=self._add_candidates())
        self._add_combo.pack(side="left", padx=(4, 4), fill="x", expand=True)
        ttk.Button(add_bar, text="添加", width=6,
                   command=self._on_add).pack(side="left")

        self.rebuild()

    # ------------------------------------------------------------------
    @property
    def cfg(self):
        return self.store.cfg

    def chain(self) -> list[dict]:
        """当前链（深拷贝，可直接写回 store）。"""
        return [dict(row.item) for row in self._rows]

    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        """按 store 中的链重建全部行。"""
        for row in self._rows:
            row.destroy()
        self._rows = []

        raw = self.cfg.get(self.key, []) or []
        if not isinstance(raw, list):
            # 配置已被损坏成字符串：提示并保持为空，不静默覆盖
            log.warning("%s 不是列表（当前 %s），请用编辑器重新配置",
                        self.key, type(raw).__name__)
            raw = []

        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            row = _FilterRow(self._list_frame, self, i, item)
            row.pack(fill="x", pady=1)
            self._rows.append(row)

        if not self._rows:
            ttk.Label(self._list_frame, text="（空：所有过滤均关闭）",
                      foreground="#7f8c8d").pack(anchor="w")
        self._refresh_add_values()

    def refresh(self) -> None:
        self.rebuild()

    def _refresh_add_values(self) -> None:
        try:
            self._add_combo.configure(values=self._add_candidates())
        except tk.TclError:                 # pragma: no cover
            pass

    def _add_candidates(self) -> list[str]:
        present = {row.name for row in self._rows}
        out = []
        for e in entries("FILTERS"):
            tag = "（已加入）" if e.name in present else ""
            out.append(f"{e.display}{tag}")
        return out

    # ------------------------------------------------------------------
    # 修改
    # ------------------------------------------------------------------
    def commit(self) -> None:
        """把当前链写回 store（真正的 list）。"""
        value = self.chain()
        self.store.update(self.key, value)
        if self._on_change is not None:
            self._on_change(value)
        self._refresh_add_values()

    def move(self, index: int, delta: int) -> None:
        """上移 / 下移一行；违反顺序约束时拒绝。"""
        target = index + delta
        if not (0 <= index < len(self._rows)) or \
                not (0 <= target < len(self._rows)):
            return
        moving = self._rows[index]
        other = self._rows[target]

        # 顺序约束：依赖"当前点云分布"的过滤器不能排在其它过滤器之前
        if delta < 0 and _depends_on_current(moving.name) \
                and not _depends_on_current(other.name):
            self._complain(
                f"“{moving.name}”依赖前序过滤后的点云分布，"
                f"不能排到“{other.name}”之前。\n\n"
                f"请先让 {other.name} 清理散点，否则台面高度与高度"
                f"直方图会被噪声干扰。")
            return

        items = self.chain()
        items[index], items[target] = items[target], items[index]
        self._write(items)

    def remove(self, index: int) -> None:
        if not (0 <= index < len(self._rows)):
            return
        items = self.chain()
        items.pop(index)
        self._write(items)

    def _on_add(self) -> None:
        text = self._add_var.get()
        if not text:
            return
        name = text.split(" — ")[0].strip()
        if "（已加入）" in text:
            return
        if name not in set(FILTERS.names()):
            return
        item: dict = {"name": name}
        if _depends_on_current(name):
            # 插到链尾：这类过滤器必须在清理类之后
            items = self.chain()
            items.append(item)
        else:
            items = self.chain()
            # 插到第一个"依赖当前点云"的过滤器之前
            pos = len(items)
            for i, existing in enumerate(items):
                if _depends_on_current(str(existing.get("name", ""))):
                    pos = i
                    break
            items.insert(pos, item)
        self._add_var.set("")
        self._write(items)

    def _write(self, items: list[dict]) -> None:
        self.store.update(self.key, items)
        if self._on_change is not None:
            self._on_change(items)
        self.rebuild()

    # ------------------------------------------------------------------
    def _complain(self, message: str) -> None:
        from tkinter import messagebox

        messagebox.showwarning("顺序不允许", message, parent=self)
