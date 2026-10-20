"""参数面板组件：按 ``ParamSpec`` / ``FieldSpec`` 自动生成控件。

设计原则
--------
**控件类型由声明决定，从不从值推断。** 这是本模块与旧
``panels.FieldTable`` 的根本区别：

.. code-block:: python

    # 旧：只有 choices / bool / 其他 三种，list 与 dict 退化成 Entry
    if fs.choices: Combobox
    elif fs.kind == "bool": Combobox
    else: Entry

    # 新：完整按 kind / choices / 范围 分派，结构型有专用编辑器
    bool            → Checkbutton
    choices         → Combobox(readonly)
    int + 范围      → Spinbox
    float + 范围    → Scale + Entry（双向联动）
    dict            → DictGroup（递归子分组）
    list            → 由调用方注册的编辑器（如过滤链编辑器）
    其他            → Entry

提交时机
--------
Horus 的滑块做法值得借鉴——**拖动中实时生效、松手才写配置**：

- ``Scale`` 拖动 → ``on_preview``（可选，用于实时调硬件）
- 松手 / 回车 / 失焦 → ``on_change(key, value)``

这样既能看到参数变化的实时效果，又不会在拖动过程中产生几十条
撤销记录或反复写配置。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from .field_spec import FieldSpec

__all__ = ["ParamGroup", "DictGroup", "SpecPanel", "param_to_field_spec",
           "make_editor", "bind_tooltip"]

#: 参数变更回调：``fn(key_path, value)``
ChangeFn = Callable[[str, Any], None]
#: 实时预览回调（拖动滑块时）：``fn(key_path, value)``
PreviewFn = Callable[[str, Any], None]

_ERROR_FG = "#c0392b"
_NORMAL_FG = "#1c1c1c"
_HINT_FG = "#7f8c8d"


def bind_tooltip(widget: tk.Misc, text: str) -> None:
    """给控件绑定悬停提示（延迟出现，避免遮挡）。"""
    if not text:
        return
    tip: dict[str, Optional[tk.Toplevel]] = {"win": None}

    def _show(_event) -> None:
        if tip["win"] is not None:
            return
        x = widget.winfo_rootx() + 12
        y = widget.winfo_rooty() + widget.winfo_height() + 6
        win = tk.Toplevel(widget)
        win.wm_overrideredirect(True)
        win.wm_geometry(f"+{x}+{y}")
        ttk.Label(win, text=text, background="#ffffe0",
                  relief="solid", borderwidth=1,
                  wraplength=360, justify="left",
                  padding=4).pack()
        tip["win"] = win

    def _hide(_event) -> None:
        win, tip["win"] = tip["win"], None
        if win is not None:
            win.destroy()

    widget.bind("<Enter>", _show, add="+")
    widget.bind("<Leave>", _hide, add="+")


# ---------------------------------------------------------------------------
def _tooltip_text(fs: FieldSpec) -> str:
    lines = [x for x in (fs.hint, f"配置项：{fs.key}") if x]
    if fs.unit:
        lines.append(f"单位：{fs.unit}")
    if fs.low is not None or fs.high is not None:
        lines.append(f"范围：{fs.low} ~ {fs.high}")
    if fs.choices:
        lines.append(f"可选：{'、'.join(str(c) for c in fs.choices)}")
    return "\n".join(lines)


def _increment(fs: FieldSpec) -> float:
    """根据取值范围推测合理的步进量。"""
    if fs.kind == "int":
        return 1
    if fs.low is None or fs.high is None:
        return 0.01
    span = float(fs.high) - float(fs.low)
    if span <= 0:
        return 0.01
    return max(span / 100.0, 1e-6)


# ---------------------------------------------------------------------------
class _Row:
    """一行的控件集合（用于统一设置错误态与读写值）。"""

    def __init__(self, fs: FieldSpec, widgets: list[tk.Misc],
                 get: Callable[[], Any],
                 set_: Callable[[Any], None],
                 error_label: Optional[ttk.Label] = None) -> None:
        self.fs = fs
        self.widgets = widgets
        self.get = get
        self.set = set_
        self.error_label = error_label


class ParamGroup(ttk.Frame):
    """一组参数控件（通常是某个配置分组或某个算法的参数）。

    Parameters
    ----------
    master : 父控件
    columns : 布局列数：0=标签 1=控件 2=错误提示
    on_change : 提交回调 ``fn(key_path, value)``
    on_preview : 实时回调（滑块拖动中）
    editor_for : ``fn(fs) -> widget | None``，为结构型配置提供专用编辑器。
        返回 ``None`` 时回退到只读提示。
    """

    def __init__(self, master: tk.Misc, *,
                 on_change: Optional[ChangeFn] = None,
                 on_preview: Optional[PreviewFn] = None,
                 editor_for: Optional[Callable[[FieldSpec],
                                               Optional[tk.Widget]]] = None,
                 **kw) -> None:
        super().__init__(master, **kw)
        self.columnconfigure(1, weight=1)
        self._on_change = on_change
        self._on_preview = on_preview
        self._editor_for = editor_for
        self._rows: dict[str, _Row] = {}
        self._index = 0
        self._specs: dict[str, FieldSpec] = {}
        self._values: dict[str, Any] = {}

    # ------------------------------------------------------------------
    @property
    def keys(self) -> list[str]:
        return list(self._rows)

    def row(self, key: str) -> Optional[_Row]:
        return self._rows.get(key)

    # ------------------------------------------------------------------
    def add_row(self, fs: FieldSpec, value: Any) -> None:
        """追加一行（标签 + 控件 + 错误提示）。"""
        if fs.key in self._rows:
            raise ValueError(f"重复的参数行: {fs.key!r}")
        self._specs[fs.key] = fs
        self._values[fs.key] = value

        row = self._index
        self._index += 1

        label = ttk.Label(self, text=fs.label, style="Key.TLabel")
        label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
        bind_tooltip(label, _tooltip_text(fs))

        holder = ttk.Frame(self)
        holder.grid(row=row, column=1, sticky="ew", pady=1)

        error = ttk.Label(self, text="", foreground=_ERROR_FG)
        error.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=1)

        spec_row = self._build_editor(holder, fs, value, error)
        self._rows[fs.key] = spec_row
        self._apply_style(spec_row, [])

        if fs.unit:
            ttk.Label(self, text=fs.unit, foreground=_HINT_FG).grid(
                row=row, column=3, sticky="w", padx=(4, 0), pady=1)

    # ------------------------------------------------------------------
    def _build_editor(self, holder: ttk.Frame, fs: FieldSpec, value: Any,
                      error: ttk.Label) -> _Row:  # noqa: C901 - 分派逻辑
        key = fs.key
        err = lambda msg: error.configure(text=msg)  # noqa: E731

        # 1) 专用编辑器优先（结构型配置、过滤链等）
        if self._editor_for is not None:
            widget = self._editor_for(fs)
            if widget is not None:
                return _Row(fs, [widget], lambda: self._values[key],
                            lambda v: self._values.__setitem__(key, v),
                            error)

        # 2) 结构型配置没有专用编辑器时，只读提示（绝不提供文本输入）
        if fs.is_structural:
            text = fs.to_text(value)
            lbl = ttk.Label(holder, text=text, foreground=_HINT_FG)
            lbl.pack(side="left")
            bind_tooltip(lbl, "该配置需由专用编辑器修改："
                              f"{fs.key}")
            return _Row(fs, [lbl], lambda: self._values[key],
                        lambda v: self._values.__setitem__(key, v), error)

        # 3) 布尔
        if fs.kind == "bool":
            var = tk.BooleanVar(value=bool(value))
            chk = ttk.Checkbutton(
                holder, variable=var,
                command=lambda k=key, v=var: self._commit_value(k, v.get()))
            chk.pack(side="left")
            return _Row(fs, [chk], lambda: var.get(),
                        lambda v: var.set(bool(v)), error)

        # 4) 候选值
        if fs.choices:
            var = tk.StringVar(value=fs.to_text(value))
            combo = ttk.Combobox(
                holder, textvariable=var, state="readonly", width=18,
                values=[str(c) for c in fs.choices])
            combo.pack(side="left")
            combo.bind("<<ComboboxSelected>>",
                       lambda _e, k=key, v=var: self._commit_text(k, v.get()))
            return _Row(fs, [combo], lambda: var.get(),
                        lambda v: var.set(fs.to_text(v)), error)

        # 5) 数值 + 范围 → 滑块（浮点）或 Spinbox（整数）
        if fs.is_numeric and fs.low is not None and fs.high is not None:
            if fs.kind == "float":
                return self._build_scale(holder, fs, value, error)
            return self._build_spinbox(holder, fs, value, error)

        # 6) 其他 → 文本输入
        var = tk.StringVar(value=fs.to_text(value))
        entry = ttk.Entry(holder, textvariable=var, width=20)
        entry.pack(side="left")
        entry.bind("<Return>",
                   lambda _e, k=key, v=var: self._commit_text(k, v.get()))
        entry.bind("<FocusOut>",
                   lambda _e, k=key, v=var: self._commit_text(k, v.get()))
        return _Row(fs, [entry], lambda: var.get(),
                    lambda v: var.set(fs.to_text(v)), error)

    def _build_spinbox(self, holder: ttk.Frame, fs: FieldSpec, value: Any,
                       error: ttk.Label) -> _Row:
        key = fs.key
        var = tk.StringVar(value=fs.to_text(value))
        spin = ttk.Spinbox(
            holder, textvariable=var, width=10,
            from_=float(fs.low), to=float(fs.high),
            increment=_increment(fs), command=lambda k=key, v=var:
            self._commit_text(k, v.get()))
        spin.pack(side="left")
        spin.bind("<Return>",
                  lambda _e, k=key, v=var: self._commit_text(k, v.get()))
        spin.bind("<FocusOut>",
                  lambda _e, k=key, v=var: self._commit_text(k, v.get()))
        return _Row(fs, [spin], lambda: var.get(),
                    lambda v: var.set(fs.to_text(v)), error)

    def _build_scale(self, holder: ttk.Frame, fs: FieldSpec, value: Any,
                     error: ttk.Label) -> _Row:
        """浮点滑块 + 数值框联动。

        拖动中调用 ``on_preview``（实时生效但不写配置），
        松手时调用 ``on_change``（写配置）。
        """
        key = fs.key
        initial = float(value if isinstance(value, (int, float))
                        else fs.low)

        var = tk.DoubleVar(value=initial)
        scale = ttk.Scale(holder, from_=float(fs.low), to=float(fs.high),
                          variable=var, orient="horizontal", length=150)
        scale.pack(side="left")

        text = tk.StringVar(value=fs.to_text(initial))
        entry = ttk.Entry(holder, textvariable=text, width=10)
        entry.pack(side="left", padx=(6, 0))

        def _on_track(_value=None) -> None:
            # 拖动：仅实时预览，不写配置
            if self._on_preview is not None:
                self._on_preview(key, round(float(var.get()), 6))
            text.set(fs.to_text(round(float(var.get()), 6)))

        def _on_release(_event=None) -> None:
            self._commit_value(key, round(float(var.get()), 6))

        def _on_entry(_event=None) -> None:
            raw = text.get()
            coerced = fs.coerce(raw)
            if not isinstance(coerced, (int, float)):
                self._show_error(key, [f"请输入数字（当前 {raw!r}）"])
                return
            var.set(float(coerced))
            self._commit_value(key, float(coerced))

        scale.bind("<B1-Motion>", _on_track)
        scale.bind("<ButtonRelease-1>", lambda _e: _on_release())
        entry.bind("<Return>", _on_entry)
        entry.bind("<FocusOut>", _on_entry)

        def _setter(v: Any) -> None:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return
            var.set(f)
            text.set(fs.to_text(f))

        return _Row(fs, [scale, entry], lambda: float(var.get()), _setter,
                    error)

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------
    def _commit_text(self, key: str, raw: str) -> None:
        fs = self._specs[key]
        value = fs.coerce(raw)
        self._commit_value(key, value)

    def _commit_value(self, key: str, value: Any) -> None:
        fs = self._specs[key]
        errors = fs.validate(value)
        self._show_error(key, errors)
        if errors:
            return
        self._values[key] = value
        row = self._rows.get(key)
        if row is not None:
            # 规范化显示（如 "007" → "7"），但不打断正在编辑的控件
            try:
                if not isinstance(row.widgets[0], (ttk.Scale,)):
                    row.set(value)
            except tk.TclError:             # pragma: no cover
                pass
        if self._on_change is not None:
            self._on_change(key, value)

    def _show_error(self, key: str, errors: list[str]) -> None:
        row = self._rows.get(key)
        if row is None or row.error_label is None:
            return
        row.error_label.configure(text=errors[0] if errors else "")
        self._apply_style(row, errors)

    def _apply_style(self, row: _Row, errors: list[str]) -> None:
        for widget in row.widgets:
            if isinstance(widget, (ttk.Entry, ttk.Spinbox, ttk.Combobox)):
                try:
                    widget.configure(
                        foreground=_ERROR_FG if errors else _NORMAL_FG)
                except tk.TclError:         # pragma: no cover
                    pass

    # ------------------------------------------------------------------
    # 外部同步
    # ------------------------------------------------------------------
    def set_value(self, key: str, value: Any) -> None:
        """外部配置变化 → 更新控件显示（不触发提交）。"""
        row = self._rows.get(key)
        if row is None:
            return
        self._values[key] = value
        try:
            row.set(value)
        except tk.TclError:                 # pragma: no cover
            return
        self._show_error(key, [])

    def values(self) -> dict[str, Any]:
        """当前各行持有的值（键为完整配置路径）。"""
        return {key: row.get() for key, row in self._rows.items()}

    def errors(self) -> dict[str, str]:
        out = {}
        for key, row in self._rows.items():
            if row.error_label is None:
                continue
            try:
                text = row.error_label.cget("text")
            except tk.TclError:             # pragma: no cover
                continue
            if text:
                out[key] = str(text)
        return out


class DictGroup(ttk.Frame):
    """结构型配置（``dict``）的递归子分组。

    把 ``reconstruction.axial`` 这样的字典展开成一组带前缀的子行，
    每个子键独立提交（``reconstruction.axial.auto_trim``），因此
    **不会覆盖整个字典**。

    Parameters
    ----------
    master : 父控件
    key_path : 字典本身的配置路径（如 ``"reconstruction.axial"``）
    value : 当前字典
    children : 子键的 ``FieldSpec`` 工厂；``fn(key_path) -> FieldSpec``
    **kw : 透传给 :class:`ParamGroup`
    """

    def __init__(self, master: tk.Misc, key_path: str, value: dict,
                 make_spec: Callable[[str, Any], FieldSpec],
                 **kw) -> None:
        super().__init__(master, **kw)
        self.key_path = key_path
        self.inner = ParamGroup(self, **kw)
        self.inner.pack(fill="x")
        self._items = value if isinstance(value, dict) else {}
        for sub_key, sub_value in self._items.items():
            full = f"{key_path}.{sub_key}"
            self.inner.add_row(make_spec(full, sub_value), sub_value)

    @property
    def keys(self) -> list[str]:
        return self.inner.keys

    def set_value(self, key: str, value: Any) -> None:
        self.inner.set_value(key, value)

    def errors(self) -> dict[str, str]:
        return self.inner.errors()


def param_to_field_spec(param, key_path: Optional[str] = None) -> FieldSpec:
    """把 ``ParamSpec`` 转成界面用的 ``FieldSpec``。

    ``ParamSpec.type`` / ``choices`` / ``low`` / ``high`` 是权威来源；
    ``kind`` 直接由类型映射，**绝不通过当前值推断**——否则值为 ``None``
    的可空参数会被误判成文本输入。

    Parameters
    ----------
    param : :class:`~triniscan.contracts.params.ParamSpec`
    key_path : 覆盖配置键（默认用 ``param.name``，即全局点号键）
    """
    kind = {
        bool: "bool",
        int: "int",
        float: "float",
        str: "str",
        dict: "dict",
        list: "list",
    }.get(getattr(param, "type", str), "str")

    key_path = key_path or getattr(param, "name", "")
    return FieldSpec(
        key=key_path,
        label=(getattr(param, "label", "") or
               key_path.rsplit(".", 1)[-1]),
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


class SpecPanel(ttk.Frame):
    """**固定算法**的参数表单（没有算法下拉框）。

    与 :class:`~triniscan.ui.algorithm_panel.AlgorithmPanel` 的分工：

    - ``AlgorithmPanel``：多个实现可选 → 下拉 + 参数
    - ``SpecPanel``：流水线里该阶段只有一个生效实现 → 只显示参数

    后者用于配准 / 网格 / 封闭：注册表里虽然可能登记了多个实现，
    但流水线当前只读取其中一个的参数，做成下拉框就是**假开关**。
    """

    def __init__(self, master: tk.Misc, spec, store, *,
                 on_change: Optional[ChangeFn] = None,
                 skip_keys: tuple[str, ...] = (),
                 note: str = "",
                 **kw) -> None:
        super().__init__(master, **kw)
        self.spec = spec
        self.store = store
        self._on_change = on_change
        self._skip = tuple(skip_keys)

        self.group = ParamGroup(self, on_change=self._commit)
        self.group.pack(fill="x")

        for param in getattr(spec, "params", ()) or ():
            if param.name in self._skip:
                continue
            self.group.add_row(param_to_field_spec(param),
                               self._cfg_get(param.name, param.default))

        if note:
            ttk.Label(self, text=note, foreground=_HINT_FG,
                      wraplength=290, justify="left").pack(
                anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    def _cfg_get(self, key: str, default: Any = None) -> Any:
        try:
            value = self.store.cfg.get(key, default)
        except Exception:                   # noqa: BLE001
            return default
        return default if value is None else value

    def _commit(self, key: str, value: Any) -> None:
        if self._on_change is not None:
            self._on_change(key, value)
        else:
            self.store.update(key, value)

    # ------------------------------------------------------------------
    def keys(self) -> list[str]:
        return self.group.keys

    def refresh(self) -> None:
        """从 store 重新读取所有值（不触发提交）。"""
        for param in getattr(self.spec, "params", ()) or ():
            if param.name in self._skip or param.name not in self.group.keys:
                continue
            self.group.set_value(param.name,
                                 self._cfg_get(param.name, param.default))


def make_editor(parent: tk.Misc, fs: FieldSpec) -> ttk.Widget:
    """为单个 ``FieldSpec`` 建一个独立控件（不含标签与错误位）。

    便携入口，供需要自定义布局的调用方（如过滤链编辑器）复用。
    """
    group = ParamGroup(parent)
    group.add_row(fs, None)
    row = group.row(fs.key)
    widget = row.widgets[0] if row else ttk.Label(parent, text="…")
    widget.pack_forget()
    return widget
