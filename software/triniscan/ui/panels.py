"""可复用的界面面板。

设计
----
每个面板是 ``ttk.Frame`` 子类，独立负责一块功能，尽量减少对主窗口的
耦合，便于二次开发与替换。

``ConfigPanel`` 与新旧两版的区别
--------------------------------
旧版**自己决定**每个参数该用什么控件、怎么解析、有没有候选值：

.. code-block:: python

    _FIELD_CHOICES = {"extraction.method": [...手写 6 个方法名...]}
    if key_path in _FIELD_CHOICES: ...
    elif isinstance(value, bool): ...
    def _coerce(...): ...

新版把这些全部交给 :mod:`triniscan.ui.field_spec`——参数表由**算法声明**
（``AlgorithmSpec.ParamSpec``）与配置树共同生成，候选值和取值范围都是
单一来源。面板只负责"摆控件"和"提示错误"：

.. code-block:: python

    fields = all_fields(cfg)        # 136 个 FieldSpec
    fs = fields["reconstruction.poisson_depth"]
    fs.kind, fs.low, fs.high        # 'int', 4, 14
    fs.coerce("9"), fs.validate(9)  # 9, []

行为变化
--------
- 越界值不再静默通过：提交时校验，错误项标红并在悬停提示里说明；
- 算法选择项（``extraction.method`` 等）的候选值来自注册表，
  新增实现无需改界面；
- ``set_value`` / ``refresh`` 与旧版签名兼容，调用方无需改动。
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from .field_spec import FieldSpec, all_fields, field_groups

#: 校验失败时的前景色
_ERROR_FG = "#c0392b"
_NORMAL_FG = ""


# ---------------------------------------------------------------------------
def _bind_tooltip(widget: tk.Misc, text: str,
                  delay_ms: int = 500) -> None:
    """给控件绑定一个简单的悬停提示（无第三方依赖）。"""
    if not text:
        return
    state: dict[str, Any] = {"tip": None, "after": None}

    def _show() -> None:
        if state["tip"] is not None:
            return
        try:
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 4
        except tk.TclError:  # pragma: no cover - 控件已销毁
            return
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(tip, text=text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1,
                 font=("", 9)).pack()
        state["tip"] = tip

    def _enter(_e: tk.Event) -> None:
        state["after"] = widget.after(delay_ms, _show)

    def _leave(_e: tk.Event) -> None:
        if state["after"] is not None:
            widget.after_cancel(state["after"])
            state["after"] = None
        if state["tip"] is not None:
            tip = state["tip"]
            state["tip"] = None
            try:
                tip.destroy()
            except tk.TclError:  # pragma: no cover
                pass

    widget.bind("<Enter>", _enter, add="+")
    widget.bind("<Leave>", _leave, add="+")
    widget.bind("<Destroy>", _leave, add="+")


# ---------------------------------------------------------------------------
class ScrollFrame(ttk.Frame):
    """带垂直滚动条的容器，内部放 ``self.body``。"""

    def __init__(self, master: tk.Misc, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical",
                                    command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body,
                                              anchor="nw")
        self.body.bind("<Configure>", self._on_body)
        self.canvas.bind("<Configure>", self._on_canvas)

    def _on_body(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._win, width=event.width)


# ---------------------------------------------------------------------------
class FieldTable(ttk.Frame):
    """参数表：分组 → 参数，值可编辑。

    与旧 ``ConfigPanel`` 的区别见模块 docstring；对外接口
    （``set_value`` / ``refresh`` / ``_vars`` / ``_commit``）保持兼容，
    因此主窗口与测试无需改动。

    Parameters
    ----------
    cfg : Config
    on_change : ``on_change(key_path, value)``，值"通过校验"后回调。
    on_error : ``on_error(key_path, messages)``，可选；默认只在界面上标红。
    """

    def __init__(self, master: tk.Misc, cfg,
                 on_change: Optional[Callable[[str, Any], None]] = None,
                 on_error: Optional[Callable[[str, list], None]] = None
                 ) -> None:
        super().__init__(master)
        self._cfg = cfg
        self._on_change = on_change
        self._on_error = on_error
        self._fields: dict[str, FieldSpec] = {}
        self._vars: dict[str, Any] = {}
        self._widgets: dict[str, ttk.Widget] = {}
        self._error_kv: dict[str, str] = {}
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._vars.clear()
        self._widgets.clear()
        self._error_kv.clear()

        self._fields = all_fields(self._cfg)
        scroller = ScrollFrame(self)
        scroller.pack(fill="both", expand=True)
        body = scroller.body

        row = 0
        for group, specs in field_groups(self._fields).items():
            ttk.Label(body, text=group, style="Group.TLabel").grid(
                row=row, column=0, columnspan=3, sticky="w",
                padx=6, pady=(8, 2))
            row += 1
            for fs in specs:
                row = self._add_row(body, row, fs)

    def _add_row(self, parent: tk.Misc, row: int, fs: FieldSpec) -> int:
        label = ttk.Label(parent, text=fs.label)
        label.grid(row=row, column=0, sticky="w", padx=(16, 6), pady=1)

        unit = f" [{fs.unit}]" if fs.unit else ""
        tip = "\n".join(x for x in (
            fs.hint,
            f"配置项：{fs.key}",
            f"类型：{fs.kind}{unit}",
            (f"范围：{fs.low} ~ {fs.high}"
             if fs.low is not None or fs.high is not None else ""),
            ("来源：算法声明" if fs.from_algorithm else ""),
        ) if x)
        _bind_tooltip(label, tip)

        current = self._cfg.get(fs.key)
        var = tk.StringVar(value=fs.to_text(current))

        if fs.is_structural:
            # 结构型配置（dict / list）**不提供文本控件**：任何单行
            # 文本表达都会在失焦时把结构覆盖成字符串，导致下游插件
            # 静默退回默认值（例：reconstruction.axial 被点一下就把
            # auto_trim 从 True 变成 False）。界面只读显示类型摘要，
            # 修改交给专用编辑器（子面板 / 过滤链编辑器）。
            widget: ttk.Widget = ttk.Label(
                parent, text=fs.to_text(current),
                foreground="#7f8c8d")
        elif fs.choices:
            widget = ttk.Combobox(
                parent, textvariable=var, width=16, state="readonly",
                values=[str(c) for c in fs.choices])
        elif fs.kind == "bool":
            widget = ttk.Combobox(parent, textvariable=var, width=16,
                                  state="readonly", values=("是", "否"))
        else:
            widget = ttk.Entry(parent, textvariable=var, width=18)

        widget.grid(row=row, column=1, sticky="w", padx=(0, 8), pady=1)
        if not fs.is_structural:
            widget.bind("<FocusOut>",
                        lambda _e, k=fs.key: self._commit(k))
            widget.bind("<Return>", lambda _e, k=fs.key: self._commit(k))
            if fs.choices or fs.kind == "bool":
                widget.bind("<<ComboboxSelected>>",
                            lambda _e, k=fs.key: self._commit(k))

        # 错误提示位
        hint = ttk.Label(parent, text="", foreground=_ERROR_FG)
        hint.grid(row=row, column=2, sticky="w", padx=(0, 6), pady=1)
        self._error_kv[fs.key] = hint

        self._vars[fs.key] = var
        self._widgets[fs.key] = widget
        if fs.unit:
            ttk.Label(parent, text=fs.unit,
                      foreground="#7f8c8d").grid(
                row=row, column=3, sticky="w", pady=1)
        return row + 1

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------
    def _commit(self, key_path: str) -> None:
        """把控件里的文本写回回调（经 :class:`FieldSpec` 转换与校验）。"""
        fs = self._fields.get(key_path)
        var = self._vars.get(key_path)
        if fs is None or var is None:
            return
        if fs.is_structural:
            # 防御：结构型配置不得从文本回写（控件本应是只读 Label，
            # 这里再挡一层，避免外部代码误调 set_value 后触发提交）。
            return
        value = fs.coerce(var.get())
        errors = fs.validate(value)
        self._mark_error(key_path, errors)
        if errors:
            if self._on_error is not None:
                try:
                    self._on_error(key_path, errors)
                except Exception:  # noqa: BLE001
                    pass
            return
        if self._on_change is not None:
            self._on_change(key_path, value)

    def _mark_error(self, key_path: str, errors: list) -> None:
        widget = self._widgets.get(key_path)
        hint = self._error_kv.get(key_path)
        if hint is not None:
            try:
                hint.configure(text=errors[0] if errors else "")
            except tk.TclError:  # pragma: no cover
                pass
        if widget is not None and isinstance(widget, (ttk.Entry,
                                                      ttk.Combobox)):
            try:
                widget.configure(foreground=_ERROR_FG if errors
                                 else _NORMAL_FG)
            except tk.TclError:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    # 外部同步
    # ------------------------------------------------------------------
    def set_value(self, key_path: str, value: Any) -> None:
        """只更新单个字段的显示（用于外部配置变化时同步）。"""
        var = self._vars.get(key_path)
        fs = self._fields.get(key_path)
        if var is None or fs is None:
            return
        var.set(fs.to_text(value))
        # 外部改动可能是合法值，清掉残留错误提示
        self._mark_error(key_path, [])

    def refresh(self, cfg) -> None:
        """用新配置刷新所有字段（字段集合变了则重建整表）。"""
        self._cfg = cfg
        new_fields = all_fields(cfg)
        if set(new_fields) != set(self._fields):
            self._build()
            return
        self._fields = new_fields
        for key_path in self._vars:
            self.set_value(key_path, cfg.get(key_path))

    # ------------------------------------------------------------------
    @property
    def fields(self) -> dict[str, FieldSpec]:
        """当前参数表（只读，便于测试与调试）。"""
        return dict(self._fields)

    def errors(self) -> dict[str, str]:
        """当前处于校验失败状态的项 ``{key: 首条错误}``。"""
        out = {}
        for key, hint in self._error_kv.items():
            try:
                text = hint.cget("text")
            except tk.TclError:  # pragma: no cover
                continue
            if text:
                out[key] = str(text)
        return out


# ---------------------------------------------------------------------------
class ConfigPanel(ttk.LabelFrame):
    """参数配置面板（``FieldTable`` 的带标题包装，保持旧接口）。"""

    def __init__(self, master: tk.Misc, cfg,
                 on_change: Optional[Callable[[str, Any], None]] = None,
                 on_error: Optional[Callable[[str, list], None]] = None
                 ) -> None:
        super().__init__(master, text="参数配置")
        self.table = FieldTable(self, cfg, on_change=on_change,
                                on_error=on_error)
        self.table.pack(fill="both", expand=True)

    # --- 转发旧接口 -----------------------------------------------
    @property
    def _vars(self) -> dict[str, Any]:
        return self.table._vars

    @property
    def _fields(self) -> dict[str, FieldSpec]:
        return self.table._fields

    def _commit(self, key_path: str) -> None:
        self.table._commit(key_path)

    def set_value(self, key_path: str, value: Any) -> None:
        self.table.set_value(key_path, value)

    def refresh(self, cfg) -> None:
        self.table.refresh(cfg)

    def errors(self) -> dict[str, str]:
        return self.table.errors()


# ---------------------------------------------------------------------------
class LogPanel(ttk.LabelFrame):
    """日志输出面板（只读文本 + 清空/保存按钮）。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="运行日志")
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="清空", command=self.clear).pack(side="left")
        ttk.Button(bar, text="保存…", command=self._save).pack(side="left",
                                                              padx=4)

        self.text = tk.Text(self, height=12, wrap="none",
                            state="disabled", font=("Consolas", 9))
        yscroll = ttk.Scrollbar(self, orient="vertical",
                                command=self.text.yview)
        self.text.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

    def append(self, message: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", message.rstrip() + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def _save(self) -> None:
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("日志文件", "*.log"), ("所有文件", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text.get("1.0", "end"))


# ---------------------------------------------------------------------------
class ResultPanel(ttk.LabelFrame):
    """结果展示面板：key–value 指标 + **过滤诊断表**。

    过滤诊断为什么必须显示
    ----------------------
    "体积偏小 / 少了某个面"这类问题，根因几乎总在某层过滤器——最典型
    的是 ``cluster_keep_top=1`` 把断开的第二个簇整块删掉。重构前这一步
    完全静默：用户只看到最终体积，看不到中间删了什么。

    这里把 :class:`~triniscan.contracts.report.MaskStack` 的每层
    "删了多少 / 为什么 / 是否依赖顺序"直接列出来，删除率 >=15% 标红。
    """

    #: 删除率超过此值即标红提醒（与 :meth:`MaskStack.summary` 一致）
    WARN_RATIO = 0.15

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="测量结果")
        self._labels: dict[str, ttk.Label] = {}
        self._grid = ttk.Frame(self)
        self._grid.pack(fill="x", padx=6, pady=4)

        self._filter_box = ttk.LabelFrame(self, text="过滤诊断")
        self._filter_box.pack(fill="both", expand=True, padx=6,
                              pady=(4, 6))
        self._filter_tree = ttk.Treeview(
            self._filter_box,
            columns=("removed", "ratio", "order", "reason"),
            show="tree headings", height=6)
        self._filter_tree.heading("#0", text="层")
        self._filter_tree.heading("removed", text="删除点数")
        self._filter_tree.heading("ratio", text="删除率")
        self._filter_tree.heading("order", text="顺序敏感")
        self._filter_tree.heading("reason", text="原因")
        self._filter_tree.column("#0", width=96, anchor="w")
        self._filter_tree.column("removed", width=84, anchor="e")
        self._filter_tree.column("ratio", width=64, anchor="e")
        self._filter_tree.column("order", width=68, anchor="center")
        self._filter_tree.column("reason", width=380, anchor="w")
        bar = ttk.Scrollbar(self._filter_box, orient="vertical",
                            command=self._filter_tree.yview)
        self._filter_tree.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self._filter_tree.pack(side="left", fill="both", expand=True)

        self._filter_tree.tag_configure("warn", foreground="#c0392b")
        self._filter_tree.tag_configure("skip", foreground="#7f8c8d")

    # ------------------------------------------------------------------
    def set_value(self, key: str, value: str) -> None:
        if key not in self._labels:
            r = len(self._labels)
            ttk.Label(self._grid, text=f"{key}:", style="Key.TLabel").grid(
                row=r, column=0, sticky="w", padx=(4, 8), pady=1)
            lbl = ttk.Label(self._grid, text="", style="Value.TLabel")
            lbl.grid(row=r, column=1, sticky="w", pady=1)
            self._labels[key] = lbl
        self._labels[key].configure(text=value)

    def clear(self) -> None:
        for lbl in self._labels.values():
            lbl.configure(text="—")
        self.set_filter_layers([])

    # ------------------------------------------------------------------
    def update_result(self, result) -> None:
        """用 ``ScanResult`` 刷新。"""
        self.set_value("体积", f"{result.volume_cm3:.2f} cm³")
        if result.cross_check_cm3 is not None:
            rel = abs(result.cross_check_cm3 - result.volume_cm3) \
                / max(result.volume_cm3, 1e-9) * 100.0
            self.set_value("体素校核",
                           f"{result.cross_check_cm3:.2f} cm³ "
                           f"(差 {rel:.1f}%)")
        self.set_value("耗时", f"{result.elapsed_s:.1f} s")
        self.set_value("有效视角", str(result.num_views))
        self.set_value("点数", f"{result.num_points}")
        self.set_value("网格水密", "是" if result.mesh_watertight else "否")

        before = getattr(result, "points_before_filter", 0) or 0
        after = getattr(result, "points_after_filter", 0) or 0
        if before:
            self.set_value(
                "过滤后点数",
                f"{after} / {before} ({after / before * 100:.1f}%)")

        if result.mesh_path:
            self.set_value("网格文件", os.path.basename(result.mesh_path))

        fingerprint = getattr(result, "calib_fingerprint", "")
        if fingerprint:
            self.set_value("标定指纹", fingerprint)

        # 封闭降级：体积已不可全信的显式提示
        if getattr(result, "close_degraded", False):
            self.set_value(
                "⚠️ 网格降级",
                getattr(result, "close_degrade_reason", "") or
                f"封闭兜底 {result.close_fallback}")
        retention = getattr(result, "mesh_face_retention", 1.0)
        if retention < 0.5:
            self.set_value("面数保留", f"{retention * 100:.1f}%")

        self.set_filter_layers(getattr(result, "filter_layers", []) or [])

    # ------------------------------------------------------------------
    def set_filter_layers(self, layers: list[dict]) -> None:
        """填充过滤诊断表。

        Parameters
        ----------
        layers : :meth:`MaskStack.to_dict` 返回的 ``layers`` 列表
        """
        tree = self._filter_tree
        tree.delete(*tree.get_children())
        for layer in layers:
            removed = int(layer.get("removed", 0))
            ratio = float(layer.get("removed_ratio", 0.0))
            applied = bool(layer.get("applied", True))
            tags = []
            if not applied:
                tags.append("skip")
            elif ratio >= self.WARN_RATIO:
                tags.append("warn")
            tree.insert(
                "", "end",
                text=str(layer.get("name", "?")),
                values=(
                    "—" if not applied else f"{removed:,}",
                    "—" if not applied else f"{ratio * 100:.1f}%",
                    "是" if layer.get("depends_on_previous") else "否",
                    str(layer.get("reason", ""))[:200],
                ),
                tags=tuple(tags),
            )
