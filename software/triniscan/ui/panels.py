"""可复用的界面面板。

每个面板是 ``ttk.Frame`` 子类，独立负责一块功能，
尽量减少对主窗口的耦合，便于二次开发与替换。
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional


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
class ConfigPanel(ttk.LabelFrame):
    """配置编辑面板。

    以"分组 → 参数"两级表格呈现，参数值可编辑。
    支持字符串、整数、浮点、布尔（下拉）、null（留空）。

    Parameters
    ----------
    on_change : 值变化时的回调 ``on_change(key_path, value)``。
    """

    def __init__(self, master: tk.Misc, cfg,
                 on_change: Optional[Callable[[str, Any], None]] = None
                 ) -> None:
        super().__init__(master, text="参数配置")
        self._cfg = cfg
        self._on_change = on_change
        self._vars: dict[str, Any] = {}
        self._types: dict[str, type] = {}

        scroller = ScrollFrame(self)
        scroller.pack(fill="both", expand=True)
        body = scroller.body

        row = 0
        for group, params in cfg.raw.items():
            if group.startswith("_") or not isinstance(params, dict):
                continue
            ttk.Label(body, text=group, style="Group.TLabel").grid(
                row=row, column=0, columnspan=2, sticky="w",
                padx=6, pady=(8, 2))
            row += 1
            for key, value in params.items():
                key_path = f"{group}.{key}"
                self._add_field(body, row, key_path, value)
                row += 1

    # ------------------------------------------------------------------
    def _add_field(self, parent: tk.Misc, row: int, key_path: str,
                   value: Any) -> None:
        short = key_path.split(".", 1)[1]
        ttk.Label(parent, text=short).grid(
            row=row, column=0, sticky="w", padx=(16, 6), pady=1)

        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            widget = ttk.Combobox(parent, textvariable=var,
                                  values=("true", "false"), width=14,
                                  state="readonly")
            var.set("true" if value else "false")
            self._types[key_path] = bool
        elif value is None:
            var = tk.StringVar(value="")
            widget = ttk.Entry(parent, textvariable=var, width=16)
            self._types[key_path] = type(None)
        else:
            var = tk.StringVar(value=str(value))
            widget = ttk.Entry(parent, textvariable=var, width=16)
            self._types[key_path] = type(value)

        widget.grid(row=row, column=1, sticky="w", padx=(0, 8), pady=1)
        widget.bind("<FocusOut>", lambda _e, k=key_path: self._commit(k))
        widget.bind("<Return>", lambda _e, k=key_path: self._commit(k))
        self._vars[key_path] = var

    def _commit(self, key_path: str) -> None:
        if self._on_change is None:
            return
        raw = self._vars[key_path].get()
        coerced = self._coerce(key_path, raw)
        self._on_change(key_path, coerced)

    def _coerce(self, key_path: str, raw: str) -> Any:
        typ = self._types.get(key_path, str)
        text = raw.strip()
        try:
            if typ is type(None):
                return None if text == "" else text
            if typ is bool:
                return text.lower() in ("1", "true", "yes", "on")
            if typ is int:
                return int(float(text))
            if typ is float:
                return float(text)
        except ValueError:
            return raw
        return raw

    def refresh(self, cfg) -> None:
        """用新配置刷新所有字段。"""
        self._cfg = cfg
        for key_path, var in self._vars.items():
            value = cfg.get(key_path)
            if isinstance(value, bool):
                var.set("true" if value else "false")
            elif value is None:
                var.set("")
            else:
                var.set(str(value))


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
    """结果展示面板：一行行 key–value，另可嵌入图像/点云预览。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="测量结果")
        self._labels: dict[str, ttk.Label] = {}
        self._grid = ttk.Frame(self)
        self._grid.pack(fill="x", padx=6, pady=4)

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
        if result.mesh_path:
            self.set_value("网格文件", os.path.basename(result.mesh_path))
