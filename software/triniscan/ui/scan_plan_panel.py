"""扫描计划面板：视角数 / 每视角转角 / 整圈接缝诊断。

为什么这个面板必需
------------------
``scan.angle_step_deg`` 是**采集与拼接共用的那个角度**：

.. code-block:: text

    acquire()            plan = resolve_scan_plan(cfg)   → 电机按 plan 走
                         apply_scan_plan(cfg, plan)      → 写回 angle_step_deg
    reconstruct_views()  按视角的真实转角拼接各视角点云
    从图像重建            angle_deg = angle_step_deg × (i + 1)

它一旦与实际转角不符，各视角点云就会沿圆周错位（点云"炸开"），之后
无论怎么调泊松深度或过滤链都救不回来。所以它必须**可见**。

配置键的三种身份（本模块严格区分）
----------------------------------
=========================  ==========================  ==========
键                          身份                        可编辑
=========================  ==========================  ==========
``scan.preset``            **输入**：视角数预设          是
``scan.num_views``         **输入**：无预设时的视角数    是
``scan.timeout_s``         **输入**：单件测量时限        是
``scan.steps_per_view``    **产物**：apply_scan_plan 写  否
``scan.angle_step_deg``    **产物**：apply_scan_plan 写  否
=========================  ==========================  ==========

产物键在这里**只读**。它们由 :func:`~triniscan.hardware.turntable.
resolve_scan_plan` 计算、由 :func:`~triniscan.hardware.turntable.
apply_scan_plan` 在采集时回写。若允许手填 1.0° 而电机走 0.4995°，
就又是一个"配置不生效"的假开关。

``preset`` / ``num_views`` 的覆盖关系
-------------------------------------
:func:`resolve_scan_plan` 的优先级是 **preset 优先**：

.. code-block:: python

    if preset not in (None, "", 0, "0"):
        num_views = int(preset)          # 忽略 num_views
    else:
        num_views = int(get("num_views", 360))

因此本面板在用户选择预设值时**同时写两个键**，在自定义时**把 preset
置空**。两个键永远互不矛盾——否则就会出现"界面改了 num_views 却毫无
效果"这类隐蔽缺陷（重构前真实存在）。

整圈接缝
--------
步进电机只能走整数微步。整圈微步数 ``S`` 若不能被视角数整除，
"每视角等分"与"走满整圈"无法同时成立，拼接会产生接缝。本面板显示的
``total_deg`` 与偏差正是 :class:`ScanPlan` 的真实计算结果——与采集时
用的是同一个函数，**不是估算**。
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Any, Optional

from ..common.logging_utils import get_logger
from ..hardware.turntable import (TurntableConfig, nearest_exact_view_count,
                                  resolve_scan_plan)
from .params import ParamGroup, bind_tooltip

log = get_logger(__name__)

__all__ = ["ScanPanel", "ScanDiagnosis", "SCAN_PRESETS",
           "MIN_VIEWS", "MAX_VIEWS"]

#: 视角数预设。
#:
#: 与 :data:`triniscan.ui.workbench_hardware.PRESETS` **语义不同**：那里
#: 是"手动分度扫描走几站"（6/8/12/…，站与站之间不抓图）；这里是"一次
#: 测量分多少视角抓图"（每视角都抓）。二者不能合并。
#:
#: 592 的依据：当前转台整圈 33152 微步，``33152 / 592 = 56`` 整除，
#: 因此 592 视角恰好走满 360°（拼接无接缝）；720 则差 −0.3475°。
SCAN_PRESETS: tuple[int, ...] = (360, 592, 720, 1080)

#: 自定义视角数的范围（下限 2 才能围成一圈）
MIN_VIEWS = 2
MAX_VIEWS = 3600

_HINT_FG = "#7f8c8d"
_WARN_FG = "#e67e22"
_OK_FG = "#27ae60"
_ERROR_FG = "#c0392b"
_LEVEL_COLORS = {"ok": _OK_FG, "warn": _WARN_FG, "error": _ERROR_FG}


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanDiagnosis:
    """一次扫描计划的计算结果与可读诊断（**纯数据，不依赖 Tk**）。

    角度计算全部委托给 :func:`resolve_scan_plan`——与
    ``ScanPipeline.acquire`` 调用的是同一个函数，所以面板显示的数值与
    电机实际转动必然一致。
    """

    num_views: int
    steps_per_view: int
    angle_step_deg: float
    total_deg: float
    full_turn_error_deg: float
    exact: bool
    suggestion: Optional[int] = None
    error: str = ""

    # ------------------------------------------------------------------
    @classmethod
    def compute(cls, scan_cfg: Any,
                table_cfg: Optional[TurntableConfig] = None
                ) -> "ScanDiagnosis":
        """由 ``scan`` 段配置计算诊断。

        Parameters
        ----------
        scan_cfg : ``cfg.raw["scan"]``（dict）或整个 Config
        table_cfg : 转台参数。调用方应传入**按当前配置**构造的实例，
            否则每视角微步数会与实际电机不符。
        """
        if table_cfg is None:
            table_cfg = TurntableConfig()
        try:
            plan = resolve_scan_plan(scan_cfg, table_cfg)
        except Exception as exc:            # noqa: BLE001 - 非法配置不应炸 UI
            return cls(num_views=0, steps_per_view=0, angle_step_deg=0.0,
                       total_deg=0.0, full_turn_error_deg=0.0, exact=False,
                       error=str(exc))

        suggestion = None
        if not plan.full_turn_exact:
            # ±400 足以覆盖到 720 附近的可整除视角数（最近的是 592/896）
            suggestion = nearest_exact_view_count(
                plan.num_views, table_cfg.steps_per_degree, max_delta=400)

        return cls(
            num_views=plan.num_views,
            steps_per_view=plan.steps_per_step,
            angle_step_deg=plan.angle_step_deg,
            total_deg=plan.total_deg,
            full_turn_error_deg=plan.full_turn_error_deg,
            exact=plan.full_turn_exact,
            suggestion=suggestion,
        )

    # ------------------------------------------------------------------
    @property
    def resolution_text(self) -> str:
        """每视角转角与整步数。"""
        if self.error:
            return "—"
        return f"{self.angle_step_deg:.6f}°  ≈ {self.steps_per_view} 微步"

    @property
    def turn_text(self) -> str:
        """整圈累计角度与接缝偏差。"""
        if self.error:
            return "—"
        if self.exact:
            return f"{self.total_deg:.4f}°  ✓ 精确整圈"
        return (f"{self.total_deg:.4f}°  ⚠ 差 "
                f"{self.full_turn_error_deg:+.4f}°")

    @property
    def turn_level(self) -> str:
        """接缝诊断级别：``error`` / ``warn`` / ``ok``。"""
        if self.error:
            return "error"
        return "ok" if self.exact else "warn"

    @property
    def suggestion_text(self) -> str:
        """说明文字：接缝偏差的成因与可整除视角数建议。"""
        if self.error:
            return f"无法计算扫描计划：{self.error}"
        if self.exact:
            return "视角数整除整圈微步数，拼接无接缝"
        if self.suggestion is None:
            return (f"{self.num_views} 视角无法整除整圈微步数，"
                    f"附近没有更合适的视角数")
        return (f"{self.num_views} 视角无法整除整圈微步数；"
                f"改用 {self.suggestion} 视角可精确整圈")


# ---------------------------------------------------------------------------
class ScanPanel(ttk.Frame):
    """测量台的「扫描计划」面板。"""

    title = "扫描计划"

    def __init__(self, master: tk.Misc, workbench) -> None:
        super().__init__(master)
        self.workbench = workbench
        self._syncing = False           # 抑制"写 store → 回调 → 再写"回环

        # ---- 视角数 ----
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="视角数", style="Key.TLabel").pack(side="left")
        self._mode = tk.StringVar()
        self._mode_combo = ttk.Combobox(
            row, textvariable=self._mode, state="readonly", width=9,
            values=[str(n) for n in SCAN_PRESETS] + ["自定义"])
        self._mode_combo.pack(side="left", padx=(8, 4))
        self._mode_combo.bind("<<ComboboxSelected>>", self._on_mode)

        self._custom = tk.StringVar()
        self._custom_spin = ttk.Spinbox(
            row, from_=MIN_VIEWS, to=MAX_VIEWS, increment=1, width=8,
            textvariable=self._custom)
        self._custom_spin.bind("<Return>", lambda _e: self._commit_custom())
        self._custom_spin.bind("<FocusOut>",
                               lambda _e: self._commit_custom())
        bind_tooltip(self._custom_spin,
                     f"自定义视角数（{MIN_VIEWS}~{MAX_VIEWS}）。"
                     "修改后会自动清空预设，保证配置自洽。")

        # ---- 派生（只读）----
        self._lbl_res = self._add_info_row("每视角")
        self._lbl_turn = self._add_info_row("整圈")

        self._lbl_advice = ttk.Label(self, text="", foreground=_WARN_FG,
                                     wraplength=300, justify="left")
        self._lbl_advice.pack(anchor="w", pady=(4, 0))

        # ---- 单件时限（输入）----
        self._timeout = ParamGroup(
            self, on_change=lambda k, v: self.workbench.store.update(k, v))
        self._timeout.pack(fill="x", pady=(8, 0))
        fs = self.workbench.field_spec("scan.timeout_s")
        if fs is not None:
            self._timeout.add_row(fs, self.cfg.get("scan.timeout_s"))

        # ---- 转台摘要（只读，指向硬件台）----
        self._lbl_table = ttk.Label(self, text="", foreground=_HINT_FG,
                                    wraplength=300, justify="left")
        self._lbl_table.pack(anchor="w", pady=(8, 0))

        self.refresh()
        self.workbench.store.add_listener(self._on_store_changed)

    # ------------------------------------------------------------------
    @property
    def cfg(self):
        return self.workbench.cfg

    def _add_info_row(self, title: str) -> ttk.Label:
        row = ttk.Frame(self)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=title, foreground=_HINT_FG).pack(side="left")
        label = ttk.Label(row, text="—", anchor="e")
        label.pack(side="right")
        return label

    # ------------------------------------------------------------------
    # 写配置
    # ------------------------------------------------------------------
    def _on_mode(self, _event: tk.Event) -> None:
        if self._syncing:
            return
        mode = self._mode.get()
        if mode == "自定义":
            self._set_custom_visible(True)
            self._commit_custom()
            return

        try:
            n = int(mode)
        except ValueError:                  # pragma: no cover - 只读下拉框
            return
        # 两个键同时写：resolve_scan_plan 里 preset 优先，只写其一会让
        # 另一个键"看起来改了但不生效"。
        self.workbench.store.update_many({
            "scan.preset": n,
            "scan.num_views": n,
        })
        self._set_custom_visible(False)
        self._refresh_derived()

    def _commit_custom(self) -> None:
        if self._syncing:
            return
        text = self._custom.get().strip()
        try:
            n = int(text)
        except ValueError:
            self._show_error(
                f"视角数必须是 {MIN_VIEWS}~{MAX_VIEWS} 的整数")
            return
        if not MIN_VIEWS <= n <= MAX_VIEWS:
            self._show_error(f"视角数超出范围 {MIN_VIEWS}~{MAX_VIEWS}")
            return

        # 自定义必须清空 preset，否则 resolve_scan_plan 仍按 preset 走。
        # 写 None（YAML 中为 `preset: null`）而非删键：Config 无删键接口，
        # 且 None 正是 resolve_scan_plan 识别的"未设置"。
        self.workbench.store.update_many({
            "scan.preset": None,
            "scan.num_views": n,
        })
        self._refresh_derived()

    def _show_error(self, text: str) -> None:
        self._lbl_advice.configure(text=text, foreground=_ERROR_FG)

    # ------------------------------------------------------------------
    # 读配置
    # ------------------------------------------------------------------
    def _on_store_changed(self, _store, key_path: str, _value: Any) -> None:
        if key_path in ("scan.preset", "scan.num_views",
                        "turntable.steps_per_rev", "turntable.microstep",
                        "turntable.gear_ratio"):
            self.refresh()

    def refresh(self) -> None:
        """从配置重新同步面板（不写回）。"""
        self._syncing = True
        try:
            preset = _as_int(self.cfg.get("scan.preset"))
            num_views = _as_int(self.cfg.get("scan.num_views")) or 360

            if preset in SCAN_PRESETS:
                self._mode.set(str(preset))
                self._set_custom_visible(False)
            else:
                self._mode.set("自定义")
                self._custom.set(str(num_views))
                self._set_custom_visible(True)

            for key in self._timeout.keys:
                self._timeout.set_value(key, self.cfg.get(key))
        finally:
            self._syncing = False
        self._refresh_derived()

    def _set_custom_visible(self, visible: bool) -> None:
        if visible:
            if not self._custom_spin.winfo_manager():
                self._custom_spin.pack(side="left")
        else:
            self._custom_spin.pack_forget()

    # ------------------------------------------------------------------
    def _refresh_derived(self) -> None:
        """重算并显示派生信息（只读）。"""
        diag = ScanDiagnosis.compute(self.cfg.raw.get("scan", {}) or {},
                                     self._table_config())
        self._lbl_res.configure(text=diag.resolution_text)
        self._lbl_turn.configure(text=diag.turn_text,
                                 foreground=_LEVEL_COLORS[diag.turn_level])
        color = _ERROR_FG if diag.error else (
            _OK_FG if diag.exact else _WARN_FG)
        self._lbl_advice.configure(text=diag.suggestion_text, foreground=color)
        self._lbl_table.configure(text=self._table_summary())

    def _table_config(self) -> TurntableConfig:
        """按当前配置构造转台参数（与实际驱动同源）。"""
        return TurntableConfig.from_config(
            self.cfg.raw.get("turntable", {}) or {})

    def _table_summary(self) -> str:
        """转台摘要：让"机械参数填错"在测量台就能看出来。

        例如减速比写成 5.0 而实际是 5.18，整圈微步数就差了 1152 步，
        所有角度按比例偏移——这类错误只看结果体积是发现不了的。
        """
        try:
            tc = self._table_config()
        except Exception as exc:            # noqa: BLE001
            return f"转台参数无效：{exc}"
        total = tc.steps_per_rev * tc.microstep * tc.gear_ratio
        return (f"转台 {tc.port}：{tc.steps_per_rev}×{tc.microstep} 微步 × "
                f"减速比 {tc.gear_ratio:g} = {total:g} 微步/圈\n"
                f"电机参数在「硬件」工作台修改")


def _as_int(value: Any) -> Optional[int]:
    """宽松转整数：``None`` / ``""`` / ``"0"`` 等空值返回 ``None``。"""
    if value in (None, "", 0, "0"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
