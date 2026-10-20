"""硬件控制台工作台：转台连接 / 角度控制 / 分度扫描。

设计要点
--------
1. **串口操作必须离开 UI 线程**。``serial`` 读写在无响应时会阻塞到
   ``timeout_s``（默认 5 s），直接在按钮回调里调用会让界面假死。这里
   复用 :class:`~triniscan.ui.worker.Worker`（后台线程 + 队列 + ``after``
   轮询）。
2. **控件状态由连接状态驱动**。未连接时动作按钮 disabled，避免用户
   反复点击却毫无反应。急停例外：连接后**始终可用**。
3. **误差要可见**。转台只能执行整数微步，输入 37.3° 的实际到位角度
   与理论值有偏差。分度扫描会显式打印"指令步数 → 实际角度"。
4. **单任务互斥**。转台是单线程设备，上一个动作未完成时忽略新指令
   （急停除外），否则串口指令会交错。
"""
from __future__ import annotations

import time
from queue import Empty, Queue
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional

import tkinter as tk

from ..common.logging_utils import get_logger
from ..hardware.turntable import Turntable, TurntableConfig
from .params import ParamGroup
from .workbench import Workbench
from .worker import TaskMessage, Worker

log = get_logger(__name__)

__all__ = ["HardwareWorkbench"]

#: 分度扫描的常用站数
PRESETS = (6, 8, 12, 16, 24, 36, 72)
#: 单次分度扫描的硬上限（防止误输入把转台转到天亮）
MAX_STEPS_PER_SCAN = 360

#: 连接面板展示的转台键
TURNTABLE_KEYS = (
    "turntable.port", "turntable.baudrate", "turntable.steps_per_rev",
    "turntable.microstep", "turntable.gear_ratio", "turntable.step_us",
    "turntable.settle_ms", "turntable.timeout_s",
)


class HardwareWorkbench(Workbench):
    """硬件工作台。"""

    title = "硬件"
    panel_width = 360

    def __init__(self, master: tk.Misc, store, **kw) -> None:
        self.turntable: Optional[Turntable] = None
        self._queue: Queue = Queue()
        self._worker: Optional[Worker] = None
        self._busy = False
        self._connected = False
        self._on_done: Callable[[Any], None] = lambda _r: None
        self._on_error: Callable[[str], None] = lambda _m: None
        super().__init__(master, store, **kw)

    # ------------------------------------------------------------------
    # 声明
    # ------------------------------------------------------------------
    def add_panels(self) -> None:
        self.add_panel("conn", _ConnectionPanel, expanded=True)
        self.add_panel("motion", _MotionPanel, expanded=True)
        self.add_panel("scan", _ScanPanel, expanded=False)

    def add_pages(self) -> None:
        self.add_page("status", _StatusPage(self.view_parent))

    # ------------------------------------------------------------------
    def on_show(self) -> None:
        self.refresh()

    def on_hide(self) -> None:
        """离开工作台时不自动断开——用户可能只是去改参数。"""
        self.refresh()

    # ------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def busy(self) -> bool:
        return self._busy

    def turntable_config(self) -> TurntableConfig:
        """按当前配置构造转台参数（不连接）。"""
        return TurntableConfig.from_config(
            self.cfg.raw.get("turntable", {}) or {})

    def field_spec(self, key: str):
        from .field_spec import all_fields
        try:
            return all_fields(self.cfg).get(key)
        except Exception:                   # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # 连接 / 断开
    # ------------------------------------------------------------------
    def connect_turntable(self) -> None:
        if self._connected or self._busy:
            return
        cfg = self.turntable_config()
        self._log(f"正在连接转台 {cfg.port} @ {cfg.baudrate} …")

        def _job(_report) -> Turntable:
            tt = Turntable(cfg)
            tt.connect()
            return tt

        def _ok(tt: Turntable) -> None:
            self.turntable = tt
            self._connected = True
            self._log(f"✓ 已连接 {tt.cfg.port}")
            self.check_firmware()

        def _err(msg: str) -> None:
            self.turntable = None
            self._connected = False
            self._log(f"✗ 连接失败: {msg}")

        self._submit(_job, _ok, _err)

    def disconnect(self) -> None:
        tt = self.turntable
        if tt is None:
            self._connected = False
            self.refresh()
            return

        def _job(_report) -> None:
            try:
                tt.close()
            except Exception:               # noqa: BLE001
                log.debug("关闭转台失败", exc_info=True)

        def _ok(_r) -> None:
            self.turntable = None
            self._connected = False
            self._log("转台已断开")

        self._submit(_job, _ok, lambda m: self._log(f"✗ 断开异常: {m}"))

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------
    def check_firmware(self) -> None:
        """读一次 ``STATUS``，把固件的步数系数与配置比对。

        固件在 ``motor_position_degrees()`` 里用**编译期**的
        ``STEPS_PER_REV`` 换算角度；它与 ``turntable.*`` 配置不一致时，
        只要走 ``STEPS`` 路径，物理角度仍是对的（角度由配置换算），
        但走 ``ROT`` 路径就会偏。这里把这个差异显式报出来。
        """
        tt = self.turntable
        if tt is None:
            return

        def _job(_report):
            return tt.read_status()

        def _ok(status) -> None:
            panel = self.panel("conn")
            if status is None:
                panel.set_firmware_status("固件未返回 STATUS（不支持或通信异常）")
                self._log("⚠ 固件未返回 STATUS")
                return
            fw = status.firmware_steps_per_rev
            cfg_total = tt.cfg.steps_per_rev_total
            text = (f"固件：{status.state}  步数 {status.steps}  "
                    f"角度 {status.deg_reported:.2f}°")
            if fw:
                diff = (cfg_total / fw - 1.0) * 100.0
                text += chr(10) + f"固件每圈 {fw:.0f} 微步 / 配置 {cfg_total:.0f}"
                if abs(diff) > 0.1:
                    text += f"  ⚠ 差 {diff:+.2f}%"
            panel.set_firmware_status(text)
            self._log(f"固件状态：{status.state}, {status.steps} 步, "
                      f"{status.deg_reported:.2f}°")

        self._submit(_job, _ok, lambda m: self._log(f"⚠ 读取状态失败: {m}"))

    def move_relative(self, angle_deg: float) -> None:
        """相对转动。

        走 ``STEPS`` 而非 ``ROT``：固件的 ``ROT`` 用整数
        ``STEPS_PER_DEG_X1000`` 换算，每度少约 1 微步；``STEPS`` 由上位机
        量化到整数微步，与扫描流水线（``rotate_steps``）完全一致。
        """
        step_count = int(round(self.turntable_config()
                               .steps_for_angle(angle_deg)))             if self.turntable is not None else 0
        self._guard(lambda tt: tt.rotate_steps(step_count),
                    f"相对转动 {angle_deg:+.2f}°（{step_count} 微步）")

    def move_to(self, angle_deg: float) -> None:
        """转到累计角度（相对归零）。"""
        if self.turntable is None:
            self._guard(None, "")
            return
        delta = float(angle_deg) - self.turntable.cumulative_deg
        if abs(delta) < 1e-9:
            self._log(f"已在 {angle_deg:.2f}°，无需转动")
            return
        self.move_relative(delta)

    def home(self) -> None:
        self._guard(lambda tt: tt.home(), "归零")

    def stop_motion(self) -> None:
        """急停：绕过互斥，允许在动作进行中发送。"""
        if self.turntable is None:
            return
        tt = self.turntable
        self._log("■ 急停 …")
        try:
            ok = tt.stop()
            self._log("✓ 已急停" if ok else "⚠ 急停指令未确认")
        except Exception as exc:             # noqa: BLE001
            self._log(f"✗ 急停失败: {exc}")

    def scan_sweep(self, count: int) -> None:
        """等分扫描一圈：先归零，再逐站转动。

        只做机械动作；与相机同步采集属于流水线职责。
        """
        count = int(count)
        if not (2 <= count <= MAX_STEPS_PER_SCAN):
            messagebox.showwarning(
                "参数不合理",
                f"分度数量应在 2～{MAX_STEPS_PER_SCAN} 之间（当前 {count}）",
                parent=self)
            return

        step = 360.0 / count
        tc = self.turntable_config()
        steps = int(round(tc.steps_for_angle(step)))
        actual = tc.nearest_angle_for_steps(steps)
        self._log(f"分度扫描 {count} 站 × {step:.3f}°"
                  f"（{steps} 微步 → 实际 {actual:.3f}°）")

        def _job(_report) -> float:
            tt = self.turntable
            tt.home()
            for i in range(count):
                # 与流水线一致：按整数微步走，避免 ROT 的整数截断误差。
                tt.rotate_steps(steps)
                # 进度直接进队列：Worker 只负责 done/error，
                # 任务内部的中间状态用它自己的队列上报。
                self._queue.put(TaskMessage(
                    "progress", payload=(i + 1) / count,
                    text=f"  第 {i + 1}/{count} 站  累计 {tt.cumulative_deg:+.1f}°"))
            return tt.cumulative_deg

        self._guard(_job, f"分度扫描 {count} 站", prepared=True)

    # ------------------------------------------------------------------
    # 后台执行
    # ------------------------------------------------------------------
    def _guard(self, action, title: str, *,
               prepared: bool = False) -> None:
        """检查连接与忙闲，然后在后台线程执行 ``action``。

        Parameters
        ----------
        action : ``(tt) -> result``；``prepared=True`` 时直接是
                 ``(report) -> result``
        """
        if self.turntable is None:
            if title:
                messagebox.showinfo("未连接", "请先连接转台。", parent=self)
            return
        if self._busy:
            self._log(f"上一个动作未完成，忽略：{title}")
            return
        if title:
            self._log(f"{title} …")

        if prepared:
            job = action
        else:
            tt = self.turntable

            def job(_report):
                return action(tt)

        def _ok(result: Any) -> None:
            if isinstance(result, float):
                self._log(f"✓ 完成，累计 {result:+.2f}°")
            else:
                self._log("✓ 完成")

        def _err(msg: str) -> None:
            self._log(f"✗ {title or '动作'}失败: {msg}")

        self._submit(job, _ok, _err)

    def _submit(self, job: Callable[[Callable], Any],
                on_done: Callable[[Any], None],
                on_error: Callable[[str], None]) -> None:
        self._drain()
        self._on_done, self._on_error = on_done, on_error
        self._busy = True
        self.refresh()

        self._worker = Worker(job, self._queue)
        self._worker.start()
        self.after(60, self._poll)

    def _drain(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass

    def _poll(self) -> None:
        finished = False
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg.kind == "progress":
                    if msg.text:
                        self._log(msg.text)
                elif msg.kind == "done":
                    finished = True
                    self._busy = False
                    self._on_done(msg.payload)
                elif msg.kind == "error":
                    finished = True
                    self._busy = False
                    self._on_error(str(msg.payload))
        except Empty:
            pass
        except tk.TclError:                 # pragma: no cover
            return

        if finished:
            self.refresh()
            return
        try:
            self.after(60, self._poll)
        except tk.TclError:                 # pragma: no cover
            pass

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        conn = self.panel("conn")
        if conn is not None:
            conn.refresh_state(self._connected, self._busy, self.turntable)
        enabled = self._connected and not self._busy
        for key in ("motion", "scan"):
            panel = self.panel(key)
            if panel is not None:
                panel.set_enabled(enabled)

    def _log(self, text: str) -> None:
        page = self._pages.get("status")
        if page is not None:
            page.append(text)


# ---------------------------------------------------------------------------
class _ConnectionPanel(ttk.Frame):
    """串口参数 + 连接状态。"""

    def __init__(self, master: tk.Misc, workbench: HardwareWorkbench) -> None:
        super().__init__(master)
        self.wb = workbench

        self.group = ParamGroup(
            self, on_change=lambda k, v: workbench.store.update(k, v))
        self.group.pack(fill="x")
        for key in TURNTABLE_KEYS:
            fs = workbench.field_spec(key) or _fallback_spec(key)
            if fs is not None:
                self.group.add_row(fs, workbench.cfg.get(key))

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(8, 0))
        self.btn_connect = ttk.Button(bar, text="连接转台",
                                      command=self.wb.connect_turntable)
        self.btn_connect.pack(side="left")
        self.btn_disconnect = ttk.Button(bar, text="断开",
                                         command=self.wb.disconnect,
                                         state="disabled")
        self.btn_disconnect.pack(side="left", padx=(6, 0))

        self.state = ttk.Label(self, text="● 未连接", foreground="#95a5a6")
        self.state.pack(anchor="w", pady=(6, 0))

        # 固件回报的步数/角度（只读）。它与配置系数不一致时最有价值。
        self.fw_state = ttk.Label(self, text="", foreground="#7f8c8d",
                                  wraplength=300, justify="left")
        self.fw_state.pack(anchor="w", pady=(2, 0))

    def set_firmware_status(self, text: str) -> None:
        """显示固件回报的步数 / 角度 / 系数比对。"""
        self.fw_state.configure(text=text)

    def refresh_state(self, connected: bool, busy: bool,
                      tt: Optional[Turntable]) -> None:
        if connected and tt is not None:
            self.state.configure(
                text=f"● 已连接 {tt.cfg.port}  累计 {tt.cumulative_deg:+.2f}°",
                foreground="#27ae60")
        elif busy:
            self.state.configure(text="◌ 执行中…", foreground="#e67e22")
        else:
            self.state.configure(text="● 未连接", foreground="#95a5a6")

        self.btn_connect.configure(
            state="disabled" if (connected or busy) else "normal")
        self.btn_disconnect.configure(
            state="normal" if (connected and not busy) else "disabled")


def _fallback_spec(key: str):
    """转台配置若没有 ``FieldSpec`` 声明，用此兜底。"""
    from .field_spec import FieldSpec

    short = key.rsplit(".", 1)[-1]
    table = {
        "port":          ("str",   None,       None,     "串口", ""),
        "baudrate":      ("int",   9600,       921600,   "波特率", ""),
        "steps_per_rev": ("int",   1,          1000,     "整步/圈", ""),
        "microstep":     ("int",   1,          256,      "细分", ""),
        "gear_ratio":    ("float", 0.1,        100.0,    "减速比", ""),
        "settle_ms":     ("int",   0,          5000,     "稳定延时", "ms"),
    }
    kind, low, high, label, unit = table.get(
        short, ("str", None, None, short, ""))
    return FieldSpec(key=key, label=label, group="", kind=kind,
                     low=low, high=high, unit=unit)


# ---------------------------------------------------------------------------
class _MotionPanel(ttk.Frame):
    """手动转动。"""

    def __init__(self, master: tk.Misc, workbench: HardwareWorkbench) -> None:
        super().__init__(master)
        self.wb = workbench

        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="角度").pack(side="left")
        self.angle = tk.StringVar(value="10.0")
        ttk.Entry(row, textvariable=self.angle, width=10).pack(
            side="left", padx=(4, 2))
        ttk.Label(row, text="°").pack(side="left")

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(6, 0))
        self.btn_rel = ttk.Button(bar, text="相对转动", width=9,
                                  command=self._on_relative)
        self.btn_rel.pack(side="left")
        self.btn_abs = ttk.Button(bar, text="转到", width=6,
                                  command=self._on_absolute)
        self.btn_abs.pack(side="left", padx=(4, 0))
        self.btn_home = ttk.Button(bar, text="归零", width=6,
                                   command=self.wb.home)
        self.btn_home.pack(side="left", padx=(4, 0))

        quick = ttk.Frame(self)
        quick.pack(fill="x", pady=(4, 0))
        for deg in (-90, -10, 10, 90, 180):
            btn = ttk.Button(quick, text=f"{deg:+d}°", width=6,
                             command=lambda d=deg: self._quick(d))
            btn.pack(side="left", padx=(0, 3))
            setattr(self, f"_quick_{deg}", btn)

        self.btn_stop = ttk.Button(self, text="■ 急停",
                                   command=self.wb.stop_motion)
        self.btn_stop.pack(fill="x", pady=(8, 0))

        self.set_enabled(False)

    def _value(self) -> Optional[float]:
        try:
            return float(self.angle.get())
        except ValueError:
            messagebox.showwarning("输入有误", "角度必须是数字。", parent=self)
            return None

    def _quick(self, deg: float) -> None:
        self.angle.set(f"{deg:.1f}")
        self._on_relative()

    def _on_relative(self) -> None:
        v = self._value()
        if v is not None:
            self.wb.move_relative(v)

    def _on_absolute(self) -> None:
        v = self._value()
        if v is not None:
            self.wb.move_to(v)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for name in ("btn_rel", "btn_abs", "btn_home"):
            getattr(self, name).configure(state=state)
        for deg in (-90, -10, 10, 90, 180):
            getattr(self, f"_quick_{deg}").configure(state=state)
        # 急停：连上就始终可点（包括动作进行中）
        self.btn_stop.configure(
            state="normal" if self.wb.connected else "disabled")


# ---------------------------------------------------------------------------
class _ScanPanel(ttk.Frame):
    """分度扫描。"""

    def __init__(self, master: tk.Misc, workbench: HardwareWorkbench) -> None:
        super().__init__(master)
        self.wb = workbench

        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="每圈站数").pack(side="left")
        self.count = tk.StringVar(value="24")
        ttk.Spinbox(row, from_=2, to=MAX_STEPS_PER_SCAN, width=6,
                    textvariable=self.count).pack(side="left", padx=(4, 4))
        ttk.Label(row, text="站", foreground="#7f8c8d").pack(side="left")

        presets = ttk.Frame(self)
        presets.pack(fill="x", pady=(4, 0))
        for n in PRESETS:
            ttk.Button(presets, text=str(n), width=4,
                       command=lambda v=n: self.count.set(str(v))).pack(
                side="left", padx=(0, 2))

        self.btn_scan = ttk.Button(self, text="开始分度扫描",
                                   command=self._on_scan)
        self.btn_scan.pack(fill="x", pady=(6, 0))

        self._hint = ttk.Label(self, text="", foreground="#7f8c8d",
                               wraplength=300, justify="left")
        self._hint.pack(anchor="w", pady=(4, 0))
        self.count.trace_add("write", lambda *_: self._update_hint())
        self._update_hint()
        self.set_enabled(False)

    def _update_hint(self) -> None:
        try:
            n = int(self.count.get())
        except ValueError:
            self._hint.configure(text="站数必须是整数")
            return
        if n < 2:
            self._hint.configure(text="站数至少为 2")
            return
        step = 360.0 / n
        try:
            tc = self.wb.turntable_config()
            steps = int(round(tc.steps_for_angle(step)))
            actual = tc.nearest_angle_for_steps(steps)
            self._hint.configure(
                text=f"每站 {step:.3f}° ≈ {steps} 微步（实际 {actual:.3f}°），"
                     f"共 {n} 站，先归零")
        except Exception:                   # noqa: BLE001
            self._hint.configure(text=f"每站 {step:.3f}°，共 {n} 站")

    def _on_scan(self) -> None:
        try:
            n = int(self.count.get())
        except ValueError:
            messagebox.showwarning("输入有误", "站数必须是整数。", parent=self)
            return
        if not messagebox.askyesno(
                "确认", f"转台将先归零，再转动约 {(n - 1) * 360.0 / n:.1f}°。"
                        f"\n继续？", parent=self):
            return
        self.wb.scan_sweep(n)

    def set_enabled(self, enabled: bool) -> None:
        self.btn_scan.configure(state="normal" if enabled else "disabled")


# ---------------------------------------------------------------------------
class _StatusPage(ttk.Frame):
    """右侧状态页：硬件动作日志。"""

    MAX_LINES = 2000

    def __init__(self, workbench: HardwareWorkbench) -> None:
        super().__init__(workbench)
        self.text = tk.Text(self, wrap="word", relief="flat",
                            background="#1e1e1e", foreground="#d4d4d4",
                            font=("Consolas", 9))
        self.text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(self, orient="vertical",
                               command=self.text.yview)
        scroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scroll.set, state="disabled")
        self.append("硬件控制台就绪。连接转台后可用。")

    def append(self, line: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", f"[{time.strftime('%H:%M:%S')}] {line}\n")
        # 长跑时裁掉旧行，避免 Text 控件无限膨胀
        lines = int(self.text.index("end-1c").split(".")[0])
        if lines > self.MAX_LINES:
            self.text.delete("1.0", f"{lines - self.MAX_LINES}.0")
        self.text.see("end")
        self.text.configure(state="disabled")
