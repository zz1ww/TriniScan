"""转台控制：通过串口与 MCU 通信。

协议
----
与固件 ``firmware/mcu_stm32_cubeide`` / ``firmware/mcu_stm32_spl`` 对应，ASCII 行协议，
``\\n`` 结尾，波特率 115200::

    PING            → PONG
    HOME            → OK
    ROT <deg>       → OK
    STEPS <n>       → OK
    ENABLE 1/0      → OK
    STATUS          → STATUS <state>
    STOP            → OK

注意
----
- 收到 ``OK`` 后还需等待机械稳定（``settle_ms``）再触发相机；
- ``rotate_deg`` 会阻塞直到 MCU 返回并完成稳定延时。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Union

from ..common.logging_utils import get_logger

log = get_logger(__name__)

try:
    import serial
    _HAS_SERIAL = True
except ImportError:  # pragma: no cover
    serial = None
    _HAS_SERIAL = False

__all__ = ["TurntableConfig", "Turntable", "ScanPlan",
           "plan_scan", "resolve_scan_plan", "apply_scan_plan",
           "nearest_exact_view_count"]


@dataclass
class TurntableConfig:
    """转台配置。"""

    port: str = "COM3"
    baudrate: int = 115200
    steps_per_rev: int = 200        # 电机整步/圈
    microstep: int = 32             # 驱动器细分
    gear_ratio: float = 5.0         # 机械减速比
    settle_ms: int = 200            # 到位稳定延时
    timeout_s: float = 5.0          # 单条指令超时

    @property
    def steps_per_degree(self) -> float:
        return self.steps_per_rev * self.microstep * self.gear_ratio / 360.0

    @property
    def steps_per_rev_total(self) -> float:
        """转一整圈所需的**微步**总数（含细分与减速比）。"""
        return self.steps_per_rev * self.microstep * self.gear_ratio

    def steps_for_angle(self, angle_deg: float) -> float:
        """给定角度对应的理论步数（浮点）。"""
        return self.steps_per_degree * float(angle_deg)

    def nearest_angle_for_steps(self, steps: int) -> float:
        """给定**整数步数**回算实际角度（度）。"""
        spd = self.steps_per_degree
        if spd <= 0:
            return 0.0
        return float(steps) / spd

    @classmethod
    def from_config(cls, cfg) -> "TurntableConfig":
        get = _make_getter(cfg)
        return cls(
            port=str(get("port", "COM3")),
            baudrate=int(get("baudrate", 115200)),
            steps_per_rev=int(get("steps_per_rev", 200)),
            microstep=int(get("microstep", 32)),
            gear_ratio=float(get("gear_ratio", 5.0)),
            settle_ms=int(get("settle_ms", 200)),
            timeout_s=float(get("timeout_s", 5.0)),
        )


class Turntable:
    """上位机转台控制器（上下文管理器）。"""

    def __init__(self,
                 config: Union[TurntableConfig, dict, object]) -> None:
        if isinstance(config, TurntableConfig):
            self.cfg = config
        else:
            self.cfg = TurntableConfig.from_config(config)
        self._ser = None
        self._cumulative_deg = 0.0

    # ------------------------------------------------------------------
    @property
    def steps_per_degree(self) -> float:
        return self.cfg.steps_per_degree

    @property
    def cumulative_deg(self) -> float:
        """累计已转角度（相对归零）。"""
        return self._cumulative_deg

    def connect(self) -> "Turntable":
        """打开串口并握手。"""
        if not _HAS_SERIAL:
            raise ImportError("需要 pyserial: pip install pyserial")
        self._ser = serial.Serial(
            self.cfg.port, self.cfg.baudrate, timeout=self.cfg.timeout_s
        )
        time.sleep(2.0)                 # 等 MCU 复位/就绪
        self._ser.reset_input_buffer()
        if not self._handshake():
            raise RuntimeError("转台握手失败（未收到 PONG）")
        log.info("转台已连接: %s @ %d", self.cfg.port, self.cfg.baudrate)
        return self

    def _handshake(self, retries: int = 3) -> bool:
        for _ in range(retries):
            self._send("PING")
            if self._readline() == "PONG":
                return True
        return False

    # ------------------------------------------------------------------
    def _send(self, cmd: str) -> None:
        if self._ser is None:
            raise RuntimeError("转台未连接")
        self._ser.write((cmd + "\n").encode("ascii"))

    def _readline(self) -> str:
        if self._ser is None:
            raise RuntimeError("转台未连接")
        return self._ser.readline().decode(errors="ignore").strip()

    def _wait_ok(self) -> bool:
        return self._readline() == "OK"

    def _settle(self) -> None:
        time.sleep(self.cfg.settle_ms / 1000.0)

    # ------------------------------------------------------------------
    def home(self) -> bool:
        """归零。"""
        self._send("HOME")
        if not self._wait_ok():
            return False
        self._settle()
        self._cumulative_deg = 0.0
        return True

    def rotate_deg(self, angle: float) -> bool:
        """相对转动指定角度（度），阻塞至完成并稳定。"""
        self._send(f"ROT {angle:.6f}")
        if not self._wait_ok():
            return False
        self._settle()
        self._cumulative_deg += angle
        return True

    def rotate_steps(self, steps: int) -> bool:
        """相对转动指定步数。"""
        self._send(f"STEPS {int(steps)}")
        if not self._wait_ok():
            return False
        self._settle()
        self._cumulative_deg += steps / self.steps_per_degree
        return True

    def enable(self, on: bool = True) -> bool:
        self._send(f"ENABLE {1 if on else 0}")
        return self._wait_ok()

    def status(self) -> str:
        self._send("STATUS")
        return self._readline()

    def stop(self) -> bool:
        self._send("STOP")
        return self._wait_ok()

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
            log.debug("转台串口已关闭")

    def __enter__(self) -> "Turntable":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()


def _make_getter(cfg):
    if hasattr(cfg, "get"):
        return cfg.get
    if isinstance(cfg, dict):
        return lambda k, d=None: cfg.get(k, d)
    return lambda k, d=None: getattr(cfg, k, d)


# ---------------------------------------------------------------------------
# 扫描角度计划：整数步数 + 角度自洽
# ---------------------------------------------------------------------------
@dataclass
class ScanPlan:
    """一次扫描的角度计划（**步数与角度自洽**）。

    为什么需要
    ----------
    步进电机只能走**整数步**，而"每视角转 360/N 度"往往不是整数步：
    以 ``steps_per_rev=200, microstep=32, gear_ratio=5.18`` 为例，整圈
    33152 微步：

    - 360 视角 → 92.09 步/视角（非整数）
    - 720 视角 → 46.04 步/视角（非整数）
    - 1080 视角 → 30.70 步/视角（非整数）

    若直接把 ``angle_step_deg`` 交给固件按角度换算，或每视角都累积
    浮点角度的舍入误差，**多视角后会出现系统性角度偏差**——表现为
    刚性物体在不同视角下的半径不一致（点云"炸开"）。

    本结构把一个视角的转角**量化为整数微步**，再**回算实际角度**；
    该实际角度同时用于：固件转动、以及拼接时的 ``angle_deg``。
    二者严格一致，误差只来自"一步以内"的量化（可忽略）。

    Attributes
    ----------
    num_views : 视角总数
    steps_per_step : 每视角的**整数**微步数
    angle_step_deg : 回算出的每视角实际角度（= steps / steps_per_degree）
    total_steps : 走完全部视角后的累计步数
    total_deg : 累计角度的理论值（应≈360°；若整圈步数无法被视角数整除，
        会与 360° 有差，见 :attr:`full_turn_exact`）
    """

    num_views: int
    steps_per_step: int
    angle_step_deg: float
    total_steps: int
    total_deg: float

    @property
    def angle_error_deg(self) -> float:
        """单步量化导致的每视角角度误差（度）。"""
        return abs(self.total_deg - 360.0) / max(self.num_views, 1)

    @property
    def full_turn_error_deg(self) -> float:
        """累计转完一圈后与 360° 的偏差（度）。

        若整圈微步数不能被视角数整除，此值可大于 0——意味着"扫完所有
        视角后转台并未精确回到起点"，多视角拼接会出现**接缝错位**。
        """
        return self.total_deg - 360.0

    @property
    def full_turn_exact(self) -> bool:
        """累计转角是否精确为 360°（无接缝错位）。"""
        return abs(self.full_turn_error_deg) < 1e-9


def nearest_exact_view_count(target: int,
                             steps_per_degree: float,
                             *,
                             max_delta: int = 3) -> Optional[int]:
    """在 ``target`` 附近寻找**能整除整圈步数**的视角数。

    为什么需要
    ----------
    若整圈微步数 ``S`` 不能被视角数 ``N`` 整除，则无法同时做到
    "每视角等分"与"转满整圈"，多视角拼接必然出现接缝错位。此时可
    微调视角数到 ``N'`` 使 ``S % N' == 0``（例如 33152 的因子有
    516、592、……）。本函数在 ``[target-max_delta, target+max_delta]``
    内找最接近且能整除的 ``N'``。

    Parameters
    ----------
    target : 期望视角数
    steps_per_degree : 转台每度微步数
    max_delta : 允许偏离 target 的最大视野数

    Returns
    -------
    int 或 None : 找到的视角数；无解则 None
    """
    total = int(round(steps_per_degree * 360.0))
    if total <= 0:
        return None
    best = None
    best_d = None
    for n in range(max(1, target - int(max_delta)),
                   target + int(max_delta) + 1):
        if total % n == 0:
            d = abs(n - target)
            if best_d is None or d < best_d:
                best, best_d = n, d
    return best


def plan_scan(num_views: int,
              steps_per_degree: float,
              *, prefer_full_turn: bool = True) -> ScanPlan:
    """为给定视角数生成**整数步**的角度计划。

    Parameters
    ----------
    num_views : 期望视角数（如 360 / 720 / 1080）
    steps_per_degree : 转台每度对应的微步数
    prefer_full_turn : True 时优先保证累计转满一整圈（360°）——
        即取 ``round(总步数 / num_views)`` 作为每视角步数，保证
        **分度均匀**（各视角间距一致），代价是总量可能略偏离 360°。

    Returns
    -------
    ScanPlan

    Notes
    -----
    取整策略：``steps_per_step = round(total_steps_per_rev / num_views)``，
    其中 ``total_steps_per_rev = steps_per_degree * 360``。这样每个
    视角的间距**完全相同**（不累积漂移），且整个过程只差最后一步的
    零头（远小于 1 步）。
    """
    num_views = int(num_views)
    if num_views <= 0:
        raise ValueError("num_views 必须为正整数")
    if steps_per_degree <= 0:
        raise ValueError("steps_per_degree 必须为正数")

    # 整圈所需微步数（理想值，可能非整数）
    total_rev = steps_per_degree * 360.0

    # 每视角步数：在"均匀分度"与"总量尽量接近整圈"之间取最优。
    # 直接 round(total_rev/n) 可能使 n*step 明显偏离 total_rev
    # （如 1080 视角时，31 步/视角 -> 累计 +3.56°）。因此在候选
    # 步数 {floor, round, ceil} 中，选使 **累计角度最接近 360°** 者，
    # 从而把误差压到"不足一步"，而不是随视角数放大。
    base = total_rev / num_views
    candidates = {max(1, int(math.floor(base))),
                  max(1, int(round(base))),
                  max(1, int(math.ceil(base)))}
    steps_per_step = min(
        candidates,
        key=lambda s: (abs(s * num_views - total_rev), -s),
    )

    angle_step_deg = float(steps_per_step) / steps_per_degree
    total_steps = steps_per_step * num_views
    total_deg = total_steps / steps_per_degree
    return ScanPlan(
        num_views=num_views,
        steps_per_step=steps_per_step,
        angle_step_deg=angle_step_deg,
        total_steps=total_steps,
        total_deg=total_deg,
    )


def resolve_scan_plan(cfg, table_cfg: Optional[TurntableConfig] = None
                      ) -> ScanPlan:
    """从配置解析扫描计划（``scan.preset`` 或 ``scan.num_views``）。

    优先级：

    1. ``scan.preset``（``720`` / ``1080`` / 任意正整数）—— 直接作为
       视角数；
    2. 否则用 ``scan.num_views``。

    解析出的 ``angle_step_deg`` 为**回算的实际角度**，调用方应把它
    写回配置（见 :func:`apply_scan_plan`）以保证后续环节自洽。
    """
    get = _make_getter(cfg)
    if table_cfg is None:
        table_cfg = TurntableConfig.from_config(
            get("turntable", {}) if hasattr(cfg, "get") else {})

    preset = get("preset", None)
    if preset is None and hasattr(cfg, "get"):
        preset = cfg.get("scan.preset", None)
    if preset not in (None, "", 0, "0"):
        try:
            num_views = int(preset)
        except (TypeError, ValueError):
            num_views = int(get("num_views", 360))
    else:
        num_views = int(get("num_views", 360))

    return plan_scan(num_views, table_cfg.steps_per_degree)


def apply_scan_plan(cfg, plan: ScanPlan) -> None:
    """把扫描计划写回配置（``scan.num_views`` / ``scan.angle_step_deg``）。

    这是"角度自洽"的落地点：固件与拼接都使用同一个
    ``angle_step_deg``，而不是"配置里写 1.0° 而电机走的是别的角"。
    """
    if hasattr(cfg, "set"):
        cfg.set("scan.num_views", plan.num_views)
        cfg.set("scan.angle_step_deg", plan.angle_step_deg)
        cfg.set("scan.steps_per_view", plan.steps_per_step)
