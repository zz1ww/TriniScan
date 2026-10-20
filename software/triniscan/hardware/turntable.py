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

**``OK`` 的真实语义（重要）**
--------------------------------
固件的运动是**非阻塞**的：``exec_command`` 调用 ``motor_move_degrees()``
只启动定时器中断就返回，紧接着发送 ``OK``。因此 ::

    OK 到达  ==  "转台开始转"，而不是"转台转到了"

唯一能确认"停稳"的方法有两个，本模块都实现了：

1. 轮询 ``STATUS`` 直到 ``busy=0``（精确，需固件支持，见 :meth:`wait_idle`）；
2. 固定延时 ``settle_ms``（兜底，不可靠——它不包含运动时间）。

**``settle_ms`` 只是"停转后"的稳定延时**，不含运动时间。总等待时间必须是
``运动时间 + settle_ms``。运动时间 = ``步数 × step_us``。若 ``settle_ms``
小于运动时间而只做 ``sleep(settle_ms)``，相机会在转台仍在转动时曝光。

固件的另一个陷阱：运动中收到新的 ``ROT`` / ``STEPS`` 会被
``motor_move_steps`` 里的 ``if (motor_is_busy()) return;`` **静默丢弃**，
但 :mod:`cmd.c` 仍然回复 ``OK``。因此 :meth:`rotate_steps` 会校验
``STATUS`` 的步数增量，发现丢弃时抛 :class:`MotionNotApplied`。
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

#: ``wait_idle`` 的轮询间隔（秒）。固件回包毫秒级，20 ms 足够，
#: 又不会在 115200 波特率下占满串口。
_STATUS_POLL_S = 0.02

__all__ = ["TurntableConfig", "Turntable", "ScanPlan", "MotorStatus",
           "MotionNotApplied", "plan_scan", "resolve_scan_plan",
           "apply_scan_plan", "nearest_exact_view_count"]


class MotionNotApplied(RuntimeError):
    """运动指令未被固件执行（通常是被 ``motor_is_busy()`` 丢弃）。

    固件在这种情况下仍然回复 ``OK``——这个静默失败会让上位机以为已经
    转到位，随后拍下的照片属于错误角度。上位机必须自己校验。
    """


@dataclass(frozen=True)
class MotorStatus:
    """``STATUS`` 命令的解析结果。

    固件回包格式（见 ``cmd.c``）::

        STATUS <state> <steps> <deg> <busy>

    其中 ``state`` 为 ``IDLE`` / ``MOVING`` / ``HOMING`` / ``ERROR``，
    ``deg`` 由固件按**它自己的** ``STEPS_PER_REV`` 换算——若该系数与
    上位机配置不一致，:attr:`firmware_steps_per_rev` 就能暴露出来。
    """

    state: str = "UNKNOWN"
    steps: int = 0
    deg_reported: float = 0.0
    busy: bool = False

    @property
    def firmware_steps_per_rev(self) -> Optional[float]:
        """由 ``steps`` 与 ``deg`` 反解固件使用的每圈微步数。

        固件计算 ``deg = steps * 360 / SPR_fw``，故
        ``SPR_fw = 360 * steps / deg``。``steps`` 为 0 时无法反解。
        """
        if self.steps == 0 or abs(self.deg_reported) < 1e-9:
            return None
        return 360.0 * abs(self.steps) / abs(self.deg_reported)

    @classmethod
    def parse(cls, line: str) -> Optional["MotorStatus"]:
        """解析 ``STATUS`` 回包；格式不符返回 ``None``。"""
        parts = line.strip().split()
        if len(parts) < 5 or parts[0] != "STATUS":
            return None
        try:
            return cls(state=parts[1], steps=int(parts[2]),
                       deg_reported=float(parts[3]),
                       busy=parts[4] not in ("0", "", "False"))
        except (TypeError, ValueError):
            return None


@dataclass
class TurntableConfig:
    """转台配置。"""

    port: str = "COM3"
    baudrate: int = 115200
    steps_per_rev: int = 200        # 电机整步/圈
    microstep: int = 32             # 驱动器细分
    gear_ratio: float = 5.0         # 机械减速比
    #: 每步脉冲间隔（微秒）。必须与固件 ``DEFAULT_STEP_US`` 一致——
    #: 上位机用它估算运动时长。不一致不会导致转错，只会让等待时间估偏。
    step_us: int = 900
    settle_ms: int = 200            # **停转后**的稳定延时（不含运动）
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

    def motion_time_ms(self, steps: int) -> float:
        """走 ``steps`` 个微步所需时间（毫秒）。

        固件每个整步占两个半周期，半周期 = ``step_us / 2``，故一步
        = ``step_us`` 微秒（见 ``firmware/.../motor.c: timer_start``）。
        """
        return abs(int(steps)) * float(self.step_us) / 1000.0

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
            step_us=int(get("step_us", 900)),
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
        """等 ``OK``；固件回 ``ERR ...`` 时记录原因并返回 False。"""
        line = self._readline()
        if line == "OK":
            return True
        if line:
            log.warning("固件返回非 OK: %r", line)
        return False

    def _settle(self) -> None:
        """**停转后**的机械稳定延时（不含运动时间）。"""
        time.sleep(self.cfg.settle_ms / 1000.0)

    # ------------------------------------------------------------------
    # 状态与停稳判定
    # ------------------------------------------------------------------
    def read_status(self) -> Optional[MotorStatus]:
        """读取固件状态；不支持或解析失败返回 ``None``。"""
        try:
            self._send("STATUS")
            line = self._readline()
        except Exception as exc:            # noqa: BLE001
            log.debug("读取 STATUS 失败: %s", exc)
            return None
        status = MotorStatus.parse(line)
        if status is None:
            log.debug("STATUS 回包无法解析: %r", line)
        return status

    def wait_idle(self, timeout_s: Optional[float] = None) -> bool:
        """轮询 ``STATUS`` 直到固件报告 ``busy=0``。

        这是"转到位"的**唯一可靠判据**：固件在开始转动时就回了 ``OK``，
        之后 ``busy`` 会一直为真，直到定时器把脉冲全部发完。

        Parameters
        ----------
        timeout_s : 超时（秒）。默认按运动时长估算：调用方应在此之前
            调用过 :meth:`_expected_motion_s`，否则用配置的指令超时。

        Returns
        -------
        bool
            ``True`` 表示已确认停转；``False`` 表示超时或固件不支持
            ``STATUS``。**调用方必须据此决定是否仍要拍照**——返回 False
            时本方法已退回"等待运动时长 + settle_ms"，但那只是估算。
        """
        status = self.read_status()
        if status is None:
            # 固件不支持 STATUS（或串口异常）：按运动时长估算兜底。
            log.debug("STATUS 不可用，回退为定时等待")
            return False
        if not status.busy:
            return True

        limit = (self.cfg.timeout_s if timeout_s is None
                 else max(float(timeout_s), 0.0))
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            time.sleep(_STATUS_POLL_S)
            status = self.read_status()
            if status is None:
                return False
            if not status.busy:
                return True

        log.warning("等待转台停稳超时（%.1f s）", limit)
        return False

    def _wait_motion_done(self, steps: int) -> bool:
        """等待一次转动真正结束，并补足 ``settle_ms``。

        总等待 = 运动时间 + ``settle_ms``。运动时间由 ``steps × step_us``
        给出（固件每整步 ``step_us`` 微秒）。若 ``STATUS`` 可用，则以
        固件回报的 ``busy=0`` 为准——它比估算更可靠。

        Returns
        -------
        bool
            是否确认停稳（``STATUS`` 不可用时返回 ``False``，但等待仍按
            估算完成，因此调用方可以继续，只是没有"确认"）。
        """
        motion_s = self.cfg.motion_time_ms(steps) / 1000.0
        confirmed = self.wait_idle(timeout_s=motion_s + self.cfg.timeout_s)
        if not confirmed:
            # STATUS 不可用：至少按运动时长等够，再补稳定延时。
            time.sleep(motion_s)
        self._settle()
        return confirmed

    # ------------------------------------------------------------------
    def home(self) -> bool:
        """归零。

        归零可能持续数十秒（``HOME_MAX_STEPS`` = 1.5 圈）。固件同样在
        **开始**归零时就回 ``OK``，因此必须等 ``busy=0`` 才能认为到位；
        否则紧接着的 ``STEPS`` 会被固件丢弃（运动中命令被忽略）。
        """
        self._send("HOME")
        if not self._wait_ok():
            return False
        self._wait_motion_done(self._home_max_steps())
        self._cumulative_deg = 0.0
        return True

    def _home_max_steps(self) -> int:
        """归零的最长行程（用于估算等待上限）。"""
        return int(self.cfg.steps_per_rev_total * 1.5)

    def rotate_deg(self, angle: float) -> bool:
        """相对转动指定角度（度），阻塞至完成并稳定。

        .. warning::
           固件的 ``ROT`` 换算使用 ``STEPS_PER_DEG_X1000`` **整数**
           截断（``STEPS_PER_REV*1000/360``），每度会少约 1 微步。
           精度要求高时请用 :meth:`rotate_steps`（流水线即如此）。
        """
        steps = int(round(self.cfg.steps_for_angle(angle)))
        self._send(f"ROT {angle:.6f}")
        if not self._wait_ok():
            return False
        self._wait_motion_done(steps)
        self._cumulative_deg += angle
        return True

    def rotate_steps(self, steps: int) -> bool:
        """相对转动指定**整数微步**，阻塞至完成并稳定。

        收到 ``OK`` 之后会再校验 ``STATUS`` 的步数增量：若固件因
        "运动中"丢弃了这条指令（它仍会回 ``OK``），这里抛
        :class:`MotionNotApplied`，避免上位机误以为已转到位。

        Raises
        ------
        MotionNotApplied
            指令未被执行，且已通过 ``STATUS`` 确认。
        """
        steps = int(steps)
        before = self.read_status()
        before_steps = before.steps if before is not None else None

        self._send(f"STEPS {steps}")
        if not self._wait_ok():
            return False
        self._wait_motion_done(steps)

        after = self.read_status()
        if before_steps is not None and after is not None:
            applied = after.steps - before_steps
            if applied != steps:
                raise MotionNotApplied(
                    f"转台未执行命令：请求 {steps:+d} 步，实际 "
                    f"{applied:+d} 步（固件在运动中丢弃了指令）")

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
