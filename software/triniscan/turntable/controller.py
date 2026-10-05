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

__all__ = ["TurntableConfig", "Turntable"]


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
