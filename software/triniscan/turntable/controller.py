"""转台控制：通过串口与 MCU 通信。"""
from __future__ import annotations

import time

import serial


class Turntable:
    """上位机转台控制器。

    与固件 firmware/mcu/turntable_controller.ino 的协议对应。
    """

    def __init__(self, cfg) -> None:
        self.port: str = cfg.get("port", "COM3")
        self.baudrate: int = cfg.get("baudrate", 115200)
        self.steps_per_rev: int = cfg.get("steps_per_rev", 200)
        self.microstep: int = cfg.get("microstep", 32)
        self.gear_ratio: float = cfg.get("gear_ratio", 5.0)
        self.settle_ms: int = cfg.get("settle_ms", 200)
        self.timeout_s: float = cfg.get("timeout_s", 5.0)

        self._ser: serial.Serial | None = None

    # ------------------------------------------------------------------
    @property
    def steps_per_degree(self) -> float:
        total = self.steps_per_rev * self.microstep * self.gear_ratio
        return total / 360.0

    # ------------------------------------------------------------------
    def connect(self) -> None:
        self._ser = serial.Serial(
            self.port, self.baudrate, timeout=self.timeout_s
        )
        time.sleep(2.0)  # 等 MCU 复位/就绪
        self._ser.reset_input_buffer()
        self._send("PING")
        resp = self._readline()
        if resp != "PONG":
            raise RuntimeError(f"转台握手失败: {resp!r}")

    def _send(self, cmd: str) -> None:
        assert self._ser is not None, "转台未连接"
        self._ser.write((cmd + "\n").encode())

    def _readline(self) -> str:
        assert self._ser is not None
        line = self._ser.readline().decode(errors="ignore").strip()
        return line

    def _wait_ok(self) -> bool:
        line = self._readline()
        return line == "OK"

    # ------------------------------------------------------------------
    def home(self) -> bool:
        self._send("HOME")
        if not self._wait_ok():
            return False
        time.sleep(self.settle_ms / 1000.0)
        return True

    def rotate_deg(self, angle: float) -> bool:
        self._send(f"ROT {angle:.6f}")
        if not self._wait_ok():
            return False
        time.sleep(self.settle_ms / 1000.0)
        return True

    def rotate_steps(self, n: int) -> bool:
        self._send(f"STEPS {int(n)}")
        if not self._wait_ok():
            return False
        time.sleep(self.settle_ms / 1000.0)
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

    def __enter__(self) -> "Turntable":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    from triniscan.core.config import Config

    tt = Turntable(Config.load().turntable)
    tt.connect()
    print("已连接转台")
    print("steps_per_degree =", tt.steps_per_degree)
    tt.home()
    print("归零完成")
    tt.rotate_deg(90)
    print("转 90 度完成")
    tt.close()
