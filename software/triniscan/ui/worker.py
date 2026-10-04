"""后台任务线程：把耗时操作从 UI 线程剥离。

Tkinter 不是线程安全的，因此约定：

- 后台线程**只**计算结果，不触碰任何控件；
- 结果通过 :class:`queue.Queue` 回传；
- UI 线程用 ``after()`` 轮询队列并刷新界面。
"""
from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable, Optional


class TaskMessage:
    """后台任务的一条消息。"""

    __slots__ = ("kind", "payload", "text")

    def __init__(self, kind: str, payload: Any = None,
                 text: str = "") -> None:
        self.kind = kind          # "log" | "progress" | "done" | "error"
        self.payload = payload
        self.text = text


class Worker:
    """在后台线程运行一个函数，并把消息推入队列。

    Parameters
    ----------
    func : 目标函数，签名为 ``func(report) -> result``，
           其中 ``report`` 是可调用的进度上报函数
           ``report(text, fraction)``。
    queue_ : 消息队列（由 UI 创建并轮询）。
    """

    def __init__(self, func: Callable[[Callable], Any],
                 queue_: "queue.Queue[TaskMessage]") -> None:
        self._func = func
        self._queue = queue_
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        """请求取消（协作式，由任务自行检查）。"""
        self._cancel.set()

    # ------------------------------------------------------------------
    def _report(self, text: str, fraction: Optional[float] = None) -> None:
        self._queue.put(TaskMessage("progress", payload=fraction, text=text))

    def start(self) -> None:
        if self.running:
            raise RuntimeError("任务已在运行")
        self._cancel.clear()

        def _run() -> None:
            try:
                result = self._func(self._report)
                self._queue.put(TaskMessage("done", payload=result))
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                self._queue.put(
                    TaskMessage("error", payload=exc, text=tb)
                )

        self._thread = threading.Thread(target=_run, daemon=True,
                                        name="triniscan-worker")
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
