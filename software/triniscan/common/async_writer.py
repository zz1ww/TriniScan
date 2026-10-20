"""异步图像落盘：把 ``imwrite`` 从采集关键路径上移走。

为什么需要
----------
采集循环的每视角时间是串行的::

    转台转动 → 等待停稳 → 抓图 → 写盘

其中写盘（1280×800 黑白 PNG 实测约 55 ms/张）**不占用相机，也不占用
转台**，纯粹是 CPU/磁盘 I/O。把它放在循环里等于让两者白等。720 视角
累计约 40 秒——而采集本身只需约 55 秒，写盘若不挪走就几乎是全部开销。

实测（本机 SSD，真实激光帧）：同步写 57.7 ms/张；异步 `submit()` 0.115
ms/张；`close()` 排空时后台线程 55.2 ms/张。

本模块把这个环节挪到独立线程：采集线程只做 ``submit()``（入队，微秒级），
写盘线程在后台消费队列。

设计约束
--------
1. **失败必须可见**：写盘线程里的异常不能吞掉。``close()`` 汇总并抛出，
   绝不允许"文件没写成功但扫描继续"。
2. **有界队列**：图像是 1280×800×3 ≈ 3 MB/张，720 张 = 2.1 GB。若写盘
   跟不上（慢磁盘），无界队列会吃光内存。队列满时 ``submit`` 阻塞，
   把压力反推回采集端——宁可慢，不可爆内存。
3. **关闭时排空**：``close()`` 必须先等队列清空再返回，否则最后几张图
   可能没落盘，而调用方已经认为采集完成。
4. **线程安全**：``submit`` 可被单线程调用（采集循环），内部用锁保护
   错误列表与计数。

用法
----
.. code-block:: python

    with AsyncImageWriter() as writer:
        for i in range(n):
            image = cam.grab()
            writer.submit(path, image)      # 立即返回
            # ... 继续转台、抓下一张 ...
    # 离开 with 时自动排空；图片在此刻保证已全部写盘

独立成模块而非塞进 ``Camera`` 的原因：落盘是 I/O 关注点，与"如何取图"
无关；标定、离线重处理等场景可以复用同一套实现，也能脱离相机单测。
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .io_utils import ensure_parent
from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["AsyncImageWriter", "ImageWriteError"]

#: 默认队列深度。30 张 × 3 MB ≈ 90 MB 峰值内存，
#: 同时给出足够的"写盘落后于采集"缓冲。
DEFAULT_QUEUE_SIZE = 30


class ImageWriteError(IOError):
    """异步写盘的聚合错误：一次作业中所有失败的图片。"""

    def __init__(self, failures: List[Tuple[str, str]]) -> None:
        self.failures = failures
        detail = "; ".join(f"{p}: {e}" for p, e in failures[:5])
        more = "" if len(failures) <= 5 else f" 等 {len(failures)} 张"
        super().__init__(f"{len(failures)} 张图片写盘失败{more}（{detail}）")


class AsyncImageWriter:
    """后台线程写图（上下文管理器）。

    Parameters
    ----------
    queue_size : 待写队列深度。满时 ``submit`` 阻塞（背压），
                 不要设成无界以免慢磁盘时内存失控。
    name : 线程名，便于日志与调试。
    """

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE,
                 name: str = "image-writer") -> None:
        if queue_size < 1:
            raise ValueError("queue_size 必须 >= 1")
        self._queue: Deque[Tuple[str, np.ndarray]] = deque()
        self._cond = threading.Condition()
        self._queue_size = int(queue_size)
        self._closed = False
        self._submitted = 0
        self._written = 0
        self._failures: List[Tuple[str, str]] = []
        self._thread = threading.Thread(target=self._run, name=name,
                                        daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def submit(self, path: str, image: np.ndarray) -> None:
        """提交一张图（不阻塞，除非队列已满）。

        Parameters
        ----------
        path : 目标路径，父目录会自动创建。
        image : ``uint8`` 图像。

        Notes
        -----
        图像**不会被复制**。调用方必须保证提交后不再原地修改该数组
        （采集循环里每帧是新数组，天然满足）。
        """
        if image is None:
            raise ValueError("image 不能为空")
        with self._cond:
            if self._closed:
                raise RuntimeError("AsyncImageWriter 已关闭")
            # 背压：队列满则等待写盘线程腾出位置
            while len(self._queue) >= self._queue_size and not self._closed:
                self._cond.wait()
            if self._closed:
                raise RuntimeError("AsyncImageWriter 已关闭")
            self._queue.append((str(path), image))
            self._submitted += 1
            self._cond.notify_all()

    def close(self) -> None:
        """排空队列并等待写盘线程结束；有失败时抛 :class:`ImageWriteError`。"""
        with self._cond:
            if self._closed:
                pass
            else:
                self._closed = True
                self._cond.notify_all()

        if self._thread.is_alive():
            self._thread.join()

        if self._failures:
            failures, self._failures = self._failures, []
            raise ImageWriteError(failures)

    @property
    def submitted(self) -> int:
        """已提交张数。"""
        return self._submitted

    @property
    def written(self) -> int:
        """已成功写盘张数。"""
        return self._written

    @property
    def pending(self) -> int:
        """队列中待写张数。"""
        with self._cond:
            return len(self._queue)

    # ------------------------------------------------------------------
    # 后台线程
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue:
                    return                      # 已关闭且已排空
                path, image = self._queue.popleft()
                self._cond.notify_all()         # 唤醒可能阻塞的 submit

            try:
                ensure_parent(path)
                if not cv2.imwrite(path, image):
                    raise IOError("cv2.imwrite 返回 False")
            except Exception as exc:            # noqa: BLE001 - 汇总后抛出
                with self._cond:
                    self._failures.append((path, str(exc)))
                log.error("写图失败 %s: %s", path, exc)
            else:
                with self._cond:
                    self._written += 1

    # ------------------------------------------------------------------
    def __enter__(self) -> "AsyncImageWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # 即使循环体已抛异常，也要把已提交的图写完（否则数据不完整）。
        try:
            self.close()
        except ImageWriteError:
            if exc_type is None:
                raise
            # 已有原始异常在抛，写盘错误只记录，不覆盖根因
            log.error("写盘阶段另有失败（已被原始异常掩盖）")
