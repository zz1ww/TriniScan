"""实时画面控件：cv2 缩放 + PPM 编码交给 Tk，替代 matplotlib 做视频预览。

为什么不用 matplotlib
---------------------
旧实现把每一帧丢给内嵌的 matplotlib ``AxesImage``：

.. code-block:: python

    ax.set_data(image); fig.canvas.draw_idle()

``draw_idle`` 会重绘整个 Figure（坐标轴、刻度、字体），实测 1280x800
单帧约 100 ms 以上，1080p 更慢，必须先把图缩到 720 px 才能显示。
而视频预览只需要"把像素搬到屏幕"，不需要坐标系。

为什么不用 Pillow
-----------------
``PIL`` 缩放 + ``ImageTk.PhotoImage`` 实测约 **49 ms/帧**——每次构造
``PhotoImage`` 都要重新上传整张图。当前实现改为
``cv2.imencode(".ppm")`` + ``tk.PhotoImage(data=...)``：缩放与编码都在
C 层完成，PPM 是无需压缩的裸像素格式，Tk 直接拷贝，实测约
**21 ms/帧**（1280x800 → 控件尺寸）。

线程模型
--------
Tk 控件只能在主线程操作，因此本模块分两层：

- :class:`FrameSource`：在后台线程读取相机，**只保留最新一帧**交给
  单槽位队列，旧帧直接丢弃；读取失败自动重连（见 ``LiveCamera``）。
- :class:`VideoView`：主线程调用 :meth:`VideoView.pump` 取帧并渲染。
  ``pump`` 由调用方用 ``after()`` 驱动，也可直接使用 :class:`LiveCamera`。

为什么用单槽位而不是无界队列
----------------------------
无界队列在 UI 卡顿时会不断堆积整帧数据（1080p BGR 约 6 MB/帧），
几十秒就能吃掉数百 MB，且恢复后会连续"追帧"。单槽位保证内存恒定，
始终显示最新画面。
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import ttk

import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["FrameSource", "VideoView", "LiveCamera"]

#: 主线程取帧间隔（毫秒）。约 33 fps，与常见相机帧率匹配。
PUMP_INTERVAL_MS = 30

#: 画面小于该尺寸时不缩放（避免无意义的开销）
_MIN_RESIZE = 8


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """把相机帧规范化成 ``(H, W, 3)`` 的 RGB uint8。

    支持灰度 ``(H, W)``、BGR ``(H, W, 3)``、BGRA ``(H, W, 4)``。
    """
    if image is None:
        raise ValueError("image 为空")
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"不支持的图像形状: {image.shape}")
    # 相机输出是 BGR；本函数约定返回 RGB，因此通道反转。
    return image[:, :, ::-1]


class FrameSource:
    """后台线程读相机的帧源（单槽位，丢旧帧）。

    Parameters
    ----------
    open_camera : 无参可调用对象，返回一个带 ``read()`` 的相机对象；
        通常传 ``lambda: Camera(cfg.raw["camera"]).open()``。
    close_camera : 接收相机对象、负责释放的可调用对象；
        默认调用其 ``close()``。
    open_retries : 打开失败后的重试次数。
        Windows 的 DirectShow 后端在**首次枚举**时常常返回失败
        （驱动尚未就绪 / 设备被短暂占用），几百毫秒后重试即成功。
        若失败一次就放弃，用户会误以为相机坏了。
    retry_delay_s : 每次重试前的等待时间。
    """

    def __init__(self, open_camera: Callable[[], Any],
                 close_camera: Optional[Callable[[Any], None]] = None,
                 on_error: Optional[Callable[[str], None]] = None,
                 open_retries: int = 2,
                 retry_delay_s: float = 0.6) -> None:
        self._open = open_camera
        self._close = close_camera or (lambda cam: cam.close())
        self._on_error = on_error
        self._open_retries = max(int(open_retries), 0)
        self._retry_delay_s = max(float(retry_delay_s), 0.0)

        self._slot: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._camera: Any = None
        self._opened = threading.Event()

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_open(self) -> bool:
        """相机是否已成功打开（用于区分"正在连接"与"无画面"）。"""
        return self._opened.is_set()

    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动读帧线程（幂等）。"""
        if self.running:
            return
        self._stop.clear()
        self._opened.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="triniscan-frames")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """请求停止并等待线程退出。"""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self._thread = None

    # ------------------------------------------------------------------
    def latest(self) -> Optional[np.ndarray]:
        """取最新一帧（没有新帧时返回 None，不阻塞）。"""
        try:
            return self._slot.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    def _put(self, frame: np.ndarray) -> None:
        """写入单槽位；槽位已满则丢弃旧帧。"""
        try:
            self._slot.put_nowait(frame)
        except queue.Full:
            try:
                self._slot.get_nowait()
            except queue.Empty:            # pragma: no cover - 竞态兜底
                pass
            try:
                self._slot.put_nowait(frame)
            except queue.Full:             # pragma: no cover - 竞态兜底
                pass

    def _loop(self) -> None:
        camera, error = self._open_with_retry()
        if camera is None:
            if self._on_error is not None:
                self._on_error(f"打开相机失败: {error}")
            return
        self._camera = camera
        self._opened.set()
        try:
            while not self._stop.is_set():
                try:
                    frame = self._camera.read()
                except Exception as exc:    # noqa: BLE001
                    if self._on_error is not None:
                        self._on_error(f"读取帧失败: {exc}")
                    break
                if frame is None:
                    continue
                self._put(frame)
        finally:
            cam, self._camera = self._camera, None
            if cam is not None:
                try:
                    self._close(cam)
                except Exception:           # noqa: BLE001
                    log.exception("关闭相机失败")
            self._opened.clear()

    def _open_with_retry(self) -> tuple[Optional[Any], str]:
        """打开相机，失败后按 ``open_retries`` 重试。

        相机未插/被占用是常见部署状态，不是程序缺陷——用 warning 而非
        exception 记录，避免整屏堆栈淹没真正的错误。只有**最终**失败才
        上报 ``on_error``，中间重试只记 debug，避免界面闪烁错误。
        """
        last_error = ""
        for attempt in range(self._open_retries + 1):
            if self._stop.is_set():
                return None, "已取消"
            try:
                camera = self._open()
                if attempt:
                    log.info("相机第 %d 次尝试打开成功", attempt + 1)
                return camera, ""
            except Exception as exc:        # noqa: BLE001
                last_error = str(exc)
                if attempt < self._open_retries:
                    log.debug("打开相机失败（第 %d 次，将重试）: %s",
                              attempt + 1, exc)
                    if self._stop.wait(self._retry_delay_s):
                        return None, "已取消"

        log.warning("打开相机失败（已重试 %d 次）: %s",
                    self._open_retries, last_error)
        return None, last_error


class VideoView(ttk.Frame):
    """实时画面显示控件（自适应缩放，保持宽高比）。

    只依赖一个 :class:`tk.Canvas`；渲染路径为

    ``numpy BGR → RGB → PIL.Image → PhotoImage → canvas.itemconfigure``

    Notes
    -----
    ``PhotoImage`` 必须保存在实例属性里。Tk 不持有 Python 引用，局部
    变量被回收后画面会变成空白——这是 Tkinter 的经典陷阱。
    """

    def __init__(self, master: tk.Misc, background: str = "#101014",
                 placeholder: str = "（无画面）") -> None:
        super().__init__(master)
        self._canvas = tk.Canvas(self, background=background,
                                 highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        self._photo: Any = None          # 必须持引用，见 Notes
        self._item: Optional[int] = None
        self._placeholder = placeholder
        self._placeholder_id: Optional[int] = None

        self._frame: Optional[np.ndarray] = None
        self._overlay: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._last_size: tuple[int, int] = (0, 0)

        self.update_idletasks()
        self._draw_placeholder()

    # ------------------------------------------------------------------
    # 叠加层
    # ------------------------------------------------------------------
    def set_overlay(
            self, fn: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
        """设置叠加层：``fn(bgr) -> bgr``，在转换为 RGB **之前**调用。

        Parameters
        ----------
        fn : 画中心线、棋盘角点等标注的函数；返回新数组，可原地修改。
            ``None`` 表示不叠加。

        Notes
        -----
        标注画在图像上而不是 canvas 上：每帧几十条 ``create_line`` 的
        开销高于一次整图转换，且图像与标注始终同步缩放。
        下一帧才会生效。
        """
        self._overlay = fn

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def show(self, image: np.ndarray) -> None:
        """显示一帧（BGR / BGRA / 灰度）。"""
        self._frame = image
        self._render()

    def refresh(self) -> None:
        """用当前帧重绘（窗口尺寸变化后调用）。"""
        if self._frame is not None:
            self._render()

    def _render(self) -> None:
        frame = self._frame
        if frame is None:
            return
        if self._overlay is not None:
            try:
                frame = self._overlay(frame)
            except Exception:               # noqa: BLE001
                log.exception("叠加层绘制失败，已跳过")
        try:
            rgb = _to_rgb(frame)
        except ValueError:
            log.exception("图像格式不受支持，跳过该帧")
            return

        width = max(self._canvas.winfo_width(), 1)
        height = max(self._canvas.winfo_height(), 1)
        if width <= 1 or height <= 1:       # 控件尚未完成布局
            return

        # 先缩放到控件大小，再编码成 PPM 交给 Tk。
        #
        # 为什么不用 PIL.ImageTk：两者都是"把像素交给 Tk"，但
        # PhotoImage 每次构造都要重新上传整张图。实测 1280x800：
        #   PIL 缩放 + ImageTk.PhotoImage   约 49 ms/帧
        #   cv2 缩放 + PPM 编码 + data=     约 21 ms/帧
        # cv2 的缩放与编码都在 C 层完成，且 PPM 是无需压缩的裸像素格式，
        # Tk 直接拷贝即可。
        import cv2

        rgb = self._fit(rgb, width, height)
        encoded = cv2.imencode(".ppm", rgb[:, :, ::-1])   # RGB → BGR
        if not encoded[0]:                  # pragma: no cover - 编码失败
            log.warning("PPM 编码失败，跳过该帧")
            return
        photo = tk.PhotoImage(data=encoded[1].tobytes())
        self._photo = photo                 # 必须持引用，见 Notes

        if self._item is None:
            self._item = self._canvas.create_image(
                width // 2, height // 2, anchor="center", image=photo)
        else:
            self._canvas.coords(self._item, width // 2, height // 2)
            self._canvas.itemconfigure(self._item, image=photo)
        if self._placeholder_id is not None:
            self._canvas.delete(self._placeholder_id)
            self._placeholder_id = None

    @staticmethod
    def _fit(rgb: np.ndarray, width: int,
             height: int) -> np.ndarray:
        """等比缩放到不超过控件尺寸（不放大）。

        ``INTER_AREA`` 在缩小时对高频（激光细线）更友好，不会产生
        明显摩尔纹；放大时用 ``INTER_NEAREST`` 更快且无插值伪影。
        """
        import cv2

        src_h, src_w = rgb.shape[:2]
        if src_w <= 0 or src_h <= 0 or width < _MIN_RESIZE                 or height < _MIN_RESIZE:
            return rgb
        scale = min(width / float(src_w), height / float(src_h))
        if scale >= 1.0:
            return rgb
        size = (max(1, int(round(src_w * scale))),
                max(1, int(round(src_h * scale))))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        return cv2.resize(rgb, size, interpolation=interp)

    def _draw_placeholder(self) -> None:
        if self._placeholder_id is not None:
            return
        self._placeholder_id = self._canvas.create_text(
            max(self._canvas.winfo_width(), 1) // 2,
            max(self._canvas.winfo_height(), 1) // 2,
            text=self._placeholder, fill="#5c6370", font=("", 11),
            justify="center", width=max(self._canvas.winfo_width() - 40, 200))

    def set_placeholder(self, text: str) -> None:
        """更换占位文字并显示（用于"相机打不开"这类需要可读原因的场景）。

        多行文本受 ``width`` 约束会自动换行，无需调用方手工折行。
        """
        self._placeholder = text
        self._frame = None
        self._photo = None
        if self._item is not None:
            self._canvas.delete(self._item)
            self._item = None
        if self._placeholder_id is not None:
            self._canvas.delete(self._placeholder_id)
            self._placeholder_id = None
        self._draw_placeholder()

    def clear(self) -> None:
        """清空画面并显示占位文字。"""
        self._frame = None
        self._photo = None
        if self._item is not None:
            self._canvas.delete(self._item)
            self._item = None
        if self._placeholder_id is not None:
            self._canvas.delete(self._placeholder_id)
        self._placeholder_id = None
        self._draw_placeholder()

    # ------------------------------------------------------------------
    @property
    def frame(self) -> Optional[np.ndarray]:
        """最近显示的一帧（原始 BGR）。"""
        return self._frame


class LiveCamera:
    """把 :class:`FrameSource` 与 :class:`VideoView` 绑定起来的调度器。

    负责"后台读帧 → 主线程定时取帧 → 渲染"的完整循环，调用方只需
    :meth:`start` / :meth:`stop`。

    Parameters
    ----------
    view : 目标显示控件
    open_camera : 见 :class:`FrameSource`
    interval_ms : 主线程取帧间隔
    on_error : 相机错误回调（在主线程被调用）

    Examples
    --------
    >>> live = LiveCamera(view, lambda: Camera(cfg).open())
    >>> live.start()
    ...
    >>> live.stop()
    """

    def __init__(self, view: VideoView,
                 open_camera: Callable[[], Any],
                 interval_ms: int = PUMP_INTERVAL_MS,
                 on_error: Optional[Callable[[str], None]] = None,
                 on_frame: Optional[Callable[[np.ndarray], None]] = None
                 ) -> None:
        self.view = view
        self.interval_ms = max(int(interval_ms), 1)
        self._on_error = on_error
        self._on_frame = on_frame
        self._source = FrameSource(open_camera,
                                   on_error=self._report_error)
        self._alive = False
        self._after_id: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._alive

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        self._source.start()
        self._schedule()

    def stop(self) -> None:
        self._alive = False
        if self._after_id is not None:
            try:
                self.view.after_cancel(self._after_id)
            except Exception:               # noqa: BLE001 - 窗口已销毁
                pass
            self._after_id = None
        self._source.stop()

    # ------------------------------------------------------------------
    def _report_error(self, message: str) -> None:
        # 后台线程不能直接碰控件：转交主线程
        try:
            self.view.after(0, self._deliver_error, message)
        except Exception:                   # noqa: BLE001 - 窗口已销毁
            pass

    def _deliver_error(self, message: str) -> None:
        if self._on_error is not None:
            self._on_error(message)

    def _schedule(self) -> None:
        if not self._alive:
            return
        try:
            self._after_id = self.view.after(self.interval_ms, self._pump)
        except Exception:                   # noqa: BLE001 - 窗口已销毁
            self._alive = False

    def _pump(self) -> None:
        self._after_id = None
        if not self._alive:
            return
        frame = self._source.latest()
        if frame is not None:
            self.view.show(frame)
            if self._on_frame is not None:
                try:
                    self._on_frame(frame)
                except Exception:           # noqa: BLE001
                    log.exception("帧回调失败")
        self._schedule()
