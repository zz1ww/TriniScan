"""标定面板轮询的健壮性测试（需图形环境，无则跳过）。

回归点
------
``_poll`` 曾用 ``latest_frame or self._latest_frame`` 选取帧，而
``latest_frame`` 是 numpy 数组，布尔求值会抛
``ValueError: truth value of an array ... is ambiguous``，导致
``_poll`` 中断且不再 ``after()`` 重排，界面**永久卡死**。

这里验证：
- 队列中同时有帧与拍摄指令时，处理不抛异常、能正确收录；
- 单条消息处理失败不会中断后续轮询（仍会重新排队）。
"""
import numpy as np
import pytest
import tkinter as tk


@pytest.fixture
def panel():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无显示环境，跳过界面测试")
    root.withdraw()
    from triniscan.ui.calibration import CalibrationPanel
    from triniscan.core.config import Config
    try:
        p = CalibrationPanel(root, Config.load())
        root.update()
        yield p, root
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def _chessboard(cols=11, rows=8, sq=70, margin=50):
    h = (rows + 1) * sq + 2 * margin
    w = (cols + 1) * sq + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[margin + r * sq:margin + (r + 1) * sq,
                    margin + c * sq:margin + (c + 1) * sq] = 0
    return np.stack([img] * 3, axis=-1)


def test_frame_then_shoot_does_not_raise(panel):
    """帧与拍摄指令同时到达时不应抛数组歧义异常（回归）。"""
    p, root = panel
    img = _chessboard()
    p._queue.put(("frame", img))
    p._queue.put(("shoot", None))
    p._poll()          # 曾经在此抛出 ValueError
    root.update()
    assert p._ensure_session().count == 1


def test_shoot_without_prior_frame_is_safe(panel):
    """没有取景帧时点拍摄，不应崩溃。"""
    p, root = panel
    p._queue.put(("shoot", None))
    p._poll()
    root.update()
    assert p._ensure_session().count == 0


def test_poll_survives_bad_message(panel):
    """单条消息处理失败不应中断轮询循环。"""
    p, root = panel
    p._queue.put(("unknown-kind", None))     # 未知类型
    p._poll()                                # 不应抛出
    root.update()
    # 之后仍能正常处理新帧
    p._queue.put(("frame", _chessboard()))
    p._queue.put(("shoot", None))
    p._poll()
    root.update()
    assert p._ensure_session().count == 1


def test_repeated_identical_shots_deduped(panel):
    """完全相同的帧会被判为重复而不再收录（预期行为）。"""
    p, root = panel
    img = _chessboard()
    for _ in range(3):
        p._queue.put(("frame", img))
        p._queue.put(("shoot", None))
        p._poll()
        root.update()
    # 第一张收录，后两张因过于相似被拒
    assert p._ensure_session().count == 1
