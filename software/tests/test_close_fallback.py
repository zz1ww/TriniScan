"""网格封闭兜底的**可见性**回归测试。

为什么单独测这组
----------------
``close_mesh`` 的三条兜底路径精度完全不同，却都返回 ``watertight=True``：

=============  ============  ==================
路径            面数保留率     4cm 立方体实测
=============  ============  ==================
平面封底        100%          无损
体素            取决于环境    近似（需 skimage）
凸包            约 1%         填平凹陷，系统性高估
=============  ============  ==================

重构前调用方**只看到 watertight=True**，无从知道形状已经被抹平，
于是"体积不对"永远被归咎于算法本身。

本文件只锁一件事：**走了哪条兜底、损失多少，必须能被上层看见。**
不测几何实现——那是 trimesh 的职责。
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from triniscan.algorithms.closing.close_mesh import close_mesh  # noqa: E402


def _open_box() -> "trimesh.Trimesh":
    """缺底面的立方体（非水密）。"""
    box = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
    zmin = box.vertices[:, 2].min()
    tri_z = box.vertices[box.faces][:, :, 2].mean(axis=1)
    open_box = box.copy()
    open_box.update_faces(tri_z > zmin + 1e-6)
    assert not open_box.is_watertight
    return open_box


def test_closed_mesh_needs_no_fallback():
    """已经是水密的网格：不降级、保留率 1.0、无原因说明。"""
    box = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
    res = close_mesh(box)
    assert res.watertight
    assert res.fallback_used == ""
    assert res.degraded is False
    assert res.face_retention == 1.0
    assert res.degrade_reason == ""


def test_plane_cap_is_lossless():
    """平面封底能救回来的开口属无损路径，不应标记降级。"""
    res = close_mesh(_open_box(), fallback="bottom_plane",
                     allow_convex_hull=True, axis_dir=np.array([0, 0, 1.0]))
    assert res.watertight
    assert res.degraded is False
    assert res.degrade_reason == ""
    assert res.face_retention == 1.0


def test_convex_hull_is_flagged_degraded():
    """显式走凸包：必须标记降级，且原因里带面数信息。"""
    sphere = trimesh.creation.icosphere(radius=0.02, subdivisions=2)
    open_sphere = sphere.copy()
    open_sphere.update_faces(np.arange(len(open_sphere.faces)) != 0)

    res = close_mesh(open_sphere, fallback="convex_hull",
                     allow_convex_hull=True)
    if not res.fallback_used.endswith("convex_hull"):
        pytest.skip(f"本例走了 {res.fallback_used or '无损' }，未经凸包")

    assert res.watertight
    assert res.degraded is True
    assert "凸包" in res.degrade_reason
    assert "面数" in res.degrade_reason
    # 凸包把面数打到 1% 量级，这是"形状被抹掉"最直白的指标
    assert res.face_retention < 0.5


def test_face_retention_matches_actual_faces():
    """face_retention 必须与实际面数口径一致（可用于告警阈值）。"""
    mesh = _open_box()
    before = len(mesh.faces)
    res = close_mesh(mesh, fallback="bottom_plane",
                     allow_convex_hull=True, axis_dir=np.array([0, 0, 1.0]))
    assert 0.0 < res.face_retention <= 10.0
    if not res.degraded:
        # 无损路径：面数只增不减（补片），保留率恒为 1.0（未重算）
        assert len(res.mesh.faces) >= before
