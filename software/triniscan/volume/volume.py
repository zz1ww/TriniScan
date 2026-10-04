"""体积计算：散度定理法（主）+ 体素法（校核）。

方法
----
1. **散度定理法**（推荐）
   对水密三角网格，体积为各三角面片的带符号体积之和::

       V = (1/6) Σ v1 · (v2 × v3)

   要求：网格水密 + 法向一致朝外。
   精度高、速度快。

2. **体素法**（校核）
   把网格体素化，统计内部体素::

       V ≈ N_inside × pitch³

   直观、对复杂孔洞友好，但精度受体素尺寸限制。
   用于交叉验证散度定理结果。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.logging_utils import get_logger

log = get_logger(__name__)

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:  # pragma: no cover
    trimesh = None
    _HAS_TRIMESH = False

__all__ = [
    "VolumeResult",
    "compute_volume",
    "compute_volume_divergence",
    "compute_volume_voxel",
    "volume_m3_to_cm3",
]


@dataclass
class VolumeResult:
    """体积计算结果。"""

    volume_m3: float
    method: str
    watertight: bool
    num_triangles: int = 0
    cross_check_m3: float | None = None     # 体素法校核值
    relative_diff: float | None = None      # |主-校核| / 主

    @property
    def volume_cm3(self) -> float:
        return self.volume_m3 * 1e6


def _require_trimesh() -> None:
    if not _HAS_TRIMESH:
        raise ImportError("需要 trimesh: pip install trimesh")


def _as_trimesh(mesh) -> "trimesh.Trimesh":
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if hasattr(mesh, "vertices") and hasattr(mesh, "triangles"):
        from .close_mesh import mesh_from_open3d
        return mesh_from_open3d(mesh)
    raise TypeError(f"不支持的网格类型: {type(mesh)}")


def compute_volume_divergence(mesh,
                              require_watertight: bool = True) -> float:
    """散度定理法计算体积，返回 m³。

    Parameters
    ----------
    mesh : trimesh.Trimesh（或 Open3D 网格）
    require_watertight : 非水密时报错（推荐 True，避免错误结果）
    """
    _require_trimesh()
    tm = _as_trimesh(mesh)

    if require_watertight and not tm.is_watertight:
        raise ValueError("网格非水密，无法可靠计算体积；请先 close_mesh()")

    return float(abs(tm.volume))


def compute_volume_voxel(mesh, pitch: float = 0.001) -> float:
    """体素法校核体积，返回 m³。"""
    _require_trimesh()
    tm = _as_trimesh(mesh)
    vox = tm.voxelized(pitch=pitch).fill()
    return float(abs(vox.volume))


def compute_volume(mesh,
                   method: str = "divergence",
                   voxel_pitch: float = 0.001,
                   cross_check: bool = True,
                   require_watertight: bool = True) -> VolumeResult:
    """计算网格体积（统一入口）。

    Parameters
    ----------
    mesh : trimesh.Trimesh（或 Open3D 网格）
    method : ``"divergence"`` 或 ``"voxel"``
    voxel_pitch : 体素法体素尺寸 (m)
    cross_check : 散度定理时是否用体素法交叉校核
    require_watertight : 散度定理是否要求水密

    Returns
    -------
    VolumeResult
    """
    _require_trimesh()
    tm = _as_trimesh(mesh)

    if method == "voxel":
        vol = compute_volume_voxel(tm, voxel_pitch)
        return VolumeResult(
            volume_m3=vol, method="voxel",
            watertight=bool(tm.is_watertight),
            num_triangles=len(tm.faces),
        )

    vol = compute_volume_divergence(tm, require_watertight)
    result = VolumeResult(
        volume_m3=vol, method="divergence",
        watertight=bool(tm.is_watertight),
        num_triangles=len(tm.faces),
    )

    if cross_check:
        try:
            vox = compute_volume_voxel(tm, voxel_pitch)
            result.cross_check_m3 = vox
            if vol > 0:
                result.relative_diff = abs(vox - vol) / vol
            log.info(
                "体积: %.3f cm³ (散度定理), 校核 %.3f cm³ (体素法), "
                "相对差 %.2f%%",
                vol * 1e6, vox * 1e6,
                (result.relative_diff or 0) * 100,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("体素法校核失败: %s", exc)
    else:
        log.info("体积: %.3f cm³ (散度定理)", vol * 1e6)

    return result


def volume_m3_to_cm3(volume_m3: float) -> float:
    """m³ → cm³。"""
    return volume_m3 * 1e6
