"""体积计算：散度定理法（主）+ 体素法（校核）。

赛题口径：实体部分实际体积，孔洞缺口不计入 → 用封闭网格积分。
"""
from __future__ import annotations

import numpy as np

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False


def compute_volume(mesh) -> float:
    """散度定理法计算体积。

    对水密网格，V = (1/6) Σ v1·(v2×v3)。

    Returns
    -------
    volume_m3 : float
    """
    if not _HAS_TRIMESH:
        raise ImportError("需要 trimesh: pip install trimesh")

    if not mesh.is_watertight:
        raise ValueError("网格非水密，无法可靠计算体积；请先 close_mesh")

    V = float(mesh.volume)  # trimesh 内部即散度定理
    return V


def compute_volume_voxel(mesh, pitch: float = 0.001) -> float:
    """体素法校核体积。

    Parameters
    ----------
    mesh : trimesh.Trimesh（需水密）
    pitch : 体素边长 (m)

    Returns
    -------
    volume_m3 : float
    """
    if not _HAS_TRIMESH:
        raise ImportError("需要 trimesh")

    vox = mesh.voxelized(pitch=pitch)
    return float(vox.volume)


def volume_cm3(volume_m3: float) -> float:
    return volume_m3 * 1e6


if __name__ == "__main__":
    # 自测：单位立方体
    if _HAS_TRIMESH:
        box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        v = compute_volume(box)
        print(f"立方体体积: {v:.6e} m³ = {volume_cm3(v):.3f} cm³ "
              f"(理论 1000 cm³)")
