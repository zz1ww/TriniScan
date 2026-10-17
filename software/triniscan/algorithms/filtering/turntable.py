"""去转盘：剔除载物台表面点。

两种方法（``reconstruction.turntable_method``）：

``axis_height``（推荐）
    直接用转轴标定体系的**台面高度** ``h_table`` 切割，不依赖点云里
    的平面结构。对"物体有大面积竖直面"（会干扰 RANSAC 选面）尤其稳健。

    注意：``h_table`` 未显式配置时，由 :func:`estimate_table_height`
    **自动猜测**——它取"下半部分点云中 h 最小的密集层"作为台面。
    如果物体底面本身就密，猜出来的可能是**物体底面**而不是台面，
    于是切掉的是物体最底下一层。这是隐藏风险，报告里会记录
    ``h_table_estimated`` 与 ``h_table_source``，便于发现。

``ransac``
    RANSAC 拟合主平面 + 法向校验（与轴夹角超过 ``align_tol_deg`` 的
    平面不认）。旧默认。

行为与重构前 ``ScanPipeline._remove_turntable`` 完全一致。
"""
from __future__ import annotations

import numpy as np

from ...contracts import Cloud, FILTERS, ParamSpec, RunContext
from .base import FilterPlugin
from .turntable_removal import (estimate_table_height,
                                remove_turntable_axis_height,
                                remove_turntable_surface)

__all__ = ["TurntableFilter"]


@FILTERS.register("turntable")
class TurntableFilter(FilterPlugin):
    """转盘表面去除。"""

    name = "turntable"
    label = "去转盘"
    # 台面高度/平面拟合都依赖当前点云分布 → 与顺序耦合
    depends_on_current = True
    params = (
        ParamSpec("reconstruction.turntable_method", str, "axis_height",
                  choices=("axis_height", "ransac"),
                  label="方法", help="axis_height: 轴基准高度（推荐）；"
                                    "ransac: 平面拟合", short="method"),
        ParamSpec("reconstruction.turntable_height_m", float, None,
                  label="台面高度",
                  help="沿轴高度(m)。留空则由点云自动估计（有猜错风险）", short="height_m"),
        ParamSpec("reconstruction.turntable_margin_m", float, 0.0005,
                  low=0.0, high=0.02, unit="m",
                  label="台面上方余量", help="h_table 上方仍视为台面的厚度", short="margin_m"),
        ParamSpec("reconstruction.turntable_band_below_m", float, 0.0015,
                  low=0.0, high=0.05, unit="m",
                  label="台面下方厚度", help="台面下方一并剔除的厚度", short="band_below_m"),
        ParamSpec("reconstruction.turntable_radius_m", float, None,
                  low=0.0, high=2.0, unit="m",
                  label="剔除半径", help="只剔除距轴此半径内的点；空表示不限", short="radius_m"),
        ParamSpec("reconstruction.turntable_plane_tol_m", float, 0.001,
                  low=0.0, high=0.02, unit="m",
                  label="[RANSAC] 平面容差", short="plane_tol_m"),
        ParamSpec("reconstruction.turntable_min_inlier", float, 0.15,
                  low=0.0, high=1.0, label="[RANSAC] 最小内点比例", short="min_inlier"),
        ParamSpec("reconstruction.turntable_align_tol_deg", float, 15.0,
                  low=0.0, high=90.0, unit="deg",
                  label="[RANSAC] 法向夹角容差", short="align_tol_deg"),
        ParamSpec("reconstruction.turntable_keep_upper_only", bool, True,
                  label="[RANSAC] 只保留平面以上", short="keep_upper_only"),
    )

    def compute_mask(self, cloud: Cloud, ctx: RunContext
                     ) -> tuple[np.ndarray, dict]:
        params = self.read_params(ctx.cfg)
        n = len(cloud)
        if n == 0:
            return np.zeros(0, dtype=bool), {"reason": "空点云",
                                             "applied": False}
        point, axis = _axis(ctx)
        method = str(params["method"]).lower()
        radius = params.get("radius_m")
        margin = float(params["margin_m"])

        if method == "axis_height":
            h_table = params.get("height_m")
            source = "配置"
            if h_table is None:
                h_table = estimate_table_height(cloud.points, point, axis)
                source = "自动估计"
            if h_table is None:
                return np.ones(n, dtype=bool), {
                    "reason": "无法估计 h_table，跳过", "applied": False}
            res = remove_turntable_axis_height(
                cloud.points, point, axis, float(h_table),
                margin=margin,
                band_below=float(params["band_below_m"]),
                radius=(float(radius) if radius else None),
            )
        else:
            h_table = None
            source = "ransac"
            res = remove_turntable_surface(
                cloud.points,
                axis_dir=axis, axis_point=point,
                ransac_dist=float(params["plane_tol_m"]),
                min_inlier_ratio=float(params["min_inlier"]),
                align_tol_deg=float(params["align_tol_deg"]),
                margin=margin,
                band_below=float(params["band_below_m"]),
                keep_upper_only=bool(params["keep_upper_only"]),
                radius=(float(radius) if radius else None),
            )

        # 旧实现：res.points 是"保留点"，需要转回掩膜
        keep = np.ones(n, dtype=bool)
        if res.removed_mask is not None and len(res.removed_mask) == n:
            keep = ~np.asarray(res.removed_mask, dtype=bool)
        elif res.num_removed > 0:
            keep = _mask_by_coords(cloud.points, res.points)

        applied = bool(res.applied and res.num_removed > 0)
        return keep, {
            "applied": applied,
            "method": method,
            "h_table_m": h_table,
            "h_table_source": source,
            "removed": int(res.num_removed),
            "reason": res.reason or "无点可剔除",
        }


# ---------------------------------------------------------------------------
def _axis(ctx: RunContext):
    if ctx.axis_point is not None and ctx.axis_dir is not None:
        p = np.asarray(ctx.axis_point, dtype=np.float64).reshape(3)
        d = np.asarray(ctx.axis_dir, dtype=np.float64).reshape(3)
    elif ctx.calib is not None:
        p = ctx.calib.axis_point
        d = ctx.calib.axis_dir
    else:
        raise ValueError("去转盘需要转轴，但 ctx 中没有 axis_point/axis_dir")
    nrm = np.linalg.norm(d)
    return p, (d / nrm if nrm > 1e-12 else d)


def _mask_by_coords(points: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """按坐标反推掩膜（结果对象没给 removed_mask 时的兜底）。"""
    if len(kept) == 0:
        return np.zeros(len(points), dtype=bool)
    key = np.ascontiguousarray(np.round(points, 9)).view(
        np.dtype((np.void, points.dtype.itemsize * 3))).ravel()
    key_keep = set(np.ascontiguousarray(np.round(kept, 9)).view(
        np.dtype((np.void, kept.dtype.itemsize * 3))).ravel().tolist())
    return np.array([k.tobytes() in key_keep for k in key], dtype=bool)
