"""重建任务：把流水线的后半段（提取→三角测量→配准→网格→体积）跑在后台。

为什么不直接调 ``ScanPipeline.run()``
-------------------------------------
``run()`` 的第一段是 :meth:`ScanPipeline.acquire`，它会**打开相机和转台**、
逐站转动并抓图。而在参数调试阶段，用户往往已经有一批采集好的图像
（如 ``data/raw`` 的 720 张），只想验证算法与参数——这时再驱动一遍硬件
既慢又没有必要，且没有硬件时完全跑不起来。

:meth:`ScanPipeline.run` 本身已经把四个阶段切成了独立方法，因此这里
**按原样复用它们**，只跳过第一段：

.. code-block:: text

    acquire()             ← 跳过（或由“完整测量”走原 run()）
    reconstruct_views()   ← 复用
    build_pointcloud()    ← 复用
    measure_volume()      ← 复用

两段代码路径共用同一套方法，不存在"UI 里跑的和命令行跑的不一样"。
"""
from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..common.io_utils import read_image
from ..common.logging_utils import get_logger
from ..core.pipeline import ScanPipeline, ScanResult
from ..hardware.turntable import (TurntableConfig, resolve_scan_plan)

log = get_logger(__name__)

__all__ = ["RebuildJob", "RebuildOutcome", "load_views_from_dir",
           "IMAGE_SUFFIXES"]

IMAGE_SUFFIXES = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")

#: 进度回调：``fn(stage_name, fraction, text)``
ProgressFn = Callable[[str, float, str], None]


# ---------------------------------------------------------------------------
@dataclass
class RebuildOutcome:
    """一次重建的完整产出（比 ``ScanResult`` 多带可视化数据）。"""

    result: Optional[ScanResult] = None
    points: Optional[np.ndarray] = None      # (N, 3) 过滤后的点云
    elapsed_s: float = 0.0
    source: str = ""
    error: str = ""
    #: 本次重建实际使用的每视角转角与视角数（来自扫描计划）。
    #: 事后核对结果时必须能回答"这组点云是用什么角度拼出来的"。
    angle_step_deg: float = 0.0
    num_views_planned: int = 0
    #: 图像张数与扫描计划不一致时的说明（空串表示一致）。
    plan_warning: str = ""

    @property
    def ok(self) -> bool:
        return self.error == "" and self.result is not None


# ---------------------------------------------------------------------------
def load_views_from_dir(raw_dir: str, angle_step_deg: float,
                        limit: Optional[int] = None,
                        progress: Optional[ProgressFn] = None
                        ) -> list[dict]:
    """读取图像目录，构造 :meth:`ScanPipeline.reconstruct_views` 需要的视角表。

    Parameters
    ----------
    raw_dir : 图像目录（按文件名排序，``view_0000.png`` 这种命名）
    angle_step_deg : 每视角转角（度）。由 ``scan.angle_step_deg`` 决定，
        必须与采集时电机实际转角一致，否则拼接会散开。
    limit : 只取前 N 张（调试用）
    progress : 进度回调

    Returns
    -------
    ``[{"index", "angle_deg", "image", "raw"}, ...]``
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"图像目录不存在: {raw_dir}")

    files = sorted(
        f for f in glob.glob(os.path.join(raw_dir, "*"))
        if os.path.splitext(f)[1].lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"目录中没有图像: {raw_dir}")
    if limit:
        files = files[:int(limit)]

    views: list[dict] = []
    total = len(files)
    for i, path in enumerate(files):
        image = read_image(path)
        if image is None:
            log.warning("读取失败，跳过: %s", path)
            continue
        views.append({
            "index": i,
            "angle_deg": float(angle_step_deg) * (i + 1),
            "image": path,
            "raw": image,
        })
        if progress is not None and (i % 20 == 0 or i == total - 1):
            progress("load", (i + 1) / total,
                     f"读取图像 {i + 1}/{total}")

    if not views:
        raise RuntimeError("所有图像都读取失败")
    log.info("已载入 %d 个视角（每视角 %.4f°）", len(views), angle_step_deg)
    return views


# ---------------------------------------------------------------------------
class RebuildJob:
    """一次重建任务的执行体（在 Worker 线程中运行）。

    Parameters
    ----------
    cfg : :class:`~triniscan.core.config.Config`
    raw_dir : 图像目录；为 ``None`` 时走完整测量（含转台采集）
    limit : 最多使用多少张图（``None`` = 全部）
    progress : ``fn(stage, fraction, text)``，由 UI 转成日志/进度条
    """

    #: 阶段权重（用于把各阶段进度合成总进度）
    STAGES = (("load", 0.10), ("triangulate", 0.35),
              ("register", 0.25), ("volume", 0.30))

    def __init__(self, cfg, raw_dir: Optional[str] = None,
                 limit: Optional[int] = None,
                 progress: Optional[ProgressFn] = None) -> None:
        self.cfg = cfg
        self.raw_dir = raw_dir
        self.limit = limit
        self._progress = progress
        self._done_weight = 0.0

    # ------------------------------------------------------------------
    def _report(self, stage: str, fraction: float, text: str) -> None:
        if self._progress is not None:
            self._progress(stage, float(fraction),
                           text or self._stage_label(stage))

    def _advance(self, stage: str, fraction: float = 1.0) -> None:
        """把某阶段标记完成，并推进总进度。"""
        weight = dict(self.STAGES).get(stage, 0.0)
        self._done_weight = min(1.0, self._done_weight + weight * fraction)
        self._report(stage, self._done_weight, "")

    @staticmethod
    def _stage_label(stage: str) -> str:
        return {
            "load": "读取图像",
            "triangulate": "提取 + 三角测量",
            "register": "配准 + 过滤",
            "volume": "网格 + 体积",
            "acquire": "采集图像",
        }.get(stage, stage)

    # ------------------------------------------------------------------
    def _resolve_angle(self) -> tuple[float, str]:
        """现场计算本次应使用的每视角转角。

        为什么不用 ``scan.angle_step_deg``
        ----------------------------------
        该键是 ``apply_scan_plan`` 在**采集时**回写的产物。从图像重建
        跳过采集，若用户之后在界面改了视角数，这个键仍是旧值，直接用
        它会让各视角按错误角度拼接（点云沿圆周错位）。

        因此这里按**当前配置**重算一次扫描计划——与 ``acquire`` 调用
        的是同一个 :func:`resolve_scan_plan`，保证"界面显示的"与
        "重建实际用的"一致。

        Returns
        -------
        (angle_step_deg, warning) : 警告非空时表示整圈无法精确闭合。
        """
        try:
            table_cfg = TurntableConfig.from_config(
                self.cfg.raw.get("turntable", {}) or {})
            plan = resolve_scan_plan(self.cfg.raw.get("scan", {}) or {},
                                     table_cfg)
        except Exception as exc:            # noqa: BLE001
            fallback = float(self.cfg.get("scan.angle_step_deg", 0.5))
            log.warning("扫描计划解析失败（%s），回退 angle_step_deg=%.6f",
                        exc, fallback)
            return fallback, f"扫描计划解析失败：{exc}"

        self._planned_views = plan.num_views
        warning = ""
        if not plan.full_turn_exact:
            warning = (f"视角数 {plan.num_views} 无法整除整圈微步数："
                       f"累计 {plan.total_deg:.4f}° "
                       f"(差 {plan.full_turn_error_deg:+.4f}°)，"
                       f"拼接可能出现接缝")
        return plan.angle_step_deg, warning

    def _check_view_count(self, count: int) -> str:
        """图像张数与扫描计划不符时给出说明（低价但高价值的校验）。"""
        planned = getattr(self, "_planned_views", 0)
        if not planned or count >= planned or self.limit:
            return ""
        return (f"实际使用 {count} 张图，而扫描计划为 {planned} 视角；"
                f"缺 {planned - count} 个视角，点云将不完整")

    # ------------------------------------------------------------------
    def run(self, _report: Optional[Callable] = None) -> RebuildOutcome:
        """执行重建，返回 :class:`RebuildOutcome`（异常会向上抛给 Worker）。

        ``_report`` 是 :class:`~triniscan.ui.worker.Worker` 传入的进度回调
        协议参数，这里不使用——进度已经通过构造时的 ``progress`` 回调
        上报，保持单一通道。
        """
        t0 = time.time()
        result = RebuildOutcome(source=self.raw_dir or "完整测量")

        pipe = ScanPipeline(self.cfg)
        self._report("load", 0.0, "载入标定…")
        pipe.load_calibrations()

        # ---- 阶段 1：取图 ----
        if self.raw_dir is None:
            self._report("acquire", 0.0, "开始完整测量（含转台采集）…")
            views = pipe.acquire()
            self._advance("load")
        else:
            step, plan_warning = self._resolve_angle()
            result.angle_step_deg = step
            result.num_views_planned = getattr(self, "_planned_views", 0)
            result.plan_warning = plan_warning
            if plan_warning:
                log.warning("%s", plan_warning)
            views = load_views_from_dir(
                self.raw_dir, step, self.limit,
                progress=lambda s, f, t: self._report(
                    "load", 0.1 * f, t))
            self._advance("load")
            mismatch = self._check_view_count(len(views))
            if mismatch:
                log.warning("%s", mismatch)
                result.plan_warning = (plan_warning + chr(10) + mismatch).strip()

        # ---- 阶段 2：提取 + 三角测量 ----
        self._report("triangulate", self._done_weight,
                     f"提取中心线（{len(views)} 视角）…")
        clouds = pipe.reconstruct_views(views)
        if not clouds:
            raise RuntimeError("没有有效视角：请检查提取参数与标定是否匹配")
        self._advance("triangulate")

        # ---- 阶段 3：配准 + 过滤链 ----
        self._report("register", self._done_weight, "配准并拼接点云…")
        pcd, _pc_path = pipe.build_pointcloud(clouds)
        points = np.asarray(pcd.points, dtype=np.float64)
        result.points = points
        self._advance("register")

        # ---- 阶段 4：网格 + 体积 ----
        self._report("volume", self._done_weight, "重建网格并计算体积…")
        vol, closed, mesh_path = pipe.measure_volume(pcd)
        self._advance("volume")

        # measure_volume 返回的是 (vol, closed, mesh_path)，但体积等元数据
        # 需要 ScanResult 统一汇总。这里有两条路：
        #   (a) 调 run() 会重新采集；
        #   (b) 复用 run() 里的汇总逻辑。
        # 选 (b)：把已算好的阶段结果填进 ScanResult，保证字段与 run() 一致。
        result.result = self._assemble(pipe, clouds, vol, closed,
                                        mesh_path, t0)
        result.elapsed_s = time.time() - t0
        self._report("volume", 1.0, "完成")
        return result

    # ------------------------------------------------------------------
    def _assemble(self, pipe: ScanPipeline, clouds: list, vol, closed,
                  mesh_path: Optional[str], t0: float) -> ScanResult:
        """按 ``ScanPipeline.run()`` 的同一口径汇总结果。

        直接调用 ``run()`` 会重复采集，因此这里复用 pipeline 在
        ``build_pointcloud`` / ``measure_volume`` 期间留下的字段
        （``_last_*``），字段含义与 ``run()`` 完全一致。
        """
        masks = getattr(pipe, "_last_masks", None)
        if masks is not None:
            layers = masks.to_dict()["layers"]
            summary = masks.summary()
            n_before, n_after = masks.n, masks.n_keep
        else:
            layers, summary = [], ""
            n_before = n_after = 0

        elapsed = time.time() - t0
        tt = getattr(pipe, "_last_turntable", {"applied": False,
                                               "removed": 0})
        return ScanResult(
            volume_cm3=float(getattr(vol, "volume_cm3", 0.0)),
            volume_m3=float(getattr(vol, "volume_m3", 0.0)),
            elapsed_s=elapsed,
            num_views=len(clouds),
            num_points=int(sum(len(c.points) for c in clouds)),
            mesh_watertight=bool(getattr(closed, "watertight", False)),
            cross_check_cm3=(float(vol.cross_check_m3) * 1e6
                             if getattr(vol, "cross_check_m3", None)
                             else None),
            hull_volume_cm3=getattr(pipe, "_last_hull_volume_cm3", None),
            stages_s={},
            mesh_path=mesh_path,
            points_path=None,
            cross_check_diff=getattr(vol, "relative_diff", None),
            register_method=str(getattr(pipe, "_last_register_method", "")),
            turntable_removed=int(tt.get("removed", 0)),
            turntable_applied=bool(tt.get("applied", False)),
            close_fallback=str(getattr(pipe, "_last_close_fallback", "")),
            close_degraded=bool(getattr(pipe, "_last_close_degraded", False)),
            close_degrade_reason=str(getattr(
                pipe, "_last_close_degrade_reason", "")),
            mesh_face_retention=float(getattr(pipe, "_last_face_retention",
                                              1.0)),
            filter_layers=layers,
            filter_summary=summary,
            points_before_filter=n_before,
            points_after_filter=n_after,
        )
