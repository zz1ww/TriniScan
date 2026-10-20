"""单件测量主流程（Pipeline）。

职责
----
把各模块串成完整测量链路，并对**单件测量总时限（<=60s）**负责：

    转-停-拍 → 中心线提取 → 三角测量 → 多视角配准（全局对齐）
    → 去转盘表面 → 网格重建 → 封闭化 → 体积计算

设计
----
- 每个阶段独立方法，便于单测与替换；
- 采集结果与中间数据可选落盘（便于复现与调试）；
- 全程记录耗时，超时给出告警；
- **标定加载即校验质量**（避免坏标定静默通过），并可严格模式中止。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common.async_writer import AsyncImageWriter
from ..hardware.camera import Camera
from ..hardware.turntable import (
    Turntable,
    TurntableConfig,
    apply_scan_plan,
    resolve_scan_plan,
)
from ..algorithms.extraction import extract_centerline
from ..algorithms.features.triangulate import triangulate
from ..algorithms.registration.register import ViewCloud, register_views
from ..algorithms.meshing.mesh import reconstruct_mesh
from ..algorithms.filtering import (FilterChain, dominant_band,
                                    estimate_table_height, filter_by_bounds,
                                    filter_by_depth, remove_small_clusters,
                                    remove_turntable_axis_height,
                                    remove_turntable_surface)
from ..algorithms.closing.close_mesh import close_mesh
from ..algorithms.volumetry.volume import compute_volume
from ..algorithms.calibration import (
    AxisCalibResult,
    load_axis_calib,
    load_camera_calib,
    load_laser_calib,
    validate_calibrations,
    format_reports,
    reports_to_dict,
)
from ..algorithms.registration.specs import SEQUENTIAL_SPEC
from ..algorithms.extraction.specs import centerline_view
from ..algorithms.meshing.specs import POISSON_SPEC
from ..algorithms.closing.specs import CLOSE_SPEC
from ..algorithms.volumetry.specs import (DIVERGENCE_SPEC, VOXEL_SPEC)
from ..algorithms.calibration.specs import quality_thresholds
from ..contracts import CalibrationBundle, Cloud, RunContext, ScanReport
from ..common.io_utils import ensure_dir
from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ScanPipeline", "ScanResult"]


def _table_supports_steps(table) -> bool:
    """探测固件是否支持 ``STEPS`` 指令。

    默认假定支持（现代固件均实现）；若握手/查询失败则回退按角度驱动。
    这里采用保守策略：仅当明确探测到不支持时才回退。
    """
    return hasattr(table, "rotate_steps")



#: 异步写图队列的默认深度（张）。30 × 3 MB ≈ 90 MB 峰值内存。
_DEFAULT_WRITER_QUEUE = 30


def _writer_queue_size(cfg) -> int:
    """异步写图队列深度。

    配置项 ``output.writer_queue`` 可覆盖；缺省 30。队列满时采集端会被
    阻塞（背压），这是有意为之——宁可慢一点，也不能在慢磁盘上把内存
    吃光（720 张 1280×800 约 2.1 GB）。
    """
    try:
        value = int(cfg.get("output.writer_queue", _DEFAULT_WRITER_QUEUE))
    except (TypeError, ValueError):
        return _DEFAULT_WRITER_QUEUE
    return max(1, value)


def _dominant_band(along: np.ndarray,
                   bins: int = 120,
                   density_ratio: float = 0.10) -> Optional[tuple]:
    """【兼容别名】→ :func:`triniscan.algorithms.filtering.dominant_band`。

    实现已迁到 filtering 包（那里是轴向裁剪插件的正主）。保留此别名是
    为了让旧测试/旧脚本 ``from triniscan.core.pipeline import _dominant_band``
    继续可用。
    """
    return dominant_band(along, bins=bins, density_ratio=density_ratio)


def _to_open3d(cloud) -> "object":
    """把 :class:`Cloud` 转成 Open3D 点云（惰性 import，避免拖慢启动）。"""
    import open3d as o3d

    points = cloud.points if isinstance(cloud, Cloud) else cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64).reshape(-1, 3))
    return pcd


@dataclass
class ScanResult:
    """一次完整测量的结果。"""

    volume_cm3: float
    volume_m3: float
    elapsed_s: float
    num_views: int
    num_points: int
    mesh_watertight: bool
    cross_check_cm3: Optional[float] = None
    hull_volume_cm3: Optional[float] = None       # 凸包体积（凸体更准）
    stages_s: dict[str, float] = field(default_factory=dict)
    mesh_path: Optional[str] = None
    points_path: Optional[str] = None
    # --- 质量与诊断信息 ---
    cross_check_diff: Optional[float] = None      # |主-校核|/主
    register_method: str = ""                      # 配准方法
    turntable_removed: int = 0                     # 去除的转盘点数
    turntable_applied: bool = False
    close_fallback: str = ""                       # 封闭兜底策略
    #: 封闭是否走了**有损**降级（体素/凸包）。True 时体积只能作参考。
    close_degraded: bool = False
    #: 降级说明（含面数保留率），直接可展示
    close_degrade_reason: str = ""
    #: 兜底后 / 兜底前的面数比。凸包通常 < 0.01
    mesh_face_retention: float = 1.0
    calib_reports: list[dict] = field(default_factory=list)
    # --- 过滤诊断（来自 MaskStack）---
    #: 每层过滤的结构化摘要，字段见 :meth:`MaskStack.to_dict`
    filter_layers: list[dict] = field(default_factory=list)
    #: 预格式化的过滤表（直接给 UI / 日志，避免各处重造格式）
    filter_summary: str = ""
    #: 原始点数（过滤前）与最终保留点数
    points_before_filter: int = 0
    points_after_filter: int = 0
    #: 标定指纹：同一指纹下两次扫描的差异才是算法差异
    calib_fingerprint: str = ""


class ScanPipeline:
    """转台扫描测量流水线。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._cam_calib = None
        self._laser_calib = None
        self._axis_calib = None
        self._calib_reports: list = []
        self._last_hull_volume_cm3: Optional[float] = None
        # 过滤链产物（新）
        self._last_masks = None
        self._last_filter_report: Optional[ScanReport] = None
        self._last_close_degraded: bool = False
        self._last_close_degrade_reason: str = ""
        self._last_face_retention: float = 1.0
        # 旧读取路径（由 _sync_legacy_filter_state 从 MaskStack 投影而来）
        self._last_bounds: dict = {"applied": False, "removed": 0}
        self._last_turntable: dict = {"applied": False, "removed": 0}

    # ------------------------------------------------------------------
    # 标定加载
    # ------------------------------------------------------------------
    def load_calibrations(self) -> None:
        """载入三类标定，并**立即做质量校验**。

        质量阈值由 :mod:`triniscan.calibration.quality` 统一定义，
        可通过配置覆盖；``calibration.strict=true`` 时不合格将中止。

        转轴有两种来源（``calibration.axis_mode``）：

        - ``"z_axis"``（默认）：**不标定**，直接假设转轴为相机系下的
          Z 轴（方向 ``z_axis_dir``、过点 ``z_axis_point``）。这是成熟
          开源转台扫描器的通行做法——转台轴本就是物理竖直轴，只要物体
          在转台中心、相机大致水平，用 Z 轴即可，省去易出错的球心标定。
        - ``"calibrated"``：从 ``calibration.axis_file`` 载入实测轴。
        """
        self._cam_calib = load_camera_calib(
            self.cfg.resolve("calibration.camera_file"))
        self._laser_calib = load_laser_calib(
            self.cfg.resolve("calibration.laser_file"))

        axis_mode = str(self.cfg.get("calibration.axis_mode", "z_axis"))
        if axis_mode == "z_axis":
            self._axis_calib = self._synthetic_z_axis()
            axis_for_quality = None      # 合成轴无需质量校验
        else:
            self._axis_calib = load_axis_calib(
                self.cfg.resolve("calibration.axis_file"))
            axis_for_quality = self._axis_calib

        thresholds = self._quality_thresholds()
        strict = bool(self.cfg.get("calibration.strict", False))
        self._calib_reports = validate_calibrations(
            camera=self._cam_calib, laser=self._laser_calib,
            axis=axis_for_quality, thresholds=thresholds, strict=strict,
        )
        log.info("\n%s", format_reports(self._calib_reports))
        if axis_mode == "z_axis":
            log.info(
                "转轴: z_axis 模式（不标定）— dir=%s, point=%s",
                np.round(self._axis_calib.direction, 4).tolist(),
                np.round(self._axis_calib.point, 4).tolist(),
            )
        else:
            log.info(
                "转轴: calibrated 模式 — RMS=%.3f mm",
                self._axis_calib.residual_rms * 1e3,
            )
        log.info(
            "标定已载入: 相机 RMS=%.3f px, 激光平面 RMS=%.3f mm",
            self._cam_calib.rms, self._laser_calib.residual_rms * 1e3,
        )

    def _synthetic_z_axis(self) -> AxisCalibResult:
        """构造"相机系 Z 轴"的合成本征轴（不标定模式）。"""
        point = np.asarray(
            self.cfg.get("calibration.z_axis_point", [0.0, 0.0, 0.0]),
            dtype=np.float64).reshape(3)
        direction = np.asarray(
            self.cfg.get("calibration.z_axis_dir", [0.0, 0.0, 1.0]),
            dtype=np.float64).reshape(3)
        nrm = np.linalg.norm(direction)
        if nrm < 1e-12:
            raise ValueError("calibration.z_axis_dir 不能为零向量")
        direction = direction / nrm
        return AxisCalibResult(
            point=point, direction=direction,
            radius=0.0, residual_rms=0.0,
            num_points=0, num_inliers=0, outlier_ratio=0.0,
            method="z_axis",
        )

    def _quality_thresholds(self) -> dict:
        """标定质量阈值覆盖（缺省用 quality 模块默认值）。

        实现见 :func:`triniscan.algorithms.calibration.specs.quality_thresholds`
        ——键名来自 ``ParamSpec`` 声明，拼写错误在 import 期暴露，而不是
        静默地"阈值不生效"。
        """
        return quality_thresholds(self.cfg)

    # ------------------------------------------------------------------
    # 阶段 1：采集
    # ------------------------------------------------------------------
    def acquire(self) -> list[dict]:
        """转-停-拍，采集各视角图像。

        角度计划
        --------
        按 ``scan.preset``（720/1080）或 ``scan.num_views`` 生成**整数步**
        的角度计划（:func:`triniscan.turntable.plan_scan`），并把实际
        每视角转角写回配置 ``scan.angle_step_deg``，保证"电机走的角"
        与"拼接用的角"严格一致，避免角度累计漂移导致点云炸开。
        """
        raw_dir = ensure_dir(self.cfg.resolve("output.raw_dir"))

        # 生成自洽的扫描计划（整数步 + 回算角度）
        table_cfg = TurntableConfig.from_config(self.cfg.raw["turntable"])
        plan = resolve_scan_plan(self.cfg.raw["scan"], table_cfg)
        apply_scan_plan(self.cfg, plan)
        step = plan.angle_step_deg
        n = plan.num_views
        log.info("扫描计划: %d 视角, 每视角 %d 微步 = %.4f° "
                 "(累计 %.3f°, 与 360° 差 %+.3f°)",
                 n, plan.steps_per_step, step, plan.total_deg,
                 plan.full_turn_error_deg)

        views = []
        with Camera(self.cfg.raw["camera"]) as cam, \
                Turntable(self.cfg.raw["turntable"]) as table, \
                AsyncImageWriter(queue_size=_writer_queue_size(self.cfg)) as writer:
            # 归零可能持续数十秒，固件同样在"开始"时就回 OK——
            # Turntable.home() 内部已轮询 STATUS 等 busy=0。
            table.home()
            # 有 plan 时用整数步直接驱动，彻底避免固件角度换算误差；
            # 固件不支持 STEPS 时回退到角度指令。
            use_steps = _table_supports_steps(table)
            if not use_steps:
                log.warning("固件不支持 STEPS，回退按角度驱动（可能有量化误差）")
            t_start = time.perf_counter()
            for i in range(n):
                if use_steps:
                    # rotate_steps 返回时已确认固件 busy=0 且补足 settle_ms，
                    # 即"转台已静止"，此时抓图不会拍进转动过程。
                    table.rotate_steps(plan.steps_per_step)
                else:
                    table.rotate_deg(step)
                path = os.path.join(raw_dir, f"view_{i:04d}.png")
                image = cam.grab()
                # 落盘交给后台线程：写图不占相机、不占转台，不应阻塞采集。
                writer.submit(path, image)
                views.append({
                    "index": i,
                    "angle_deg": step * (i + 1),
                    "image": path,
                    "raw": image,
                })
                if (i + 1) % 10 == 0 or i + 1 == n:
                    elapsed = time.perf_counter() - t_start
                    per = elapsed / (i + 1)
                    log.debug("采集 %d/%d (%.3f°)  %.1f ms/张，"
                              "预计剩余 %.0f s",
                              i + 1, n, step * (i + 1), per * 1000,
                              per * (n - i - 1))
            # with 退出时排空队列：此刻起所有图片保证已写盘。
            elapsed = time.perf_counter() - t_start
        log.info("采集完成: %d 视角，用时 %.1f s（%.1f ms/张）",
                 len(views), elapsed, elapsed / max(1, len(views)) * 1000)
        return views

    # ------------------------------------------------------------------
    # 阶段 2：提取 + 三角测量
    # ------------------------------------------------------------------
    def reconstruct_views(self, views: list[dict]) -> list[ViewCloud]:
        """对每个视角提取中心线并三角测量为三维点。

        三角测量后立即施加**相机系深度门限**（``reconstruction.depth_filter``），
        在最早的环节剔除环境光/反光造成的远点——这是"飞碟"点云的主因，
        越早剔除，后续拼接/网格越干净。
        """
        if self._cam_calib is None:
            self.load_calibrations()

        K = self._cam_calib.K
        dist = self._cam_calib.dist
        plane = self._laser_calib.plane
        # 提取参数：从声明读取（带类型转换与范围校验）。
        # 重构前直接传 cfg.raw["extraction"]，越界值要跑到算法内部才报错。
        ex_cfg = centerline_view(self.cfg)
        min_pts = int(self.cfg.get("extraction.min_points", 50))

        # 深度门限配置（相机系，单位 m）
        use_depth = bool(self.cfg.get("reconstruction.depth_filter", False))
        depth_min = float(self.cfg.get("reconstruction.depth_min_m", 0.0))
        depth_max = float(self.cfg.get("reconstruction.depth_max_m", 0.0))

        clouds = []
        n_removed = 0
        for v in views:
            pixels = extract_centerline(v["raw"], ex_cfg)
            points, mask = triangulate(pixels, K, plane, dist,
                                       return_mask=True)
            if use_depth and len(points):
                fr = filter_by_depth(points, depth_min=depth_min,
                                     depth_max=depth_max)
                points = fr.points
                n_removed += fr.num_removed
            if len(points) < min_pts:
                log.warning("视角 %s: 有效点不足 (%d)，跳过",
                            v["index"], len(points))
                continue
            clouds.append(ViewCloud(angle_deg=v["angle_deg"], points=points))
        if use_depth and n_removed:
            log.info("深度过滤合计剔除 %d 点（%d 视角）",
                     n_removed, len(views))
        log.info("三角测量完成: %d/%d 视角有效", len(clouds), len(views))
        return clouds

    # ------------------------------------------------------------------
    # 阶段 3：配准（全局对齐）+ 去转盘
    # ------------------------------------------------------------------
    def build_pointcloud(self, clouds: list[ViewCloud]):
        if self._axis_calib is None:
            self.load_calibrations()
        reg_params = SEQUENTIAL_SPEC.read(self.cfg)
        reg = register_views(
            clouds,
            self._axis_calib.point,
            self._axis_calib.direction,
            return_result=True,
            **reg_params,
        )
        log.debug("配准参数: %s", reg_params)

        # --- 过滤链（边界 / 连通域 / 去转盘 / 轴向裁剪 …）---
        cloud, masks, ctx = self._run_filter_chain(reg.pointcloud)
        self._last_masks = masks
        self._last_filter_report = ctx.report

        pcd = _to_open3d(cloud)
        pc_dir = ensure_dir(self.cfg.resolve("output.pointcloud_dir"))
        pc_path = os.path.join(pc_dir, "cloud.ply")
        try:
            import open3d as o3d
            o3d.io.write_point_cloud(pc_path, pcd)
        except Exception as exc:  # noqa: BLE001
            log.warning("点云保存失败: %s", exc)
            pc_path = None

        self._last_register_method = reg.method
        self._sync_legacy_filter_state(masks)
        return pcd, pc_path

    # ------------------------------------------------------------------
    # 过滤链：唯一的过滤入口
    # ------------------------------------------------------------------
    def make_context(self) -> RunContext:
        """构造本次扫描的运行上下文（配置 + 标定 + 报告）。"""
        calib = None
        if self._cam_calib is not None or self._axis_calib is not None:
            calib = CalibrationBundle(
                camera=self._cam_calib,
                laser=self._laser_calib,
                axis=self._axis_calib,
                reports=list(self._calib_reports),
                axis_mode=str(self.cfg.get("calibration.axis_mode", "z_axis")),
            )
        return RunContext(
            cfg=self.cfg,
            report=ScanReport(),
            calib=calib,
            axis_point=(None if self._axis_calib is None
                        else np.asarray(self._axis_calib.point)),
            axis_dir=(None if self._axis_calib is None
                      else np.asarray(self._axis_calib.direction)),
        )

    def _run_filter_chain(self, pcd):
        """对配准后的点云执行完整过滤链。

        这是**唯一**的过滤入口。链路顺序来自配置
        （``reconstruction.filter_chain``）；未写该键时自动回退到旧式扁平
        键，行为与重构前逐字一致。

        Returns
        -------
        (Cloud, MaskStack, RunContext)
        """
        ctx = self.make_context()
        cloud = Cloud(np.asarray(pcd.points, dtype=np.float64),
                      frame="world", source="registration")
        chain = FilterChain.from_config(self.cfg)
        log.info("过滤链: %s", " → ".join(chain.names()) or "(空)")
        out, masks = chain.apply(cloud, ctx)
        if masks is not None:
            log.info("\n%s", masks.summary())
        return out, masks, ctx

    # ------------------------------------------------------------------
    # 【兼容层】旧的四个私有过滤方法
    # ------------------------------------------------------------------
    # 主流程已改为 ``FilterChain``（见 :meth:`build_pointcloud`）。
    # 下面四个方法保留是为了：旧测试、外部脚本、以及 UI 里"单独试一下
    # 某个过滤"的调用点。它们现在只是**把 Open3D 对象翻译成 Cloud，
    # 交给对应插件执行，再翻译回去**，因此行为与新链路严格一致。
    #
    # 新代码请直接用 :meth:`_run_filter_chain` 或 FilterChain。
    # ------------------------------------------------------------------
    def _run_single_filter(self, name: str, pcd):
        """对单个过滤器跑一次，返回 ``(pcd, info)``（兼容旧签名）。"""
        from ..contracts import FILTERS
        from ..contracts import RunContext

        if self._axis_calib is None:
            self.load_calibrations()
        ctx = self.make_context()
        cloud = Cloud(np.asarray(pcd.points, dtype=np.float64),
                      frame="world", source=name)
        plugin = FILTERS.get(name)()
        out, report = plugin.run(cloud, ctx)
        info = dict(report.extra)
        info.setdefault("applied", False)
        info.setdefault("removed", report.n_in - report.n_out)
        if not info["applied"]:
            return pcd, info
        if report.n_out == 0:
            return _to_open3d(out), info
        return _to_open3d(out), info

    def _filter_bounds(self, pcd):
        """【兼容】按世界系半径 + 轴向高度剔除越界点 → ``(pcd, info)``。

        转调 :class:`~triniscan.algorithms.filtering.bounds.BoundsFilter`。
        三个门限全空时不启用（``applied=False``）。
        """
        return self._run_single_filter("bounds", pcd)

    def _filter_clusters(self, pcd):
        """【兼容】连通域去噪 → ``(pcd, info)``。

        转调 :class:`~triniscan.algorithms.filtering.cluster.ClusterFilter`。
        ``reconstruction.cluster_filter=false`` 时插件不生效。

        ⚠️ ``cluster_keep_top=1`` 会把除最大簇以外的**全部**点删掉，
        若物体在某处断成两簇，少的那一簇就是"少一面"的来源。
        """
        if not bool(self.cfg.get("reconstruction.cluster_filter", False)):
            return pcd, {"applied": False, "removed": 0, "reason": "未启用"}
        return self._run_single_filter("cluster", pcd)

    def _remove_turntable(self, pcd):
        """【兼容】去转盘表面 → ``(pcd, info)``。

        转调 :class:`~triniscan.algorithms.filtering.turntable.TurntableFilter`，
        方法由 ``reconstruction.turntable_method``（axis_height / ransac）决定。
        """
        return self._run_single_filter("turntable", pcd)

    def _clip_axial_band(self, pcd):
        """【兼容】按沿轴高度 / 半径裁剪 → ``(pcd, info)``。

        转调 :class:`~triniscan.algorithms.filtering.axial.AxialTrimFilter`。
        """
        return self._run_single_filter("axial", pcd)

    # ------------------------------------------------------------------
    def _sync_legacy_filter_state(self, masks) -> None:
        """把 MaskStack 汇总回旧的 ``_last_turntable`` / ``_last_bounds``。

        旧代码（日志、结果文件、UI）从这两个属性读过滤统计；新链路只产出
        ``MaskStack``。这里做一次投影，保证**两条读取路径数值一致**。

        ``removed`` 取的是**该层自身**删掉的点数（增量），而不是累计值：
        累计值在"第 2 层之后"已经包含了第 1 层的删除量，拿去做诊断只会
        误导。累计量请看 :meth:`MaskStack.cumulative`。
        """
        if masks is None:
            self._last_bounds = {"applied": False, "removed": 0}
            self._last_turntable = {"applied": False, "removed": 0}
            return

        cascade = masks.cumulative()        # {name: 该层之后的剩余点数}

        def _layer(name):
            layer = masks.layers.get(name)
            if layer is None:
                return {"applied": False, "removed": 0, "reason": "未执行"}
            prev = cascade.get(name, len(layer.mask)) + layer.removed
            return {
                "applied": bool(layer.applied),
                "removed": int(layer.removed),
                "reason": layer.reason,
                "input": int(prev),
            }

        self._last_bounds = _layer("bounds")
        tt = _layer("turntable")
        tt["cluster"] = _layer("cluster")
        tt["axial"] = _layer("axial")
        self._last_turntable = tt

    # ------------------------------------------------------------------
    # 阶段 4：重建 + 封闭 + 体积
    # ------------------------------------------------------------------
    def measure_volume(self, pcd) -> tuple:
        # 参数一律经 spec 读取：默认值单一来源、越界立即报错。
        mesh_params = POISSON_SPEC.read(self.cfg)
        mesh_result = reconstruct_mesh(pcd, **mesh_params)
        log.debug("重建参数: %s", mesh_params)

        close_params = CLOSE_SPEC.read(self.cfg)
        closed = close_mesh(mesh_result.mesh,
                            axis_dir=self._axis_calib.direction,
                            **close_params)

        # 主算法与校核体素尺寸分别来自各自 spec，避免两处默认值漂移
        div_params = DIVERGENCE_SPEC.read(self.cfg)
        voxel_params = VOXEL_SPEC.read(self.cfg)
        vol = compute_volume(
            closed.mesh,
            method=str(self.cfg.get("volume.method", "divergence")),
            voxel_pitch=float(voxel_params["pitch"]),
            cross_check=bool(self.cfg.get("volume.cross_check", True)),
            require_watertight=bool(div_params["require_watertight"]),
        )

        # 凸包体积：对凸体（方块、柱、球等）比 Poisson 更贴近真值。
        # 二者不一致时说明点云含凹陷/噪声，值得人工复核。
        self._last_hull_volume_cm3 = None
        try:
            import trimesh
            hull = trimesh.PointCloud(
                np.asarray(pcd.points)).convex_hull
            self._last_hull_volume_cm3 = float(hull.volume * 1e6)
        except Exception as exc:  # noqa: BLE001
            log.debug("凸包体积计算失败: %s", exc)

        mesh_dir = ensure_dir(self.cfg.resolve("output.mesh_dir"))
        mesh_path = os.path.join(mesh_dir, "mesh.ply")
        try:
            closed.mesh.export(mesh_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("网格保存失败: %s", exc)
            mesh_path = None
        # 同时导出 OBJ（通用 CAD/审阅格式，便于第三者打开核对）
        try:
            obj_path = os.path.join(mesh_dir, "mesh.obj")
            closed.mesh.export(obj_path)
            log.info("网格已保存: %s / %s", mesh_path, obj_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("OBJ 保存失败: %s", exc)
        self._last_close_fallback = closed.fallback_used
        self._last_close_degraded = bool(getattr(closed, "degraded", False))
        self._last_close_degrade_reason = str(
            getattr(closed, "degrade_reason", ""))
        self._last_face_retention = float(
            getattr(closed, "face_retention", 1.0))
        if self._last_close_degraded:
            # 降级不是错误，但必须让用户看见——否则"体积偏小/偏大"
            # 会永远被归咎于算法，而真正原因是封闭兜底抹掉了形状。
            log.warning("网格封闭降级: %s", self._last_close_degrade_reason)
        return vol, closed, mesh_path

    # ------------------------------------------------------------------
    # 顶层流程
    # ------------------------------------------------------------------
    def run(self) -> ScanResult:
        """执行一次完整测量。"""
        t_start = time.time()
        stages: dict[str, float] = {}

        t = time.time()
        views = self.acquire()
        stages["acquire"] = time.time() - t

        t = time.time()
        clouds = self.reconstruct_views(views)
        if not clouds:
            raise RuntimeError("没有有效视角，测量失败")
        stages["triangulate"] = time.time() - t

        t = time.time()
        pcd, pc_path = self.build_pointcloud(clouds)
        stages["register"] = time.time() - t

        t = time.time()
        vol, closed, mesh_path = self.measure_volume(pcd)
        stages["volume"] = time.time() - t

        elapsed = time.time() - t_start
        limit = float(self.cfg.get("scan.timeout_s", 60))
        if elapsed > limit:
            log.warning("测量耗时 %.1f s 超过限制 %.0f s", elapsed, limit)

        reg_method = getattr(self, "_last_register_method", "")
        tt = getattr(self, "_last_turntable", {"applied": False, "removed": 0})
        masks = self._last_masks
        if masks is not None:
            filter_layers = masks.to_dict()["layers"]
            filter_summary = masks.summary()
            n_before, n_after = masks.n, masks.n_keep
        else:
            filter_layers, filter_summary = [], ""
            n_before = n_after = 0
        fingerprint = ""
        if self._cam_calib is not None:
            try:
                fingerprint = CalibrationBundle(
                    camera=self._cam_calib, laser=self._laser_calib,
                    axis=self._axis_calib,
                    axis_mode=str(self.cfg.get("calibration.axis_mode",
                                               "z_axis")),
                ).fingerprint()
            except Exception as exc:      # noqa: BLE001 - 指纹失败不影响测量
                log.debug("标定指纹计算失败: %s", exc)

        result = ScanResult(
            volume_cm3=vol.volume_cm3,
            volume_m3=vol.volume_m3,
            elapsed_s=elapsed,
            num_views=len(clouds),
            num_points=int(sum(len(c.points) for c in clouds)),
            mesh_watertight=closed.watertight,
            cross_check_cm3=(vol.cross_check_m3 * 1e6
                             if vol.cross_check_m3 else None),
            hull_volume_cm3=self._last_hull_volume_cm3,
            stages_s=stages,
            mesh_path=mesh_path,
            points_path=pc_path,
            cross_check_diff=vol.relative_diff,
            register_method=reg_method,
            turntable_removed=int(tt.get("removed", 0)),
            turntable_applied=bool(tt.get("applied", False)),
            close_fallback=getattr(self, "_last_close_fallback", ""),
            close_degraded=getattr(self, "_last_close_degraded", False),
            close_degrade_reason=getattr(
                self, "_last_close_degrade_reason", ""),
            mesh_face_retention=getattr(self, "_last_face_retention", 1.0),
            calib_reports=reports_to_dict(self._calib_reports),
            filter_layers=filter_layers,
            filter_summary=filter_summary,
            points_before_filter=n_before,
            points_after_filter=n_after,
            calib_fingerprint=fingerprint,
        )
        self._log_result(result)
        return result

    @staticmethod
    def _log_result(result: ScanResult) -> None:
        log.info("=" * 52)
        log.info("测量结果: %.2f cm³", result.volume_cm3)
        log.info("耗时: %.1f s  视角: %d  点: %d  水密: %s",
                 result.elapsed_s, result.num_views,
                 result.num_points, result.mesh_watertight)
        if result.cross_check_cm3 is not None:
            log.info("体素法校核: %.2f cm³（相对差 %.2f%%）",
                     result.cross_check_cm3,
                     (result.cross_check_diff or 0) * 100)
        if result.turntable_applied:
            log.info("已去除转盘点: %d", result.turntable_removed)
        if result.close_fallback:
            log.info("网格封闭兜底策略: %s", result.close_fallback)
        if result.hull_volume_cm3 is not None:
            log.info("凸包体积（凸体参考）: %.2f cm³", result.hull_volume_cm3)
        log.info("各阶段: %s", ", ".join(
            f"{k}={v:.1f}s" for k, v in result.stages_s.items()))
        log.info("=" * 52)