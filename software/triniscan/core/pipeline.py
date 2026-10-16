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

from ..camera import Camera
from ..turntable import Turntable
from ..extraction import extract_centerline
from ..reconstruction import (
    ViewCloud,
    register_views,
    reconstruct_mesh,
    remove_turntable_surface,
    triangulate,
    filter_by_depth,
    filter_by_bounds,
)
from ..volume import close_mesh, compute_volume
from ..calibration import (
    AxisCalibResult,
    load_axis_calib,
    load_camera_calib,
    load_laser_calib,
    validate_calibrations,
    format_reports,
    reports_to_dict,
)
from ..common.io_utils import ensure_dir
from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ScanPipeline", "ScanResult"]


def _dominant_band(along: np.ndarray,
                   bins: int = 120,
                   density_ratio: float = 0.10) -> Optional[tuple]:
    """在"沿轴坐标"直方图中找出最大密度连通区间（物体主体）。

    转台扫描中，物体本体沿轴投影会形成一条**连续的高密度区间**，而背景/
    支架表现为分离的小簇。本函数返回主体区间 ``(lo, hi)``（单位同输入），
    用于自动剔除物体上下方的孤立环带。

    Parameters
    ----------
    along : (N,) 各点沿轴坐标
    bins : 直方图分箱数
    density_ratio : 连通判据——某箱计数 > 该比例×峰值得以延续

    Returns
    -------
    (lo, hi) 或 None（无法判定）
    """
    along = np.asarray(along, dtype=np.float64).ravel()
    if along.size < 20 or along.ptp() < 1e-9:
        return None
    hist, edges = np.histogram(along, bins=bins)
    peak = int(hist.max())
    if peak <= 0:
        return None
    thr = max(1, int(round(peak * density_ratio)))
    # 找最长连续"高密度"段
    best = (0, -1)
    i = 0
    while i < len(hist):
        if hist[i] > thr:
            j = i
            while j + 1 < len(hist) and hist[j + 1] > thr:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j + 1
        else:
            i += 1
    if best[1] < best[0]:
        return None
    lo = float(edges[best[0]])
    hi = float(edges[best[1] + 1])
    return lo, hi


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
    calib_reports: list[dict] = field(default_factory=list)


class ScanPipeline:
    """转台扫描测量流水线。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._cam_calib = None
        self._laser_calib = None
        self._axis_calib = None
        self._calib_reports: list = []
        self._last_hull_volume_cm3: Optional[float] = None

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
        """从配置拼装阈值覆盖（缺省则用 quality 模块默认值）。"""
        def _pair(prefix: str) -> Optional[dict]:
            good = self.cfg.get(f"quality.{prefix}_good", None)
            warn = self.cfg.get(f"quality.{prefix}_warn", None)
            if good is None and warn is None:
                return None
            d = {}
            if good is not None:
                d["good"] = good
            if warn is not None:
                d["warn"] = warn
            return d

        overrides = {}
        for name, key in (("camera", "camera_rms"),
                          ("laser", "laser_rms"),
                          ("axis", "axis_rms")):
            pair = _pair(key)
            if pair:
                overrides[name] = pair
        return overrides

    # ------------------------------------------------------------------
    # 阶段 1：采集
    # ------------------------------------------------------------------
    def acquire(self) -> list[dict]:
        """转-停-拍，采集各视角图像。"""
        raw_dir = ensure_dir(self.cfg.resolve("output.raw_dir"))
        step = float(self.cfg.get("scan.angle_step_deg", 2.0))
        n = int(self.cfg.get("scan.num_views", 180))

        views = []
        with Camera(self.cfg.raw["camera"]) as cam, \
                Turntable(self.cfg.raw["turntable"]) as table:
            table.home()
            for i in range(n):
                table.rotate_deg(step)
                path = os.path.join(raw_dir, f"view_{i:04d}.png")
                image = cam.grab_and_save(path)
                views.append({
                    "index": i,
                    "angle_deg": step * (i + 1),
                    "image": path,
                    "raw": image,
                })
                log.debug("采集 %d/%d (%.1f°)", i + 1, n, step * (i + 1))
        log.info("采集完成: %d 视角", len(views))
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
        ex_cfg = self.cfg.raw.get("extraction", {})
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
        voxel = float(self.cfg.get("reconstruction.voxel_size", 0.0005))
        reg = register_views(
            clouds,
            self._axis_calib.point,
            self._axis_calib.direction,
            voxel_size=voxel,
            icp_refine=bool(self.cfg.get("reconstruction.icp_refine", True)),
            global_align=bool(
                self.cfg.get("reconstruction.global_align", True)),
            pose_graph=bool(
                self.cfg.get("reconstruction.pose_graph", True)),
            edge_span=int(self.cfg.get("reconstruction.edge_span", 3)),
            iterative_iters=int(
                self.cfg.get("reconstruction.iterative_iters", 3)),
            return_result=True,
        )
        pcd = reg.pointcloud

        # --- 世界系物理边界过滤（半径 / 轴向）---
        pcd, bounds_info = self._filter_bounds(pcd)

        # --- 去转盘表面 ---
        removal_info = {"applied": False, "removed": 0}
        if bool(self.cfg.get("reconstruction.remove_turntable", True)):
            pcd, removal_info = self._remove_turntable(pcd)

        # --- 轴向裁剪（剔除背景/支架孤立环带）---
        pcd, axial_info = self._clip_axial_band(pcd)
        removal_info = dict(removal_info)
        removal_info["axial"] = axial_info

        pc_dir = ensure_dir(self.cfg.resolve("output.pointcloud_dir"))
        pc_path = os.path.join(pc_dir, "cloud.ply")
        try:
            import open3d as o3d
            o3d.io.write_point_cloud(pc_path, pcd)
        except Exception as exc:  # noqa: BLE001
            log.warning("点云保存失败: %s", exc)
            pc_path = None

        self._last_register_method = reg.method
        self._last_turntable = removal_info
        self._last_bounds = bounds_info
        return pcd, pc_path

    def _filter_bounds(self, pcd):
        """按世界系物理边界（半径 + 轴向）剔除越界点，返回 ``(pcd, info)``。

        与"去转盘"不同，这里剔除的是**转台半径之外 / 高度范围之外**的
        虚假点（反光、环境光）。半径门限需显式配置
        ``reconstruction.bounds_radius_m`` 才生效，避免误伤大物体。
        """
        import open3d as o3d

        r_max = float(self.cfg.get("reconstruction.bounds_radius_m", 0.0) or 0.0)
        z_min = self.cfg.get("reconstruction.bounds_z_min_m", None)
        z_max = self.cfg.get("reconstruction.bounds_z_max_m", None)
        info = {"applied": False, "removed": 0}
        if r_max <= 0 and z_min is None and z_max is None:
            return pcd, info

        pts = np.asarray(pcd.points, dtype=np.float64)
        res = filter_by_bounds(
            pts, axis_point=self._axis_calib.point,
            axis_dir=self._axis_calib.direction,
            radius_max=r_max,
            z_min=(float(z_min) if z_min is not None else None),
            z_max=(float(z_max) if z_max is not None else None),
        )
        info = {"applied": res.applied, "removed": res.num_removed,
                "reason": res.reason}
        if not res.applied or res.num_removed == 0:
            return pcd, info
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(res.points)
        return new_pcd, info

    def _remove_turntable(self, pcd):
        """对世界系点云做转盘去除（几何拟合），返回 ``(pcd, info)``。"""
        import open3d as o3d

        pts = np.asarray(pcd.points, dtype=np.float64)
        radius = self.cfg.get("reconstruction.turntable_radius_m", None)
        res = remove_turntable_surface(
            pts,
            axis_dir=self._axis_calib.direction,
            axis_point=self._axis_calib.point,
            ransac_dist=float(
                self.cfg.get("reconstruction.turntable_plane_tol_m", 0.001)),
            min_inlier_ratio=float(
                self.cfg.get("reconstruction.turntable_min_inlier", 0.15)),
            align_tol_deg=float(
                self.cfg.get("reconstruction.turntable_align_tol_deg", 15.0)),
            margin=float(
                self.cfg.get("reconstruction.turntable_margin_m", 0.0005)),
            band_below=float(
                self.cfg.get("reconstruction.turntable_band_below_m", 0.0015)),
            keep_upper_only=bool(
                self.cfg.get("reconstruction.turntable_keep_upper_only",
                             True)),
            radius=(float(radius) if radius else None),
        )
        info = {
            "applied": res.applied, "removed": res.num_removed,
            "reason": res.reason,
        }
        if not res.applied or res.num_removed == 0:
            return pcd, info

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(res.points)
        return new_pcd, info

    def _clip_axial_band(self, pcd):
        """按"沿轴高度 + 到轴半径"裁剪点云（剔除背景/支架等孤立环带）。

        动机
        ----
        转台扫描时激光常同时打到**物体后方的背景/支架**，在点云里表现为
        若干条与物体分离的环带（沿轴高度不同、半径也不同）。这类点用
        RANSAC 平面拟合往往抓错目标（会误抓物体的竖直面）。而对"物体
        相对转轴的几何"而言，**沿轴高度分布**最能分离它们。

        策略（可配置，全部为 0/None 时不生效）：

        1. ``axial.height_min_m`` / ``height_max_m``：硬门限，直接切掉
           高度范围外的点（最直接、最可控）；
        2. ``axial.radius_max_m``：切掉离轴过远的散点；
        3. ``axial.auto_trim``：自动取"沿轴高度直方图的主体连通区间"，
           把上方/下方孤立的小簇一并切掉（物体高度未知时的兜底）。

        Returns
        -------
        (pcd, info)  info 含 applied/removed/reason
        """
        import open3d as o3d

        if self._axis_calib is None:
            return pcd, {"applied": False, "removed": 0, "reason": "无轴"}

        pts = np.asarray(pcd.points, dtype=np.float64)
        if len(pts) == 0:
            return pcd, {"applied": False, "removed": 0, "reason": "空点云"}

        axis = np.asarray(self._axis_calib.direction, dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        origin = np.asarray(self._axis_calib.point, dtype=np.float64)
        rel = pts - origin
        along = rel @ axis
        perp = rel - np.outer(along, axis)
        radius = np.linalg.norm(perp, axis=1)

        keep = np.ones(len(pts), dtype=bool)
        reasons = []

        h_min = self.cfg.get("reconstruction.axial.height_min_m", None)
        h_max = self.cfg.get("reconstruction.axial.height_max_m", None)
        r_max = self.cfg.get("reconstruction.axial.radius_max_m", None)
        auto = bool(self.cfg.get("reconstruction.axial.auto_trim", False))

        if auto and h_min is None and h_max is None and r_max is None:
            hi = _dominant_band(along)
            if hi is not None:
                lo_mm, hi_mm = hi
                m = (along >= lo_mm) & (along <= hi_mm)
                if m.sum() >= 20:
                    keep &= m
                    reasons.append(
                        f"自动主体区间 [{lo_mm*1e3:.0f},"
                        f"{hi_mm*1e3:.0f}] mm")

        if h_min is not None:
            keep &= along >= float(h_min)
            reasons.append(f"h>={float(h_min)*1e3:.0f}mm")
        if h_max is not None:
            keep &= along <= float(h_max)
            reasons.append(f"h<={float(h_max)*1e3:.0f}mm")
        if r_max is not None:
            keep &= radius <= float(r_max)
            reasons.append(f"r<={float(r_max)*1e3:.0f}mm")

        removed = int((~keep).sum())
        if removed == 0:
            return pcd, {"applied": False, "removed": 0,
                         "reason": "无越界点"}
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(pts[keep])
        log.info("轴向裁剪: 剔除 %d/%d 点（%s）",
                 removed, len(pts), "，".join(reasons))
        return new_pcd, {"applied": True, "removed": removed,
                         "reason": "，".join(reasons)}

    # ------------------------------------------------------------------
    # 阶段 4：重建 + 封闭 + 体积
    # ------------------------------------------------------------------
    def measure_volume(self, pcd) -> tuple:
        mesh_result = reconstruct_mesh(
            pcd,
            poisson_depth=int(
                self.cfg.get("reconstruction.poisson_depth", 9)),
            density_quantile=float(
                self.cfg.get("reconstruction.density_quantile", 0.02)),
            adaptive_depth=bool(
                self.cfg.get("reconstruction.adaptive_depth", True)),
            orient_max_nn=int(
                self.cfg.get("reconstruction.orient_max_nn", 30)),
        )
        closed = close_mesh(
            mesh_result.mesh,
            fallback=str(self.cfg.get("reconstruction.close_fallback",
                                      "bottom_plane")),
            axis_dir=self._axis_calib.direction,
            allow_convex_hull=bool(
                self.cfg.get("reconstruction.allow_convex_hull", True)),
        )
        vol = compute_volume(
            closed.mesh,
            method=str(self.cfg.get("volume.method", "divergence")),
            voxel_pitch=float(self.cfg.get("volume.voxel_pitch", 0.001)),
            cross_check=bool(self.cfg.get("volume.cross_check", True)),
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
            calib_reports=reports_to_dict(self._calib_reports),
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