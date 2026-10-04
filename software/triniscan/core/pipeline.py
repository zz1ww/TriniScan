"""单件测量主流程（Pipeline）。

职责
----
把各模块串成完整测量链路，并对**单件测量总时限（<=60s）**负责：

    转-停-拍 → 中心线提取 → 三角测量 → 多视角配准
    → 网格重建 → 封闭化 → 体积计算

设计
----
- 每个阶段独立方法，便于单测与替换；
- 采集结果与中间数据可选落盘（便于复现与调试）；
- 全程记录耗时，超时给出告警。
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
    triangulate,
)
from ..volume import close_mesh, compute_volume
from ..calibration import (
    load_axis_calib,
    load_camera_calib,
    load_laser_calib,
)
from ..common.io_utils import ensure_dir
from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ScanPipeline", "ScanResult"]


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
    stages_s: dict[str, float] = field(default_factory=dict)
    mesh_path: Optional[str] = None
    points_path: Optional[str] = None


class ScanPipeline:
    """转台扫描测量流水线。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._cam_calib = None
        self._laser_calib = None
        self._axis_calib = None

    # ------------------------------------------------------------------
    # 标定加载
    # ------------------------------------------------------------------
    def load_calibrations(self) -> None:
        """载入三类标定（测量前必须完成标定）。"""
        self._cam_calib = load_camera_calib(
            self.cfg.resolve("calibration.camera_file"))
        self._laser_calib = load_laser_calib(
            self.cfg.resolve("calibration.laser_file"))
        self._axis_calib = load_axis_calib(
            self.cfg.resolve("calibration.axis_file"))
        log.info(
            "标定已载入: 相机 RMS=%.3f px, 激光平面 RMS=%.3f mm",
            self._cam_calib.rms, self._laser_calib.residual_rms * 1e3,
        )

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
        """对每个视角提取中心线并三角测量为三维点。"""
        if self._cam_calib is None:
            self.load_calibrations()

        K = self._cam_calib.K
        dist = self._cam_calib.dist
        plane = self._laser_calib.plane
        ex_cfg = self.cfg.raw.get("extraction", {})
        min_pts = int(self.cfg.get("extraction.min_points", 50))

        clouds = []
        for v in views:
            pixels = extract_centerline(v["raw"], ex_cfg)
            points, mask = triangulate(pixels, K, plane, dist,
                                       return_mask=True)
            if len(points) < min_pts:
                log.warning("视角 %s: 有效点不足 (%d)，跳过",
                            v["index"], len(points))
                continue
            clouds.append(ViewCloud(angle_deg=v["angle_deg"], points=points))
        log.info("三角测量完成: %d/%d 视角有效", len(clouds), len(views))
        return clouds

    # ------------------------------------------------------------------
    # 阶段 3：配准
    # ------------------------------------------------------------------
    def build_pointcloud(self, clouds: list[ViewCloud]):
        if self._axis_calib is None:
            self.load_calibrations()
        voxel = float(self.cfg.get("reconstruction.voxel_size", 0.0005))
        pcd = register_views(
            clouds,
            self._axis_calib.point,
            self._axis_calib.direction,
            voxel_size=voxel,
            icp_refine=bool(self.cfg.get("reconstruction.icp_refine", True)),
        )
        pc_dir = ensure_dir(self.cfg.resolve("output.pointcloud_dir"))
        pc_path = os.path.join(pc_dir, "cloud.ply")
        try:
            import open3d as o3d
            o3d.io.write_point_cloud(pc_path, pcd)
        except Exception as exc:  # noqa: BLE001
            log.warning("点云保存失败: %s", exc)
            pc_path = None
        return pcd, pc_path

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
        )
        closed = close_mesh(mesh_result.mesh)
        vol = compute_volume(
            closed.mesh,
            method=str(self.cfg.get("volume.method", "divergence")),
            voxel_pitch=float(self.cfg.get("volume.voxel_pitch", 0.001)),
            cross_check=bool(self.cfg.get("volume.cross_check", True)),
        )

        mesh_dir = ensure_dir(self.cfg.resolve("output.mesh_dir"))
        mesh_path = os.path.join(mesh_dir, "mesh.ply")
        try:
            closed.mesh.export(mesh_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("网格保存失败: %s", exc)
            mesh_path = None
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

        result = ScanResult(
            volume_cm3=vol.volume_cm3,
            volume_m3=vol.volume_m3,
            elapsed_s=elapsed,
            num_views=len(clouds),
            num_points=int(sum(len(c.points) for c in clouds)),
            mesh_watertight=closed.watertight,
            cross_check_cm3=(vol.cross_check_m3 * 1e6
                             if vol.cross_check_m3 else None),
            stages_s=stages,
            mesh_path=mesh_path,
            points_path=pc_path,
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
            log.info("体素法校核: %.2f cm³", result.cross_check_cm3)
        log.info("各阶段: %s", ", ".join(
            f"{k}={v:.1f}s" for k, v in result.stages_s.items()))
        log.info("=" * 52)
