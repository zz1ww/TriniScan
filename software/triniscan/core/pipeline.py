"""单件测量主流程（Pipeline）。"""
from __future__ import annotations

import os
import time

import numpy as np

from triniscan.camera import Camera
from triniscan.turntable import Turntable
from triniscan.extraction import extract_centerline
from triniscan.reconstruction import triangulate, register_views, \
    reconstruct_mesh
from triniscan.volume import close_mesh, compute_volume, volume_cm3
from triniscan.calibration.camera_calib import load_camera_calib
from triniscan.calibration.laser_calib import load_laser_calib
from triniscan.calibration.axis_calib import load_axis_calib


class ScanPipeline:
    """转-停-拍 → 提取 → 三角测量 → 配准 → 重建 → 封闭 → 体积。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.get("_root", ".")

    # ------------------------------------------------------------------
    def _abspath(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def _load_calib(self):
        cam = load_camera_calib(
            self._abspath(self.cfg.get("calibration.camera_file")))
        plane = load_laser_calib(
            self._abspath(self.cfg.get("calibration.laser_file")))
        axis_pt, axis_dir = load_axis_calib(
            self._abspath(self.cfg.get("calibration.axis_file")))
        return cam, plane, axis_pt, axis_dir

    # ------------------------------------------------------------------
    def acquire(self, cam: Camera, tt: Turntable) -> list[dict]:
        """转-停-拍，采集各视角图像与角度。"""
        raw_dir = self._abspath(self.cfg.get("output.raw_dir"))
        os.makedirs(raw_dir, exist_ok=True)

        step = self.cfg.get("scan.angle_step_deg", 2.0)
        n = self.cfg.get("scan.num_views", 180)

        tt.home()
        views = []
        for i in range(n):
            tt.rotate_deg(step)
            path = os.path.join(raw_dir, f"view_{i:04d}.png")
            img = cam.grab_and_save(path)
            views.append({"angle_deg": step * (i + 1),
                          "image": path, "raw": img})
            print(f"[采集] {i + 1}/{n} angle={step * (i + 1):.1f}°")
        return views

    # ------------------------------------------------------------------
    def process(self, views: list[dict]) -> dict:
        """提取 → 三角测量 → 配准 → 重建 → 封闭 → 体积。"""
        cam_calib, plane, axis_pt, axis_dir = self._load_calib()
        K, dist = cam_calib["K"], cam_calib["dist"]

        ex_cfg = self.cfg.raw["extraction"]
        from triniscan.core.config import Config
        ex_cfg = Config(ex_cfg) if not isinstance(ex_cfg, Config) else ex_cfg

        views_pts = []
        for v in views:
            cl = extract_centerline(v["raw"], ex_cfg)
            pts = triangulate(cl, K, plane, dist)
            views_pts.append({"angle_deg": v["angle_deg"], "points": pts})

        voxel = self.cfg.get("reconstruction.voxel_size", 0.0005)
        pcd = register_views(views_pts, axis_pt, axis_dir, voxel_size=voxel)

        mesh = reconstruct_mesh(
            pcd,
            poisson_depth=self.cfg.get("reconstruction.poisson_depth", 9),
            density_quantile=self.cfg.get(
                "reconstruction.density_quantile", 0.02),
        )
        tm = close_mesh(mesh)
        vol = compute_volume(tm)
        return {"volume_cm3": volume_cm3(vol), "mesh": tm, "pcd": pcd}

    # ------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        cam = Camera(self.cfg.raw["camera"])
        tt = Turntable(self.cfg.raw["turntable"])
        try:
            cam.open()
            tt.connect()
            views = self.acquire(cam, tt)
            result = self.process(views)
        finally:
            cam.close()
            tt.close()
        result["elapsed_s"] = time.time() - t0
        print(f"\n体积: {result['volume_cm3']:.3f} cm³  "
              f"用时: {result['elapsed_s']:.1f} s")
        return result


if __name__ == "__main__":
    from triniscan.core.config import Config, find_project_root

    cfg = Config.load()
    cfg.raw["_root"] = find_project_root()
    ScanPipeline(cfg).run()
