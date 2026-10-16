"""一键全流程测量：采集 → 三角测量 → 拼接 → 网格 → 体积。

用法::

    # 完整测量（驱动相机 + 转台）
    python -m triniscan.tools.measure

    # 回放已采集的图像（离线重建，不连硬件）
    python -m triniscan.tools.measure --replay data/raw

    # 只检查标定与配置，不采集
    python -m triniscan.tools.measure --check

产物（默认写入配置 ``output`` 段对应目录）::

    data/raw/            原始图像（每视角一张）
    data/pointclouds/cloud.ply
    data/meshes/mesh.ply + mesh.obj
    data/results/measure.json + measure.txt

设计说明
--------
- **单一入口**：把"标定 → 采集 → 重建 → 体积"串成一条可复现流水线，
  方便现场一键出结果，也便于赛后复现；
- **离线回放**：``--replay`` 直接用磁盘上已有图像重建，无需接硬件，
  既可用于调试算法，也可用于答辩演示；
- **结果落盘**：体积、点云、网格、报告一并写入默认输出目录，供第三方
  复核（OBJ 为通用格式）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from ..common.logging_utils import get_logger, set_level
from ..core.config import Config
from ..core.pipeline import ScanPipeline

log = get_logger(__name__)

__all__ = ["main", "run_replay"]


def _load_images(directory: str) -> list[dict]:
    """读取目录下所有图像，构造流水线所需的视角列表。"""
    import cv2

    files = sorted(glob.glob(os.path.join(directory, "*.png"))
                   + glob.glob(os.path.join(directory, "*.jpg")))
    if not files:
        raise FileNotFoundError(f"目录中没有图像: {directory}")
    step = 360.0 / len(files)
    views = []
    for i, path in enumerate(files):
        img = cv2.imread(path)
        if img is None:
            log.warning("无法读取，跳过: %s", path)
            continue
        views.append({
            "index": i,
            "angle_deg": step * (i + 1),
            "image": path,
            "raw": img,
        })
    if not views:
        raise RuntimeError("没有任何可用的图像")
    return views


def run_replay(cfg: Config, directory: str = "", save_report: bool = True):
    """离线回放：用磁盘上已有的图像跑完整重建 + 体积流程。"""
    directory = directory or cfg.resolve("output.raw_dir")
    log.info("离线回放: %s", directory)

    pipeline = ScanPipeline(cfg)
    pipeline.load_calibrations()
    views = _load_images(directory)
    log.info("载入 %d 个视角", len(views))

    clouds = pipeline.reconstruct_views(views)
    if not clouds:
        raise RuntimeError("没有有效视角，测量失败")
    pcd, pc_path = pipeline.build_pointcloud(clouds)
    vol, closed, mesh_path = pipeline.measure_volume(pcd)

    result = {
        "volume_cm3": round(vol.volume_cm3, 3),
        "volume_m3": vol.volume_m3,
        "num_views": len(clouds),
        "num_points": int(len(np.asarray(pcd.points))),
        "mesh_watertight": bool(closed.watertight),
        "close_fallback": closed.fallback_used,
        "cross_check_cm3": (round(vol.cross_check_m3 * 1e6, 3)
                            if vol.cross_check_m3 else None),
        "cross_check_diff": vol.relative_diff,
        "hull_volume_cm3": (round(getattr(pipeline,
                             "_last_hull_volume_cm3"), 3)
                            if getattr(pipeline,
                                       "_last_hull_volume_cm3", None)
                            else None),
        "mesh_path": mesh_path,
        "points_path": pc_path,
        "mode": "replay",
    }
    if save_report:
        _write_report(cfg, result)
    _print_result(result)
    return result


def run_capture(cfg: Config, save_report: bool = True):
    """完整测量：驱动相机 + 转台，全自动出结果。"""
    pipeline = ScanPipeline(cfg)
    pipeline.load_calibrations()
    result = pipeline.run()
    payload = {
        "volume_cm3": round(result.volume_cm3, 3),
        "volume_m3": result.volume_m3,
        "num_views": result.num_views,
        "num_points": result.num_points,
        "mesh_watertight": result.mesh_watertight,
        "close_fallback": result.close_fallback,
        "cross_check_cm3": (round(result.cross_check_cm3, 3)
                            if result.cross_check_cm3 else None),
        "cross_check_diff": result.cross_check_diff,
        "hull_volume_cm3": (round(result.hull_volume_cm3, 3)
                            if result.hull_volume_cm3 else None),
        "mesh_path": result.mesh_path,
        "points_path": result.points_path,
        "elapsed_s": result.elapsed_s,
        "stages_s": result.stages_s,
        "mode": "capture",
    }
    if save_report:
        _write_report(cfg, payload)
    _print_result(payload)
    return payload


def _write_report(cfg: Config, result: dict) -> None:
    """把结果写入 ``output.result_dir``（JSON + 文本两版）。"""
    from ..common.io_utils import ensure_dir

    out_dir = ensure_dir(cfg.resolve("output.result_dir"))
    result = dict(result)
    result["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    json_path = os.path.join(out_dir, "measure.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    lines = ["=" * 52,
             "TriniScan 测量报告",
             "=" * 52,
             f"时间      : {result['created_at']}",
             f"模式      : {result.get('mode', '-')}",
             f"体积      : {result['volume_cm3']:.2f} cm³ "
             f"({result['volume_m3']:.6f} m³)",
             f"视角/点数 : {result['num_views']} / "
             f"{result['num_points']}",
             f"水密      : {result['mesh_watertight']}",
             f"补洞策略  : {result.get('close_fallback') or '-'}"]
    if result.get("hull_volume_cm3"):
        lines.append(f"凸包体积  : {result['hull_volume_cm3']:.2f} cm³"
                     "（凸体更准，供参考）")
    if result.get("cross_check_cm3"):
        lines.append(f"体素校核  : {result['cross_check_cm3']:.2f} cm³ "
                     f"(相对差 "
                     f"{(result.get('cross_check_diff') or 0) * 100:.2f}%)")
    if result.get("elapsed_s"):
        lines.append(f"耗时      : {result['elapsed_s']:.1f} s")
    lines.append(f"网格      : {result.get('mesh_path')}")
    lines.append(f"点云      : {result.get('points_path')}")
    lines.append("=" * 52)
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "measure.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt + "\n")
    log.info("报告已写入: %s / measure.txt", json_path)


def _print_result(result: dict) -> None:
    print()
    print("=" * 46)
    print(f"体积: {result['volume_cm3']:.2f} cm³ "
          f"({result['volume_m3']:.6f} m³)")
    print(f"视角: {result['num_views']}  点数: {result['num_points']}")
    print(f"水密: {result['mesh_watertight']}  "
          f"补洞: {result.get('close_fallback') or '-'}")
    if result.get("hull_volume_cm3"):
        print(f"凸包体积（凸体参考）: {result['hull_volume_cm3']:.2f} cm³")
    if result.get("cross_check_cm3"):
        print(f"体素校核: {result['cross_check_cm3']:.2f} cm³ "
              f"(相对差 {(result.get('cross_check_diff') or 0) * 100:.2f}%)")
    if result.get("elapsed_s"):
        print(f"耗时: {result['elapsed_s']:.1f} s")
    print("=" * 46)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triniscan-measure",
        description="TriniScan 一键全流程测量（采集/回放 → 点云 → 体积）",
    )
    p.add_argument("--config", default=None,
                   help="配置文件（默认 software/config/default.yaml）")
    p.add_argument("--replay", nargs="?", const="", default=None,
                   metavar="DIR",
                   help="离线回放：不连硬件，用 DIR 下已有图像重建"
                        "（省略 DIR 则用 output.raw_dir）")
    p.add_argument("--check", action="store_true",
                   help="只检查配置与标定文件，不采集/重建")
    p.add_argument("--no-report", action="store_true",
                   help="不写结果报告文件")
    p.add_argument("--log-level", default=None,
                   help="日志级别 DEBUG/INFO/WARNING/ERROR")
    return p


def _check(cfg: Config) -> int:
    ok = True
    for key in ("calibration.camera_file",
                "calibration.laser_file",
                "calibration.axis_file"):
        path = cfg.resolve(key)
        exists = os.path.exists(path)
        ok &= exists
        log.info("[%s] %s", "OK" if exists else "缺失", path)
    log.info("转轴模式: %s", cfg.get("calibration.axis_mode"))
    log.info("采样: %s 视角 × %s°",
             cfg.get("scan.num_views"), cfg.get("scan.angle_step_deg"))
    log.info("检查结果: %s", "通过" if ok else "存在缺失")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.log_level:
        set_level(args.log_level)

    try:
        cfg = Config.load(args.config)
    except FileNotFoundError as exc:
        log.error("配置文件不存在: %s", exc)
        return 2

    if args.check:
        return _check(cfg)

    try:
        if args.replay is not None:
            run_replay(cfg, args.replay, save_report=not args.no_report)
        else:
            run_capture(cfg, save_report=not args.no_report)
    except Exception as exc:  # noqa: BLE001
        log.exception("测量失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
