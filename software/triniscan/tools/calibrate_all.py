"""一键标定工具：相机内参 + 激光平面 + 转轴。

用法::

    python -m triniscan.tools.calibrate_all camera --dir <棋盘格目录>
    python -m triniscan.tools.calibrate_all laser  --dir <激光棋盘格目录>
    python -m triniscan.tools.calibrate_all axis   --points <特征点.npy>

说明
----
- 相机标定：输入目录中的棋盘格图像；
- 激光平面：输入目录中"棋盘格+激光"图像，自动提取中心线；
- 转轴：输入各角度下特征点三维坐标（由三角测量得到）的 .npy 文件。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from ..calibration import (
    calibrate_axis,
    calibrate_camera_from_files,
    calibrate_laser_plane,
    load_camera_calib,
    load_laser_calib,
)
from ..core.config import Config
from ..common.logging_utils import get_logger

log = get_logger(__name__)


def _cmd_camera(args, cfg) -> int:
    out = args.out or cfg.resolve("calibration.camera_file")
    calibrate_camera_from_files(
        images_glob=os.path.join(args.dir, "*.png"),
        pattern_cols=args.cols,
        pattern_rows=args.rows,
        square_size_m=args.square,
        out_path=out,
        show=args.show,
    )
    return 0


def _cmd_laser(args, cfg) -> int:
    cam_file = cfg.resolve("calibration.camera_file")
    if not os.path.exists(cam_file):
        log.error("请先完成相机标定: %s", cam_file)
        return 1
    cam = load_camera_calib(cam_file)

    files = sorted(glob.glob(os.path.join(args.dir, "*.png")))
    pairs = [{"image": f} for f in files]
    out = args.out or cfg.resolve("calibration.laser_file")
    calibrate_laser_plane(
        pairs, cam.K, cam.dist,
        pattern_cols=args.cols, pattern_rows=args.rows,
        square_size_m=args.square, out_path=out,
    )
    return 0


def _cmd_axis(args, cfg) -> int:
    points = np.load(args.points)
    out = args.out or cfg.resolve("calibration.axis_file")
    calibrate_axis(points, out_path=out)
    return 0


def _cmd_axis_live(args, cfg) -> int:
    """现场转轴标定：读一组「各角度含黑球」的图像目录，自动出轴。"""
    import cv2

    from ..calibration import AxisLiveCollector, SpotNotFound

    cam_file = args.camera or cfg.resolve("calibration.camera_file")
    laser_file = args.laser or cfg.resolve("calibration.laser_file")
    if not os.path.exists(cam_file):
        log.error("请先完成相机标定: %s", cam_file)
        return 1
    if not os.path.exists(laser_file):
        log.error("请先完成激光平面标定: %s", laser_file)
        return 1

    cam = load_camera_calib(cam_file)
    laser = load_laser_calib(laser_file)

    radius = (args.radius if args.radius is not None
              else float(cfg.get("axis_calib.sphere_radius_m", 0.020)))
    method = args.method or str(cfg.get("axis_calib.method", "peak"))

    files = sorted(glob.glob(os.path.join(args.dir, "*.png"))
                   + glob.glob(os.path.join(args.dir, "*.jpg")))
    if not files:
        log.error("目录中没有图像: %s", args.dir)
        return 1

    collector = AxisLiveCollector(
        cam.K, laser.plane, sphere_radius_m=radius,
        method=method, dist=cam.dist)
    collector._algo_cfg = {
        "spot_threshold": float(
            cfg.get("axis_calib.spot_threshold", 60.0)),
        "spot_win_px": int(cfg.get("axis_calib.spot_win_px", 25)),
        "min_radius_px": float(cfg.get("axis_calib.min_radius_px", 8.0)),
        "max_radius_px": float(cfg.get("axis_calib.max_radius_px", 240.0)),
    }

    step = 360.0 / len(files)
    for idx, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            log.warning("无法读取: %s", path)
            continue
        try:
            collector.add_image(img, angle_deg=idx * step)
            log.info("视角 %d/%d 已收录 (%s)", idx + 1, len(files), path)
        except SpotNotFound as exc:
            log.warning("视角 %d 跳过: %s", idx + 1, exc)

    if collector.count < 3:
        log.error("有效视角不足（%d < 3）", collector.count)
        return 1

    out = args.out or cfg.resolve("calibration.axis_file")
    archive = cfg.resolve("axis_calib.archive_dir")
    points_path = os.path.join(archive, "axis_points.npy")
    result = collector.calibrate(out_path=out, save_points_to=points_path)

    log.info("转轴标定完成: 半径=%.1f mm, 残差 RMS=%.3f mm",
             result.radius * 1e3, result.residual_rms * 1e3)
    log.info("现场点已存档: %s", points_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TriniScan 一键标定")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_chessboard_args(p):
        p.add_argument("--dir", required=True, help="图像目录")
        p.add_argument("--cols", type=int, default=9)
        p.add_argument("--rows", type=int, default=6)
        p.add_argument("--square", type=float, default=0.020)
        p.add_argument("--out", default=None)
        p.add_argument("--show", action="store_true")

    p_cam = sub.add_parser("camera", help="相机内参标定")
    _add_chessboard_args(p_cam)

    p_laser = sub.add_parser("laser", help="激光平面标定")
    _add_chessboard_args(p_laser)

    p_axis = sub.add_parser("axis", help="转轴标定（读 .npy 特征点）")
    p_axis.add_argument("--points", required=True, help="特征点 .npy")
    p_axis.add_argument("--out", default=None)

    p_live = sub.add_parser(
        "axis-live", help="转轴现场标定（读含黑球的图像目录）")
    p_live.add_argument("--dir", required=True,
                        help="各角度含标记球的图像目录")
    p_live.add_argument("--method", default=None, choices=["peak", "sphere"],
                        help="取点方法：peak 最亮极点 / sphere 球心")
    p_live.add_argument("--radius", type=float, default=None,
                        help="标记球半径 (m)，默认取配置")
    p_live.add_argument("--camera", default=None, help="相机标定 .npz")
    p_live.add_argument("--laser", default=None, help="激光平面 .npz")
    p_live.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    cfg = Config.load(args.config)

    dispatch = {"camera": _cmd_camera, "laser": _cmd_laser,
                "axis": _cmd_axis, "axis-live": _cmd_axis_live}
    return dispatch[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
