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

    p_axis = sub.add_parser("axis", help="转轴标定")
    p_axis.add_argument("--points", required=True, help="特征点 .npy")
    p_axis.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    cfg = Config.load(args.config)

    dispatch = {"camera": _cmd_camera, "laser": _cmd_laser,
                "axis": _cmd_axis}
    return dispatch[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
