"""中心线提取可视化工具。

对单张激光图像跑通三种提取方法，叠加显示并打印统计，
用于现场调参与验收。

用法::

    python -m triniscan.tools.extract_preview <image> [--save out.png]
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from ..extraction.centerline import (
    available_methods,
    extract_centerline,
)
from ..common.logging_utils import get_logger

log = get_logger(__name__)

_COLORS = {
    "steger": (0, 0, 255),          # 红
    "gray_centroid": (0, 255, 0),   # 绿
    "peak": (255, 0, 0),            # 蓝
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="中心线提取预览")
    parser.add_argument("image", help="输入激光图像路径")
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--save", default=None, help="保存叠加结果")
    parser.add_argument("--method", default=None,
                        help="仅测某一方法，默认全部")
    args = parser.parse_args(argv)

    img = cv2.imread(args.image)
    if img is None:
        log.error("无法读取图像: %s", args.image)
        return 1

    vis = img.copy()
    methods = [args.method] if args.method else available_methods() + \
        ["gray_centroid"] if args.method else available_methods()

    for method in methods:
        cfg = {"method": method, "threshold": args.threshold,
               "smooth_sigma": args.sigma}
        pts = extract_centerline(img, cfg)
        color = _COLORS.get(method, (255, 255, 255))
        for (u, v) in pts:
            cv2.circle(vis, (int(round(u)), int(round(v))), 1, color, -1)
        spread = float(np.std(pts[:, 1])) if len(pts) else 0.0
        log.info("%-14s: %4d 点, v 标准差 %.3f px",
                 method, len(pts), spread)

    if args.save:
        cv2.imwrite(args.save, vis)
        log.info("已保存: %s", args.save)
    else:
        cv2.imshow("centerline preview", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
