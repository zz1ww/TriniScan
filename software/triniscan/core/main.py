"""程序入口。

用法:
    python -m triniscan.core.main --config software/config/default.yaml
    python -m triniscan.core.main --check      # 仅检查环境/配置
"""
from __future__ import annotations

import argparse
import sys

from triniscan.core.config import Config, find_project_root


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TriniScan 体积测量系统")
    p.add_argument("--config", default="software/config/default.yaml",
                   help="配置文件路径")
    p.add_argument("--check", action="store_true",
                   help="仅检查配置与环境")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config.load(args.config)
    cfg.raw["_root"] = find_project_root()

    print("=" * 50)
    print("TriniScan — 激光三角测量三维体积测量系统")
    print("=" * 50)
    print(f"配置: {args.config}")
    print(f"相机: index={cfg.get('camera.index')} "
          f"{cfg.get('camera.width')}x{cfg.get('camera.height')}")
    print(f"转台: {cfg.get('turntable.port')} @ {cfg.get('turntable.baudrate')}")
    print(f"视角: {cfg.get('scan.num_views')} × {cfg.get('scan.angle_step_deg')}°")

    if args.check:
        print("\n[检查] 配置加载正常。")
        return 0

    from triniscan.core.pipeline import ScanPipeline
    ScanPipeline(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
