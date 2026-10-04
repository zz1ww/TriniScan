"""程序入口。

用法::

    python -m triniscan.core.main                    # 用默认配置跑测量
    python -m triniscan.core.main --check            # 仅检查配置与标定
    python -m triniscan.core.main --config my.yaml   # 指定配置
    python -m triniscan.core.main --log-level DEBUG  # 调试日志
"""
from __future__ import annotations

import argparse
import sys

from .config import Config, find_project_root
from ..common.logging_utils import get_logger, set_level

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triniscan",
        description="TriniScan — 激光三角测量三维体积测量系统",
    )
    p.add_argument("--config", default=None,
                   help="配置文件路径（默认 software/config/default.yaml）")
    p.add_argument("--check", action="store_true",
                   help="仅检查配置与标定文件，不测量")
    p.add_argument("--log-level", default=None,
                   help="日志级别: DEBUG/INFO/WARNING/ERROR")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.log_level:
        set_level(args.log_level)

    try:
        cfg = Config.load(args.config)
    except FileNotFoundError as exc:
        log.error("配置文件不存在: %s", exc)
        return 2

    log.info("=" * 52)
    log.info("TriniScan — 激光三角测量三维体积测量系统")
    log.info("=" * 52)
    log.info("项目根: %s", cfg.root)
    log.info("配置: %s", cfg.get("_meta.config_path"))
    _print_summary(cfg)

    if args.check:
        return _run_check(cfg)

    from .pipeline import ScanPipeline

    pipeline = ScanPipeline(cfg)
    try:
        pipeline.load_calibrations()
        result = pipeline.run()
    except Exception as exc:  # noqa: BLE001
        log.exception("测量失败: %s", exc)
        return 1

    print(f"\n最终体积: {result.volume_cm3:.2f} cm³ "
          f"(用时 {result.elapsed_s:.1f} s)")
    return 0


def _print_summary(cfg: Config) -> None:
    log.info("相机: index=%s %sx%s",
             cfg.get("camera.index"),
             cfg.get("camera.width"), cfg.get("camera.height"))
    log.info("转台: %s @ %s",
             cfg.get("turntable.port"), cfg.get("turntable.baudrate"))
    log.info("视角: %s × %s°",
             cfg.get("scan.num_views"), cfg.get("scan.angle_step_deg"))


def _run_check(cfg: Config) -> int:
    import os
    ok = True
    for key in ("calibration.camera_file",
                "calibration.laser_file",
                "calibration.axis_file"):
        path = cfg.resolve(key)
        exists = os.path.exists(path)
        ok &= exists
        log.info("[%s] %s", "OK" if exists else "缺失", path)
    log.info("检查结果: %s", "通过" if ok else "存在缺失")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
