"""图形界面入口：``python -m triniscan.ui``。"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triniscan.ui",
        description="TriniScan 图形界面",
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from .app import launch
    return launch(args.config, args.log_level)


if __name__ == "__main__":
    sys.exit(main())
