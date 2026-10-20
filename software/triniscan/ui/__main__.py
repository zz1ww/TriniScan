"""图形界面入口：``python -m triniscan.ui``。

两套界面并存：

- ``--ui new``（默认）：工作台界面（左侧可折叠面板 + 右侧实时画面），
  含 测量 / 标定 / 硬件 三个工作台；
- ``--ui classic``：原有两页界面（测量 + 标定），保留作回退。

要退回旧界面，加 ``--ui classic`` 即可。
"""
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
    parser.add_argument("--ui", choices=("classic", "new"), default="new",
                        help="界面版本：new=工作台界面（默认），"
                             "classic=原两页界面")
    args = parser.parse_args(argv)

    if args.ui == "classic":
        from .app import launch as _launch
    else:
        from .app_v2 import launch as _launch
    return _launch(args.config, args.log_level)


if __name__ == "__main__":
    sys.exit(main())
