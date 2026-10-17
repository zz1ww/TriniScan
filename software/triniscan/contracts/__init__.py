"""契约层：跨层共享的数据结构、插件协议、注册表、报告。

**零业务依赖**：只 import 标准库与 numpy，不 import open3d / cv2 /
trimesh，也不 import 项目内其它包。因此任何层都可以安全依赖它，
不会产生循环依赖，也不会拖慢启动（仅 ~5ms）。

分层约定
--------
::

    contracts/      ← 谁都依赖它，它不依赖任何人
      ↑
    common/         ← 纯工具（几何、IO、日志）
      ↑
    algorithms/     ← 可插拔实现（依赖 contracts + common）
      ↑
    stages/         ← 流程步骤（依赖 contracts + algorithms）
      ↑
    pipeline/       ← 编排（依赖 stages + config）
      ↑
    ui/ tools/      ← 应用入口（只依赖 pipeline + config）
"""
from __future__ import annotations

from .algorithm import AlgorithmSpec
from .errors import (ConfigError, HardwareError, InputError, PluginError,
                     TriniScanError)
from .frames import CalibrationBundle, Cloud, ViewFrame
from .params import ParamSet, ParamSpec, coerce
from .plugin import Plugin, PluginBase, RunContext
from .registry import (CALIBRATORS, CLOSERS, EXTRACTORS, FILTERS, MESHERS,
                       REGISTRARS, VOLUMETERS, Registry)
from .report import MaskLayer, MaskStack, ScanReport, StageReport, Timer

__all__ = [
    # algorithm
    "AlgorithmSpec",
    # errors
    "TriniScanError", "ConfigError", "InputError", "HardwareError",
    "PluginError",
    # params
    "ParamSpec", "ParamSet", "coerce",
    # plugin
    "Plugin", "PluginBase", "RunContext",
    # registry
    "Registry", "EXTRACTORS", "FILTERS", "REGISTRARS", "MESHERS",
    "CLOSERS", "VOLUMETERS", "CALIBRATORS",
    # report
    "StageReport", "ScanReport", "MaskStack", "MaskLayer", "Timer",
    # frames
    "ViewFrame", "Cloud", "CalibrationBundle",
]
