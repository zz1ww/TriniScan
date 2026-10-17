"""标定的参数声明。

标定分为"采集侧参数"（标定工具用）与"质量判据"（每次扫描载入标定后
都要用它判断这批标定能不能用）。后者此前散在 ``pipeline._quality_thresholds``
里手工拼键名，拼错的后果是**阈值静默失效**。

这里把三类标定的质量阈值统一声明，并附上"阈值 → 判定"的语义：
超过 ``warn`` 给警告，超过 ``good``… 实际约定见
:mod:`triniscan.algorithms.calibration.quality`。
"""
from __future__ import annotations

from triniscan.contracts import AlgorithmSpec, CALIBRATORS, ParamSpec

from .axis_calib import calibrate_axis, load_axis_calib
from .camera_calib import calibrate_camera, load_camera_calib
from .laser_calib import calibrate_laser_plane, load_laser_calib

__all__ = ["CAMERA_CALIB_SPEC", "LASER_CALIB_SPEC", "AXIS_CALIB_SPEC",
           "QUALITY_SPEC", "specs"]


CAMERA_CALIB_SPEC = AlgorithmSpec(
    name="camera",
    label="相机内参标定",
    func=calibrate_camera,
    help="棋盘格张正友标定；产出 K / dist / 每视角重投影误差",
    tags=("标定", "离线"),
    params=(
        ParamSpec("camera_calib.pattern_cols", int, 11, low=3, high=40,
                  short="pattern_cols", label="棋盘内角列数"),
        ParamSpec("camera_calib.pattern_rows", int, 8, low=3, high=40,
                  short="pattern_rows", label="棋盘内角行数"),
        ParamSpec("camera_calib.square_size_m", float, 0.01,
                  low=1e-4, high=1.0, unit="m", short="square_size_m",
                  label="方格边长"),
        ParamSpec("camera_calib.use_sb", bool, True, short="use_sb",
                  label="亚像素角点",
                  help="关闭后标定 RMS 会明显变大"),
    ),
)

LASER_CALIB_SPEC = AlgorithmSpec(
    name="laser",
    label="激光平面标定",
    func=calibrate_laser_plane,
    help="多姿态靶标上的激光条 → 拟合相机系激光平面",
    tags=("标定", "离线"),
    params=(
        ParamSpec("laser_calib.robust", bool, True, short="robust",
                  label="稳健拟合",
                  help="用 IRLS 抑制离群点"),
        ParamSpec("laser_calib.view_reject_mm", float, 1.0,
                  low=0.0, high=50.0, unit="mm", short="view_reject_mm",
                  label="单视角剔除阈值",
                  help="某视角残差超过此值则整视角剔除"),
        ParamSpec("laser_calib.outlier_mm", float, 0.5,
                  low=0.0, high=50.0, unit="mm", short="outlier_mm",
                  label="点级离群阈值"),
        ParamSpec("laser_calib.channel", str, "red_laser",
                  choices=("red_laser", "gray"), short="channel",
                  label="颜色通道"),
    ),
)

AXIS_CALIB_SPEC = AlgorithmSpec(
    name="axis",
    label="转轴标定",
    func=calibrate_axis,
    help="标准球绕转轴的多个视角球心 → 拟合圆周 → 轴点与轴向",
    tags=("标定", "离线"),
    params=(
        ParamSpec("axis_calib.method", str, "depth",
                  choices=("depth", "sphere"), short="method",
                  label="标定方法",
                  help="depth=仅用激光深度（快）；sphere=球心拟合"),
        ParamSpec("axis_calib.num_views", int, 24, low=6, high=72,
                  short="num_views", label="标定视角数"),
        ParamSpec("axis_calib.step_deg", float, 30.0, low=1.0, high=120.0,
                  unit="deg", short="step_deg", label="每步角度"),
        ParamSpec("axis_calib.max_residual_mm", float, 1.0,
                  low=0.0, high=20.0, unit="mm", short="max_residual_mm",
                  label="最大残差",
                  help="超过视为标定不合格"),
        ParamSpec("axis_calib.robust", bool, True, short="robust",
                  label="稳健拟合"),
    ),
)

QUALITY_SPEC = AlgorithmSpec(
    name="quality_gate",
    label="标定质量判据",
    func=lambda **kw: kw,
    help="载入标定后据此判定 good/warn/fail；strict 模式下不合格会中止扫描",
    tags=("质量", "校验"),
    params=(
        ParamSpec("quality.camera_rms_good", float, 0.3,
                  low=0.0, high=50.0, unit="px", short="camera_rms_good",
                  label="相机 RMS 合格线"),
        ParamSpec("quality.camera_rms_warn", float, 0.5,
                  low=0.0, high=50.0, unit="px", short="camera_rms_warn",
                  label="相机 RMS 警告线"),
        ParamSpec("quality.laser_rms_good", float, 0.3,
                  low=0.0, high=50.0, unit="mm", short="laser_rms_good",
                  label="激光平面 RMS 合格线"),
        ParamSpec("quality.laser_rms_warn", float, 0.5,
                  low=0.0, high=50.0, unit="mm", short="laser_rms_warn",
                  label="激光平面 RMS 警告线"),
        ParamSpec("quality.axis_rms_good", float, 0.1,
                  low=0.0, high=50.0, unit="mm", short="axis_rms_good",
                  label="转轴 RMS 合格线"),
        ParamSpec("quality.axis_rms_warn", float, 0.3,
                  low=0.0, high=50.0, unit="mm", short="axis_rms_warn",
                  label="转轴 RMS 警告线"),
    ),
)


specs = (CAMERA_CALIB_SPEC, LASER_CALIB_SPEC, AXIS_CALIB_SPEC, QUALITY_SPEC)

for _spec in specs:
    CALIBRATORS.add_spec(_spec)


def quality_thresholds(cfg) -> dict:
    """从配置构造 ``validate_calibrations`` 需要的阈值覆盖字典。

    返回 ``{"camera": {"good":…, "warn":…}, "laser": {...}, "axis": {...}}``；
    某项两个阈值都没配置时**不出现**该键（用 quality 模块默认值）。

    这是对重构前 ``ScanPipeline._quality_thresholds`` 的替代实现：
    那里手工拼 ``f"{prefix}_good"`` 字符串，前缀写错不会报错。
    这里用声明过的 ``ParamSpec.name``，拼写错误在 import 期就暴露。
    """
    out: dict[str, dict] = {}
    for family in ("camera", "laser", "axis"):
        good_key = f"quality.{family}_rms_good"
        warn_key = f"quality.{family}_rms_warn"
        pair = {}
        for key, name in ((good_key, "good"), (warn_key, "warn")):
            spec = next((p for p in QUALITY_SPEC.params if p.name == key), None)
            if spec is None:
                continue
            value = cfg.get(key, None)
            if value is not None:
                pair[name] = spec.coerce(value)
        if pair:
            out[family] = pair
    return out
