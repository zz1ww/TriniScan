"""标定质量评估：阈值判定 + 报告。

设计目标
--------
把"标定结果是否可用"从各模块的零散判断收敛到**唯一的阈值与判定逻辑**，
避免出现"文件存在就当有效"的隐患（曾导致残差 16 mm 的激光标定一路跑到底）。

- :data:`DEFAULT_THRESHOLDS`  各标定的优/良/合格阈值（单一来源）；
- :class:`QualityLevel`       判定等级（OK / WARN / FAIL）；
- :class:`QualityReport`      单个标定的评估报告；
- :func:`assess_camera` / :func:`assess_laser` / :func:`assess_axis`
- :func:`validate_calibrations`  三标定联合校验（可严格模式抛错）；
- :func:`format_reports`      人类可读文本（日志 / 界面）；
- :func:`reports_to_dict`     可序列化（落盘 / UI）。

阈值依据（详见 docs/04_标定方案.md 的误差预算）
----------------------------------------------
- 相机内参 RMS：优 < 0.3 px，合格 < 0.5 px；
- 激光平面 RMS：优 < 0.3 mm，合格 < 0.5 mm（最关键，直接进三角测量）；
- 转轴 RMS：优 < 0.1 mm，合格 < 0.3 mm。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ...common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "QualityLevel",
    "QualityReport",
    "DEFAULT_THRESHOLDS",
    "assess_camera",
    "assess_laser",
    "assess_axis",
    "validate_calibrations",
    "format_reports",
    "reports_to_dict",
]


class QualityLevel(str, Enum):
    """标定质量等级。"""

    OK = "ok"        # 优秀，可直接使用
    WARN = "warn"    # 合格但偏差偏大，建议留意/视情重标
    FAIL = "fail"    # 不合格，应重新标定

    @property
    def symbol(self) -> str:
        return {QualityLevel.OK: "✓", QualityLevel.WARN: "!",
                QualityLevel.FAIL: "✗"}[self]

    @property
    def rank(self) -> int:
        return {QualityLevel.OK: 0, QualityLevel.WARN: 1,
                QualityLevel.FAIL: 2}[self]


# ---------------------------------------------------------------------------
# 阈值（单一来源）
# ---------------------------------------------------------------------------
#: 各标定的好/合格阈值。键含义：``good`` 以下为 OK，``warn`` 以下为 WARN，
#: 超过 ``warn`` 为 FAIL。单位：相机 px，激光/转轴 m。
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "camera": {"good": 0.30, "warn": 0.50},        # px
    "laser": {"good": 0.30e-3, "warn": 0.50e-3},   # m
    "axis": {"good": 0.10e-3, "warn": 0.30e-3},    # m
}


def _thresholds(overrides: Optional[dict] = None) -> dict:
    """合并用户阈值覆盖（深拷贝避免污染默认值）。"""
    merged = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    if overrides:
        for name, vals in overrides.items():
            if name in merged and isinstance(vals, dict):
                merged[name].update(
                    {k: float(v) for k, v in vals.items() if k in ("good", "warn")}
                )
    return merged


def _level(value: float, th: dict) -> QualityLevel:
    """按阈值给出等级；非有限值判为 FAIL。"""
    if value is None or not np.isfinite(value):
        return QualityLevel.FAIL
    if value <= th["good"]:
        return QualityLevel.OK
    if value <= th["warn"]:
        return QualityLevel.WARN
    return QualityLevel.FAIL


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
@dataclass
class QualityReport:
    """单个标定的质量报告。"""

    name: str
    level: QualityLevel
    metric: str                     # 指标名，如 "重投影 RMS"
    value: float                    # 数值
    unit: str                       # 单位
    message: str = ""               # 人类可读结论
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.level is not QualityLevel.FAIL

    def one_line(self) -> str:
        return (f"{self.level.symbol} {self.name}: {self.metric} "
                f"{self.value:.4g} {self.unit} [{self.level.value}]"
                + (f" — {self.message}" if self.message else ""))


# ---------------------------------------------------------------------------
# 各标定评估
# ---------------------------------------------------------------------------
def assess_camera(result, thresholds: Optional[dict] = None) -> QualityReport:
    """评估相机内参标定（依据重投影 RMS 与逐图误差）。"""
    th = _thresholds(thresholds)["camera"]
    rms = float(getattr(result, "rms", float("nan")))
    level = _level(rms, th)

    details: dict[str, Any] = {
        "num_images": int(getattr(result, "num_images", 0)),
        "good": th["good"], "warn": th["warn"],
    }
    per_view = list(getattr(result, "per_view_errors", []) or [])
    if per_view:
        arr = np.asarray(per_view, dtype=np.float64)
        details["per_view_max"] = float(arr.max())
        details["per_view_mean"] = float(arr.mean())
        # 若单张误差显著偏离其余（用中位数+稳健尺度判断），提示存在坏图
        if arr.size >= 4:
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826
            # 稳健尺度为 0（多数值相同）时，用中位数本身的相对倍数兜底
            if (mad > 1e-12 and arr.max() > med + 5.0 * mad) or \
                    (mad <= 1e-12 and med > 1e-12
                     and arr.max() > 3.0 * med):
                details["suspicious_view"] = int(np.argmax(arr))
    if details["num_images"] and details["num_images"] < 6:
        level = max(level, QualityLevel.WARN, key=lambda l: l.rank)

    msg = {
        QualityLevel.OK: "内参良好",
        QualityLevel.WARN: "内参偏差偏大，建议补拍/检查对焦与畸变建模",
        QualityLevel.FAIL: "内参不合格，请重新标定（检查对焦、靶标平整、视场覆盖）",
    }[level]
    return QualityReport("相机内参", level, "重投影 RMS", rms, "px",
                         msg, details)


def assess_laser(result, thresholds: Optional[dict] = None) -> QualityReport:
    """评估激光平面标定（依据拟合残差 RMS 与视角数）。"""
    th = _thresholds(thresholds)["laser"]
    rms_m = float(getattr(result, "residual_rms", float("nan")))
    level = _level(rms_m, th)

    num_views = int(getattr(result, "num_views", 0))
    num_points = int(getattr(result, "num_points", 0))
    details: dict[str, Any] = {
        "num_views": num_views, "num_points": num_points,
        "view_point_counts": list(getattr(result, "view_point_counts", []) or []),
        "good_mm": th["good"] * 1e3, "warn_mm": th["warn"] * 1e3,
    }
    # 视角不足（<4）或视角点分布严重不均 → 至少 WARN
    if num_views < 4:
        level = max(level, QualityLevel.WARN, key=lambda l: l.rank)
        details["reason_view_count"] = f"有效视角仅 {num_views}（建议 >= 4）"
    counts = details["view_point_counts"]
    if counts and min(counts) < 0.2 * max(counts):
        level = max(level, QualityLevel.WARN, key=lambda l: l.rank)
        details["reason_imbalance"] = "各视角贡献点数严重不均"

    msg = {
        QualityLevel.OK: "激光平面良好",
        QualityLevel.WARN: "激光平面残差偏大，建议重标或检查棋盘格/中心线",
        QualityLevel.FAIL: ("激光平面不合格，几乎不可用：请检查靶标尺寸、"
                            "棋盘格参数、激光线提取与多姿态覆盖"),
    }[level]
    return QualityReport("激光平面", level, "拟合残差 RMS",
                         rms_m * 1e3, "mm", msg, details)


def assess_axis(result, thresholds: Optional[dict] = None,
                ref_direction: Optional[np.ndarray] = None,
                ref_tol_deg: float = 25.0) -> QualityReport:
    """评估转轴标定（依据圆拟合残差 RMS 与点数）。

    Parameters
    ----------
    result : AxisCalibResult
    thresholds : 阈值覆盖
    ref_direction : (3,) 可选参考方向（如激光线走向/重力），用于校验
        拟合轴的方向是否合理。二者夹角超过 ``ref_tol_deg`` 判为 FAIL
        ——这能抓住"拟合圆退化、方向乱指"的坏标定。
    ref_tol_deg : 方向一致性容差（度）
    """
    th = _thresholds(thresholds)["axis"]
    rms_m = float(getattr(result, "residual_rms", float("nan")))
    level = _level(rms_m, th)

    num_points = int(getattr(result, "num_points", 0))
    radius_m = float(getattr(result, "radius", 0.0))
    details: dict[str, Any] = {
        "num_points": num_points,
        "radius_m": radius_m,
        "good_mm": th["good"] * 1e3, "warn_mm": th["warn"] * 1e3,
    }
    if num_points and num_points < 6:
        level = max(level, QualityLevel.WARN, key=lambda l: l.rank)
        details["reason_point_count"] = f"特征点仅 {num_points}（建议 >= 8）"

    # 半径过小 = 特征点几乎挤在一起 = 取点失败（本系统曾因此写出坏轴）
    if abs(radius_m) * 1e3 < 2.0:
        level = QualityLevel.FAIL
        details["reason_radius"] = (
            f"拟合圆半径仅 {radius_m * 1e3:.2f}mm，特征点几乎重合")

    # 方向一致性：拟合轴应与参考方向（激光线走向）大致平行
    if ref_direction is not None:
        d = np.asarray(getattr(result, "direction", [0, 0, 1]),
                       dtype=np.float64).reshape(3)
        r = np.asarray(ref_direction, dtype=np.float64).reshape(3)
        nd, nr = np.linalg.norm(d), np.linalg.norm(r)
        if nd > 1e-12 and nr > 1e-12:
            cos = abs(float(d @ r) / (nd * nr))
            cos = max(-1.0, min(1.0, cos))
            angle_deg = float(np.degrees(np.arccos(cos)))
            details["axis_ref_angle_deg"] = angle_deg
            if angle_deg > ref_tol_deg:
                level = QualityLevel.FAIL
                details["reason_direction"] = (
                    f"拟合轴与参考方向夹角 {angle_deg:.1f}° "
                    f"> {ref_tol_deg:.0f}°，方向不合理")

    msg = {
        QualityLevel.OK: "转轴良好",
        QualityLevel.WARN: "转轴残差偏大，建议增加采样角度或检查标记点提取",
        QualityLevel.FAIL: "转轴不合格，请重新标定（检查标记球、角度记录与环境光）",
    }[level]
    return QualityReport("转轴", level, "圆拟合残差 RMS",
                         rms_m * 1e3, "mm", msg, details)


# ---------------------------------------------------------------------------
# 联合校验
# ---------------------------------------------------------------------------
def validate_calibrations(
    camera=None,
    laser=None,
    axis=None,
    thresholds: Optional[dict] = None,
    strict: bool = False,
) -> list[QualityReport]:
    """对已加载的标定做联合质量校验。

    Parameters
    ----------
    camera, laser, axis : 各标定结果对象（可为 None，None 表示缺失）
    thresholds : 阈值覆盖（可选）
    strict : True 时若有任何 FAIL 则抛 :class:`RuntimeError`

    Returns
    -------
    list[QualityReport]

    Raises
    ------
    RuntimeError
        ``strict=True`` 且存在不合格项时。
    """
    reports: list[QualityReport] = []
    if camera is not None:
        reports.append(assess_camera(camera, thresholds))
    if laser is not None:
        reports.append(assess_laser(laser, thresholds))
    if axis is not None:
        reports.append(assess_axis(axis, thresholds))

    for rep in reports:
        log.info("标定质量 %s", rep.one_line())

    if strict:
        bad = [r for r in reports if r.level is QualityLevel.FAIL]
        if bad:
            detail = "\n".join(r.one_line() for r in bad)
            raise RuntimeError(
                "标定质量不合格，已按严格模式中止：\n" + detail)
    return reports


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def format_reports(reports: list[QualityReport]) -> str:
    """把报告列表格式化为多行文本（日志 / 界面用）。"""
    if not reports:
        return "（无标定质量报告）"
    lines = ["标定质量报告："]
    for r in reports:
        lines.append("  " + r.one_line())
        for key, val in r.details.items():
            if key in ("good", "warn"):
                continue
            lines.append(f"      · {key}: {val}")
    return "\n".join(lines)


def reports_to_dict(reports: list[QualityReport]) -> list[dict]:
    """把报告列表转为可序列化结构（JSON / UI）。"""
    out = []
    for r in reports:
        out.append({
            "name": r.name,
            "level": r.level.value,
            "metric": r.metric,
            "value": float(r.value) if np.isfinite(r.value) else None,
            "unit": r.unit,
            "message": r.message,
            "details": _jsonable(r.details),
        })
    return out


def _jsonable(obj: Any) -> Any:
    """把 numpy 标量/数组转为原生可序列化类型。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
