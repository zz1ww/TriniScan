"""报告与掩膜：让"每个点为什么被删"可追溯、可落盘。

问题
----
重构前的点云清洗是**黑箱串行删除**：

.. code-block:: python

    pcd, _ = self._filter_bounds(pcd, ...)     # 换了新对象
    pcd, _ = self._filter_clusters(pcd, ...)   # 换了新对象，索引断了
    pcd, _ = self._remove_turntable(pcd, ...)  # 又换了
    pcd, _ = self._clip_axial_band(pcd, ...)

三个后果：

1. **索引丢失** —— 第一个过滤器删完后，第二个看到的是新点云，
   它在架构上就无法回答"第 137 号点被谁删了"；
2. **只能开关二分猜** —— 排查"少一面"只能一遍遍改配置重扫；
3. **算了的掩膜没人用** —— ``point_filter.py`` 每个函数都返回
   ``keep_mask``，但全项目没有一行代码读它。

方案
----
:class:`MaskStack` 让所有过滤器**针对同一份原始点云**各自给出掩膜，
最后一次性逻辑与。:class:`StageReport` / :class:`ScanReport` 把每个
阶段的输入/输出/参数/告警记录成结构化数据，扫描结束落盘 JSON。

关键性质
--------
- **顺序无关**：掩膜是逻辑与，可交换；换过滤顺序结果一致（除少数
  依赖上一阶段点云内容的算法，见各插件 ``needs_current_cloud`` 声明）。
- **可干跑**：``MaskStack.dry_run=True`` 时只统计不删点，用于诊断。
- **可回退**：每层掩膜独立保存，可任选子集重新组合。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["MaskLayer", "MaskStack", "StageReport", "ScanReport", "Timer"]


# ---------------------------------------------------------------------------
# 掩膜
# ---------------------------------------------------------------------------
@dataclass
class MaskLayer:
    """一层过滤的结果。

    Attributes
    ----------
    name : 层名（通常等于插件注册名）
    mask : (N,) bool，True 表示**保留**。N 必须等于原始点数。
    reason : 人可读的原因（会进报告）
    params : 实际生效的参数快照（便于复现）
    applied : 该层是否真的做了判断（未启用/无约束时 False）
    extra : 算法特有指标（簇数量、簇大小、h_table 估计值…）
    depends_on_previous : 该层的判断基于"上一层过滤后的点云"，
        因此**不能与其他层交换顺序**。默认为 False（纯局部判断）。
    """

    name: str
    mask: np.ndarray
    reason: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    applied: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    depends_on_previous: bool = False

    @property
    def removed(self) -> int:
        return int((~self.mask).sum())

    def to_dict(self, n_total: int) -> dict:
        return {
            "name": self.name,
            "applied": self.applied,
            "removed": self.removed,
            "removed_ratio": round(self.removed / n_total, 6)
            if n_total else 0.0,
            "reason": self.reason,
            "params": _jsonable(self.params),
            "extra": _jsonable(self.extra),
            "depends_on_previous": self.depends_on_previous,
        }


class MaskStack:
    """所有过滤层的掩膜集合（针对同一份原始点云）。

    Examples
    --------
    >>> stack = MaskStack(n=1000)
    >>> stack.add("cluster", keep)                  # (1000,) bool
    >>> kept = stack.combined()
    >>> print(stack.summary())
    """

    def __init__(self, n: int, dry_run: bool = False) -> None:
        self.n = int(n)
        self.dry_run = dry_run
        self.layers: dict[str, MaskLayer] = {}
        self._order: list[str] = []

    # ------------------------------------------------------------------
    def add(self, name: str, mask: np.ndarray, *,
            reason: str = "", params: Optional[dict] = None,
            applied: bool = True, extra: Optional[dict] = None,
            depends_on_previous: bool = False) -> MaskLayer:
        """登记一层掩膜。

        Raises
        ------
        ValueError
            掩膜形状与原始点数不符 —— 这通常是"过滤器在上一层的
            子集上计算，却拿回来当全量掩膜"的 bug，必须立即暴露。
        """
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.shape != (self.n,):
            raise ValueError(
                f"[{name}] 掩膜形状 {mask.shape} 与原始点数 ({self.n},) 不符；"
                "过滤器必须基于**原始点云**产出掩膜，而不是上一层的结果")
        layer = MaskLayer(name=name, mask=mask, reason=reason,
                          params=dict(params or {}), applied=applied,
                          extra=dict(extra or {}),
                          depends_on_previous=depends_on_previous)
        if name not in self.layers:
            self._order.append(name)
        self.layers[name] = layer
        return layer

    def add_pass(self, name: str, reason: str = "未启用") -> MaskLayer:
        """登记一层"不做任何删除"的掩膜（未启用 / 无约束）。"""
        return self.add(name, np.ones(self.n, dtype=bool),
                        reason=reason, applied=False)

    # ------------------------------------------------------------------
    def combined(self) -> np.ndarray:
        """所有层逻辑与。无任何层时返回全 True。"""
        out = np.ones(self.n, dtype=bool)
        for name in self._order:
            out &= self.layers[name].mask
        return out

    @property
    def n_keep(self) -> int:
        """最终保留点数（所有层逻辑与）。"""
        return int(self.combined().sum())

    @property
    def n_removed(self) -> int:
        """最终剔除点数。"""
        return self.n - self.n_keep

    def cumulative(self) -> dict[str, int]:
        """按登记顺序返回"每层之后的剩余点数"，用于发现耦合。"""
        out: dict[str, int] = {}
        cur = np.ones(self.n, dtype=bool)
        for name in self._order:
            cur = cur & self.layers[name].mask
            out[name] = int(cur.sum())
        return out

    def applying_order(self) -> list[str]:
        """返回登记顺序（诊断用）。"""
        return list(self._order)

    # ------------------------------------------------------------------
    def summary(self, warn_ratio: float = 0.15) -> str:
        """生成可读的删除统计表 —— **排查"少一面"的入口**。

        Parameters
        ----------
        warn_ratio : 单层删除比例超过此值时打 ``⚠️``
        """
        if self.n == 0:
            return "点云为空"
        lines = [f"点云掩膜汇总（原始 {self.n} 点"
                 + (f"，dry-run 未真正删除" if self.dry_run else "") + "）"]
        for name in self._order:
            layer = self.layers[name]
            if not layer.applied:
                lines.append(f"  {name:16s} —      （{layer.reason}）")
                continue
            pct = layer.removed / self.n * 100
            flag = " ⚠️" if pct >= warn_ratio * 100 else ""
            lines.append(f"  {name:16s} 删 {layer.removed:7d} "
                         f"({pct:5.1f}%){flag}")
            if layer.reason:
                lines.append(f"  {'':16s}   {layer.reason}")
        kept = self.n_keep
        lines.append(f"  {'最终保留':16s} {kept:10d} "
                     f"({kept / self.n * 100:.1f}%)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_total": self.n,
            "n_kept": self.n_keep,
            "layers": [self.layers[n].to_dict(self.n) for n in self._order],
        }


# ---------------------------------------------------------------------------
# 阶段报告
# ---------------------------------------------------------------------------
@dataclass
class StageReport:
    """一个处理阶段的执行记录。

    与只记耗时的旧 ``stages: dict[str, float]`` 相比，这里额外记录
    **输入/输出数量、实际生效参数、告警、降级路径**——这些正是排查
    "结果不对但不知道哪一步错"所必需的信息。
    """

    stage: str
    n_in: Optional[int] = None
    n_out: Optional[int] = None
    n_removed: Optional[int] = None
    elapsed_s: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def warn(self, message: str) -> "StageReport":
        self.warnings.append(message)
        return self

    def fallback(self, message: str) -> "StageReport":
        """记录一次"降级路径"（如法向一致化回退、假轴兜底）。

        这些在重构前都是**静默发生**的，是隐藏风险的温床。
        """
        self.fallbacks.append(message)
        return self

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "n_removed": self.n_removed,
            "elapsed_s": round(self.elapsed_s, 3),
            "params": _jsonable(self.params),
            "warnings": list(self.warnings),
            "fallbacks": list(self.fallbacks),
            "extra": _jsonable(self.extra),
        }


@dataclass
class ScanReport:
    """一次完整扫描的全部阶段记录（可落盘 JSON）。"""

    stages: list[StageReport] = field(default_factory=list)
    masks: Optional[MaskStack] = None
    calib_fingerprint: str = ""
    total_elapsed_s: float = 0.0

    def begin(self, stage: str) -> Timer:
        """开始计时一个阶段；``with`` 结束自动 append 报告。"""
        return Timer(self, stage)

    def add(self, report: StageReport) -> None:
        self.stages.append(report)

    def to_dict(self) -> dict:
        return {
            "calib_fingerprint": self.calib_fingerprint,
            "total_elapsed_s": round(self.total_elapsed_s, 3),
            "stages": [s.to_dict() for s in self.stages],
            "masks": self.masks.to_dict() if self.masks else None,
        }

    def to_json(self, path: str) -> str:
        """落盘到 JSON（UTF-8，缩进 2）。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def to_text(self) -> str:
        lines = []
        for s in self.stages:
            head = f"[{s.stage}] {s.elapsed_s:.2f}s"
            if s.n_in is not None and s.n_out is not None:
                head += f"  {s.n_in} → {s.n_out}"
            lines.append(head)
            for w in s.warnings:
                lines.append(f"     ⚠️ {w}")
            for fb in s.fallbacks:
                lines.append(f"     ↪️ 降级: {fb}")
        if self.masks is not None:
            lines.append(self.masks.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
class Timer:
    """``with report.begin("acquire") as t:`` 自动计时并登记。

    用法::

        with report.begin("filter") as t:
            t.params = {...}
            t.n_in = 1000
            ...
            t.n_out = 800
    """

    def __init__(self, report: ScanReport, stage: str) -> None:
        self.report = report
        self.stage = stage
        self.params: dict = {}
        self.n_in: Optional[int] = None
        self.n_out: Optional[int] = None
        self.extra: dict = {}
        self.warnings: list[str] = []
        self.fallbacks: list[str] = []
        self._t0 = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        elapsed = time.perf_counter() - self._t0
        n_removed = None
        if self.n_in is not None and self.n_out is not None:
            n_removed = self.n_in - self.n_out
        self.report.add(StageReport(
            stage=self.stage, n_in=self.n_in, n_out=self.n_out,
            n_removed=n_removed, elapsed_s=elapsed,
            params=self.params, warnings=self.warnings,
            fallbacks=self.fallbacks, extra=self.extra))


# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    """把 numpy / 元组 / 嵌套 dict 转成 JSON 可序列化结构。"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
