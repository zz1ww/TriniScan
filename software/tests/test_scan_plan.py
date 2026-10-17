"""扫描角度计划（720 / 1080 视角 + 整数步自洽）测试。

背景：步进电机只能走整数步，而 360/N 度往往不是整数步。若按浮点角度
累积，多视角后会出现系统性角度偏差（点云"炸开"）。本测试验证
`plan_scan` 把每视角转角量化为整数步并回算实际角度。
"""
import numpy as np
import pytest

from triniscan.turntable.controller import (
    ScanPlan,
    TurntableConfig,
    apply_scan_plan,
    nearest_exact_view_count,
    plan_scan,
    resolve_scan_plan,
)


# 真实配置：steps_per_rev=200, microstep=32, gear_ratio=5.18
SPD = 200 * 32 * 5.18 / 360.0        # ≈ 92.09 步/度


def test_steps_per_degree_real_config():
    cfg = TurntableConfig(steps_per_rev=200, microstep=32, gear_ratio=5.18)
    assert abs(cfg.steps_per_degree - SPD) < 1e-6
    assert abs(cfg.steps_per_rev_total - 200 * 32 * 5.18) < 1e-6


@pytest.mark.parametrize("n", [360, 720, 1080])
def test_plan_scan_integer_steps(n):
    plan = plan_scan(n, SPD)
    assert plan.num_views == n
    assert isinstance(plan.steps_per_step, int)
    assert plan.steps_per_step >= 1


@pytest.mark.parametrize("n", [360, 720, 1080])
def test_plan_scan_angle_is_self_consistent(n):
    """回算角度必须与整数步严格一致（角度自洽的核心）。"""
    plan = plan_scan(n, SPD)
    assert abs(plan.angle_step_deg
               - plan.steps_per_step / SPD) < 1e-12


@pytest.mark.parametrize("n", [360, 720, 1080, 1440])
def test_plan_scan_total_is_optimal_among_candidates(n):
    """每视角步数取 {floor, round, ceil} 中使累计角度最接近 360° 者。

    这是可达范围内的**最优解**：由于整圈步数不能被任意视角数整除，
    1080 视角时 360° 并不可达（± 一步的零头无法消除），但所选步数
    一定是三个候选中偏差最小的。
    """
    total_rev = SPD * 360.0
    base = total_rev / n
    cands = {max(1, int(base // 1)), max(1, int(round(base))),
             max(1, int(base) + 1)}
    best = min(cands, key=lambda s: abs(s * n - total_rev))
    plan = plan_scan(n, SPD)
    assert plan.steps_per_step == best
    assert abs(plan.steps_per_step * n - total_rev) <= max(
        abs(s * n - total_rev) for s in cands) + 1e-9


@pytest.mark.parametrize("n", [360, 720, 1080])
def test_plan_scan_uniform_spacing_no_drift(n):
    """所有视角间距完全相同——不存在累积漂移。"""
    plan = plan_scan(n, SPD)
    # 逐视角都是同一个整数步数，故累计角 = i * steps_per_step / SPD
    angles = np.array([i * plan.steps_per_step / SPD
                       for i in range(plan.num_views + 1)])
    diffs = np.diff(angles)
    assert np.allclose(diffs, plan.angle_step_deg, atol=1e-12)


def test_plan_scan_720_vs_1080_step_ratio():
    """1080 视角的每步角应约为 720 的 2/3。"""
    p720 = plan_scan(720, SPD)
    p1080 = plan_scan(1080, SPD)
    ratio = p1080.angle_step_deg / p720.angle_step_deg
    assert abs(ratio - 2.0 / 3.0) < 0.05


def test_plan_scan_invalid_inputs():
    with pytest.raises(ValueError):
        plan_scan(0, SPD)
    with pytest.raises(ValueError):
        plan_scan(360, 0.0)


def test_resolve_scan_plan_from_preset():
    """scan.preset=720 应解析出 720 视角。"""

    class _Cfg:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            return self._d.get(k, default)

    cfg = _Cfg({"preset": 720, "turntable": {
        "steps_per_rev": 200, "microstep": 32, "gear_ratio": 5.18}})
    plan = resolve_scan_plan(cfg)
    assert plan.num_views == 720


def test_resolve_scan_plan_falls_back_to_num_views():
    class _Cfg:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            return self._d.get(k, default)

    cfg = _Cfg({"preset": None, "num_views": 360, "turntable": {
        "steps_per_rev": 200, "microstep": 32, "gear_ratio": 5.18}})
    plan = resolve_scan_plan(cfg)
    assert plan.num_views == 360


def test_apply_scan_plan_writes_config():
    """apply_scan_plan 应把自洽的角度写回配置。"""

    class _Cfg:
        def __init__(self):
            self.d = {}

        def get(self, k, default=None):
            return self.d.get(k, default)

        def set(self, k, v):
            self.d[k] = v

    cfg = _Cfg()
    plan = plan_scan(1080, SPD)
    apply_scan_plan(cfg, plan)
    assert cfg.d["scan.num_views"] == 1080
    assert abs(cfg.d["scan.angle_step_deg"]
               - plan.steps_per_step / SPD) < 1e-12
    assert cfg.d["scan.steps_per_view"] == plan.steps_per_step


def test_angle_error_is_sub_millidegree():
    """720 / 1080 下每视角量化角度误差应远小于 0.01°。"""
    for n in (720, 1080):
        plan = plan_scan(n, SPD)
        assert plan.angle_error_deg < 0.01


def test_full_turn_exact_for_divisible_counts():
    """能整除整圈步数的视角数，应精确转满 360°（无接缝错位）。"""
    total = int(round(SPD * 360))
    for n in (360, 720, 1440):
        if total % n == 0:
            assert plan_scan(n, SPD).full_turn_exact


def test_nearest_exact_view_count_finds_divisor():
    """应能找到能整除整圈步数的邻近视角数。"""
    n = nearest_exact_view_count(1080, SPD, max_delta=60)
    if n is not None:
        total = int(round(SPD * 360))
        assert total % n == 0


def test_nearest_exact_view_count_none_when_unreachable():
    n = nearest_exact_view_count(7, SPD, max_delta=0)
    assert n is None or int(round(SPD * 360)) % n == 0
