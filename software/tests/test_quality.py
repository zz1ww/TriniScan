"""标定质量评估单元测试。"""
import numpy as np
import pytest

from triniscan.calibration.quality import (
    DEFAULT_THRESHOLDS,
    QualityLevel,
    assess_axis,
    assess_camera,
    assess_laser,
    format_reports,
    reports_to_dict,
    validate_calibrations,
)


class _Fake:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _cam(rms, images=15, per_view=None):
    return _Fake(rms=rms, num_images=images, per_view_errors=per_view or [])


def _laser(rms_m, views=6, points=600, counts=None):
    return _Fake(residual_rms=rms_m, num_views=views, num_points=points,
                 view_point_counts=counts or [100] * views)


def _axis(rms_m, points=12):
    return _Fake(residual_rms=rms_m, num_points=points, radius=0.05)


# ---------------------------------------------------------------------------
def test_camera_ok_warn_fail():
    assert assess_camera(_cam(0.2)).level is QualityLevel.OK
    assert assess_camera(_cam(0.4)).level is QualityLevel.WARN
    assert assess_camera(_cam(0.9)).level is QualityLevel.FAIL


def test_laser_ok_warn_fail():
    assert assess_laser(_laser(0.2e-3)).level is QualityLevel.OK
    assert assess_laser(_laser(0.4e-3)).level is QualityLevel.WARN
    assert assess_laser(_laser(16e-3)).level is QualityLevel.FAIL


def test_axis_ok_warn_fail():
    assert assess_axis(_axis(0.05e-3)).level is QualityLevel.OK
    assert assess_axis(_axis(0.2e-3)).level is QualityLevel.WARN
    assert assess_axis(_axis(2e-3)).level is QualityLevel.FAIL


def test_laser_few_views_downgraded():
    # 残差优秀但视角不足 → 至少 WARN
    rep = assess_laser(_laser(0.1e-3, views=2))
    assert rep.level.rank >= QualityLevel.WARN.rank


def test_camera_warns_on_few_images():
    rep = assess_camera(_cam(0.1, images=4))
    assert rep.level.rank >= QualityLevel.WARN.rank


def test_camera_flags_suspicious_view():
    # 3 张正常(0.1) + 1 张明显坏图(5.0) → 稳健尺度应识别出坏图
    rep = assess_camera(_cam(0.3, per_view=[0.1, 0.1, 0.1, 5.0]))
    assert "suspicious_view" in rep.details
    assert rep.details["suspicious_view"] == 3


def test_nan_is_fail():
    assert assess_camera(_cam(float("nan"))).level is QualityLevel.FAIL


def test_strict_mode_raises_on_fail():
    with pytest.raises(RuntimeError):
        validate_calibrations(camera=_cam(2.0), strict=True)


def test_non_strict_returns_reports():
    reports = validate_calibrations(camera=_cam(2.0), laser=_laser(0.1e-3),
                                    strict=False)
    assert len(reports) == 2
    assert any(r.level is QualityLevel.FAIL for r in reports)


def test_threshold_override():
    reports = validate_calibrations(
        laser=_laser(0.4e-3),
        thresholds={"laser": {"good": 0.5e-3, "warn": 1.0e-3}},
    )
    assert reports[0].level is QualityLevel.OK


def test_format_and_serialize():
    reports = [assess_camera(_cam(0.2)), assess_laser(_laser(16e-3))]
    text = format_reports(reports)
    assert "标定质量报告" in text
    data = reports_to_dict(reports)
    assert data[0]["level"] == "ok"
    assert data[1]["unit"] == "mm"
    assert data[1]["value"] == pytest.approx(16.0, rel=0.1)


def test_default_thresholds_sane():
    assert DEFAULT_THRESHOLDS["camera"]["good"] < \
        DEFAULT_THRESHOLDS["camera"]["warn"]
    assert DEFAULT_THRESHOLDS["laser"]["warn"] == pytest.approx(0.5e-3)
