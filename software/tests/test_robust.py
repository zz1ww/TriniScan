"""稳健估计（IRLS / RANSAC / MAD）单元测试。"""
import numpy as np
import pytest

from triniscan.common.robust import (
    huber_weights,
    irls_plane_fit,
    mad_sigma,
    ransac_plane_fit,
    robust_circle_3d_fit,
    robust_plane_fit,
    tukey_weights,
)


def _random_plane_points(n=200, noise=1e-4, seed=0, plane=None):
    """在已知平面上生成带噪声点。"""
    rng = np.random.default_rng(seed)
    if plane is None:
        normal = np.array([0.2, -0.3, 0.93])
        normal /= np.linalg.norm(normal)
        d = -0.25
        plane = np.array([*normal, d])
    n_vec = plane[:3]
    p0 = -plane[3] * n_vec
    a = np.cross(n_vec, [1, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n_vec, a)
    uv = rng.uniform(-0.05, 0.05, size=(n, 2))
    pts = p0 + uv[:, :1] * a + uv[:, 1:] * b
    pts = pts + rng.normal(scale=noise, size=pts.shape)
    return pts, plane


# ---------------------------------------------------------------------------
def test_mad_sigma_robust_to_outliers():
    clean = np.zeros(1000)
    noisy = np.concatenate([clean, np.full(50, 1000.0)])  # 5% 离群
    # MAD 估计不随离群点爆炸
    assert mad_sigma(noisy) < 1.0


def test_mad_sigma_zero_for_constant():
    assert mad_sigma(np.full(10, 3.0)) == 0.0
    assert mad_sigma(np.array([])) == 0.0


def test_huber_weights_bounds():
    r = np.array([0.0, 1.0, 2.0, 10.0])
    w = huber_weights(r, delta=1.0)
    assert w[0] == 1.0 and w[1] == 1.0
    assert w[2] < 1.0 and w[3] < w[2]
    assert np.all((w > 0) & (w <= 1))


def test_tukey_weights_hard_reject():
    r = np.array([0.0, 1.0, 100.0])
    w = tukey_weights(r, c=4.685)
    assert w[0] == 1.0
    assert w[2] == 0.0  # 超阈值直接置零


# ---------------------------------------------------------------------------
def test_irls_plane_recovers_with_outliers():
    pts, plane = _random_plane_points(noise=1e-5, seed=1)
    rng = np.random.default_rng(2)
    idx = rng.choice(len(pts), size=40, replace=False)
    pts[idx] += rng.normal(scale=0.02, size=(40, 3))  # 20% 大离群
    fit = irls_plane_fit(pts)
    # 法向方向一致（允许符号）
    assert abs(np.dot(fit.plane[:3], plane[:3])) > 0.99
    assert fit.num_inliers >= len(pts) - 50


def test_ransac_plane_beats_lsq_with_outliers():
    pts, plane = _random_plane_points(noise=1e-5, seed=3)
    rng = np.random.default_rng(4)
    idx = rng.choice(len(pts), size=80, replace=False)
    pts[idx] += rng.normal(scale=0.05, size=(80, 3))  # 40% 离群
    fit = robust_plane_fit(pts, method="ransac", outlier_threshold=0.002)
    assert abs(np.dot(fit.plane[:3], plane[:3])) > 0.999
    # 与最小二乘对比：RANSAC 更接近真实平面
    lsq = robust_plane_fit(pts, method="svd")
    assert abs(np.dot(fit.plane[:3], plane[:3])) >= \
        abs(np.dot(lsq.plane[:3], plane[:3]))


def test_robust_plane_fit_methods():
    pts, _ = _random_plane_points(noise=1e-5, seed=5)
    for m in ("svd", "irls", "ransac"):
        fit = robust_plane_fit(pts, method=m)
        assert fit.plane.shape == (4,)
        assert np.isclose(np.linalg.norm(fit.plane[:3]), 1.0)


def test_robust_plane_fit_bad_method():
    pts, _ = _random_plane_points()
    with pytest.raises(ValueError):
        robust_plane_fit(pts, method="nope")


def test_plane_sign_convention():
    """相机原点代入平面方程应为负（法向背离相机）。"""
    pts, _ = _random_plane_points(noise=1e-6, seed=6)
    fit = robust_plane_fit(pts, method="irls")
    assert fit.plane[3] <= 0


# ---------------------------------------------------------------------------
def test_ransac_circle_recovers_axis():
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    r = 0.05
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta),
                           np.full_like(theta, 0.03)])
    rng = np.random.default_rng(7)
    idx = rng.choice(len(pts), size=5, replace=False)
    pts[idx] += rng.normal(scale=0.01, size=(5, 3))  # 跳点
    fit = robust_circle_3d_fit(pts, method="ransac", threshold=0.002)
    assert np.allclose(fit.center, [0, 0, 0.03], atol=2e-3)
    assert abs(fit.normal[2]) > 0.99
    assert np.isclose(fit.radius, 0.05, atol=2e-3)


def test_circle_lsq_exact():
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([0.04 * np.cos(theta), 0.04 * np.sin(theta),
                           np.zeros_like(theta)])
    fit = robust_circle_3d_fit(pts, method="lsq")
    assert fit.residual_rms < 1e-9


def test_circle_too_few_points():
    with pytest.raises(ValueError):
        robust_circle_3d_fit(np.zeros((2, 3)))
