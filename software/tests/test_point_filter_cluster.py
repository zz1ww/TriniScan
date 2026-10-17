"""连通域（欧氏聚类）去噪测试。"""
import numpy as np

from triniscan.reconstruction.point_filter import remove_small_clusters


def _cloud(points):
    return np.asarray(points, dtype=np.float64)


def test_removes_isolated_small_cluster():
    """一个大簇（物体）+ 一个远处小簇（反光），小簇应被剔除。"""
    rng = np.random.default_rng(0)
    obj = rng.normal(0, 0.005, size=(500, 3))        # 大簇
    blob = rng.normal([0.2, 0.2, 0.2], 0.002, size=(30, 3))  # 远处小簇
    pts = np.vstack([obj, blob])
    res = remove_small_clusters(pts, eps=0.01, min_points=5, keep_top=1)
    assert res.applied
    # 保留点应全部靠近原点
    assert np.abs(res.points).max() < 0.1
    assert res.num_removed >= 30


def test_keeps_all_when_no_small_cluster():
    """只有一个大簇时，不应删除任何点。"""
    rng = np.random.default_rng(1)
    pts = rng.normal(0, 0.005, size=(500, 3))
    res = remove_small_clusters(pts, eps=0.01, min_points=5, keep_top=1)
    assert not res.applied
    assert res.num_removed == 0


def test_keep_top_two_clusters():
    """keep_top=2 时应保留两个最大的簇。"""
    rng = np.random.default_rng(2)
    a = rng.normal([0.0, 0.0, 0.0], 0.003, size=(300, 3))
    b = rng.normal([0.3, 0.0, 0.0], 0.003, size=(200, 3))
    c = rng.normal([0.6, 0.0, 0.0], 0.002, size=(20, 3))
    pts = np.vstack([a, b, c])
    res = remove_small_clusters(pts, eps=0.02, min_points=5, keep_top=2)
    assert res.applied
    # a、b 保留、c 剔除
    assert len(res.points) > 450


def test_removes_dbscan_noise_label():
    """DBSCAN 的 -1（噪声）点应被剔除。"""
    rng = np.random.default_rng(3)
    obj = rng.normal(0, 0.004, size=(400, 3))
    noise = np.array([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])   # 孤立点
    pts = np.vstack([obj, noise])
    res = remove_small_clusters(pts, eps=0.01, min_points=5, keep_top=1)
    assert np.abs(res.points).max() < 0.1


def test_small_cloud_skipped():
    """点数过少时跳过（避免误删）。"""
    pts = np.zeros((10, 3))
    res = remove_small_clusters(pts, eps=0.01, min_points=5)
    assert not res.applied


def test_empty_cloud():
    res = remove_small_clusters(np.zeros((0, 3)))
    assert not res.applied
    assert res.num_removed == 0


def test_eps_too_small_returns_safely():
    """eps/min_points 过严导致无簇时，应**安全返回**而非删光点云。"""
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 0.1, size=(400, 3))
    res = remove_small_clusters(pts, eps=0.001, min_points=3, keep_top=1)
    # 无有效簇 -> 不启用过滤，原样返回（安全兜底）
    assert not res.applied
    assert res.num_removed == 0
    assert len(res.points) == len(pts)
