"""端到端对比：基线 vs 新模块（条纹增强 + 轴基准去转盘 + 连通域去噪）。

从 data/raw 读入已采集视角，跑完整流水线（不触碰真机）。
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "software"))
os.chdir(ROOT)

from triniscan.core.config import Config
from triniscan.core.pipeline import ScanPipeline


def load_views(limit=None):
    files = sorted(glob.glob("data/raw/view_*.png"))
    if limit:
        files = files[:limit]
    step = 360.0 / len(files)
    out = []
    for i, p in enumerate(files):
        out.append({"index": i, "angle_deg": step * (i + 1),
                    "image": p,
                    "raw": cv2.imread(p, cv2.IMREAD_UNCHANGED)})
    return out


def describe(tag, path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print(f"  [{tag}] 空点云")
        return None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = (hi - lo) * 1000
    r = np.linalg.norm(pts[:, :2], axis=1) * 1000
    z = pts[:, 2] * 1000
    print(f"  [{tag}] 点数 {len(pts):>7d}")
    print(f"    包围盒: {size[0]:7.1f} x {size[1]:7.1f} x {size[2]:7.1f} mm")
    print(f"    Z 范围: [{z.min():7.1f}, {z.max():7.1f}] mm")
    print(f"    半径:   中位 {np.median(r):6.1f} | "
          f"95% {np.percentile(r, 95):6.1f} | 最大 {r.max():6.1f} mm")
    return {"points": len(pts), "bbox": size,
            "r_median": float(np.median(r)),
            "r95": float(np.percentile(r, 95)),
            "r_max": float(r.max()), "z": (float(z.min()), float(z.max()))}


def run(tag, overrides=None, limit=None):
    print(f"\n{'='*72}\n{tag}\n{'='*72}")
    cfg = Config.load("software/config/default.yaml")
    for k, v in (overrides or {}).items():
        cfg.set(k, v)
    pipe = ScanPipeline(cfg)
    pipe.acquire = lambda: load_views(limit)
    t0 = time.time()
    res = pipe.run()
    dt = time.time() - t0
    print(f"  >> 体积 {res.volume_cm3:.2f} cm³ | 点 {res.num_points} | "
          f"视角 {res.num_views} | 水密 {res.mesh_watertight}")
    print(f"  >> 转盘去除: {res.turntable_applied} ({res.turntable_removed} 点) | "
          f"配准 {res.register_method} | 封闭 {res.close_fallback}")
    if res.cross_check_cm3:
        print(f"  >> 体素校核 {res.cross_check_cm3:.2f} cm³ "
              f"(相对差 {res.cross_check_diff*100:.1f}%)")
    info = describe(tag, res.points_path) if res.points_path else None
    print(f"  >> 用时 {dt:.1f}s")
    return {"vol": res.volume_cm3, "info": info, "res": res}


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    base = run("① 基线（新功能全关）", limit=limit)
    enh = run("② 条纹增强", {
        "extraction.enhance.enable": True,
    }, limit=limit)
    both = run("③ 全开（增强+连通域+轴基准去转盘）", {
        "extraction.enhance.enable": True,
        "reconstruction.cluster_filter": True,
        "reconstruction.turntable_method": "axis_height",
    }, limit=limit)

    print(f"\n{'='*72}\n对比汇总\n{'='*72}")
    rows = [("体积 cm³", "vol"), ]
    print(f"{'配置':<14}{'点数':>9}{'体积cm³':>10}{'半径中位':>10}"
          f"{'半径95%':>10}{'半径max':>10}{'X mm':>9}{'Y mm':>9}{'Z mm':>9}")
    for name, d in (("基线", base), ("增强", enh), ("全开", both)):
        i = d["info"]
        if not i:
            continue
        print(f"{name:<14}{i['points']:>9d}{d['vol']:>10.2f}"
              f"{i['r_median']:>10.1f}{i['r95']:>10.1f}{i['r_max']:>10.1f}"
              f"{i['bbox'][0]:>9.1f}{i['bbox'][1]:>9.1f}{i['bbox'][2]:>9.1f}")
