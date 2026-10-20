"""激光中心线实验台 —— 命令行入口。

流程::

    读 config.yaml / 命令行参数
      → 逐张读图 → 预处理 → 提取中心线
      → 每张图存叠加图（原图 + 红点）
      → 三角测量还原三维点
      → 按角度旋转拼接成一个点云
      → 存 cloud.ply / cloud.xyz / summary.json

用法见 ``python lab/run.py --help``。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

# 让 ``python lab/run.py`` 能 import 同目录模块，也能找到 software/ 下的标定
_LAB = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_LAB)                 # TriniScan/
sys.path.insert(0, _LAB)

import methods                                # noqa: E402
from methods import preprocess                # noqa: E402


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:                       # pragma: no cover
        raise SystemExit("需要 PyYAML：pip install pyyaml")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _parse_params(items: list[str] | None) -> dict:
    """``--param k=v`` → dict（值自动转 int/float/bool）。"""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--param 格式应为 k=v，收到 {item!r}")
        key, raw = item.split("=", 1)
        text = raw.strip()
        if text.lower() in ("true", "false"):
            out[key.strip()] = text.lower() == "true"
        else:
            try:
                out[key.strip()] = int(text)
            except ValueError:
                try:
                    out[key.strip()] = float(text)
                except ValueError:
                    out[key.strip()] = text
    return out


# ---------------------------------------------------------------------------
# 标定读取（自己解析 npz，不依赖主程序包）
# ---------------------------------------------------------------------------
def _load_calib(cfg_calib: dict) -> tuple[np.ndarray, np.ndarray | None,
                                          np.ndarray, np.ndarray]:
    """返回 ``K, dist, plane, axis_point/dir``。

    ``calib_camera.npz`` / ``calib_laser.npz`` / ``calib_axis.npz``
    的键名由主程序约定，这里按常见键名宽松读取。
    """
    def load(path: str) -> dict:
        full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
        if not os.path.exists(full):
            raise SystemExit(f"标定文件不存在: {full}")
        with np.load(full, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    cam = load(cfg_calib["camera"])
    K = np.asarray(cam.get("K", cam.get("camera_matrix")), dtype=np.float64)
    dist = cam.get("dist", cam.get("dist_coeffs"))
    dist = None if dist is None else np.asarray(dist, dtype=np.float64)

    las = load(cfg_calib["laser"])
    plane = np.asarray(las.get("plane", las.get("laser_plane")),
                       dtype=np.float64).ravel()[:4]
    if plane.size != 4:
        raise SystemExit(f"激光平面应为 4 个系数，实际 {plane.size}")

    axis = load(cfg_calib["axis"])
    point = np.asarray(axis.get("point", axis.get("axis_point")),
                       dtype=np.float64).ravel()[:3]
    direction = np.asarray(
        axis.get("direction", axis.get("axis_dir")), dtype=np.float64).ravel()[:3]

    return K, dist, plane, (point, direction)


# ---------------------------------------------------------------------------
# 三角测量
# ---------------------------------------------------------------------------
def triangulate(pixels: np.ndarray, K: np.ndarray,
                dist: np.ndarray | None, plane: np.ndarray) -> np.ndarray:
    """像素 + 激光平面 → 相机系三维点 ``(N, 3)``。

    做法：像素 → 归一化射线（含畸变校正）→ 与平面求交。
    """
    if len(pixels) == 0:
        return np.zeros((0, 3))

    uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    has_dist = (dist is not None and np.any(np.asarray(dist) != 0))

    if has_dist:
        try:
            import cv2
        except ImportError:                   # pragma: no cover
            cv2 = None
        if cv2 is not None:
            # undistortPoints 直接返回**归一化像平面坐标**（已除以内参），
            # 因此不能再乘 inv(K)。
            rays = cv2.undistortPoints(
                uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            rays = np.column_stack([rays, np.ones(len(uv))])
        else:
            print("  [警告] 有畸变系数但未安装 cv2，跳过去畸变")
            rays = np.column_stack([
                (uv - K[:2, 2]) / np.diag(K)[:2], np.ones(len(uv))])
    else:
        # 无畸变：像素 → 归一化坐标需要显式除以内参
        rays = np.column_stack([
            (uv - K[:2, 2]) / np.diag(K)[:2], np.ones(len(uv))])

    a, b, c, d = plane
    denom = rays @ np.array([a, b, c])
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    scale = -d / denom                        # 相机在原点，t=0
    pts = rays * scale[:, None]
    return pts[np.isfinite(pts).all(axis=1)]


# ---------------------------------------------------------------------------
# 拼接
# ---------------------------------------------------------------------------
def rotate_about_axis(points: np.ndarray, angle_deg: float,
                      axis_point: np.ndarray,
                      axis_dir: np.ndarray) -> np.ndarray:
    """把点云绕给定轴旋转（罗德里格斯公式）。"""
    axis = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
    theta = np.deg2rad(angle_deg)
    k = axis
    Kx = np.array([[0, -k[2], k[1]],
                   [k[2], 0, -k[0]],
                   [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)
    return (points - axis_point) @ R.T + axis_point


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def write_ply(path: str, points: np.ndarray,
              colors: np.ndarray | None = None) -> None:
    """写二进制 PLY（无需 open3d）。"""
    n = len(points)
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green",
                   "property uchar blue"]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        xyz = np.asarray(points, dtype=np.float32)
        if colors is None:
            f.write(np.ascontiguousarray(xyz).tobytes())
        else:
            rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"),
                                     ("z", "<f4"), ("red", "u1"),
                                     ("green", "u1"), ("blue", "u1")])
            rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            rgb = np.asarray(colors, dtype=np.uint8)
            rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(rec.tobytes())


def save_overlay(path: str, image: np.ndarray,
                 points: np.ndarray) -> None:
    """原图 + 检出点（红色），用于肉眼核对每张图的提取效果。"""
    try:
        import cv2
    except ImportError:                       # pragma: no cover
        print("  [警告] 未安装 opencv-python，跳过叠加图")
        return
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for u, v in np.asarray(points).reshape(-1, 2):
        cv2.circle(vis, (int(round(u)), int(round(v))), 2,
                   (0, 0, 255), -1)
    cv2.imwrite(path, vis)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="激光中心线实验台：提取 + 点云",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="可用算法:\n" + methods.describe_all())
    p.add_argument("--config", default=os.path.join(_LAB, "config.yaml"))
    p.add_argument("--list", action="store_true", help="列出算法后退出")
    p.add_argument("--method", default=None, help="算法名")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None, help="预处理 σ")
    p.add_argument("--channel", default=None)
    p.add_argument("--invert", action="store_true", default=None)
    p.add_argument("--roi", default=None, help="x,y,w,h")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 张")
    p.add_argument("--dir", dest="image_dir", default=None, help="图像目录")
    p.add_argument("--out", dest="out_dir", default=None, help="输出目录")
    p.add_argument("--param", action="append", metavar="K=V",
                   help="算法专属参数，可重复")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--no-stack", action="store_true",
                   help="只提取，不拼点云")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print("可用算法：")
        print(methods.describe_all())
        return 0

    cfg = _load_yaml(args.config)

    # 命令行覆盖配置
    if args.method:
        cfg.setdefault("extract", {})["method"] = args.method
    if args.threshold is not None:
        cfg.setdefault("extract", {})["threshold"] = args.threshold
    if args.min_pixels is not None:
        cfg.setdefault("extract", {})["min_pixels"] = args.min_pixels
    if args.sigma is not None:
        cfg.setdefault("preprocess", {})["sigma"] = args.sigma
    if args.channel:
        cfg.setdefault("preprocess", {})["channel"] = args.channel
    if args.invert is not None:
        cfg.setdefault("preprocess", {})["invert"] = args.invert
    if args.roi:
        cfg.setdefault("preprocess", {})["roi"] = [
            int(v) for v in args.roi.split(",")]
    if args.limit is not None:
        cfg.setdefault("input", {})["limit"] = args.limit
    if args.image_dir:
        cfg.setdefault("input", {})["dir"] = args.image_dir
    if args.out_dir:
        cfg.setdefault("output", {})["dir"] = args.out_dir
    if args.no_overlay:
        cfg.setdefault("output", {})["save_overlay"] = False
    if args.param:
        _deep_update(cfg.setdefault("extract", {}).setdefault("params", {}),
                     _parse_params(args.param))

    ex = cfg.get("extract", {})
    pre = cfg.get("preprocess", {})
    method_name = str(ex.get("method", "row_peak"))
    method = methods.get(method_name)
    threshold = float(ex.get("threshold", 60.0))
    min_pixels = int(ex.get("min_pixels", 1))
    algo_params = {**method.defaults(), **(ex.get("params") or {})}

    # ---- 找图 ----
    img_dir = cfg.get("input", {}).get("dir", "data/raw")
    img_dir = img_dir if os.path.isabs(img_dir) else os.path.join(_ROOT, img_dir)
    ext = str(cfg.get("input", {}).get("ext", ".png")).lower()
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(ext))
    limit = int(cfg.get("input", {}).get("limit", 0) or 0)
    if limit > 0:
        files = files[:limit]
    if not files:
        raise SystemExit(f"目录里没有 {ext} 文件: {img_dir}")

    out_dir = cfg.get("output", {}).get("dir", "lab/outputs")
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(_ROOT, out_dir)
    overlay_dir = os.path.join(out_dir, "overlay")
    os.makedirs(out_dir, exist_ok=True)
    if cfg.get("output", {}).get("save_overlay", True):
        os.makedirs(overlay_dir, exist_ok=True)

    print(f"算法     : {method_name} — {method.label}")
    print(f"参数     : threshold={threshold} min_pixels={min_pixels} "
          f"{algo_params}")
    print(f"预处理   : channel={pre.get('channel', 'gray')} "
          f"sigma={pre.get('sigma', 1.5)} invert={pre.get('invert', False)} "
          f"roi={pre.get('roi')}")
    print(f"图像     : {len(files)} 张 ← {img_dir}")
    print(f"输出     : {out_dir}")
    print("-" * 60)

    try:
        import cv2
    except ImportError:                       # pragma: no cover
        raise SystemExit("需要 opencv-python：pip install opencv-python")

    # ---- 标定（不拼点云时可以跳过）----
    do_stack = not args.no_stack
    K = dist = plane = None
    axis_point = axis_dir = None
    if do_stack:
        K, dist, plane, (axis_point, axis_dir) = _load_calib(cfg["calib"])

    # ---- 逐张处理 ----
    per_image = []
    all_points = []
    angle_step = cfg.get("stack", {}).get("angle_step_deg")
    if angle_step is None:
        angle_step = 360.0 / max(len(files), 1)

    for i, name in enumerate(files):
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [{i:4d}] 读图失败，跳过: {name}")
            continue
        gray = preprocess(img, channel=pre.get("channel", "gray"),
                          sigma=float(pre.get("sigma", 1.5)),
                          invert=bool(pre.get("invert", False)),
                          roi=pre.get("roi"))
        pts2d = method.call(gray, threshold=threshold,
                            min_pixels=min_pixels, **algo_params)

        if cfg.get("output", {}).get("save_overlay", True):
            stem = os.path.splitext(name)[0]
            save_overlay(os.path.join(overlay_dir, f"{stem}_overlay.png"),
                         img, pts2d)

        n3d = 0
        if do_stack and len(pts2d):
            pts3d = triangulate(pts2d, K, dist, plane)
            n3d = len(pts3d)
            if n3d:
                angle = angle_step * (i + 1)
                all_points.append(rotate_about_axis(
                    pts3d, -angle, axis_point, axis_dir))

        per_image.append({"image": name, "points_2d": int(len(pts2d)),
                          "points_3d": int(n3d)})
        print(f"  [{i:4d}] {name:<18} 提取 {len(pts2d):4d} 点"
              + (f"，三维 {n3d:4d}" if do_stack else ""))

    # ---- 汇总点云 ----
    cloud = np.vstack(all_points) if all_points else np.zeros((0, 3))

    voxel = float(cfg.get("stack", {}).get("voxel_size", 0.0) or 0.0)
    if voxel > 0 and len(cloud):
        keys = np.floor(cloud / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        cloud = cloud[np.sort(idx)]
        print(f"体素下采样 ({voxel} m) → {len(cloud)} 点")

    ply_path = os.path.join(out_dir, "cloud.ply")
    xyz_path = os.path.join(out_dir, "cloud.xyz")
    write_ply(ply_path, cloud)
    np.savetxt(xyz_path, cloud, fmt="%.6f")

    summary = {
        "method": method_name,
        "label": method.label,
        "params": {"threshold": threshold, "min_pixels": min_pixels,
                   **algo_params},
        "preprocess": pre,
        "angle_step_deg": angle_step,
        "num_images": len(files),
        "num_points": int(len(cloud)),
        "per_image": per_image,
    }
    with open(os.path.join(out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"完成: {len(cloud):,} 点")
    print(f"  点云  → {ply_path}")
    print(f"  文本  → {xyz_path}")
    if cfg.get("output", {}).get("save_overlay", True):
        print(f"  叠加图 → {overlay_dir}/")
    print(f"  汇总  → {os.path.join(out_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
