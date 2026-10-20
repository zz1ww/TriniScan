"""差分法激光平面标定（黑白相机专用）。

用法：
    python tools/calib_laser_diff.py <图片目录> [--thresh 60] [--write]

图片目录内的 .jpg/.png 按文件名排序，每 2 张为一对：
    奇数位 = 关激光，偶数位 = 开激光

原理：
    差分图 = max(亮图 - 暗图, 0)
    静止的棋盘格/背景/手 被完全扣除，只剩激光线。
    黑白相机没有颜色信息时，这是最可靠的激光提取方式。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

# 允许从仓库根目录直接运行
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from triniscan.algorithms.calibration.camera_calib import (  # noqa: E402
    _detect_corners, load_camera_calib, make_object_points)
from triniscan.algorithms.calibration.laser_calib import (  # noqa: E402
    _clip_centerline_to_board, _intersect_pixels_with_plane,
    calibrate_laser_plane, laser_plane_from_chessboard)


def _list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(folder, e)))
        out.extend(glob.glob(os.path.join(folder, e.upper())))
    return sorted(set(out))


def extract_line(diff: np.ndarray, thresh: float = 60.0,
                 min_px: int = 20, max_halfwidth: float = 7.0) -> np.ndarray:
    """从差分图提取激光线，返回按列的中心点 (N,2)。

    **不要只取最大连通域**：激光穿过黑白格时亮度差异很大，线会在
    黑白交界处断开（实测一张图能碎成 30 多段）。

    **也不要用"竖直"判据**：激光斜穿棋盘时同一段碎片的 x 跨度很大，
    ``h > 3w`` 会把它们误杀。正确判据是"细"——用距离变换求每个
    连通域的最大内切半径，激光线 < 6 像素，白纸/手/木框等块状物
    远大于此。这个判据与方向无关。
    """
    mask = (diff > thresh).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_px:
            continue
        sub = (lbl[y:y + h, x:x + w] == i).astype(np.uint8)
        # 最大内切半径：块状物远大于线宽
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        if dist.max() > max_halfwidth:
            continue
        keep[y:y + h, x:x + w] |= (sub > 0)

    ys, xs = np.where(keep)
    if xs.size == 0:
        return np.zeros((0, 2))

    cols: dict[int, list[int]] = {}
    for x, y in zip(xs, ys):
        cols.setdefault(int(x), []).append(int(y))
    pts = []
    for x in sorted(cols):
        yy = np.array(cols[x], dtype=np.int64)
        w = diff[yy, x].astype(np.float64)
        if w.sum() <= 0:
            continue
        pts.append((float(x), float((yy * w).sum() / w.sum())))
    return np.array(pts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="含成对图片的目录")
    ap.add_argument("--thresh", type=float, default=60.0,
                    help="差分阈值（默认 60）")
    ap.add_argument("--camera", default="config/calib_camera.npz")
    ap.add_argument("--out", default="config/calib_laser.npz")
    ap.add_argument("--write", action="store_true",
                    help="合格时写入标定文件")
    ap.add_argument("--min-inside", type=int, default=30,
                    help="每视角至少多少个点在棋盘内")
    args = ap.parse_args()

    cam = load_camera_calib(args.camera)
    K, dist = cam.K, cam.dist
    objp = make_object_points(11, 8, 0.01).astype(np.float32)

    files = _list_images(args.folder)
    if len(files) < 4:
        print(f"图片不足：{len(files)} 张（至少 2 对）")
        return 2
    pairs = [(files[i], files[i + 1]) for i in range(0, len(files) - 1, 2)]
    print(f"共 {len(files)} 张图，{len(pairs)} 对\n")

    good_pairs = []
    for k, (dark_f, lit_f) in enumerate(pairs, 1):
        dark = cv2.imread(dark_f, cv2.IMREAD_GRAYSCALE)
        lit = cv2.imread(lit_f, cv2.IMREAD_GRAYSCALE)
        if dark is None or lit is None or dark.shape != lit.shape:
            print(f"[{k}] 读取失败或尺寸不一致，跳过")
            continue
        diff = np.clip(lit.astype(np.int16) - dark.astype(np.int16),
                       0, 255).astype(np.uint8)

        # 用暗图（无激光）检测棋盘，更干净
        corners = _detect_corners(dark, (11, 8), use_sb=True)
        if corners is None:
            print(f"[{k}] 棋盘格检测失败，跳过")
            continue
        c2 = corners.reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            objp, c2, K.astype(np.float32), dist.astype(np.float32))
        if not ok:
            print(f"[{k}] solvePnP 失败，跳过")
            continue

        line = extract_line(diff, args.thresh)
        inside = _clip_centerline_to_board(
            line, c2.astype(np.float64), 0.0) if len(line) else line
        n_ok = len(inside) >= args.min_inside

        print(f"[{k}] {os.path.basename(lit_f)}")
        print(f"     棋盘 x {c2[:,0].min():.0f}~{c2[:,0].max():.0f} "
              f"y {c2[:,1].min():.0f}~{c2[:,1].max():.0f}")
        if len(line):
            print(f"     激光 x {line[:,0].min():.0f}~{line[:,0].max():.0f} "
                  f"y {line[:,1].min():.0f}~{line[:,1].max():.0f} "
                  f"列数 {len(line)}  棋盘内 {len(inside)}")
        else:
            print("     未提取到激光线")
        if not n_ok:
            print(f"     ✗ 棋盘内点不足 {args.min_inside}，该视角不参与拟合")
            continue

        plane = laser_plane_from_chessboard(rvec, tvec)
        pts = _intersect_pixels_with_plane(inside, K, dist, plane)
        # 关键：bgr 用**暗图**（无激光），否则 calibrate_laser_plane
        # 内部重新检测棋盘时，激光线穿过棋盘会让 SB 检测失败。
        # 中心线则通过 "centerline" 键直接给定，无需从图里再提取。
        good_pairs.append({"bgr": dark, "centerline": line})
        print(f"     ✓ 收录 {len(pts)} 点")

    if len(good_pairs) < 2:
        print(f"\n合格视角 {len(good_pairs)} < 2，无法标定。")
        return 1

    # 复用现有标定函数（IRLS + 视角剔除）
    t0 = time.time()
    res = calibrate_laser_plane(
        good_pairs, K, dist,
        pattern_cols=11, pattern_rows=8, square_size_m=0.01,
        out_path=args.out if args.write else None,
        view_reject_mm=1.0, outlier_mm=0.5, clip_margin_px=0.0,
        channel="gray",
    )
    rms_mm = res.residual_rms * 1e3
    print(f"\n=== 结果 ===")
    print(f"视角 {res.num_views}   点数 {res.num_points}   "
          f"残差 RMS {rms_mm:.3f} mm   ({time.time()-t0:.1f}s)")
    if args.write and rms_mm <= 1.0:
        print(f"已写入 {args.out}")
    elif args.write:
        print(f"残差 {rms_mm:.3f} mm 超过 1.0 mm，未写入")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
