"""相机内参 + 畸变标定（张正友棋盘格法）。"""
from __future__ import annotations

import glob
import os
from typing import Optional

import cv2
import numpy as np


def _make_object_points(cols: int, rows: int, square: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square
    return objp


def calibrate_camera(
    images_glob: str,
    pattern_cols: int = 9,
    pattern_rows: int = 6,
    square_size_m: float = 0.020,
    out_path: Optional[str] = None,
    show: bool = False,
) -> dict:
    """用棋盘格图像标定相机。

    Returns
    -------
    dict: {"K", "dist", "image_size", "rms"}
    """
    pattern_size = (pattern_cols, pattern_rows)
    objp = _make_object_points(pattern_cols, pattern_rows, square_size_m)

    obj_points, img_points = [], []
    files = sorted(glob.glob(images_glob))
    if not files:
        raise FileNotFoundError(f"未找到标定图像: {images_glob}")

    image_size = None
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ok:
            print(f"[跳过] 未找到角点: {f}")
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        obj_points.append(objp)
        img_points.append(corners)

        if show:
            cv2.drawChessboardCorners(gray, pattern_size, corners, ok)
            cv2.imshow("corners", gray)
            cv2.waitKey(100)

    if len(obj_points) < 3:
        raise RuntimeError("有效标定图像不足（<3）")

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    result = {
        "K": K,
        "dist": dist,
        "image_size": np.array(image_size),
        "rms": rms,
    }
    print(f"标定完成: RMS={rms:.4f} px, 使用 {len(obj_points)} 张图")

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)),
                    exist_ok=True)
        np.savez(out_path, **result)
        print(f"已保存: {out_path}")

    return result


def load_camera_calib(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    from triniscan.core.config import Config

    cfg = Config.load()
    calibrate_camera(
        images_glob=cfg.get("camera_calib.images_glob"),
        pattern_cols=cfg.get("camera_calib.pattern_cols", 9),
        pattern_rows=cfg.get("camera_calib.pattern_rows", 6),
        square_size_m=cfg.get("camera_calib.square_size_m", 0.020),
        out_path=cfg.get("calibration.camera_file"),
    )
