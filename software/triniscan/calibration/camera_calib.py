"""相机内参 + 畸变标定（张正友棋盘格法）。

目标
----
求相机内参矩阵 ``K`` 与畸变系数 ``dist``，把像素坐标映射为归一化射线::

        ┌ fx  0  cx ┐
    K = │ 0  fy  cy │        dist = [k1, k2, p1, p2, k3]
        └ 0   0   1 ┘

标定注意事项
------------
- 标定时的**焦距、光圈、工作距离、分辨率**必须与实测一致；
- 棋盘格须平整，覆盖整个视场（含边角）以正确约束径向畸变；
- 姿态多样（平移、倾斜、旋转），建议 15~25 张；
- 重投影 RMS 误差应 < 0.5 px。
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from ..common.io_utils import ensure_parent, read_image, save_calibration
from ..common.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "CameraCalibResult",
    "calibrate_camera",
    "calibrate_camera_from_files",
    "detect_chessboard_quality",
    "ChessboardDetection",
    "save_camera_calib",
    "load_camera_calib",
    "make_object_points",
]

_FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ChessboardDetection:
    """单幅图像的棋盘格检测与质量评价结果。"""

    found: bool                        # 是否检测到棋盘格
    corners: Optional[np.ndarray]      # (N,2) 亚像素角点，未检测到为 None
    area_ratio: float                  # 角点凸包面积 / 图像面积
    min_side_px: float                 # 相邻角点最小间距（像素）
    sharpness: float                   # 拉普拉斯方差（清晰度）

    @property
    def num_corners(self) -> int:
        return 0 if self.corners is None else int(len(self.corners))


@dataclass
class CameraCalibResult:
    """相机标定结果。"""

    K: np.ndarray                      # (3,3) 内参
    dist: np.ndarray                   # (1,N) 或 (N,) 畸变
    image_size: tuple[int, int]        # (width, height)
    rms: float                         # 重投影 RMS (px)
    num_images: int = 0                # 成功使用的图像数
    per_view_errors: list[float] = field(default_factory=list)

    def undistort_points(self, pixels: np.ndarray) -> np.ndarray:
        """对 (N,2) 像素去畸变，返回仍在内参 K 定义下的像素坐标。"""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pixels, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """对整幅图像去畸变。"""
        return cv2.undistort(image, self.K, self.dist,
                             None, self.K)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def make_object_points(cols: int, rows: int, square: float) -> np.ndarray:
    """构造棋盘格角点的世界坐标 (Z=0)。"""
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square)
    return objp


def _detect_corners(gray: np.ndarray, pattern_size: tuple[int, int]):
    """检测并亚像素细化棋盘格角点。"""
    found, corners = cv2.findChessboardCorners(gray, pattern_size, _FIND_FLAGS)
    if not found:
        return None
    return cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA
    )


# ---------------------------------------------------------------------------
# 主标定流程
# ---------------------------------------------------------------------------
def calibrate_camera(
    images: Sequence[np.ndarray],
    pattern_cols: int = 9,
    pattern_rows: int = 6,
    square_size_m: float = 0.020,
    out_path: Optional[str] = None,
    show: bool = False,
) -> CameraCalibResult:
    """用一组棋盘格图像标定相机。

    Parameters
    ----------
    images : 图像序列（BGR 或灰度 np.ndarray）
    pattern_cols, pattern_rows : 棋盘格**内角点**列数、行数
    square_size_m : 方格实际尺寸（米）
    out_path : 非空则保存结果
    show : 是否可视化角点检测结果

    Returns
    -------
    CameraCalibResult
    """
    pattern_size = (pattern_cols, pattern_rows)
    objp = make_object_points(pattern_cols, pattern_rows, square_size_m)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    for i, img in enumerate(images):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        corners = _detect_corners(gray, pattern_size)
        if corners is None:
            log.warning("图像 %d: 未检测到棋盘格，跳过", i)
            continue

        obj_points.append(objp)
        img_points.append(corners)

        if show:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            cv2.imshow("corners", vis)
            cv2.waitKey(200)

    if show:
        cv2.destroyAllWindows()

    if len(obj_points) < 3:
        raise RuntimeError(
            f"有效标定图像不足（{len(obj_points)} < 3），无法标定"
        )
    if image_size is None:
        raise RuntimeError("无有效图像")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # 逐视角重投影误差
    per_view = _per_view_errors(
        obj_points, img_points, rvecs, tvecs, K, dist
    )

    result = CameraCalibResult(
        K=K,
        dist=dist,
        image_size=image_size,
        rms=float(rms),
        num_images=len(obj_points),
        per_view_errors=per_view,
    )

    log.info("相机标定完成: RMS=%.4f px, 使用 %d 张图像",
             result.rms, result.num_images)
    if result.rms > 0.5:
        log.warning("重投影误差偏大 (%.3f px)，建议重新采集标定图", result.rms)

    if out_path:
        save_camera_calib(out_path, result)
    return result


def calibrate_camera_from_files(
    images_glob: str,
    pattern_cols: int = 9,
    pattern_rows: int = 6,
    square_size_m: float = 0.020,
    out_path: Optional[str] = None,
    show: bool = False,
) -> CameraCalibResult:
    """从文件通配符读取图像并标定。"""
    files = sorted(glob.glob(images_glob))
    if not files:
        raise FileNotFoundError(f"未找到标定图像: {images_glob}")

    images = []
    for f in files:
        img = read_image(f, must_exist=False)
        if img is None:
            log.warning("无法读取: %s", f)
            continue
        images.append(img)

    log.info("读入 %d 张标定图像", len(images))
    return calibrate_camera(
        images, pattern_cols, pattern_rows, square_size_m, out_path, show
    )


def detect_chessboard_quality(
    image: np.ndarray,
    pattern_cols: int = 9,
    pattern_rows: int = 6,
) -> "ChessboardDetection":
    """检测单幅图像中的棋盘格并给出质量评价。

    供交互式标定拍摄使用：拍一张即可立刻知道该帧是否可用。

    Parameters
    ----------
    image : 图像（BGR 或灰度 np.ndarray）
    pattern_cols, pattern_rows : 棋盘格内角点列数、行数

    Returns
    -------
    ChessboardDetection
        含是否成功、角点像素、以及若干质量指标。
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY)
    pattern_size = (int(pattern_cols), int(pattern_rows))
    corners = _detect_corners(gray, pattern_size)
    if corners is None:
        return ChessboardDetection(
            found=False, corners=None, area_ratio=0.0,
            min_side_px=0.0, sharpness=0.0,
        )

    pts = corners.reshape(-1, 2).astype(np.float64)
    # 棋盘格在图像中占据的面积占比（粗略：角点凸包面积 / 图像面积）
    try:
        hull = cv2.convexHull(pts.astype(np.float32))
        area_ratio = float(cv2.contourArea(hull)) / float(
            gray.shape[0] * gray.shape[1])
    except cv2.error:  # pragma: no cover - 极端退化情形
        area_ratio = 0.0
    # 相邻角点最小间距（像素），用于判断是否离得太远/太小
    min_side = _min_adjacent_spacing(pts, pattern_cols, pattern_rows)
    # 清晰度：拉普拉斯方差（越大越清晰）
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return ChessboardDetection(
        found=True, corners=pts, area_ratio=area_ratio,
        min_side_px=min_side, sharpness=sharpness,
    )


def _min_adjacent_spacing(pts: np.ndarray, cols: int,
                          rows: int) -> float:
    """角点阵列中相邻（水平/垂直）角点的最小间距（像素）。"""
    grid = pts.reshape(rows, cols, 2)
    d = []
    if cols > 1:
        d.append(np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel())
    if rows > 1:
        d.append(np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel())
    if not d:
        return 0.0
    return float(np.min(np.concatenate(d)))


def _per_view_errors(obj_points, img_points, rvecs, tvecs,
                     K, dist) -> list[float]:
    errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], K, dist
        )
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / \
            len(projected)
        errors.append(float(err))
    return errors


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------
def save_camera_calib(path: str, result: CameraCalibResult) -> None:
    save_calibration(
        path,
        arrays={
            "K": result.K,
            "dist": result.dist,
            "image_size": np.array(result.image_size, dtype=np.int64),
            "rms": np.array(result.rms),
            "num_images": np.array(result.num_images),
            "per_view_errors": np.array(result.per_view_errors),
        },
        meta={"type": "camera_intrinsics"},
    )


def load_camera_calib(path: str) -> CameraCalibResult:
    from ..common.io_utils import load_calibration
    arrays, _ = load_calibration(path)
    image_size = tuple(int(v) for v in arrays["image_size"])
    return CameraCalibResult(
        K=arrays["K"],
        dist=arrays["dist"],
        image_size=image_size,
        rms=float(arrays.get("rms", np.nan)),
        num_images=int(arrays.get("num_images", 0)),
        per_view_errors=list(
            np.asarray(arrays.get("per_view_errors", [])).tolist()
        ),
    )
