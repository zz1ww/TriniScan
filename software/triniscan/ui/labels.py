"""配置项的中文标签与说明。

界面（``ui/panels.py`` 的 :class:`ConfigPanel`）据此把配置里的英文键名
显示为中文，便于不熟悉配置文件的同学使用。

约定
----
- ``GROUP_LABELS``   分组名 → 中文
- ``FIELD_LABELS``   ``"分组.键"`` → 中文标签
- ``FIELD_HINTS``    ``"分组.键"`` → 简短说明（鼠标悬停提示）

未登记的键会**原样显示英文**，因此新增配置项不会导致界面报错，
只是暂时没有中文而已。
"""
from __future__ import annotations

__all__ = ["GROUP_LABELS", "FIELD_LABELS", "FIELD_HINTS",
           "group_label", "field_label", "field_hint"]


# ---------------------------------------------------------------------------
GROUP_LABELS: dict[str, str] = {
    "camera": "相机",
    "turntable": "转台",
    "scan": "扫描",
    "calibration": "标定文件",
    "camera_calib": "相机标定",
    "axis_calib": "转轴标定",
    "laser": "激光",
    "extraction": "中心线提取",
    "reconstruction": "三维重建",
    "volume": "体积计算",
    "output": "输出目录",
    "logging": "日志",
}


FIELD_LABELS: dict[str, str] = {
    # --- 相机 ---
    "camera.index": "相机编号",
    "camera.backend": "采集后端",
    "camera.width": "图像宽度",
    "camera.height": "图像高度",
    "camera.auto_exposure": "自动曝光",
    "camera.exposure": "曝光值",
    "camera.gain": "增益",
    "camera.auto_wb": "自动白平衡",
    "camera.discard_frames": "丢弃帧数",

    # --- 转台 ---
    "turntable.port": "串口号",
    "turntable.baudrate": "波特率",
    "turntable.steps_per_rev": "每圈整步数",
    "turntable.microstep": "细分倍数",
    "turntable.gear_ratio": "减速比",
    "turntable.settle_ms": "到位稳定延时",
    "turntable.timeout_s": "指令超时",

    # --- 扫描 ---
    "scan.angle_step_deg": "每视角转角",
    "scan.num_views": "视角数",
    "scan.timeout_s": "单件测量时限",

    # --- 标定文件 ---
    "calibration.camera_file": "相机标定文件",
    "calibration.laser_file": "激光平面标定文件",
    "calibration.axis_file": "转轴标定文件",

    # --- 相机标定 ---
    "camera_calib.pattern_cols": "棋盘格列数",
    "camera_calib.pattern_rows": "棋盘格行数",
    "camera_calib.square_size_m": "方格边长",
    "camera_calib.images_glob": "标定图路径",

    # --- 转轴标定 ---
    "axis_calib.sphere_radius_m": "标记球半径",
    "axis_calib.method": "取点方法",
    "axis_calib.num_views": "采样角度数",
    "axis_calib.step_deg": "每步转角",
    "axis_calib.min_views": "最少有效视角",
    "axis_calib.settle_ms": "到位稳定延时",
    "axis_calib.spot_threshold": "亮点灰度阈值",
    "axis_calib.spot_win_px": "亮点搜索窗口",
    "axis_calib.min_radius_px": "球最小半径",
    "axis_calib.max_radius_px": "球最大半径",
    "axis_calib.archive_dir": "现场点存档目录",

    # --- 激光 ---
    "laser.wavelength_nm": "波长",
    "laser.power_mw": "功率",
    "laser.safety_class": "安全等级",

    # --- 中心线提取 ---
    "extraction.method": "提取方法",
    "extraction.threshold": "亮度阈值",
    "extraction.smooth_sigma": "预处理平滑",
    "extraction.steger_sigma": "Steger 尺度",
    "extraction.min_pixels": "最少条纹像素",
    "extraction.min_points": "每视角最少点数",
    "extraction.invert": "暗线模式",
    "extraction.roi": "感兴趣区域",

    # --- 三维重建 ---
    "reconstruction.voxel_size": "体素尺寸",
    "reconstruction.icp_refine": "ICP 精配准",
    "reconstruction.poisson_depth": "Poisson 深度",
    "reconstruction.density_quantile": "密度裁剪分位",

    # --- 体积计算 ---
    "volume.method": "计算方法",
    "volume.voxel_pitch": "校核体素尺寸",
    "volume.cross_check": "体素法校核",
    "volume.unit": "单位",

    # --- 输出目录 ---
    "output.raw_dir": "原始图像目录",
    "output.pointcloud_dir": "点云目录",
    "output.mesh_dir": "网格目录",
    "output.result_dir": "结果目录",
    "output.calib_dir": "标定图目录",
    "output.log_dir": "日志目录",

    # --- 日志 ---
    "logging.level": "日志级别",
}


FIELD_HINTS: dict[str, str] = {
    "camera.index": "多摄像头时依次尝试 0、1、2",
    "camera.backend": "Windows 用 dshow，其它平台用 any",
    "camera.auto_exposure": "必须关闭，否则激光亮度会漂移",
    "camera.exposure": "依相机而定，需实测：过曝则调小",
    "camera.auto_wb": "必须关闭，保证各视角颜色一致",
    "camera.discard_frames": "每次抓图前丢弃的帧数，等曝光稳定",

    "turntable.port": "设备管理器 → 端口，查看实际 COM 口",
    "turntable.steps_per_rev": "NEMA17 步距角 1.8° → 200",
    "turntable.microstep": "驱动器细分设定，须与实际跳线一致",
    "turntable.gear_ratio": "机械减速比，须与实际机构一致",
    "turntable.settle_ms": "转台到位后等待稳定的时间",

    "scan.angle_step_deg": "每视角转过的角度，越小越密但越慢",
    "scan.num_views": "视角数，应等于 360 / 每视角转角",

    "camera_calib.pattern_cols": "棋盘格内角点数（列），如 9",
    "camera_calib.pattern_rows": "棋盘格内角点数（行），如 6",
    "camera_calib.square_size_m": "方格实际边长，单位米",
    "camera_calib.images_glob": "批量标定的图像通配路径",

    "axis_calib.sphere_radius_m": "标记球半径（米）；乒乓球半径 0.020",
    "axis_calib.method": "peak=最亮极点法（关环境光）；"
                         "sphere=球心法（亮背景）",
    "axis_calib.num_views": "转台采样角度数，建议 12",
    "axis_calib.min_views": "少于该数量则判定标定失败",
    "axis_calib.spot_threshold": "亮点相对背景的灰度下限",
    "axis_calib.spot_win_px": "亮点亚像素定位的搜索窗口半径",

    "laser.wavelength_nm": "激光波长，用于安全说明",
    "laser.power_mw": "激光功率，注意安全",
    "laser.safety_class": "激光安全等级（需在装置上标明）",

    "extraction.method": "steger 最精确，gray_centroid 最快",
    "extraction.threshold": "激光亮度绝对阈值，过低会引入噪点",
    "extraction.invert": "亮背景上的暗线请设为 true",

    "reconstruction.voxel_size": "下采样/配准体素，越小越精细但越慢",
    "reconstruction.icp_refine": "相邻视角做 ICP 精配准，提高拼接精度",
    "reconstruction.poisson_depth": "Poisson 八叉树深度，越大越细",
    "reconstruction.density_quantile": "裁剪最低密度面片的分位",

    "volume.method": "divergence 散度定理；voxel 体素法",
    "volume.cross_check": "用体素法交叉校核，建议开启",
    "volume.unit": "显示单位，如 cm3",

    "logging.level": "DEBUG / INFO / WARNING / ERROR",
}


# ---------------------------------------------------------------------------
def group_label(group: str) -> str:
    """分组的中文名（未登记则原样返回）。"""
    return GROUP_LABELS.get(group, group)


def field_label(key_path: str) -> str:
    """字段的中文标签（未登记则返回去掉分组后的英文键）。"""
    if key_path in FIELD_LABELS:
        return FIELD_LABELS[key_path]
    return key_path.split(".", 1)[-1]


def field_hint(key_path: str) -> str:
    """字段的说明文字（未登记返回空串）。"""
    return FIELD_HINTS.get(key_path, "")
