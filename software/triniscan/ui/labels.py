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
    "laser_calib": "激光平面标定",
    "laser": "激光",
    "quality": "标定质量阈值",
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
    "camera.fourcc": "像素格式",
    "camera.fourcc_mono": "黑白像素格式",
    "camera.mono": "黑白相机",
    "camera.fps": "目标帧率",
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
    "scan.preset": "视角数预设",
    "scan.preset": "720 或 1080 视角；启用后自动回算自洽的每视角转角",
    "scan.angle_step_deg": "每视角转角",
    "scan.num_views": "视角数",
    "scan.steps_per_view": "每视角微步数",
    "scan.timeout_s": "单件测量时限",

    # --- 标定文件 ---
    "calibration.camera_file": "相机标定文件",
    "calibration.laser_file": "激光平面标定文件",
    "calibration.axis_file": "转轴标定文件",
    "calibration.axis_mode": "转轴来源",
    "calibration.z_axis_point": "Z轴过点",
    "calibration.z_axis_dir": "Z轴方向",
    "calibration.strict": "严格校验标定",

    # --- 相机标定 ---
    "camera_calib.pattern_cols": "棋盘格列数",
    "camera_calib.pattern_rows": "棋盘格行数",
    "camera_calib.square_size_m": "方格边长",
    "camera_calib.images_glob": "标定图路径",
    "camera_calib.use_sb": "SB 检测器",
    "camera_calib.auto_prune": "自动剔除坏图",
    "camera_calib.min_images_for_pruning": "剔除最少张数",

    # --- 转轴标定 ---
    "axis_calib.sphere_radius_m": "标记球半径",
    "axis_calib.method": "取点方法",
    "axis_calib.save_images": "保存标定图",
    "axis_calib.num_views": "采样角度数",
    "axis_calib.step_deg": "每步转角",
    "axis_calib.min_views": "最少有效视角",
    "axis_calib.max_residual_mm": "残差上限(mm)",
    "axis_calib.min_radius_mm": "半径下限(mm)",
    "axis_calib.settle_ms": "到位稳定延时",
    "axis_calib.spot_threshold": "亮点灰度阈值",
    "axis_calib.spot_win_px": "亮点搜索窗口",
    "axis_calib.spot_channel": "亮点取色通道",
    "axis_calib.min_radius_px": "球最小半径",
    "axis_calib.max_radius_px": "球最大半径",
    "axis_calib.robust": "稳健拟合",
    "axis_calib.archive_dir": "现场点存档目录",

    # --- 激光平面标定 ---
    "laser_calib.robust": "稳健拟合",
    "laser_calib.robust_method": "稳健方法",
    "laser_calib.view_reject_mm": "视角剔除阈值",
    "laser_calib.outlier_mm": "离群距离阈值",
    "laser_calib.save_images": "保存标定原图",
    "laser_calib.clip_margin_px": "中心线裁剪边距",
    "laser_calib.channel": "激光标定提取通道",

    # --- 标定质量阈值 ---
    "quality.camera_rms_good": "相机 RMS 优秀",
    "quality.camera_rms_warn": "相机 RMS 合格",
    "quality.laser_rms_good": "激光 RMS 优秀",
    "quality.laser_rms_warn": "激光 RMS 合格",
    "quality.axis_rms_good": "转轴 RMS 优秀",
    "quality.axis_rms_warn": "转轴 RMS 合格",

    # --- 激光 ---
    "laser.wavelength_nm": "波长",
    "laser.power_mw": "功率",
    "laser.safety_class": "安全等级",

    # --- 中心线提取 ---
    "extraction.method": "提取方法",
    "extraction.threshold": "亮度阈值",
    "extraction.smooth_sigma": "预处理平滑",
    "extraction.subpixel": "亚像素插值",
    "extraction.steger_sigma": "Steger 尺度",
    "extraction.steger_merge": "条纹归并策略",
    "extraction.steger_min_sep_px": "中心最小间距",
    "extraction.min_pixels": "最少条纹像素",
    "extraction.min_points": "每视角最少点数",
    "extraction.invert": "暗线模式",
    "extraction.roi": "感兴趣区域",
    "extraction.channel": "提取通道",
    "extraction.enhance_enable": "启用条纹增强",
    "extraction.enhance_threshold": "增强阈值",
    "extraction.enhance_blur_ksize": "平滑核尺寸",
    "extraction.enhance_window": "峰值窗口半宽",
    "extraction.enhance_window_enable": "峰值窗口掩膜",
    "extraction.enhance_double_threshold_enable": "双阈值平滑",
    "extraction.enhance_scale_to_width": "按宽度缩放",
    "extraction.enhance_min_peak": "最小峰值",
    "extraction.channel": "中心线提取通道",
    "laser_calib.channel": "激光标定提取通道",

    # --- 三维重建 ---
    "reconstruction.voxel_size": "体素尺寸",
    "reconstruction.icp_refine": "ICP 精配准",
    "reconstruction.depth_filter": "深度过滤",
    "reconstruction.depth_min_m": "深度下限(m)",
    "reconstruction.depth_max_m": "深度上限(m)",
    "reconstruction.bounds_radius_m": "半径上限(m)",
    "reconstruction.bounds_z_min_m": "轴向下限(m)",
    "reconstruction.bounds_z_max_m": "轴向上限(m)",
    "reconstruction.global_align": "全局对齐",
    "reconstruction.pose_graph": "位姿图优化",
    "reconstruction.edge_span": "位姿图邻接数",
    "reconstruction.iterative_iters": "迭代配准轮数",
    "reconstruction.poisson_depth": "Poisson 深度",
    "reconstruction.adaptive_depth": "深度自适应",
    "reconstruction.axial": "轴向裁剪",
    "reconstruction.axial.auto_trim": "自动主体区间",
    "reconstruction.axial.height_min_m": "轴向下限(m)",
    "reconstruction.axial.height_max_m": "轴向上限(m)",
    "reconstruction.axial.radius_max_m": "半径上限(m)",
    "reconstruction.orient_max_nn": "法向近邻数",
    "reconstruction.density_quantile": "密度裁剪分位",
    "reconstruction.close_fallback": "封闭兜底策略",
    "reconstruction.allow_convex_hull": "允许凸包兜底",
    "reconstruction.remove_turntable": "去除转盘点",
    "reconstruction.turntable_plane_tol_m": "转盘面距离容差",
    "reconstruction.turntable_min_inlier": "转盘最小内点占比",
    "reconstruction.turntable_align_tol_deg": "转盘法向夹角容差",
    "reconstruction.turntable_margin_m": "转盘面厚度容差",
    "reconstruction.turntable_band_below_m": "转盘面下方剔除厚度",
    "reconstruction.turntable_keep_upper_only": "仅保留盘面上方",
    "reconstruction.turntable_radius_m": "转盘半径限制",
    "reconstruction.turntable_method": "转盘去除方法",
    "reconstruction.turntable_height_m": "转盘面高度",
    "reconstruction.cluster_filter": "连通域去噪",
    "reconstruction.cluster_eps_m": "聚类半径(m)",
    "reconstruction.cluster_min_points": "聚类最少点数",
    "reconstruction.cluster_keep_top": "保留簇数",
    "reconstruction.filter_chain": "过滤链",

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
    "camera.backend": "msmf 帧率高(推荐)；dshow 锁曝光稳但帧率低",
    "camera.fourcc": "MJPG 压缩省带宽、帧率高；YUY2 不压缩、帧率低",
    "camera.fourcc_mono": "黑白相机首选（如 Y800/GREY），失败回退像素格式",
    "camera.mono": "黑白（单色）相机时开启；提取通道自动降级为 gray",
    "camera.fps": "目标采集帧率，0 表示不限制",
    "camera.auto_exposure": "必须关闭，否则激光亮度会漂移",
    "camera.exposure": "依相机而定，需实测：过曝则调小",
    "camera.auto_wb": "必须关闭，保证各视角颜色一致",
    "camera.discard_frames": "每次抓图前丢弃的帧数，等曝光稳定",

    "turntable.port": "设备管理器 → 端口，查看实际 COM 口",
    "turntable.steps_per_rev": "NEMA17 步距角 1.8° → 200",
    "turntable.microstep": "驱动器细分设定，须与实际跳线一致",
    "turntable.gear_ratio": "机械减速比，须与实际机构一致",
    "turntable.settle_ms": "转台到位后等待稳定的时间",

    "scan.preset": "720 或 1080 视角；启用后自动回算自洽的每视角转角",
    "scan.angle_step_deg": "每视角转过的角度，越小越密但越慢",
    "scan.num_views": "视角数，可选 180/150/120/90/60（转角自动=360/视角数）",
    "scan.steps_per_view": "每个视角转台走过的微步数（33152 微步=整圈，改动会重算转角）",

    "camera_calib.pattern_cols": "棋盘格内角点数（列），如 9",
    "camera_calib.pattern_rows": "棋盘格内角点数（行），如 6",
    "camera_calib.square_size_m": "方格实际边长，单位米",
    "camera_calib.images_glob": "批量标定的图像通配路径",
    "camera_calib.use_sb": "是否优先使用 sector-based 检测器（精度更高，旧版 OpenCV 自动回退）",
    "camera_calib.auto_prune": "是否自动剔除重投影误差过大的坏图（模糊/反光/板子翘曲）",
    "camera_calib.min_images_for_pruning": "视角少于该数时不做自动剔除（保留足够约束）",

    "axis_calib.sphere_radius_m": "标记球半径（米）；乒乓球半径 0.020",
    "axis_calib.method": "depth=球心高度约束反解（推荐：亮背景黑球，"
                         "只需球心像素，抗半径抖动，最稳）；"
                         "sphere=球心法（亮背景，依赖 Hough 像素半径，"
                         "半径抖会放大深度误差）；"
                         "peak=最亮极点法（需激光打中球，仅少数角度有效）",
    "axis_calib.save_images": "保存每个角度的原图，便于复查失败原因",
    "axis_calib.num_views": "转台采样角度数，建议 12",
    "axis_calib.min_views": "少于该数量则判定标定失败",
    "axis_calib.max_residual_mm": "圆拟合残差 RMS 上限，超过判失败",
    "axis_calib.min_radius_mm": "拟合圆半径下限；过小说明取点失败（点挤在一起）",
    "axis_calib.spot_threshold": "亮点相对背景的灰度下限",
    "axis_calib.spot_win_px": "亮点亚像素定位的搜索窗口半径",
    "axis_calib.spot_channel": "peak 法取亮点用的通道：red_laser 抑制亮背景最稳，green_laser/r/g/b 次之",
    "axis_calib.robust": "RANSAC 抗跳点，建议开启",

    "laser_calib.robust": "IRLS/RANSAC 抗离群，强烈建议开启",
    "laser_calib.robust_method": "irls 平滑加权；ransac 更耐大比例离群",
    "laser_calib.view_reject_mm": "单视角残差中位数超此值则剔除该视角",
    "laser_calib.outlier_mm": "稳健拟合的离群点距离阈值",
    "laser_calib.save_images": "保存激光标定原图，便于复查中心线提取",
    "laser_calib.channel": "red_laser=彩色相机用颜色差分（推荐）；gray=黑白相机",
    "laser_calib.clip_margin_px": "把中心线裁剪到棋盘格凸包内的额外边距；0=紧贴棋盘边界，可去除棋盘外的伪中心线点",

    "quality.camera_rms_good": "相机重投影 RMS 优秀线（px）",
    "quality.camera_rms_warn": "相机重投影 RMS 合格线（px），超过判不合格",
    "quality.laser_rms_good": "激光平面残差优秀线（mm）",
    "quality.laser_rms_warn": "激光平面残差合格线（mm），超过判不合格",
    "quality.axis_rms_good": "转轴残差优秀线（mm）",
    "quality.axis_rms_warn": "转轴残差合格线（mm），超过判不合格",

    "extraction.steger_merge": "column=每列一个中心；normal=按法向归并（倾斜条纹）",
    "extraction.steger_min_sep_px": "normal 归并下两中心的最小间距（像素）",
    "extraction.enhance_enable": "开启后先做阈值→平滑→阈值+峰值窗口掩膜，抑制镜面反光簇",
    "extraction.enhance_threshold": "增强用亮度阈值，通常与提取阈值一致",
    "extraction.enhance_blur_ksize": "邻域均值平滑核（奇数），越大越能压掉孤立反光斑",
    "extraction.enhance_window": "只保留每行峰值左右该半宽内的像素，抑制远端反光",
    "extraction.enhance_window_enable": "关闭则只做双阈值平滑、不做峰值窗口",
    "extraction.enhance_double_threshold_enable": "关闭则跳过阈值→平滑→阈值，仅做窗口掩膜",
    "extraction.enhance_scale_to_width": "按图像宽度自动缩放核尺寸，使 720p/1080p 效果一致",
    "extraction.enhance_min_peak": "该行峰值低于此值则整行清零（滤除弱反光行）",
    "extraction.channel": "red_laser=彩色相机用颜色差分（推荐）；gray=黑白相机",

    "reconstruction.global_align": "多视角全局对齐，替代顺序 ICP 消除累积误差",
    "reconstruction.pose_graph": "优先用位姿图优化（失败自动回退迭代配准）",
    "reconstruction.edge_span": "每个视角向后连接的邻居视角数",
    "reconstruction.iterative_iters": "兜底迭代配准的轮数",
    "reconstruction.adaptive_depth": "按点云规模自适应 Poisson 深度",
    "reconstruction.orient_max_nn": "法向一致化的近邻数上限",
    "reconstruction.close_fallback": "bottom_plane=平面封底(推荐)；convex_hull/voxel",
    "reconstruction.allow_convex_hull": "平面封底仍失败时是否允许凸包兜底",
    "reconstruction.remove_turntable": "用平面拟合剔除载物台表面点",
    "reconstruction.turntable_plane_tol_m": "RANSAC 判定盘面内点的距离阈值(m)",
    "reconstruction.turntable_min_inlier": "盘面内点占比低于此值则不认为是转盘",
    "reconstruction.turntable_align_tol_deg": "盘面法向与转轴允许夹角(度)",
    "reconstruction.turntable_margin_m": "盘面上下该厚度内的点视为转盘",
    "reconstruction.turntable_band_below_m": "盘面下方该厚度内的点一并剔除",
    "reconstruction.turntable_keep_upper_only": "直接只保留盘面上方的点",
    "reconstruction.turntable_radius_m": "仅剔除距转轴小于该半径的点；null 不限",
    "reconstruction.turntable_method": "ransac=平面拟合；axis_height=轴基准高度（推荐，抗竖直面）",
    "reconstruction.turntable_height_m": "轴基准法：转盘面沿轴高度(m)；null 时从点云自动估计",
    "reconstruction.cluster_filter": "剔除反光簇/支架残片等密集小团，保留最大簇",
    "reconstruction.cluster_eps_m": "DBSCAN 邻域半径，略大于平均点间距",
    "reconstruction.cluster_min_points": "成为核心点的最少邻居数",
    "reconstruction.cluster_keep_top": "保留点数最多的前 k 个簇",
    "reconstruction.filter_chain": "按顺序执行的过滤器列表，如 [{name: depth}, {name: cluster, eps_m: 0.003}]；留空则回退到上面各项的开关",

    "laser.wavelength_nm": "激光波长，用于安全说明",
    "laser.power_mw": "激光功率，注意安全",
    "laser.safety_class": "激光安全等级（需在装置上标明）",

    "extraction.method": "row_peak 逐行峰值+抛物线(默认,竖直线激光最优,抗零散亮点)；row_centroid 逐行重心；steger 最精确但最慢(不依赖扫描方向)；gray_centroid 逐列均衡；parabola 逐列最快亚像素；peak 逐列最粗",
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
