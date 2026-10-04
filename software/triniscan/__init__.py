"""TriniScan — 激光三角测量三维体积测量系统。

第十五届全国大学生光电设计竞赛 · 赛题二

包结构::

    triniscan/
    ├── common/        通用基础层（几何、日志、IO）
    ├── core/          配置与主控流程
    ├── camera/        相机采集
    ├── turntable/     转台控制
    ├── calibration/   相机内参 / 激光平面 / 转轴标定
    ├── extraction/    激光中心线亚像素提取
    ├── reconstruction/三角测量、多视角配准、网格重建
    ├── volume/        网格封闭化与体积计算
    └── ui/            图形界面（可选）
"""

__version__ = "0.2.0"
__author__ = "TriniScan Team"
