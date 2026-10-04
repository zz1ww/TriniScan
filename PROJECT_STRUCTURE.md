# 项目结构说明

```
TriniScan/                                  ← 项目根
│
├── README.md                    ★ 项目总说明（入口）
├── PROJECT_STRUCTURE.md         ★ 本文件：目录结构说明
├── .gitignore                   版本控制忽略规则
│
├── docs/                        ★ 设计与研究文档
│   ├── 01_系统总体设计.md
│   ├── 02_硬件选型与参数.md
│   ├── 03_软件架构设计.md
│   ├── 04_标定方案.md
│   ├── 05_体积计算方案.md
│   ├── 06_开发计划与里程碑.md
│   ├── 07_竞赛规则要点.md
│   └── ciclop机械件说明.md
│
├── hardware/                    硬件
│   ├── mechanical/              机械
│   │   └── ciclop/              开源扫描仪模型（参考复用）
│   │       ├── stp/             STEP 模型 (.stp)
│   │       └── fcstd/           FreeCAD 源文件 (.fcstd)
│   ├── electronics/             电路（原理图/接线，待补充）
│   └── datasheets/              器件数据手册（待补充）
│
├── firmware/                    嵌入式固件
│   └── mcu/
│       ├── README.md            固件说明 + 串口协议
│       ├── config.h             引脚/运动学参数
│       └── turntable_controller.ino   转台控制器（Arduino）
│
├── software/                    上位机软件
│   ├── requirements.txt         依赖
│   ├── config/
│   │   └── default.yaml         ★ 主配置文件
│   └── triniscan/               主程序包
│       ├── __init__.py
│       ├── core/                核心
│       │   ├── config.py        配置加载
│       │   ├── pipeline.py      ★ 单件测量主流程
│       │   └── main.py          程序入口
│       ├── camera/              相机
│       │   └── capture.py       采集/锁参数/抓图
│       ├── turntable/           转台
│       │   └── controller.py    串口控制
│       ├── calibration/         标定
│       │   ├── camera_calib.py  相机内参
│       │   ├── laser_calib.py   激光平面（★）
│       │   └── axis_calib.py    转轴（★）
│       ├── extraction/          提取
│       │   └── centerline.py    激光中心线（★）
│       ├── reconstruction/      重建
│       │   ├── triangulate.py   三角测量（★）
│       │   ├── register.py      多视角配准
│       │   └── mesh.py          Poisson 网格重建
│       ├── volume/              体积
│       │   ├── close_mesh.py    网格封闭化（★）
│       │   └── volume.py        体积积分（★）
│       ├── ui/                  图形界面（可选）
│       └── tools/               独立工具脚本
│   └── tests/                   测试
│
├── assets/
│   ├── calibration_targets/     标定靶（棋盘格图像放这）
│   └── samples/                 示例数据
│
├── reference/
│   ├── competition/             竞赛官方文件（PDF）
│   └── 开源推荐.txt
│
├── scripts/                     运行/安装脚本
│   ├── run.sh / run.bat         启动
│   └── setup.bat                环境安装
│
├── data/                        数据（不纳入 git）
│   ├── raw/                     原始图像
│   ├── pointclouds/             点云
│   ├── meshes/                  网格
│   └── results/                 结果
│
└── logs/                        日志（不纳入 git）
```

## 关键文件速查

| 我想… | 看/改这里 |
|---|---|
| 了解项目 | `README.md` |
| 看硬件选型 | `docs/02_硬件选型与参数.md` |
| 看软件架构 | `docs/03_软件架构设计.md` |
| 看标定方法 | `docs/04_标定方案.md` |
| 看体积算法 | `docs/05_体积计算方案.md` |
| 改参数 | `software/config/default.yaml` |
| 改引脚/运动学 | `firmware/mcu/config.h` |
| 改转台协议 | `firmware/mcu/turntable_controller.ino` + `software/triniscan/turntable/controller.py` |
| 跑主流程 | `software/triniscan/core/pipeline.py` |
| 复现机械件 | `hardware/mechanical/ciclop/` |

## 单位约定

- 长度：米 (m)
- 体积：输出立方厘米 (cm³)
- 角度：度 (°)
- 时间：秒 (s)

## 数据流

```
配置 → 转台拍照(data/raw) → 提取中心线 → 三角测量
    → 配准点云(data/pointclouds) → 重建网格(data/meshes)
    → 封闭 → 体积(data/results)
```

## 开发顺序（详见 docs/06）

1. 转-停-拍打通
2. 三标定
3. 单视角点云
4. 多视角重建
5. 体积计算
6. 自动化 ≤60s
7. 误差分析 + 海报
