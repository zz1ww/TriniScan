# 项目结构说明

```
TriniScan/                                  ← 项目根
│
├── README.md                    ★ 项目总说明（入口）
├── PROJECT_STRUCTURE.md         ★ 本文件：目录结构说明
├── .gitignore                   版本控制忽略规则
│
├── docs/                        ★ 设计与研究文档
│   ├── 01_系统总体设计.md        目标、技术路线、坐标系、红线
│   ├── 02_硬件选型与参数.md      选型、带宽与像素格式、角度分辨率
│   ├── 03_软件架构设计.md        设计原则与演进（历史，实现见 09）
│   ├── 04_标定方案.md            三标定方法、文件格式、清单
│   ├── 05_体积计算方案.md        网格、封闭化、体积积分
│   ├── 06_开发计划与里程碑.md    阶段划分与验收
│   ├── 07_竞赛规则要点.md        赛题要求与得分点
│   ├── 08_开发与测试.md          环境、运行、测试、GUI、规范
│   ├── 09_架构说明（重构后）.md  ★ 当前架构（分层/契约/注册表/配置）
│   ├── 10_采集链路与性能.md      相机格式协商、异步落盘、实测耗时
│   ├── 11_界面架构与操作.md      工作台模型、面板、操作流程
│   └── ciclop机械件说明.md
│
├── software/                    上位机软件
│   ├── pyproject.toml           项目定义 / 依赖 / 工具链
│   ├── requirements.txt         运行依赖（pip 用）
│   ├── config/
│   │   ├── default.yaml         ★ 主配置（唯一真实来源）
│   │   ├── calib_*.npz          本机标定结果（不入库）
│   │   └── calib_manifest.json  标定清单（新鲜度检测）
│   ├── scripts/                 gui.bat 等启动脚本
│   ├── tools/                   独立脚本（标定、预览、测量、实验）
│   ├── tests/                   单元测试（39 个文件 / 424 用例）
│   └── triniscan/               主程序包
│       ├── contracts/           契约层（不依赖项目内其它包）
│       │   ├── plugin.py            插件协议、PluginBase、RunContext
│       │   ├── frames.py            ViewFrame / Cloud / CalibrationBundle
│       │   ├── report.py            MaskStack / StageReport / ScanReport
│       │   ├── params.py            ParamSpec（默认值/范围/候选值）
│       │   ├── algorithm.py         AlgorithmSpec（声明式算法）
│       │   ├── registry.py          7 张注册表
│       │   └── errors.py            统一异常
│       ├── common/              纯工具（无业务知识）
│       │   ├── geometry.py          射线/平面/旋转/拟合
│       │   ├── io_utils.py          路径、图像、标定序列化
│       │   ├── logging_utils.py     统一日志
│       │   ├── robust.py            稳健拟合（RANSAC/IRLS）
│       │   └── async_writer.py      ★ 后台线程写图（有界队列 + 背压）
│       ├── algorithms/          可插拔实现（每个域一个子包）
│       │   ├── extraction/          中心线提取 + 条纹增强
│       │   ├── filtering/           过滤器插件（6 个）
│       │   ├── features/            三角测量
│       │   ├── registration/        多视角配准
│       │   ├── meshing/             泊松 / 滚球
│       │   ├── closing/             网格封闭化
│       │   ├── volumetry/           散度 / 体素
│       │   └── calibration/         三类标定 + 质量判据 + 棋盘格转轴
│       ├── pipeline/            编排
│       │   └── calibration.py       标定加载、质量门限、降级记录
│       ├── hardware/            设备驱动
│       │   ├── camera.py            UVC 打开、格式协商、通道判定、取帧
│       │   └── turntable.py         串口驱动、整数步角度计划
│       ├── core/                配置与主流程
│       │   ├── config.py            点号路径配置 + 项目根发现
│       │   ├── config_store.py      配置唯一来源（界面↔业务共享）
│       │   ├── pipeline.py          ★ ScanPipeline（采集→重建→点云→体积）
│       │   └── main.py              CLI 入口
│       ├── calib_manifest.py    标定清单与新鲜度
│       ├── ui/                  图形界面
│       │   ├── __main__.py          入口（--ui new | classic）
│       │   ├── app_v2.py            ★ 新主窗口（工作台容器）
│       │   ├── workbench.py         ★ 工作台基类（面板/页面/工具条）
│       │   ├── workbench_measure.py 测量台
│       │   ├── workbench_calib.py   标定台
│       │   ├── workbench_hardware.py 硬件台
│       │   ├── rebuild_job.py       后台重建（跳过采集）
│       │   ├── field_spec.py        配置+算法 → 字段描述
│       │   ├── params.py            按字段描述生成控件
│       │   ├── labels.py            中文标签与说明
│       │   ├── video.py             实时画面（cv2 + PPM）
│       │   ├── result_view.py       结果页
│       │   ├── worker.py            后台线程 + 队列 + 轮询
│       │   ├── app.py               旧两页界面（classic 回退）
│       │   └── ...                  参数面板、过滤链编辑器等
│       └── camera/ turntable/ calibration/ extraction/
│           reconstruction/ volume/  兼容 shim（旧路径仍可用）
│
├── firmware/                    嵌入式固件（两套等价实现，协议一致）
│   ├── mcu_stm32_cubeide/       STM32F103C8T6 + HAL（实测通过，推荐）
│   └── mcu_stm32_spl/           STM32F103C8T6 + 标准库（Keil5）
│
├── hardware/                    机械与电路
│   ├── mechanical/ciclop/       开源扫描仪模型（参考复用）
│   ├── electronics/             电路（待补充）
│   └── datasheets/              数据手册（待补充）
│
├── lab/                         算法实验台（独立于主程序）
│   ├── run.py                   从已拍好的图跑提取→三角→点云
│   ├── methods.py               实验用提取方法
│   ├── config.yaml
│   └── outputs/                 实验输出（不入库）
│
├── assets/                      标定靶、示例数据
├── reference/                   竞赛官方文件
├── scripts/                     运行/安装脚本
├── data/                        raw / pointclouds / meshes / results（不入库）
└── logs/                        运行日志（不入库）
```

## 关键文件速查

| 我想… | 看/改这里 |
|---|---|
| 了解项目 | `README.md` |
| 看当前架构 | `docs/09_架构说明（重构后）.md` |
| 看采集性能与踩坑 | `docs/10_采集链路与性能.md` |
| 看界面结构与操作 | `docs/11_界面架构与操作.md` |
| 看硬件选型/带宽 | `docs/02_硬件选型与参数.md` |
| 看标定方法 | `docs/04_标定方案.md` |
| 看体积算法 | `docs/05_体积计算方案.md` |
| 改参数 | `software/config/default.yaml` |
| 改相机取图 | `software/triniscan/hardware/camera.py` |
| 改转台/角度计划 | `software/triniscan/hardware/turntable.py` |
| 改主流程 | `software/triniscan/core/pipeline.py` |
| 加一个算法 | 对应 `algorithms/<域>/` + `specs.py` 声明 |
| 加一个界面功能 | 对应 `ui/workbench_*.py` 的 panel |
| 改引脚/运动学 | `firmware/mcu_stm32_cubeide/Core/Inc/config.h` |
| 改转台固件 | `firmware/mcu_stm32_cubeide/Core/Src/motor.c` |
| 改转台协议 | `firmware/.../Core/Src/cmd.c` + `hardware/turntable.py` |
| 复现机械件 | `hardware/mechanical/ciclop/` |

## 单位约定

- 长度：米 (m)；像素：px；体积输出：立方厘米 (cm³)
- 角度：度 (°)，内部换算为整数微步
- 时间：秒 (s)，延时用毫秒 (ms)

## 数据流

```
default.yaml ─► ConfigStore ─► ScanPipeline
                                   │
   转台整数步 ─► 相机取帧 ─► 异步落盘(data/raw)
                                   │
       中心线提取 ─► 三角测量 ─► 转轴变换 ─► 过滤链
                                   │
        配准(data/pointclouds) ─► 网格(data/meshes)
                                   │
              封闭化 ─► 体积 + 体素校核(data/results)
```

## 开发顺序（详见 docs/06）

1. 转-停-拍打通
2. 三标定
3. 单视角点云
4. 多视角重建
5. 体积计算
6. 自动化 ≤60s
7. 误差分析 + 海报
