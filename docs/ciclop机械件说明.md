# ciclop 机械件说明

本目录 `hardware/mechanical/ciclop/` 保存的是开源 3D 扫描仪 **ciclop**
（基于 RepRap 项目）的机械模型，作为本项目机械结构的**参考与复用**来源。

## 文件清单

### STEP 模型 (`stp/`)
| 文件 | 说明 | 本项目用途 |
|---|---|---|
| `camera-holder.stp` | 相机固定支架 | 相机固定（俯仰可调） |
| `laser-holder.stp` | 激光固定支架 | 线激光固定（倾角可调） |
| `disc-holder.stp` | 转盘/承载盘支架 | 转台承载盘 |
| `motor-holder.stp` | 电机固定支架 | NEMA17 电机固定 |
| `shaft-coupling.stp` | 联轴器 | 电机轴 → 转台 |
| `bearing-clip.stp` | 轴承卡扣 | 转轴轴承 |
| `back-cover.stp` | 后盖 | 结构件 |
| `pattern-holder.stp` | 标定图案支架 | 标定靶固定 |
| `pattern-holder-v1.fcstd` | 标定图案支架 v1 | — |
| `pattern-surface.stp` | 标定面 | 标定参考面 |
| `laser-wireway.stp` | 激光走线槽 | 走线 |
| `motor-wireway.stp` | 电机走线槽 | 走线 |
| `motor-wireway(1).stp` | 电机走线槽（副本） | — |

### FreeCAD 源文件 (`fcstd/`)
| 文件 | 说明 |
|---|---|
| `Shaft-coupling.fcstd` | 联轴器 |
| `back-cover.fcstd` | 后盖 |
| `disc-holder.fcstd` | 转盘支架 |
| `laser-holder.fcstd` | 激光支架 |
| `laser-wireway.fcstd` | 激光走线槽 |
| `motor-holder.fcstd` | 电机支架 |
| `motor-wireway.fcstd` | 电机走线槽 |

## 使用说明

1. **STEP (.stp)**：通用三维格式，可用 FreeCAD / SolidWorks / Fusion360 打开，
   用于查看、修改、3D 打印。
2. **FreeCAD (.fcstd)**：FreeCAD 原生工程文件，可编辑参数化模型。
3. 建议使用 **FreeCAD**（开源免费）打开与编辑。

> 注意：`OpenSCAD-2021.01-x86-64-Installer.exe` 为 OpenSCAD 安装包，
> 未纳入本仓库（体积大且非模型文件）。如需请自行安装。

## 本项目机械设计对应关系

```
ciclop 原始结构            TriniScan 本项目
─────────────────         ──────────────────
camera-holder      →      相机固定支架（俯仰锁死）
laser-holder       →      线激光固定支架（倾角锁死）
disc-holder        →      转台承载盘
motor-holder       →      NEMA17 电机座
shaft-coupling     →      电机-转台联轴
bearing-clip       →      转轴轴承固定
```

## 待办

- [ ] 根据本项目工作距离（~300mm）与机体尺寸（≤50cm³）重新校核支架尺寸
- [ ] 设计底板/框架（铝型材）承载全部模块
- [ ] 设计遮光罩/遮光箱
- [ ] 输出本项目装配图与爆炸图
