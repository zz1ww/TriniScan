# 激光中心线实验台（lab）

从**已拍好的一组图**里提取激光中心线、三角测量、拼成点云。不依赖 `triniscan` 主程序包，也不走 UI。
用途：快速对比不同中心线提取算法（灰度重心 / 抛物线 / Steger 等），不污染主程序。

## 用法

```bash
cd TriniScan

python lab/run.py --list        # 列出可用算法
python lab/run.py               # 用默认配置跑
python lab/run.py --config lab/config.yaml --method steger
```

## 参数入口

全部在 `lab/config.yaml`，包括输入图目录、算法名、阈值、三角测量参数。

## 文件

| 路径 | 作用 |
|---|---|
| `run.py` | 入口，串起 预处理 → 中心线提取 → 三角测量 → 点云 |
| `methods.py` | 各中心线提取算法实现（注册在 `METHODS` 里） |
| `_steger.py` | Steger 算法（旧实现，保留对照） |
| `config.yaml` | 配置 |
| `outputs/` | 运行输出（点云 / 图 / 日志），**不入库** |
| `_debug/` | 中间调试图，**不入库** |

## 加一个新算法

1. 在 `methods.py` 写 `def 名字(gray, **kwargs) -> (xs, ys)`（亚像素坐标，按行）
2. 注册到文件末尾的 `METHODS` 字典
3. `python lab/run.py --list` 能看到，就可用 `--method 名字` 跑

## 注意

- `outputs/` 与 `_debug/` 已列入 `.gitignore`，体积可能到几百 MB，别误删成"整理"
- 本实验台的结论要回写到 `docs/05_体积计算方案.md` 与 `docs/10_采集链路与性能.md`
