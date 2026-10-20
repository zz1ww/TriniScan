# 激光中心线实验台

从**已拍好的一组图**里提取激光中心线、三角测量、拼成点云。不依赖 `triniscan` 主程序包，也不走 UI。

## 用法

```bash
cd TriniScan

# 列出可用算法
python lab/run.py --list

# 用默认配置跑
python lab/run.py

# 换算法 + 调阈值
python lab/run.py --method steger --threshold 80 --param sigma=2.0

# 只跑前 20 张（先验证链路）
python lab/run.py --limit 20

# 换配置文件 / 换输出目录
python lab/run.py --config lab/config.yaml --out lab/outputs/test1
```

## 输出

```
lab/outputs/
├── overlay/            每张图的提取结果（原图 + 红点）
├── cloud.ply           拼接后的点云
├── cloud.xyz           同上，文本格式
└── summary.json        每张图的点数、总点数、参数快照
```

## 加一个算法

在 `methods.py` 里写一个函数，然后 `register(...)`：

```python
@register("my_method", label="我的方法", params={"win": (5, "窗口")})
def my_method(gray, threshold, min_pixels, win=5):
    ...
    return np.array([[u, v], ...])   # (N, 2)
```

命令行即可用 `--method my_method --param win=7`。
