"""STEP(.stp/.step) 零件信息提取工具。

纯正则解析 ISO-10303-21 (AP214)，**无需安装任何 CAD**，
提取每个零件的：

- 包围盒尺寸（长×宽×高，mm）
- 坐标偏移（零件原点相对文件坐标原点）
- 几何要素计数（平面/圆柱面/圆，用于判断含多少孔/回转特征）

注意
----
STEP 的 ``CARTESIAN_POINT`` 包含朝向矩阵的原点、装配偏移等参考点，
因此本工具给出的包围盒是"包含所有定义点的范围"，用于快速了解尺寸
量级与接口位置，**不能替代精确测量**。若需精确值请用 CAD 打开。

用法::

    python -m triniscan.tools.step_info <目录或文件> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

import numpy as np

# 匹配 #123=CARTESIAN_POINT('name',(x,y,z)); 允许跨行
_POINT_RE = re.compile(
    r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*"
    r"([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
    re.IGNORECASE,
)
_DIR_RE = re.compile(r"DIRECTION\s*\(\s*'[^']*'\s*,\s*\(\s*"
                     r"([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)",
                     re.IGNORECASE)
_CIRCLE_RE = re.compile(r"=\s*CIRCLE\s*\(", re.IGNORECASE)
_CYL_RE = re.compile(r"=\s*CYLINDRICAL_SURFACE\s*\(", re.IGNORECASE)
_PLANE_RE = re.compile(r"=\s*PLANE\s*\(", re.IGNORECASE)


@dataclass
class StepInfo:
    """单个 STEP 文件的分析结果。"""

    file: str
    size_xyz_mm: tuple = (0.0, 0.0, 0.0)
    bbox_min_mm: tuple = (0.0, 0.0, 0.0)
    bbox_max_mm: tuple = (0.0, 0.0, 0.0)
    n_points: int = 0
    n_planes: int = 0
    n_cylinders: int = 0
    n_circles: int = 0
    unit: str = "mm"

    @property
    def volume_bbox_cm3(self) -> float:
        """包围盒体积（cm³），用于快速估料。"""
        x, y, z = self.size_xyz_mm
        return x * y * z / 1000.0


def analyse_step(path: str) -> StepInfo:
    """解析一个 STEP 文件，返回 :class:`StepInfo`。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    xs, ys, zs = [], [], []
    for m in _POINT_RE.finditer(text):
        try:
            xs.append(float(m.group(1)))
            ys.append(float(m.group(2)))
            zs.append(float(m.group(3)))
        except ValueError:
            continue

    info = StepInfo(file=os.path.basename(path))
    if not xs:
        return info

    pts = np.column_stack([xs, ys, zs])
    # 用密集区间而非 min/max：STEP 含装配偏移与朝向矩阵的孤立参考点，
    # 直接取极值会得到上千 mm 的假尺寸。取 [0.5%, 99.5%] 分位可剔除。
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    info.bbox_min_mm = tuple(round(float(v), 2) for v in lo)
    info.bbox_max_mm = tuple(round(float(v), 2) for v in hi)
    info.size_xyz_mm = tuple(round(float(hi[i] - lo[i]), 2)
                             for i in range(3))
    info.n_points = len(xs)
    info.n_planes = len(_PLANE_RE.findall(text))
    info.n_cylinders = len(_CYL_RE.findall(text))
    info.n_circles = len(_CIRCLE_RE.findall(text))
    return info


def analyse_dir(target: str) -> list[StepInfo]:
    """分析目录下（或单个文件）的所有 STEP 文件。"""
    if os.path.isfile(target):
        return [analyse_step(target)]
    results = []
    for name in sorted(os.listdir(target)):
        if name.lower().endswith((".stp", ".step")):
            results.append(analyse_step(os.path.join(target, name)))
    return results


def format_table(items: list[StepInfo]) -> str:
    """把结果格式化为等宽表格。"""
    header = (f"{'文件':<26}{'长 x 宽 x 高 (mm)':<28}"
              f"{'包围盒 (cm3)':<14}{'平面':>5}{'圆柱':>5}{'圆':>5}")
    lines = [header, "-" * len(header)]
    for it in items:
        size = " × ".join(f"{v:.1f}" for v in it.size_xyz_mm)
        lines.append(
            f"{it.file:<26}{size:<26}{it.volume_bbox_cm3:<14.1f}"
            f"{it.n_planes:>5}{it.n_cylinders:>5}{it.n_circles:>5}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="STEP 零件信息提取（无需 CAD）")
    parser.add_argument("target", help="STEP 文件或包含 STEP 的目录")
    parser.add_argument("--json", default=None, help="结果另存为 JSON")
    args = parser.parse_args(argv)

    # Windows 控制台默认 GBK，强制 UTF-8 避免中文/符号报错
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    if not os.path.exists(args.target):
        print(f"路径不存在: {args.target}", file=sys.stderr)
        return 1

    items = analyse_dir(args.target)
    if not items:
        print("未找到 STEP 文件", file=sys.stderr)
        return 1

    print(format_table(items))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(it) for it in items], f, ensure_ascii=False,
                      indent=2)
        print(f"\n已保存: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
