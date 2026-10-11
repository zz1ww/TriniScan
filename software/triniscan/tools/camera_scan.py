"""相机索引扫描工具。

用途
----
电脑上接了多个摄像头时，无法确定该用哪个 ``camera.index``。
本工具依次尝试打开 0..N，报告每个索引能否打开、实际分辨率、
以及能否读到图像，并把抓到的首帧存成图片便于肉眼确认。

用法::

    # 扫描 0..5
    python -m triniscan.tools.camera_scan

    # 扫描 0..9，抓图存到指定目录
    python -m triniscan.tools.camera_scan --max 9 --out data/camera_check

    # 只测某个索引
    python -m triniscan.tools.camera_scan --index 1

    # 指定后端（Windows 默认 dshow，Linux/Mac 用 any）
    python -m triniscan.tools.camera_scan --backend any
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np


# 常见 usb 相机在 OpenCV 下的控制项 ID（不同后端可能不同）
_BACKEND_MAP = {
    "dshow": cv2.CAP_DSHOW,     # Windows DirectShow
    "msmf": cv2.CAP_MSMF,       # Windows Media Foundation
    "any": cv2.CAP_ANY,         # 自动
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),  # Linux
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),  # macOS
}


def _open(index: int, backend: str):
    """尝试打开相机，返回 VideoCapture 或 None。"""
    flag = _BACKEND_MAP.get(backend, cv2.CAP_ANY)
    cap = cv2.VideoCapture(index, flag)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def scan_one(index: int, backend: str, out_dir: str | None,
             warmup: int = 5) -> dict | None:
    """打开指定索引并采集一帧；成功返回信息字典，失败返回 None。"""
    cap = _open(index, backend)
    if cap is None:
        return None

    try:
        # 尝试请求一个常见分辨率（相机可能不支持，会静默回落）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        for _ in range(warmup):        # 丢弃前几帧让曝光稳定
            cap.read()

        t0 = time.time()
        ok, frame = cap.read()
        dt = time.time() - t0

        if not ok or frame is None:
            return {"index": index, "opened": True, "readable": False}

        h, w = frame.shape[:2]
        info = {
            "index": index,
            "opened": True,
            "readable": True,
            "width": w,
            "height": h,
            "mean": float(frame.mean()),
            "min": int(frame.min()),
            "max": int(frame.max()),
            "latency_ms": dt * 1e3,
        }

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"index_{index}_{w}x{h}.png")
            cv2.imwrite(path, frame)
            info["saved"] = path

        return info
    finally:
        cap.release()


def _fmt(info: dict) -> str:
    if not info.get("readable"):
        return (f"  index {info['index']}: 能打开但**读不到图像**"
                f"（可能被占用/驱动异常）")
    line = (f"  index {info['index']}: {info['width']}x{info['height']}  "
            f"亮度均值={info['mean']:.1f} 范围=[{info['min']},{info['max']}]  "
            f"延迟={info['latency_ms']:.0f}ms")
    if "saved" in info:
        line += f"\n      → 已存图: {info['saved']}"
    return line


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="camera-scan",
        description="扫描可用摄像头并报告索引/分辨率/是否可读图")
    p.add_argument("--max", type=int, default=5,
                   help="扫描的最大索引（含），默认 5（即 0..5）")
    p.add_argument("--index", type=int, default=None,
                   help="只测指定索引")
    p.add_argument("--backend", default="dshow",
                   choices=list(_BACKEND_MAP),
                   help="采集后端，Windows 一般为 dshow")
    p.add_argument("--out", default=None,
                   help="抓图保存目录（便于肉眼确认哪个是目标相机）")
    p.add_argument("--warmup", type=int, default=5,
                   help="抓图前丢弃的帧数，默认 5")
    args = p.parse_args(argv)

    indices = [args.index] if args.index is not None else \
        list(range(args.max + 1))

    print(f"扫描相机索引 {indices[0]}..{indices[-1]} "
          f"(backend={args.backend})")
    print("-" * 60)

    found = []
    for i in indices:
        try:
            info = scan_one(i, args.backend, args.out, args.warmup)
        except Exception as exc:  # noqa: BLE001 - 单个相机失败不应中断扫描
            print(f"  index {i}: 打开异常 — {exc}")
            continue
        if info is None:
            print(f"  index {i}: 无设备")
            continue
        found.append(info)
        print(_fmt(info))

    print("-" * 60)
    usable = [f for f in found if f.get("readable")]
    if usable:
        print(f"发现 {len(usable)} 个可用相机："
              f"{[f['index'] for f in usable]}")
        # 测量相机通常是分辨率最高、且不是 640x480/1280x720 的那个。
        # 按分辨率面积排序，给出"更可能是测量相机"的推荐（仅供参考）。
        best = max(usable, key=lambda f: f["width"] * f["height"])
        print(f"\n其中分辨率最高的是 index {best['index']}"
              f"（{best['width']}x{best['height']}），"
              f"通常就是外置测量相机。")
        if args.out:
            print(f"抓图已保存到 {args.out}/，**请打开图片逐个确认**"
                  f"哪个画面是你的测量场景（转台/激光），")
            print(f"然后把 config/default.yaml 里的 camera.index "
                  f"设为该索引。")
        else:
            print("\n建议加 --out <目录> 抓图，肉眼确认哪个才是测量相机：")
            print("  python -m triniscan.tools.camera_scan --out "
                  "data/camera_check")
        return 0
    print("未发现可用相机。请检查：① USB 是否插好；"
          "② 是否被微信/QQ/其他程序占用；③ 换个 backend 再试。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
