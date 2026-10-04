#!/usr/bin/env bash
# TriniScan 图形界面启动脚本 (Linux/macOS/Git-Bash)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/software:${PYTHONPATH}"
python -m triniscan.ui --config software/config/default.yaml "$@"
