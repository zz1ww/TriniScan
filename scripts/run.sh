#!/usr/bin/env bash
# TriniScan 启动脚本 (Linux/macOS/Git-Bash)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/software:${PYTHONPATH}"
python -m triniscan.core.main --config software/config/default.yaml "$@"
