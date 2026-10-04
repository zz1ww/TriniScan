@echo off
REM TriniScan 环境安装脚本 (Windows)
echo === 创建虚拟环境 ===
python -m venv .venv
call .venv\Scripts\activate.bat
echo === 安装依赖 ===
python -m pip install --upgrade pip
pip install -r software\requirements.txt
echo === 完成 ===
echo 运行: scripts\run.bat
