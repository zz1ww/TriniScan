@echo off
REM TriniScan 启动脚本 (Windows)
cd /d "%~dp0\.."
set PYTHONPATH=%cd%\software;%PYTHONPATH%
python -m triniscan.core.main --config software\config\default.yaml %*
