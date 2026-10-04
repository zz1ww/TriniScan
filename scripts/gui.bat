@echo off
REM TriniScan 图形界面启动脚本 (Windows)
cd /d "%~dp0\.."
set PYTHONPATH=%cd%\software;%PYTHONPATH%
python -m triniscan.ui --config software\config\default.yaml %*
