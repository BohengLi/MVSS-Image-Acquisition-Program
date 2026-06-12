@echo off
chcp 65001 >nul
cd /d "%~dp0"
python stereo_capture_only.py
pause
