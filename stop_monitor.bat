@echo off
taskkill /FI "WINDOWTITLE eq KPL-Monitor*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq KPL Monitor Loop*" /T /F >nul 2>&1
if exist "%~dp0data\realtime\monitor.pid" del /f "%~dp0data\realtime\monitor.pid"
echo [OK] Monitor stopped (if it was running).
pause
