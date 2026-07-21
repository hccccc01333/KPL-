@echo off
rem Run by double-click in Explorer, or: cmd /c start_monitor.bat
cd /d "%~dp0"
start "KPL-Monitor" "%ComSpec%" /k "%~dp0scripts\launch_monitor.bat"
echo [OK] KPL monitor started in a new window.
echo      Close that window to stop, or run stop_monitor.bat
