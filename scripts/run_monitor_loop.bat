@echo off
rem Run in cmd.exe (double-click start_monitor.bat). Do NOT run in Git Bash.
title KPL Monitor Loop
cd /d "%~dp0"

if not exist "..\data\realtime\logs" mkdir "..\data\realtime\logs"

:loop
echo.
echo [%date% %time%] official_match_monitor.py starting...
python official_match_monitor.py
echo [%date% %time%] exited, restart in 30s. Close this window to stop.
ping 127.0.0.1 -n 31 >nul
goto loop
