# KPL monitor launcher (ASCII-safe for Windows PowerShell 5.1)
param([switch]$NewWindow, [switch]$Stop)

$root = Split-Path $PSScriptRoot -Parent
$bat = Join-Path $root "start_monitor.bat"
$stopBat = Join-Path $root "stop_monitor.bat"

if ($Stop) {
    & $stopBat
    exit $LASTEXITCODE
}

if ($NewWindow) {
    Start-Process -FilePath $bat -WorkingDirectory $root
    exit 0
}

& (Join-Path $PSScriptRoot "run_monitor_loop.py")
