# start_tonight.ps1
# One-click overnight run: kills any stale Python, then opens
# the trainer watchdog and monitor watchdog in separate windows.

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "Killing any stale Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2

Write-Host "Launching trainer watchdog..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList `
    "-NoExit", "-ExecutionPolicy", "Bypass", `
    "-Command", "Set-Location '$ProjectDir'; .\train_watchdog.ps1" `
    -WindowStyle Normal

Start-Sleep 2

Write-Host "Launching monitor watchdog..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList `
    "-NoExit", "-ExecutionPolicy", "Bypass", `
    "-Command", "Set-Location '$ProjectDir'; .\monitor_watchdog.ps1" `
    -WindowStyle Normal

Write-Host ""
Write-Host "Both watchdogs launched." -ForegroundColor Green
Write-Host "Check progress anytime:"
Write-Host "  Get-Content '$ProjectDir\logs\watchdog_train.log' -Tail 10"
Write-Host "  Get-Content '$ProjectDir\records\vision_training_log.jsonl' | Select-Object -Last 3"
