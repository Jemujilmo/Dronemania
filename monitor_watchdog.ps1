# monitor_watchdog.ps1
# Auto-restarts monitor.py on any crash.
# Waits 10s on first start to give the trainer time to write the first checkpoint.
# Logs to logs/watchdog_monitor.log

$ProjectDir = $PSScriptRoot
$Python     = "$ProjectDir\.venv\Scripts\python.exe"
$LogFile    = "$ProjectDir\logs\watchdog_monitor.log"
$MonitorArgs = @("monitor.py")

Set-Location $ProjectDir
New-Item -ItemType Directory -Force -Path "$ProjectDir\logs" | Out-Null

Write-Host "[monitor watchdog] Waiting 10s for trainer to initialise..." -ForegroundColor Yellow
Start-Sleep 10

$run = 0
while ($true) {
    $run++
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $msg = "[$ts] Run #$run - starting monitor"
    Write-Host $msg -ForegroundColor Cyan
    Add-Content $LogFile $msg

    & $Python @MonitorArgs 2>&1

    $code = $LASTEXITCODE
    $ts2  = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

    if ($code -eq 0) {
        $msg2 = "[$ts2] Run #$run - clean exit (code 0). Restarting in 3s..."
        Write-Host $msg2 -ForegroundColor Green
        Start-Sleep 3
    } else {
        $msg2 = "[$ts2] Run #$run - CRASHED (code $code). Restarting in 5s..."
        Write-Host $msg2 -ForegroundColor Red
        Start-Sleep 5
    }
    Add-Content $LogFile $msg2
}
