# train_watchdog.ps1
# Auto-restarts self_train.py --vision on any crash.
# Logs each crash with timestamp to logs/watchdog_train.log
# Run via start_tonight.ps1 or manually in its own window.

$ProjectDir = $PSScriptRoot
$Python     = "$ProjectDir\.venv\Scripts\python.exe"
$LogFile    = "$ProjectDir\logs\watchdog_train.log"
$TrainArgs  = @("self_train.py", "--vision")

Set-Location $ProjectDir
New-Item -ItemType Directory -Force -Path "$ProjectDir\logs" | Out-Null

$run = 0
while ($true) {
    $run++
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $msg = "[$ts] Run #$run - starting trainer"
    Write-Host $msg -ForegroundColor Cyan
    Add-Content $LogFile $msg

    & $Python @TrainArgs 2>&1

    $code = $LASTEXITCODE
    $ts2  = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

    if ($code -eq 99) {
        # Another trainer holds the lock — wait quietly and retry
        $msg2 = "[$ts2] Run #$run - lock held by another trainer (code 99). Waiting 120s..."
        Write-Host $msg2 -ForegroundColor Yellow
        Add-Content $LogFile $msg2
        Start-Sleep 120
        $run--   # don't count lock-wait as a separate run number
        continue
    } elseif ($code -eq 0) {
        $msg2 = "[$ts2] Run #$run - clean exit (code 0). Restarting..."
        Write-Host $msg2 -ForegroundColor Green
    } else {
        $msg2 = "[$ts2] Run #$run - CRASHED (code $code). Restarting in 5s..."
        Write-Host $msg2 -ForegroundColor Red
        Start-Sleep 5
    }
    Add-Content $LogFile $msg2
}
