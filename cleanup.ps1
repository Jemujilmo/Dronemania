# cleanup.ps1 - kills all Dronemania watchdog windows and Python, then does a clean launch
# Run this from a fresh PowerShell window as admin if needed

Write-Host "=== DRONEMANIA CLEAN LAUNCH ===" -ForegroundColor Yellow

# Kill all Python first
Write-Host "Killing all Python..." -ForegroundColor Cyan
taskkill /F /IM python.exe 2>$null
Start-Sleep 1

# Kill all PowerShell windows running watchdog scripts or Dronemania Python
$killed = 0
Get-CimInstance Win32_Process -Filter "name='powershell.exe'" | ForEach-Object {
    $cmd = $_.CommandLine
    $wdPid = $_.ProcessId
    # Skip VS Code terminals and this script itself
    if ($cmd -like '*train_watchdog*' -or $cmd -like '*monitor_watchdog*' -or
        ($cmd -like '*Dronemania*' -and $cmd -notlike '*VS Code*' -and $cmd -notlike '*cleanup*' -and $wdPid -ne $PID)) {
        Write-Host "  Killing PS PID $wdPid" -ForegroundColor Red
        taskkill /F /PID $wdPid 2>$null
        $killed++
    }
}
Write-Host "Killed $killed watchdog windows."
Start-Sleep 2

# Restore fallback checkpoint ONLY if latest is missing or corrupt.
# Never blindly overwrite a valid model — that destroys training progress.
$latest   = "trained_models\rl_vision_policy_latest.zip"
$bestSave = "trained_models\rl_vision_policy_best.zip"   # saved when avg_gates hits a new high
$fallback = "trained_models\rl_vision_policy_2gates_checkpoint.zip"  # old emergency fallback
Write-Host "Checking checkpoint..." -ForegroundColor Cyan

$needRestore = $false
if (-not (Test-Path $latest)) {
    Write-Host "  latest.zip missing - will restore from best/fallback" -ForegroundColor Yellow
    $needRestore = $true
} else {
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $latest).Path)
        $zip.Dispose()
        $sz = (Get-Item $latest).Length
        Write-Host "  latest.zip valid ($sz bytes) - keeping existing model" -ForegroundColor Green
    } catch {
        Write-Host "  latest.zip corrupt - will restore from best/fallback" -ForegroundColor Yellow
        $needRestore = $true
    }
}
if ($needRestore) {
    if (Test-Path $bestSave) {
        Copy-Item $bestSave $latest -Force
        Write-Host "  Restored from best checkpoint: $bestSave" -ForegroundColor Green
    } elseif (Test-Path $fallback) {
        Copy-Item $fallback $latest -Force
        Write-Host "  Restored from emergency fallback: $fallback" -ForegroundColor Yellow
    } else {
        Write-Host "  No restore source found - will start fresh" -ForegroundColor Red
    }
}

# Remove stale trainer lock (left behind if Python was hard-killed)
$lockFile = "trained_models\trainer.lock"
if (Test-Path $lockFile) {
    Remove-Item $lockFile -Force
    Write-Host "Removed stale trainer.lock" -ForegroundColor Yellow
}

# Final check
Start-Sleep 1
$pyCount = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "Python processes before launch: $pyCount"
if ($pyCount -gt 0) {
    Write-Host "WARNING: Python still running, killing again..." -ForegroundColor Red
    taskkill /F /IM python.exe 2>$null
    Start-Sleep 2
}

# Launch
Write-Host ""
Write-Host "Launching overnight training..." -ForegroundColor Green
.\start_tonight.ps1
