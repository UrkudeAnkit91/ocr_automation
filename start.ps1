param([switch]$NoFrontend)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# Kill old processes
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'server.py' } | Stop-Process -Force
Get-Process -Name node -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Start-Sleep 1

# Start backend
Write-Host "Starting backend..." -ForegroundColor Green
$backend = Start-Process -NoNewWindow -FilePath python -ArgumentList "server.py" -WorkingDirectory $root -PassThru
Write-Host "  Backend PID: $($backend.Id)"

# Wait for backend
for ($i = 0; $i -lt 15; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/status" -UseBasicParsing -TimeoutSec 2
        Write-Host "  Backend ready!" -ForegroundColor Green
        break
    } catch {
        Start-Sleep 1
    }
}

if (-not $NoFrontend) {
    Write-Host "Starting frontend..." -ForegroundColor Green
    $frontend = Start-Process -NoNewWindow -FilePath pwsh.exe -ArgumentList "-NoLogo","-NoProfile","-Command","ng serve --host 127.0.0.1 --port 4200" -WorkingDirectory "$root\frontend" -PassThru
    Write-Host "  Frontend PID: $($frontend.Id)"
    Start-Sleep 10
}

Write-Host ""
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:4200" -ForegroundColor Cyan
Write-Host ""
Write-Host "Trigger OCR: curl -X POST http://localhost:8000/api/trigger" -ForegroundColor Yellow
Write-Host "Stop: Ctrl+C in each window, or kill PIDs $($backend.Id)" -ForegroundColor Yellow
