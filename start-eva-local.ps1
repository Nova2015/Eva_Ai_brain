$ErrorActionPreference = "Stop"

$Root = "D:\Eva_Ai_brain"
$Bootstrap = "$Root\scripts\eva_local_bootstrap.py"

$Services = @(
    @{ Name="EVA Backend"; Port=8000; Url="http://127.0.0.1:8000/docs"; Folder=$Root; Python="$Root\.venv311\Scripts\python.exe"; Service="backend" },
    @{ Name="EVA ML + RAG"; Port=7860; Url="http://127.0.0.1:7860/docs"; Folder="$Root\eva-ml-service"; Python="$Root\eva-ml-service\.venv\Scripts\python.exe"; Service="ml" },
    @{ Name="EVA Agent"; Port=8100; Url="http://127.0.0.1:8100/docs"; Folder="$Root\src"; Python="$Root\.openhands-venv\Scripts\python.exe"; Service="agent" }
)

function Test-EvaHttp {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400)
    }
    catch {
        return $false
    }
}

function Test-EvaTcp {
    param([int]$Port)

    return (Test-NetConnection 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
}

function Wait-EvaHttp {
    param(
        [string]$Name,
        [string]$Url,
        [int]$Timeout = 60
    )

    $timer = [Diagnostics.Stopwatch]::StartNew()

    while ($timer.Elapsed.TotalSeconds -lt $Timeout) {
        if (Test-EvaHttp $Url) {
            Write-Host "$Name : READY" -ForegroundColor Green
            return $true
        }

        Start-Sleep -Seconds 2
    }

    Write-Host "$Name : TIMEOUT - Check service window" -ForegroundColor Red
    return $false
}

function Start-EvaService {
    param($Service)

    if (Test-EvaHttp $Service.Url) {
        Write-Host "$($Service.Name) : Already running and healthy" -ForegroundColor Green
        return
    }

    if (Test-EvaTcp $Service.Port) {
        Write-Host "$($Service.Name) : Port $($Service.Port) is occupied, but expected URL is not healthy. Skipping to avoid duplicate start." -ForegroundColor Red
        return
    }

    if (!(Test-Path $Service.Python)) {
        Write-Host "$($Service.Name) : Python not found: $($Service.Python)" -ForegroundColor Red
        return
    }

    $child = "Set-Location -LiteralPath '$($Service.Folder)'; & '$($Service.Python)' '$Bootstrap' '$($Service.Service)'"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($child))

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-EncodedCommand",
        $encoded
    )

    Write-Host "$($Service.Name) : Launch command sent" -ForegroundColor Yellow
}

if (!(Test-Path "$Root\.env")) {
    throw "Root .env file not found."
}

if (!(Test-Path $Bootstrap)) {
    throw "Bootstrap script not found: $Bootstrap"
}

# Guard variables are set before child services start.
$env:EVA_LOCAL_ONLY = "1"
$env:EVA_LOCAL_GUARD_MODE = "log"
$env:EVA_LOCAL_ALLOW_LAN = "1"
$env:EVA_LOCAL_GUARD_LOG = "$Root\eva_blocked_hosts.log"
$env:PYTHONPATH = "$Root;$Root\src;$Root\eva-ml-service"

Write-Host "`n=== EVA LOCAL STARTUP ===" -ForegroundColor Cyan

# Launch backend, ML and agent.
foreach ($service in $Services) {
    Start-EvaService $service
}

# Launch React frontend.
$frontendUrl = "http://127.0.0.1:3000/"

if (Test-EvaHttp $frontendUrl) {
    Write-Host "EVA Frontend : Already running and healthy" -ForegroundColor Green
}
elseif (Test-EvaTcp 3000) {
    Write-Host "EVA Frontend : Port 3000 occupied, but frontend is not healthy. Skipping." -ForegroundColor Red
}
else {
    $frontendFolder = "$Root\frontend"

    if (!(Test-Path "$frontendFolder\package.json")) {
        Write-Host "Frontend package.json not found." -ForegroundColor Red
    }
    else {
        $frontendCommand = "Set-Location -LiteralPath '$frontendFolder'; `$env:PORT='3000'; `$env:REACT_APP_API_BASE='http://127.0.0.1:8000'; `$env:REACT_APP_RAG_API_BASE='http://127.0.0.1:7860'; npm start"
        $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($frontendCommand))

        Start-Process powershell.exe -ArgumentList @(
            "-NoExit",
            "-EncodedCommand",
            $encoded
        )

        Write-Host "EVA Frontend : Launch command sent" -ForegroundColor Yellow
    }
}

# Wait for each service and verify its actual HTTP response.
Write-Host "`n=== HTTP READINESS CHECK ===" -ForegroundColor Cyan

$results = @()

foreach ($service in $Services) {
    $ready = Wait-EvaHttp $service.Name $service.Url 60

    $results += [PSCustomObject]@{
        Service = $service.Name
        Port = $service.Port
        Status = if ($ready) { "READY" } else { "NOT READY" }
    }
}

$frontendReady = Wait-EvaHttp "EVA Frontend" $frontendUrl 60

$results += [PSCustomObject]@{
    Service = "EVA Frontend"
    Port = 3000
    Status = if ($frontendReady) { "READY" } else { "NOT READY" }
}

Write-Host "`n=== EVA SERVICE STATUS ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize

Write-Host "Local Guard: LOG mode (monitor only; not a firewall)" -ForegroundColor Yellow

Write-Host "`nFrontend : http://127.0.0.1:3000"
Write-Host "Backend  : http://127.0.0.1:8000/docs"
Write-Host "ML/RAG   : http://127.0.0.1:7860/docs"
Write-Host "Agent    : http://127.0.0.1:8100/docs"

if (@($results | Where-Object { $_.Status -ne "READY" }).Count -gt 0) {
    Write-Host "`nSome services are not ready. Check their PowerShell windows." -ForegroundColor Red
}
else {
    Write-Host "`nAll EVA services are ready!" -ForegroundColor Green
}