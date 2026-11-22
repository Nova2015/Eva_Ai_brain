$ErrorActionPreference = "Stop"

function Test-EvaPort {
    param([int]$Port)

    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Start-EvaWindow {
    param(
        [string]$Name,
        [int]$Port,
        [string]$Folder,
        [string]$Command
    )

    if (Test-EvaPort $Port) {
        Write-Host "$Name : Already ON (port $Port)" -ForegroundColor Green
        return
    }

    $script = "Set-Location -LiteralPath '$Folder'; $Command"
    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($script)
    )

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit", "-EncodedCommand", $encoded
    )

    Write-Host "$Name : Starting (port $Port)" -ForegroundColor Yellow
}

function Start-EvaAgent {
    if (Test-EvaPort 8100) {
        Write-Host "Agent Service : Already ON (port 8100)" -ForegroundColor Green
        return
    }

    $script = @'
$ErrorActionPreference = "Stop"
$env:AGENT_WORKSPACE = "D:\eva_agent_workspace"
$env:OPENHANDS_SUPPRESS_BANNER = "1"

$line = Get-Content "D:\Eva_Ai_brain\.env" |
    Where-Object { $_ -match '^\s*AGENT_API_KEY\s*=' } |
    Select-Object -First 1

if ($line) {
    $env:AGENT_API_KEY = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
}

Set-Location "D:\Eva_Ai_brain\src"
& "D:\Eva_Ai_brain\.openhands-venv\Scripts\python.exe" -m uvicorn agent_service:app --host 127.0.0.1 --port 8100
'@

    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($script)
    )

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit", "-EncodedCommand", $encoded
    )

    Write-Host "Agent Service : Starting (port 8100)" -ForegroundColor Yellow
}

Start-EvaWindow "RAG Backend" 8000 `
    "D:\Eva_Ai_brain" `
    "& '.\.venv311\Scripts\python.exe' -m uvicorn src.app:app --host 127.0.0.1 --port 8000"

Start-EvaWindow "ML Service" 7860 `
    "D:\Eva_Ai_brain\eva-ml-service" `
    "& '.\.venv\Scripts\python.exe' app.py"

Start-EvaAgent
