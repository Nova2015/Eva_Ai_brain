$ErrorActionPreference = "Continue"

Write-Host "`nStopping EVA local services..." -ForegroundColor Cyan

$all = @(Get-CimInstance Win32_Process)

$targets = @(
    $all | Where-Object {
        $_.CommandLine -match 'D:\\Eva_Ai_brain\\scripts\\eva_local_bootstrap\.py'
    }
)

$roots = foreach ($p in $targets) {
    $parent = $all | Where-Object { $_.ProcessId -eq $p.ParentProcessId } |
        Select-Object -First 1

    if (!$parent -or $parent.CommandLine -notmatch 'eva_local_bootstrap\.py') {
        $p
    }
}

# Identify the React launcher, not unrelated Node applications.
$react = @($all | Where-Object {
    $_.CommandLine -match 'D:\\Eva_Ai_brain\\frontend\\node_modules.*react-scripts.*start'
})

$reactIds = @($react | ForEach-Object { $_.ProcessId })

$reactRoots = foreach ($p in $react) {
    $parent = $all | Where-Object { $_.ProcessId -eq $p.ParentProcessId } |
        Select-Object -First 1

    if (!$parent -or $parent.ProcessId -notin $reactIds) {
        $p
    }
}

$roots = @($roots) + @($reactRoots)

if ($roots.Count -eq 0) {
    Write-Host "No EVA launcher processes found." -ForegroundColor Yellow
}

foreach ($p in $roots) {
    Write-Host "Stopping PID $($p.ProcessId): $($p.Name)" -ForegroundColor Yellow
    & taskkill.exe /PID $p.ProcessId /T /F
}

Start-Sleep -Seconds 3

Write-Host "`nRemaining EVA listeners:" -ForegroundColor Cyan

foreach ($port in 8000,7860,8100,3000) {
    $test = Test-NetConnection 127.0.0.1 -Port $port -WarningAction SilentlyContinue
    if ($test.TcpTestSucceeded) {
        Write-Host "$port : STILL LISTENING" -ForegroundColor Red
    } else {
        Write-Host "$port : STOPPED" -ForegroundColor Green
    }
}