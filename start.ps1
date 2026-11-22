Get-Content "E:\code\Rag project\.env" | ForEach-Object {
    if ($_ -match '^([^#][^=]*)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}
cd "E:\code\Rag project\src"
python -m uvicorn app:app --reload --port 8002