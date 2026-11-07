$ErrorActionPreference = 'Stop'

$ollama = (Get-Command ollama -CommandType Application -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    $ollama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
}
if (-not (Test-Path -LiteralPath $ollama -PathType Leaf)) {
    throw 'Install Ollama for Windows from https://ollama.com/download/windows, then rerun Live setup.'
}

Push-Location (Join-Path $PSScriptRoot '..')
try {
    $python = if (Test-Path -LiteralPath '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }
    $model = & $python -c 'from speedlimit.air_quality import OLLAMA_MODEL; print(OLLAMA_MODEL)'
    if ($LASTEXITCODE -ne 0 -or -not $model) {
        throw 'Could not read the project model setting. Run Live setup with Python available.'
    }
    $server = 'http://127.0.0.1:11434'
    try {
        $models = Invoke-RestMethod "$server/api/tags" -TimeoutSec 3
    } catch {
        $logs = New-Item -ItemType Directory -Force -Path 'artifacts'
        $start = [System.Diagnostics.ProcessStartInfo]::new($ollama, 'serve')
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.WorkingDirectory = $logs.FullName
        $start.EnvironmentVariables['OLLAMA_HOST'] = '127.0.0.1:11434'
        $start.EnvironmentVariables['OLLAMA_NO_CLOUD'] = '1'
        $null = [System.Diagnostics.Process]::Start($start)
        for ($attempt = 0; $attempt -lt 15 -and $null -eq $models; $attempt++) {
            Start-Sleep -Seconds 1
            try { $models = Invoke-RestMethod "$server/api/tags" -TimeoutSec 2 } catch {}
        }
        if ($null -eq $models) { throw 'Ollama did not start on the local port 11434.' }
    }
    if ($model -notin $models.models.name) {
        # Point the CLI at the same local server even if the parent shell uses a different host.
        $previousHost = $env:OLLAMA_HOST
        try {
            $env:OLLAMA_HOST = '127.0.0.1:11434'
            & $ollama pull $model
            if ($LASTEXITCODE -ne 0) { throw "Ollama could not download $model. Retry Live setup to resume." }
        } finally {
            $env:OLLAMA_HOST = $previousHost
        }
    }
    Write-Output "SpeedLimit Live is provisioned with $model."
} finally {
    Pop-Location
}
