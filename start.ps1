param([string]$Python = "python", [int]$Port = 8501)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    & $Python -m streamlit run (Join-Path $PSScriptRoot "app.py") --server.address 127.0.0.1 --server.port $Port --browser.gatherUsageStats false
} finally {
    Pop-Location
}
