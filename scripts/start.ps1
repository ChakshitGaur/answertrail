# Starts the app (API + web UI on one server) and opens it in your browser.
# Usage:  powershell -ExecutionPolicy Bypass -File D:\projects\answertrail\scripts\start.ps1
# Asks for the Gemini key (hidden). The key lives only in the server window and is never saved to disk.
# Add -Streamlit to also start the older Streamlit UI on http://localhost:8501.

param([switch]$Streamlit)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

$env:EMBEDDING_BACKEND = "fastembed"

# If a key is saved in .env (see .env.example) the app reads it itself, so there is nothing to ask.
$envFile = Join-Path $root ".env"
$hasSavedKey = (Test-Path $envFile) -and (Select-String -Path $envFile -Pattern '^\s*(LLM_API_KEY|ANTHROPIC_API_KEY)\s*=\s*\S' -Quiet)
if ($hasSavedKey) {
    Write-Host "Using the key saved in .env" -ForegroundColor Green
    $key = $null
} else {
    $secure = Read-Host "Gemini API key (hidden, press Enter to start without one)" -AsSecureString
    $key = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}

if ($key) {
    $env:LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    $env:LLM_MODEL = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "gemini-3.6-flash" }
    $env:LLM_FALLBACK_MODELS = if ($env:LLM_FALLBACK_MODELS) { $env:LLM_FALLBACK_MODELS } else { "gemini-3.5-flash-lite,gemini-3.8-flash" }
    $env:LLM_API_KEY = $key
} elseif (-not $hasSavedKey) {
    Write-Host "No key: answers will be raw passages." -ForegroundColor Yellow
}

# child windows inherit the environment variables set above
Start-Process $py -ArgumentList "`"$root\scripts\run_api.py`"" -WorkingDirectory $root
if ($Streamlit) {
    Start-Process $py -ArgumentList "-m streamlit run `"$root\ui\app.py`" --server.port 8501 --browser.gatherUsageStats false" -WorkingDirectory $root
}

Write-Host "`nStarted. The first start takes ~15 seconds (it loads the search model)." -ForegroundColor Green
Start-Sleep -Seconds 12
Start-Process "http://localhost:8000"
