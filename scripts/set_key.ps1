# Saves your Gemini key into .env (hidden prompt, nothing printed). Run it once.
# Usage:  powershell -ExecutionPolicy Bypass -File D:\projects\rag-docqa\scripts\set_key.ps1

$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) { Copy-Item (Join-Path $root ".env.example") $envFile }

$secure = Read-Host "Paste your Gemini API key (hidden)" -AsSecureString
$key = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)).Trim()
if (-not $key) { Write-Host "No key entered. Nothing changed." -ForegroundColor Yellow; exit 1 }

$lines = Get-Content $envFile
$found = $false
$lines = $lines | ForEach-Object {
    if ($_ -match '^\s*LLM_API_KEY\s*=') { $found = $true; "LLM_API_KEY=$key" } else { $_ }
}
if (-not $found) { $lines += "LLM_API_KEY=$key" }
[IO.File]::WriteAllLines($envFile, $lines, (New-Object Text.UTF8Encoding $false))  # UTF-8, no BOM

Write-Host "Saved to $envFile. Now restart the server (or tell Claude 'done')." -ForegroundColor Green
