param([switch]$AcceptNonCommercial)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$secretDir = Join-Path $root '.secrets'
[void][System.IO.Directory]::CreateDirectory($secretDir)
foreach ($name in @('worker_token', 'mcp_token')) {
    $path = Join-Path $secretDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        $bytes = New-Object byte[] 32
        $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
        $token = [Convert]::ToBase64String($bytes)
        # CreateNew prevents overwriting another setup process's token.
        $stream = [System.IO.File]::Open($path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
        try {
            $encoded = [System.Text.Encoding]::UTF8.GetBytes($token + "`n")
            $stream.Write($encoded, 0, $encoded.Length)
        } finally { $stream.Dispose() }
    }
    if ([string]::IsNullOrWhiteSpace([System.IO.File]::ReadAllText($path))) {
        throw "Empty secret file: $name. Replace it with a securely generated token before starting."
    }
}
$envPath = Join-Path $root '.env'
if (-not (Test-Path -LiteralPath $envPath)) {
    [System.IO.File]::Copy((Join-Path $root '.env.example'), $envPath, $false)
}
if ($AcceptNonCommercial) {
    $text = [System.IO.File]::ReadAllText($envPath)
    if ($text -match '(?m)^ACCEPT_NONCOMMERCIAL_MODELS=.*$') {
        $text = [regex]::Replace($text, '(?m)^ACCEPT_NONCOMMERCIAL_MODELS=.*$', 'ACCEPT_NONCOMMERCIAL_MODELS=true')
    } else {
        $text += "`nACCEPT_NONCOMMERCIAL_MODELS=true`n"
    }
    [System.IO.File]::WriteAllText($envPath, $text, (New-Object System.Text.UTF8Encoding $false))
}
Write-Host 'Setup complete. Secrets preserved, never printed. Model use requires explicit non-commercial acceptance.'
