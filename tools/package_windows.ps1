# Run on Windows after PyInstaller. All checks exercise the shipped executable.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$exe = Join-Path $root "dist/Ayah Studio/Ayah Studio.exe"
$check = Start-Process -FilePath $exe -ArgumentList "--self-check" -PassThru
if (-not $check.WaitForExit(120000)) {
    $check.Kill()
    throw "Frozen runtime self-check timed out"
}
if ($check.ExitCode -ne 0) { throw "Frozen runtime self-check failed: $($check.ExitCode)" }

$app = Start-Process -FilePath $exe -PassThru
try {
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        $app.Refresh()
        if ($app.HasExited) { throw "GUI exited before opening: $($app.ExitCode)" }
        if ($app.MainWindowHandle -ne 0 -and $app.MainWindowTitle -eq "Ayah Studio" -and $app.Responding) {
            $ready = $true
            break
        }
    }
    if (-not $ready) { throw "Native Ayah Studio window did not open" }
    Write-Host "Frozen runtime and native window checks passed"
} finally {
    if (-not $app.HasExited) { Stop-Process -Id $app.Id -Force }
}

$version = uv run --locked --extra build python -c "import importlib.metadata; print(importlib.metadata.version('ayah-studio'))"
if ($LASTEXITCODE -ne 0) { throw "Cannot determine version" }
$name = "Ayah-Studio-$version-Windows-x64.zip"
$zip = Join-Path $root "dist/$name"
Compress-Archive -Path "dist/Ayah Studio" -DestinationPath $zip -Force
$hash = (Get-FileHash -Path $zip -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText("$zip.sha256", "$hash  $name`n", [System.Text.UTF8Encoding]::new($false))
Write-Host "Ready: $zip"
