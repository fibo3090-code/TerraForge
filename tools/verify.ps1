# TerraForge verification: full test suite + dual-viewpoint GPU captures.
# Run after ANY visually-affecting change, then compare the two PNGs against
# the previous pair from the same viewpoints before claiming success.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

cargo nextest run
if ($LASTEXITCODE -ne 0) { Write-Host "TESTS FAILED" -ForegroundColor Red; exit 1 }

cargo build --release
if ($LASTEXITCODE -ne 0) { exit 1 }

$env:BEVY_ASSET_ROOT = (Get-Location).Path

$env:TERRAFORGE_CAPTURE = "test_output/verify_close.png"
Remove-Item Env:TERRAFORGE_CAPTURE_VIEW -ErrorAction SilentlyContinue
& ./target/release/terraforge.exe

$env:TERRAFORGE_CAPTURE = "test_output/verify_wide.png"
$env:TERRAFORGE_CAPTURE_VIEW = "wide"
& ./target/release/terraforge.exe

Remove-Item Env:TERRAFORGE_CAPTURE, Env:TERRAFORGE_CAPTURE_VIEW -ErrorAction SilentlyContinue
Write-Host "OK - captures: test_output/verify_close.png, test_output/verify_wide.png" -ForegroundColor Green
