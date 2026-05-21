param(
    [string]$ExePath = ".\dist\unv-modal-viewer\unv-modal-viewer.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) {
    throw "Executable not found: $ExePath"
}

Write-Host "Frozen GUI executable exists: $((Resolve-Path $ExePath).Path)"
Write-Host "Manual smoke test:"
Write-Host "  & `"$ExePath`" path\to\modal_test_file.unv"
