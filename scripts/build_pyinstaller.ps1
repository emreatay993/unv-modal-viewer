param(
    [switch]$SkipTests,
    [switch]$NoClean,
    [switch]$Console
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv or run the PyInstaller commands manually from an activated venv."
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    uv venv --python 3.12
}

uv pip install -e ".[test,build]"

if (-not $SkipTests) {
    uv run pytest
}

$Args = @(".\unv_modal_viewer.spec", "--noconfirm")
if (-not $NoClean) {
    $Args += "--clean"
}
if ($Console) {
    Write-Warning "The checked-in spec is windowed. Use the console diagnostic script for console builds."
}

uv run pyinstaller @Args

$Exe = Join-Path $ProjectRoot "dist\unv-modal-viewer\unv-modal-viewer.exe"
if (-not (Test-Path $Exe)) {
    throw "Build finished without the expected executable: $Exe"
}

Write-Host "Built: $Exe"
