$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$bashExe = "C:\msys64\usr\bin\bash.exe"

if (-not (Test-Path $bashExe)) {
    throw "MSYS2 bash not found at $bashExe. Run scripts/setup_windows_deps.ps1 first."
}

$buildCmd = "set -e; cd '$repoRoot'; export PATH=/ucrt64/bin:`$PATH; make clean && make"
& $bashExe -lc $buildCmd

Write-Host "Build complete: ar_tag_detector" -ForegroundColor Green
