$ErrorActionPreference = "Stop"

param(
    [string]$SingleTagSource = "Tag0.mp4",
    [string]$MultiTagSource = "multipleTags.mp4",
    [switch]$InstallDeps,
    [switch]$GenerateDebug,
    [string]$DebugRoot = "debug_out_pre_ship"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$singleDebugDir = Join-Path $DebugRoot "single_tag"
$multiDebugDir = Join-Path $DebugRoot "multi_tag"

Set-Location $repoRoot

Write-Host "Pre-ship checklist (Windows): starting..." -ForegroundColor Cyan

if ($InstallDeps) {
    Write-Host "1/5 Installing/verifying dependencies..." -ForegroundColor Cyan
    & "$PSScriptRoot\setup_windows_deps.ps1" -Install
} else {
    Write-Host "1/5 Verifying dependencies..." -ForegroundColor Cyan
    & "$PSScriptRoot\setup_windows_deps.ps1"
}

Write-Host "2/5 Building project..." -ForegroundColor Cyan
& "$PSScriptRoot\build_windows.ps1"

Write-Host "3/5 Running single-tag Task 1 check on $SingleTagSource..." -ForegroundColor Cyan
& "$PSScriptRoot\test_windows.ps1" -Source $SingleTagSource -Mode task1

Write-Host "4/5 Running multi-tag Task 1 check on $MultiTagSource..." -ForegroundColor Cyan
& "$PSScriptRoot\test_windows.ps1" -Source $MultiTagSource -Mode task1

if ($GenerateDebug) {
    Write-Host "5/5 Generating debug frames in $DebugRoot..." -ForegroundColor Cyan
    if (-not (Test-Path $DebugRoot)) {
        New-Item -Path $DebugRoot -ItemType Directory | Out-Null
    }

    & "$PSScriptRoot\test_windows.ps1" -Source $SingleTagSource -Mode task1 -DebugDir $singleDebugDir
    & "$PSScriptRoot\test_windows.ps1" -Source $MultiTagSource -Mode task1 -DebugDir $multiDebugDir
} else {
    Write-Host "5/5 Skipping debug frame generation (pass -GenerateDebug to enable)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Pre-ship checklist complete." -ForegroundColor Green
Write-Host "Manual acceptance (must pass before shipping):" -ForegroundColor Yellow
Write-Host "- Tag0.mp4: single tag labeled with stable ID + rotation in most frames."
Write-Host "- multipleTags.mp4: all 3 tags labeled when visible in the same frame."
Write-Host "- Quad/corners aligned to printed tag boundary."

