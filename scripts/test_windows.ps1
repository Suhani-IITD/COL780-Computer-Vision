param(
    [string]$Source = "Tag0.mp4",
    [string]$Template = "",
    [string]$Intrinsics = "camera_intrinsics.yml",
    [string]$Obj = "",
    [ValidateSet("task1", "task2", "task3", "all")]
    [string]$Mode = "task1",
    [string]$DebugDir = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$bashExe = "C:\msys64\usr\bin\bash.exe"
if (-not (Test-Path $bashExe)) {
    throw "MSYS2 bash not found at $bashExe. Run scripts/setup_windows_deps.ps1 first."
}

$binaryPath = Join-Path $repoRoot "ar_tag_detector"
if (-not (Test-Path $binaryPath)) {
    Write-Host "Binary not found. Building first..." -ForegroundColor Yellow
    & "$PSScriptRoot\build_windows.ps1"
}

if (-not (Test-Path (Join-Path $repoRoot $Source)) -and $Source -ne "0") {
    throw "Source not found: $Source"
}

$argsList = @($Source)

if ($Template -ne "" -or $Obj -ne "" -or (Test-Path (Join-Path $repoRoot $Intrinsics))) {
    if ($Template -ne "") { $argsList += $Template } else { $argsList += "-" }

    if ($Obj -ne "") {
        if (Test-Path (Join-Path $repoRoot $Intrinsics)) {
            $argsList += $Intrinsics
            $argsList += $Obj
        } else {
            $argsList += $Obj
        }
    } elseif (Test-Path (Join-Path $repoRoot $Intrinsics)) {
        $argsList += $Intrinsics
    }
}

$argString = ($argsList | ForEach-Object { "'$_'" }) -join " "
if ($DebugDir -ne "") {
    $runCmd = "set -e; cd '$repoRoot'; export PATH=/ucrt64/bin:`$PATH; ./ar_tag_detector $argString --mode $Mode --debug-dir '$DebugDir'"
    Write-Host "Running: ./ar_tag_detector $($argsList -join ' ') --mode $Mode --debug-dir $DebugDir" -ForegroundColor Cyan
} else {
    $runCmd = "set -e; cd '$repoRoot'; export PATH=/ucrt64/bin:`$PATH; ./ar_tag_detector $argString --mode $Mode"
    Write-Host "Running: ./ar_tag_detector $($argsList -join ' ') --mode $Mode" -ForegroundColor Cyan
}

& $bashExe -lc $runCmd
