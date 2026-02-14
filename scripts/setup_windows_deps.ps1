param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"

function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Ok($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }

$repoRoot = Split-Path -Parent $PSScriptRoot
$msysRoot = "C:\msys64"
$bashExe = Join-Path $msysRoot "usr\bin\bash.exe"

Info "Checking Windows prerequisites for native build (MSYS2 + MinGW UCRT64)..."

if (-not (Test-Path $bashExe)) {
    Warn "MSYS2 not found at $bashExe"
    if ($Install) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Info "Installing MSYS2 via winget..."
            winget install -e --id MSYS2.MSYS2 --accept-package-agreements --accept-source-agreements
        } else {
            throw "winget not found. Install MSYS2 manually from https://www.msys2.org/"
        }
    } else {
        Write-Host "Install MSYS2 manually: https://www.msys2.org/"
        Write-Host "Then re-run: .\\scripts\\setup_windows_deps.ps1 --Install"
        exit 1
    }
}

if (-not (Test-Path $bashExe)) {
    throw "MSYS2 still not found after install attempt."
}

$installCmd = @'
set -e
pacman -Sy --noconfirm --needed \
  make \
  mingw-w64-ucrt-x86_64-toolchain \
  mingw-w64-ucrt-x86_64-opencv \
  mingw-w64-ucrt-x86_64-pkgconf \
  mingw-w64-ucrt-x86_64-python \
  mingw-w64-ucrt-x86_64-python-numpy \
  mingw-w64-ucrt-x86_64-python-opencv
'@

if ($Install) {
    Info "Installing/Updating packages in MSYS2 UCRT64..."
    & $bashExe -lc $installCmd
}

$checkCmd = @'
set -e
export PATH=/ucrt64/bin:$PATH
missing=0
for c in g++ make pkg-config python; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "missing_cmd:$c"
    missing=1
  fi
done
if ! pkg-config --exists opencv4; then
  echo "missing_pkg:opencv4"
  missing=1
fi
python - <<PY || missing=1
import numpy
import cv2
print("python_ok")
PY
if [ "$missing" -ne 0 ]; then
  exit 1
fi
'@

Info "Validating toolchain..."
try {
    & $bashExe -lc $checkCmd
    Ok "Windows dependencies are ready."
    Write-Host "Build with: .\\scripts\\build_windows.ps1"
} catch {
    Warn "Some dependencies are missing."
    Write-Host "Run: .\\scripts\\setup_windows_deps.ps1 --Install"
    exit 1
}
