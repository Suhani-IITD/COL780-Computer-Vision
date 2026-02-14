#!/usr/bin/env bash
set -euo pipefail

INSTALL_MODE="false"
if [[ "${1:-}" == "--install" ]]; then
  INSTALL_MODE="true"
elif [[ "${1:-}" != "" ]]; then
  echo "Usage: $0 [--install]"
  exit 2
fi

missing_packages=()

check_cmd() {
  local cmd="$1"
  local pkg="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing_packages+=("$pkg")
    return 1
  fi
  return 0
}

echo "[1/4] Checking base build tools..."
check_cmd g++ g++ || true
check_cmd make make || true
check_cmd pkg-config pkg-config || true

echo "[2/4] Checking OpenCV C++ (pkg-config opencv4)..."
if ! pkg-config --exists opencv4 2>/dev/null; then
  missing_packages+=("libopencv-dev")
fi

echo "[3/4] Checking Python tools for calibration..."
check_cmd python3 python3 || true
if command -v python3 >/dev/null 2>&1; then
  if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
PY
  then
    missing_packages+=("python3-numpy")
  fi

  if ! python3 - <<'PY' >/dev/null 2>&1
import cv2
PY
  then
    missing_packages+=("python3-opencv")
  fi
fi

echo "[4/4] Summary"
if [[ ${#missing_packages[@]} -eq 0 ]]; then
  echo "All required dependencies are present."
  echo "You can build with: make clean && make"
  exit 0
fi

# Deduplicate package list
readarray -t uniq_missing < <(printf '%s\n' "${missing_packages[@]}" | awk '!seen[$0]++')

echo "Missing packages: ${uniq_missing[*]}"

if [[ "$INSTALL_MODE" != "true" ]]; then
  echo "Run with --install to install missing packages (Ubuntu/Debian)."
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "Error: --install currently supports apt-get based systems only."
  exit 1
fi

echo "Installing missing packages..."
sudo apt-get update
sudo apt-get install -y "${uniq_missing[@]}"

echo "Re-checking after install..."
exec "$0"
