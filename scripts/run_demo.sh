#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-Tag0.mp4}"
TEMPLATE="${2:-}"
INTRINSICS="${3:-camera_intrinsics.yml}"
OBJ="${4:-}"

make all

CMD=("./ar_tag_detector" "$SOURCE")

need_placeholder="false"
if [[ -n "$TEMPLATE" || -n "$OBJ" || -f "$INTRINSICS" ]]; then
  need_placeholder="true"
fi

if [[ "$need_placeholder" == "true" ]]; then
  CMD+=("${TEMPLATE:--}")

  if [[ -n "$OBJ" ]]; then
    if [[ -f "$INTRINSICS" ]]; then
      CMD+=("$INTRINSICS" "$OBJ")
    else
      CMD+=("$OBJ")
    fi
  elif [[ -f "$INTRINSICS" ]]; then
    CMD+=("$INTRINSICS")
  fi
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
