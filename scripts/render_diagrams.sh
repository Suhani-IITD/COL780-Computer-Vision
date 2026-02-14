#!/usr/bin/env bash
set -euo pipefail

mkdir -p docs/diagrams

for f in docs/diagrams/*.mmd; do
  base="${f%.mmd}"
  echo "Rendering $f -> $base.svg"
  mmdc -i "$f" -o "$base.svg" -b transparent

done

echo "Done. Rendered SVG diagrams in docs/diagrams/."
