#!/usr/bin/env bash
set -u
R=/home/victoryang00/CXLMemUring
IN=${1:-$R/llama.mlir}
"$R/build/bin/cira" "$IN" \
  --pass-pipeline='builtin.module(cira-region-formation{require-profitable=false})' \
  -o /dev/null --debug-only=cira-region-formation 2>&1 |
  sed -E 's/ at .*//; s/%[0-9]+/%N/g' | sort | uniq -c | sort -rn | head -20
