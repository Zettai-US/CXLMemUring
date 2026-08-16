#!/usr/bin/env bash
set -u
R=/home/victoryang00/CXLMemUring
OUT=$R/build/lowered
mkdir -p "$OUT"

FILES="$R/llama.mlir $R/llama-graph.mlir $R/llama-grammar.mlir \
       $R/bat_storage.mlir $R/rel_propagate.mlir $R/dataframe_performance.mlir"

echo "### linking"
/usr/bin/time -f "  merge: %e s, %M KB peak" "$R/build/bin/cira-link" \
  --report="$OUT/relationship.txt" -o "$OUT/all.linked.mlir" $FILES \
  2>&1 | grep -vE "^\[cira-link\] parsing" | tail -5
ls -la "$OUT/all.linked.mlir"

echo "### region formation on the linked module"
/usr/bin/time -f "  regions: %e s, %M KB peak" "$R/build/bin/cira" \
  "$OUT/all.linked.mlir" \
  --pass-pipeline='builtin.module(cira-region-formation{require-profitable=false})' \
  -o "$OUT/all.regions.mlir" 2>&1 | tail -5

echo "### region stats"
grep -c 'cira\.offload' "$OUT/all.regions.mlir"
