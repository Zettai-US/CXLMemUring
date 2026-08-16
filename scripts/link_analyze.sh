#!/usr/bin/env bash
set -u
R=/home/victoryang00/CXLMemUring
OUT=$R/build/lowered
mkdir -p "$OUT"
cd "$R"

FILES="$R/llama.mlir $R/llama-chat.mlir $R/llama-graph.mlir $R/llama-grammar.mlir \
       $R/bat_storage.mlir $R/rel_propagate.mlir $R/dataframe_performance.mlir"

/usr/bin/time -v "$R/build/bin/cira-link" --analyze-only \
  --report="$OUT/relationship.txt" $FILES 2>"$OUT/link-analyze.log"
echo "exit=$?"
grep -E "Maximum resident|Elapsed \(wall" "$OUT/link-analyze.log"
cat "$OUT/relationship.txt"
