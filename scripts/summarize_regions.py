#!/usr/bin/env python3
"""Summarize the cira.offload regions formed in a (linked) MLIR module."""
import re
import sys
from collections import Counter, defaultdict

MIN_CHAIN_DEPTH = 4  # OffloadCostModelParams::MIN_CHAIN_DEPTH

path = sys.argv[1]
pat = re.compile(
    r'cira\.offload.*?cira\.chain_depth = (\d+).*?cira\.slice_size = (\d+)'
    r'(?:.*?cira\.source = "([^"]*)")?'
)

per_file = defaultdict(lambda: {"n": 0, "depths": Counter(), "slice": 0, "profitable": 0})
total = 0
with open(path, errors="replace") as fh:
    for line in fh:
        if "cira.offload" not in line:
            continue
        m = pat.search(line)
        if not m:
            continue
        depth, size, src = int(m.group(1)), int(m.group(2)), m.group(3) or "<unknown>"
        e = per_file[src.split("/")[-1]]
        e["n"] += 1
        e["depths"][depth] += 1
        e["slice"] += size
        e["profitable"] += depth >= MIN_CHAIN_DEPTH
        total += 1

print(f"{'source translation unit':<26}{'regions':>9}{'profitable':>12}{'avg depth':>11}{'max depth':>11}{'avg slice':>11}")
for src, e in sorted(per_file.items(), key=lambda kv: -kv[1]["n"]):
    depths = e["depths"]
    avg_d = sum(d * c for d, c in depths.items()) / e["n"]
    print(f"{src:<26}{e['n']:>9}{e['profitable']:>12}{avg_d:>11.2f}{max(depths):>11}{e['slice'] / e['n']:>11.1f}")

alld = Counter()
for e in per_file.values():
    alld.update(e["depths"])
print(f"\ntotal regions: {total}, "
      f"profitable (depth >= {MIN_CHAIN_DEPTH}): {sum(c for d, c in alld.items() if d >= MIN_CHAIN_DEPTH)}")
print("\ndependence-chain depth histogram")
for d in sorted(alld):
    print(f"  depth {d:>3}: {alld[d]:>7}  {'#' * min(60, alld[d] * 60 // max(alld.values()))}")
