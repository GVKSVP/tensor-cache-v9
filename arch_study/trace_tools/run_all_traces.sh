#!/bin/bash
# Step 3d: trace all layers of a trace plan, then label them.
#   phase 1 (in parallel): O3 trace runs (4 workers) | page tables (1) | tensor maps (1)
#   phase 2: extract ROI tables + Belady/LRU labels for every successful run
# Usage: run_all_traces.sh PLAN TAG     (outputs results/<TAG>, results/<TAG>_maps,
#                                         datasets/ai_kernels_<TAG>.csv, results/<TAG>.log)
set -u
cd "$(dirname "$0")/.."
PY=~/gem5-project/.venv/bin/python
PLAN=$1; TAG=$2
$PY -u run_ai_kernels.py --plan "$PLAN" --out datasets/ai_kernels_$TAG.csv \
    --results results/$TAG --workers 4 > results/$TAG.log 2>&1 &
$PY -u trace_tools/run_ptable.py --plan "$PLAN" --results results/${TAG}_maps --workers 1 \
    > results/${TAG}_ptable.log 2>&1 &
$PY -u trace_tools/run_map.py --plan "$PLAN" --results results/${TAG}_maps --workers 1 \
    > results/${TAG}_map.log 2>&1 &
wait
echo "PHASE1 DONE $(date -u)" >> results/$TAG.log
$PY - "$TAG" <<'PYEOF' > results/${TAG}_label.log 2>&1
import csv, subprocess, sys
tag = sys.argv[1]
py = sys.executable
for r in csv.DictReader(open(f"datasets/ai_kernels_{tag}.csv")):
    if r["status"] != "SUCCESS":
        continue
    d = f"results/{tag}/{r['run_id']}_{r['layer_id']}"
    for cmd in ([py, "trace_tools/extract_roi.py", d, "--maps", f"results/{tag}_maps"],
                [py, "trace_tools/belady.py", d, "--l2-prefetch-fills"]):
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout.strip(), p.stderr.strip()[-300:], flush=True)
print("LABELS DONE")
PYEOF
echo "ALL DONE $(date -u)" >> results/$TAG.log
