#!/usr/bin/env python3
"""make_flush_plan.py - Step 1 fair re-test plan: LRU, BRRIP and ours (tensor_l3)
on chosen layers, using the same rows as the existing plans (repl_v1 for BRRIP,
tensorl3_v1 for ours; LRU = the BRRIP row with --repl lru). Run with
run_ai_kernels_flush.py (true cache flush at ROI begin). Output must not exist."""
import argparse, csv, sys
from pathlib import Path
K = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--layers", required=True, help="comma-separated layer_ids")
ap.add_argument("--prefix", default="FLU")
o = ap.parse_args()
out = K / o.out
if out.exists():
    sys.exit(f"ERROR: {out} exists (plans are never overwritten)")
want = o.layers.split(",")
def rows(f, pred):
    with open(K / f, newline="") as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, {x["layer_id"]: x for x in r if pred(x)}
fields, br = rows("experiment_plan_repl_v1.csv", lambda x: x["config_id"].endswith("_repl-brrip"))
_, br2 = rows("experiment_plan_repl_chain_v1.csv", lambda x: x["config_id"].endswith("_repl-brrip"))
br.update(br2)
_, ours = rows("experiment_plan_tensorl3_v1.csv", lambda x: True)
plan, n = [], 0
for L in want:
    b = br[L]
    assert b["sim_flags"].endswith("--repl brrip"), b["sim_flags"]
    lru = dict(b); lru["sim_flags"] = b["sim_flags"][:-len("brrip")] + "lru"
    lru["config_id"] = b["config_id"].replace("_repl-brrip", "_repl-lru")
    for r in (lru, dict(b), dict(ours[L])):
        n += 1
        r["run_id"] = f"{o.prefix}{n:03d}"
        r["config_id"] += "_flush"
        r["notes"] = (r.get("notes", "") + "; Step1 fair re-test: true cache flush at ROI begin").lstrip("; ")
        plan.append({k: r.get(k, "") for k in fields})
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(plan)
print(f"wrote {out} ({len(plan)} runs)")
