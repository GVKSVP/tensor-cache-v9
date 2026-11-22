#!/usr/bin/env python3
"""
make_trace_plan.py - plan for Step 3 trace runs: the 65 LRU layers on config F
(same rows as make_repl_plan.py uses) with "--trace l2l3" added, so gem5
records every request entering L2 and L3. All other settings are unchanged,
so each trace run must reproduce its Step-2 LRU result exactly.

Usage: make_trace_plan.py --out PLAN.csv [--layers A,B]
"""

import argparse
import csv
import sys
from pathlib import Path

from make_repl_plan import DEFAULT_SOURCES, SUPERSEDED

KDIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    opt = ap.parse_args()
    out = Path(opt.out)
    if not out.is_absolute():
        out = KDIR / out
    if out.exists():
        sys.exit(f"ERROR: {out} already exists (plans are never overwritten)")

    rows, fields = [], None
    for src in DEFAULT_SOURCES:
        with open(KDIR / src, newline="") as f:
            reader = csv.DictReader(f)
            fields = fields or reader.fieldnames
            rows += [r for r in reader if r["layer_id"] not in SUPERSEDED]
    want = [x for x in opt.layers.split(",") if x]
    if want:
        missing = set(want) - {r["layer_id"] for r in rows}
        if missing:
            sys.exit(f"ERROR: layers not found: {sorted(missing)}")
        rows = [r for r in rows if r["layer_id"] in want]

    plan = []
    for i, r in enumerate(rows, 1):
        n = dict(r)
        n["run_id"] = f"TRC{i:03d}"
        n["config_id"] = f"{r['config_id']}_trace"
        n["sim_flags"] = f"{r['sim_flags']} --trace l2l3"
        n["notes"] = f"L2+L3 access trace (LRU); {r['notes']}".strip("; ")
        plan.append(n)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(plan)
    print(f"wrote {out}: {len(plan)} trace runs")


if __name__ == "__main__":
    main()
