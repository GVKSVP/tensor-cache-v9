#!/usr/bin/env python3
"""
make_tensor_plan.py - Step 5: plan for the learned tensor-aware policy
(gem5 TensorAwareRP, --repl tensor) on the same rows as the LRU plans, with
each row's tensor hint table (trace_tools/make_hints.py) passed in sim_flags.

run_id    = <prefix><index>
config_id = <source config>_repl-tensor-<tag>   (tag names the model, e.g. v3t30)
Output must not exist yet (plans are never overwritten).
Usage: make_tensor_plan.py --out PLAN.csv --tag v3t30 --hints DIR
                           [--layers A,B] [--sources P1,P2] [--prefix TEN]
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hints", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES + ["experiment_plan_chain_v1.csv"]))
    ap.add_argument("--prefix", default="TEN")
    ap.add_argument("--slice", default="0", help="--tensor-slice beta for L3 (0 = off)")
    ap.add_argument("--slice-l2", default="0", help="--tensor-slice-l2 beta for L2 (0 = off)")
    ap.add_argument("--l2-tree", default="main", choices=["main", "demand"])
    ap.add_argument("--mode", default="tensor", choices=["tensor", "tensor_l3"],
                    help="tensor = L2+L3, tensor_l3 = L3 only (L2 LRU)")
    opt = ap.parse_args()
    out = Path(opt.out)
    if not out.is_absolute():
        out = KDIR / out
    if out.exists():
        sys.exit(f"ERROR: {out} already exists (plans are never overwritten)")
    hints = Path(opt.hints).resolve()
    rows, fields = [], None
    for src in opt.sources.split(","):
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
        h = hints / f"{r['layer_id']}.txt"
        if not h.exists():
            sys.exit(f"ERROR: no hint table {h}")
        n = dict(r)
        n["run_id"] = f"{opt.prefix}{i:03d}"
        n["config_id"] = f"{r['config_id']}_repl-{opt.mode}-{opt.tag}"
        n["sim_flags"] = f"{r['sim_flags']} --repl {opt.mode} --tensor-hints {h}" + \
            (f" --tensor-slice {opt.slice}" if float(opt.slice) > 0 else "") + \
            (f" --tensor-slice-l2 {opt.slice_l2}" if float(opt.slice_l2) > 0 else "") + \
            (f" --tensor-l2-tree {opt.l2_tree}" if opt.l2_tree != "main" else "")
        lv = "L2+L3" if opt.mode == "tensor" else "L3 (L2 LRU)"
        n["notes"] = f"{lv} replacement=tensor-aware ({opt.tag}; L1 LRU); {r['notes']}".strip("; ")
        plan.append(n)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(plan)
    print(f"wrote {out}: {len(plan)} runs")


if __name__ == "__main__":
    main()
