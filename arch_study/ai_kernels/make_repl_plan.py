#!/usr/bin/env python3
"""
make_repl_plan.py - build replacement-policy plans from the existing LRU plans.

Each source plan row (config F, LRU) is copied once per policy with
"--repl <policy>" appended to sim_flags (sim_config.py applies it to L2 and L3
only; L1 stays LRU). Everything else - kernel, args, units, mode, caches - is
unchanged, so every policy run is directly comparable with its LRU row.

The source plans are only read. The output plan must not exist yet.

Usage:
  make_repl_plan.py --policies srrip,brrip --out PLAN.csv [--layers A,B] [--sources P1,P2]
"""

import argparse
import csv
import sys
from pathlib import Path

KDIR = Path(__file__).resolve().parent

# The 65 LRU layers on config F (27 + 19 + 7 + 13, minus the superseded
# 20-head LLAMA8B_attn_dec_4k, replaced by LLAMA8B_attn_dec_4k_full in v3).
DEFAULT_SOURCES = [
    "experiment_plan_ai_p870d_full_pf_27_v1.csv",
    "experiment_plan_ai_v2_p870d_pf_v1.csv",
    "experiment_plan_ai_v3_p870d_pf_v1.csv",
    "experiment_plan_ai_v4_p870d_pf_v1.csv",
]
SUPERSEDED = {"LLAMA8B_attn_dec_4k"}
POLICIES = ("lru", "fifo", "random", "treeplru", "srrip", "brrip", "drrip", "ship", "shippc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="comma list of layer_ids (default: all)")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    opt = ap.parse_args()

    policies = [p for p in opt.policies.split(",") if p]
    bad = [p for p in policies if p not in POLICIES]
    if bad:
        sys.exit(f"ERROR: unknown policies {bad}; allowed: {', '.join(POLICIES)}")
    out = Path(opt.out)
    if not out.is_absolute():
        out = KDIR / out
    if out.exists():
        sys.exit(f"ERROR: {out} already exists (plans are never overwritten)")

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
    ids = [r["layer_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("ERROR: duplicate layer_ids in sources")

    plan = []
    for pol in policies:
        for i, r in enumerate(rows, 1):
            n = dict(r)
            n["run_id"] = f"{pol.upper()}{i:03d}"
            n["config_id"] = f"{r['config_id']}_repl-{pol}"
            n["sim_flags"] = f"{r['sim_flags']} --repl {pol}"
            n["notes"] = f"L2+L3 replacement={pol} (L1 LRU); {r['notes']}".strip("; ")
            plan.append(n)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(plan)
    print(f"wrote {out}: {len(plan)} runs = {len(policies)} policies x {len(rows)} layers")


if __name__ == "__main__":
    main()
