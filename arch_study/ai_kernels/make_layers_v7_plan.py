#!/usr/bin/env python3
"""make_layers_v7_plan.py - single-layer check of the final rule (v7):
the 65 single layers of experiment_plan_repl_v1.csv (BRRIP rows) as
LRU, DRRIP and ours (v7 + hints from results/tensor_hints_layers), all with
the L2 prefetcher notify setting. Writes 3 new plan files (never overwrites)."""
import csv, sys
from pathlib import Path
K = Path(__file__).resolve().parent
H = K.parent / "results" / "tensor_hints_layers"
V7 = ("--tensor-slice 0 --tensor-slice-ws 0.9 --tensor-duel-skip-first 1 --tensor-duel-lru-default 1 "
      "--tensor-slice-l2 0 --tensor-l2-pf-fix 1 --tensor-l2-no-duel 1 --tensor-l2-tree-off 1 "
      "--tensor-l2-slice-ws 0.9 --tensor-l2-keep-pf 0 --tensor-l2-slice-contig 1 --tensor-l2-contig-min 0.35 "
      "--tensor-l3-pack 1 --tensor-coord 1 --l2-pf-on-access 1 --tensor-l2-pack 2 --tensor-dead-demote 1")
src = [x for x in csv.DictReader(open(K / "experiment_plan_repl_v1.csv")) if x["config_id"].endswith("_repl-brrip")]
fields = list(src[0].keys())
for tag, pre, rule, extra in [("lru", "LYL", "lru", "--l2-pf-on-access 1"),
                              ("drrip", "LYD", "drrip", "--l2-pf-on-access 1"),
                              ("ours", "LYO", "tensor", V7)]:
    out = K / f"experiment_plan_layers_v7_{tag}.csv"
    if out.exists():
        sys.exit(f"{out} exists")
    rows = []
    for i, x in enumerate(src, 1):
        r = dict(x)
        assert r["sim_flags"].endswith("--repl brrip")
        r["sim_flags"] = r["sim_flags"][:-len("brrip")] + rule + " " + extra
        if rule == "tensor":
            r["sim_flags"] += f" --tensor-hints {H / (r['layer_id'] + '.txt')}"
        r["run_id"] = f"{pre}{i:03d}"
        r["config_id"] = r["config_id"].replace("_repl-brrip", f"_repl-{rule}_flush")
        rows.append(r)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    print("wrote", out.name, len(rows))
