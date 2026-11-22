#!/usr/bin/env python3
"""unseen_v9_summary.py - DRAM traffic of every rule vs LRU on the 4 UNSEEN blocks.
Reads datasets/ai_kernels_unseen_v9_lru.csv + ai_kernels_unseen_v9_rules.csv,
writes NEW file datasets/unseen_v9_summary.csv (refuses to overwrite)."""
from pathlib import Path
import pandas as pd
S = Path(__file__).resolve().parent.parent
out = S / "datasets" / "unseen_v9_summary.csv"
assert not out.exists(), f"{out} exists"
d = pd.concat([pd.read_csv(S / "datasets" / f) for f in ("ai_kernels_unseen_v9_lru.csv", "ai_kernels_unseen_v9_rules.csv")])
d = d[d.status == "SUCCESS"].copy()
d["rule"] = d.run_id.str.split("_").str[1]
d["dram_MB"] = (d.dram_bytes_read + d.dram_bytes_written) / 1e6
W = d.pivot(index="layer_id", columns="rule", values="dram_MB")
R = W.div(W["lru"], axis=0).round(4)
order = [c for c in ["lru", "fifo", "random", "treeplru", "srrip", "brrip", "drrip", "ship", "shippc", "v9"] if c in R]
R = R[order]
o = R.drop(columns=["lru", "v9"], errors="ignore")
R["best_other"] = o.idxmin(axis=1); R["best_other_vs_lru"] = o.min(axis=1)
R.to_csv(out)
print(R.to_string()); print("written:", out)
