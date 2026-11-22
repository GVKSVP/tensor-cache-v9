#!/usr/bin/env python3
"""make_unseen_v9_plan.py - plans for the UNSEEN test with the final v9 setup.
Writes NEW files only (refuses to overwrite):
  experiment_plan_unseen_v9_lru.csv    4 LRU runs (also used for profiling/hints)
  experiment_plan_unseen_v9_rules.csv  36 runs: 8 standard rules + v9 on the 4 blocks
Workload rows come from experiment_plan_unseen_lru.csv; the v9 flags are copied
from the final v9 BLOCK_bert row, only the hint file path changes."""
from pathlib import Path
import pandas as pd

K = Path(__file__).resolve().parent
STUDY = K.parent
HINTS = STUDY / "results" / "tensor_hints_unseen_v9"
BASE = ("--cpu o3 --freq 2GHz --mem ddr5_4400 --dram-page-policy open_adaptive "
        "--l1-lat 4 --l2-lat 16 --l3-lat 40 --prefetch stride")
STD = ["fifo", "random", "treeplru", "srrip", "brrip", "drrip", "ship", "shippc"]

u = pd.read_csv(K / "experiment_plan_unseen_lru.csv")
v9 = pd.read_csv(K / "experiment_plan_v9_full.csv")
v9f = v9.loc[v9.layer_id == "BLOCK_bert", "sim_flags"].iloc[0]
assert v9f.startswith(BASE + " --repl tensor") and "--tensor-hints " in v9f
v9f = v9f.split("--tensor-hints ")[0].strip()
u["notes"] = "UNSEEN block (block_unseen.c); shapes in src/block_unseen.c"

def rows(rule, flags, cfg):
    t = u.copy()
    t["run_id"] = f"UNV_{rule}_" + t.layer_id
    t["sim_flags"] = flags if isinstance(flags, str) else t.layer_id.map(flags)
    t["config_id"] = cfg
    return t

lru = rows("lru", BASE + " --repl lru --l2-pf-on-access 1", "F_P870D_full_L2stride_repl-lru_flush14")
rest = [rows(r, BASE + f" --repl {r} --l2-pf-on-access 1", f"F_P870D_full_L2stride_repl-{r}_flush14") for r in STD]
rest.append(rows("v9", lambda L: f"{v9f} --tensor-hints {HINTS}/{L}.txt", "F_P870D_full_L2stride_repl-tensor_flush_v9"))
for name, df in [("experiment_plan_unseen_v9_lru.csv", lru), ("experiment_plan_unseen_v9_rules.csv", pd.concat(rest))]:
    p = K / name
    if p.exists():
        print("exists, kept:", p); continue
    df.to_csv(p, index=False); print("written:", p, len(df), "runs")
