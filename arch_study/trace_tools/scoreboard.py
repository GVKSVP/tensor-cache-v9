#!/usr/bin/env python3
"""
scoreboard.py - final comparison of replacement rules on config F (gem5).
DRAM traffic (bytes read + written in the ROI) of every rule divided by LRU,
per layer / chain, plus geomean, total, worst case and number of cases more
than 2% worse than LRU. Also runtime (sim_ticks) ratios.

LRU     : datasets/ai_kernels_trace_v1.csv + ai_kernels_chain_trace_v1.csv
          (tracing does not change timing, checked in Step 3d)
9 rules : datasets/ai_kernels_repl_v1.csv + ai_kernels_repl_chain_v1.csv
ours    : any dataset given with --ours NAME=CSV (e.g. tensor_l3=...tensorl3_v1.csv)
Rows marked "debug" were used while debugging the design (CHAIN_vit_mlp,
SDXL_ffn_1280): reported, but also summarised without them.
Usage: scoreboard.py --ours tensor_l3=datasets/ai_kernels_tensorl3_v1.csv [--out CSV]
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEBUG = {"CHAIN_vit_mlp", "SDXL_ffn_1280"}
TEST_PREFIX = ("SDXL_", "VIT_", "MBV2_", "LLAMA70B_", "CHAIN_vit_mlp", "CHAIN_mbv2_block")


def rows(path):
    with open(ROOT / path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "SUCCESS"]


def traffic(r):
    return float(r["dram_bytes_read"] or 0) + float(r["dram_bytes_written"] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", action="append", default=[], help="NAME=CSV")
    ap.add_argument("--out", default="")
    opt = ap.parse_args()
    base = {r["layer_id"]: r for f in ("datasets/ai_kernels_trace_v1.csv",
                                       "datasets/ai_kernels_chain_trace_v1.csv") for r in rows(f)}
    res = defaultdict(dict)                       # layer -> rule -> row
    for f in ("datasets/ai_kernels_repl_v1.csv", "datasets/ai_kernels_repl_chain_v1.csv"):
        for r in rows(f):
            res[r["layer_id"]][r["config_id"].split("_repl-")[1]] = r
    names = ["fifo", "random", "treeplru", "srrip", "brrip", "drrip", "ship", "shippc"]
    for spec in opt.ours:
        name, path = spec.split("=", 1)
        for r in rows(path):
            res[r["layer_id"]][name] = r
        names.append(name)
    layers = sorted(L for L in base if L in res)
    table = []
    for L in layers:
        b = base[L]
        row = {"layer_id": L, "lru_dram_MB": round(traffic(b) / 1e6, 2),
               "set": "test" if L.startswith(TEST_PREFIX) else "train",
               "debug": int(L in DEBUG)}
        for n in names:
            if n in res[L]:
                row[n] = round(traffic(res[L][n]) / max(traffic(b), 1), 4)
                row[n + "_time"] = round(float(res[L][n]["sim_ticks"]) / float(b["sim_ticks"]), 4)
        table.append(row)
    if opt.out:
        with open(ROOT / opt.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
            w.writeheader()
            w.writerows(table)

    def summary(sel, title):
        print(f"\n{title} ({len(sel)} layers/chains; DRAM traffic vs LRU, lower is better)")
        print(f"{'rule':<12}{'geomean':>9}{'total':>8}{'worst':>8}{'>2% worse':>11}{'best of all':>13}{'time gm':>9}")
        full = [r for r in sel if all(n in r for n in names)]
        for n in names:
            v = [r[n] for r in full]
            tot = sum(r[n] * r["lru_dram_MB"] for r in full) / sum(r["lru_dram_MB"] for r in full)
            best = sum(1 for r in full if r[n] <= min(r[m] for m in names) + 1e-9)
            tg = math.exp(sum(math.log(r[n + "_time"]) for r in full) / len(full))
            print(f"{n:<12}{math.exp(sum(map(math.log, v)) / len(v)):>9.3f}{tot:>8.3f}{max(v):>8.3f}"
                  f"{sum(x > 1.02 for x in v):>11}{best:>13}{tg:>9.3f}")
        if len(full) < len(sel):
            print(f"  ({len(sel) - len(full)} rows missing a rule were left out)")
    summary(table, "ALL")
    summary([r for r in table if r["set"] == "test" and not r["debug"]], "UNSEEN MODELS (not used in debugging)")
    summary([r for r in table if r["set"] == "train"], "TRAINING MODELS")


if __name__ == "__main__":
    main()
