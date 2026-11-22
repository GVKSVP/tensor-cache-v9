#!/usr/bin/env python3
"""summarize_traces.py - one row per traced layer: tensor coverage, offline LRU vs
gem5 (model check), and Belady OPT vs LRU misses at L2 and L3, overall and per
tensor role. Writes datasets/<tag>_summary.csv (new file) and prints a digest."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

tag = sys.argv[1] if len(sys.argv) > 1 else "trace_v1"
root = Path(__file__).resolve().parent.parent
rows, role_tot = [], defaultdict(lambda: defaultdict(int))
for r in csv.DictReader(open(root / f"datasets/ai_kernels_{tag}.csv")):
    if r["status"] != "SUCCESS":
        continue
    d = root / f"results/{tag}/{r['run_id']}_{r['layer_id']}"
    roi = json.loads((d / "roi_summary.json").read_text())
    stats = {}
    for line in open(d / "stats.txt"):
        if "End Simulation" in line:
            break
        p = line.split()
        if len(p) > 1 and p[0] in ("system.l2.overallMshrMisses::total", "system.l3.overallMshrMisses::total"):
            stats[p[0].split(".")[1]] = int(p[1])
    row = {"layer_id": r["layer_id"], "model": r["model"], "kernel": r["kernel"],
           "l2_coverage_pct": roi["l2"]["tensor_coverage_pct"]}
    for lvl in ("l2", "l3"):
        b = json.loads((d / f"{lvl}_belady_summary.json").read_text())
        row[f"{lvl}_accesses"] = b["accesses"]
        row[f"{lvl}_lru_misses"] = b["lru_misses"]
        row[f"{lvl}_opt_misses"] = b["opt_misses"]
        row[f"{lvl}_gem5_mshr_misses"] = stats[lvl]
        row[f"{lvl}_opt_saving_pct"] = round(100 * (1 - b["opt_misses"] / max(b["lru_misses"], 1)), 2)
        for role, v in b["per_role"].items():
            role_tot[(lvl, role)]["lru"] += v["lru_misses"]
            role_tot[(lvl, role)]["opt"] += v["opt_misses"]
    rows.append(row)
out = root / f"datasets/{tag}_summary.csv"
if out.exists():
    sys.exit(f"{out} exists; not overwritten")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"wrote {out} ({len(rows)} layers)")
for lvl in ("l2", "l3"):
    lru = sum(r[f"{lvl}_lru_misses"] for r in rows); opt = sum(r[f"{lvl}_opt_misses"] for r in rows)
    g5 = sum(r[f"{lvl}_gem5_mshr_misses"] for r in rows)
    print(f"{lvl}: offline LRU {lru:,} vs gem5 {g5:,} ({100*(lru/g5-1):+.1f}%) | OPT {opt:,} ({100*(1-opt/lru):.1f}% fewer than LRU)")
    print("   per role (LRU -> OPT misses):", {k[1]: f"{v['lru']:,}->{v['opt']:,}" for k, v in sorted(role_tot.items()) if k[0] == lvl and v['lru'] > 1000})
print("coverage min %.2f%%" % min(r["l2_coverage_pct"] for r in rows))
print("layers with OPT saving >5%  L2:", sum(r["l2_opt_saving_pct"] > 5 for r in rows), " L3:", sum(r["l3_opt_saving_pct"] > 5 for r in rows))
print("top L3 opportunities:")
for r in sorted(rows, key=lambda r: -(r["l3_lru_misses"] - r["l3_opt_misses"]))[:10]:
    print(f"  {r['layer_id']:<26} L3 {r['l3_lru_misses']:>9,} -> {r['l3_opt_misses']:>9,} ({r['l3_opt_saving_pct']:5.1f}%)   L2 {r['l2_opt_saving_pct']:5.1f}%")
