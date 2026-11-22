#!/usr/bin/env python3
"""
belady.py - offline cache replay of an ROI trace with Belady's OPT (the
perfect, future-knowing policy) and with LRU, producing per-access labels
for training a replacement policy.

Input : <run>/l2_roi.parquet or l3_roi.parquet (from extract_roi.py)
Output: <run>/<level>_labels.parquet  with, per access that touches the cache:
          seq, tick, line, set, cmd_name, pc, tensor_idx, tensor, role,
          next_use      (seq distance to the next access of the same line; -1 = never)
          opt_hit       (1 if OPT hits on this access)
          opt_keep      (1 if OPT keeps the line until its next use = Hawkeye-style
                         "cache-friendly" label; 0 if it is evicted first or never reused)
          lru_hit       (1 if LRU hits on this access)
        <run>/<level>_belady_summary.json  (hit/miss totals overall and per role)

Cache model: set-associative, set = line % sets (gem5 default indexing),
allocate on every miss (reads, upgrades, writebacks); CleanEvict packets carry
no data and are skipped. With --l2-prefetch-fills (L2 only) the L2 prefetch
requests seen in the L3 trace (HardPFReq) are merged in as fills, because the
L2 prefetcher fills L2 from inside the cache and is invisible in the L2 trace.
"""

import argparse
import heapq
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GEOMETRY = {"l2": (4 * 2**20, 16), "l3": (16 * 2**20, 16)}  # config F
LINE = 64
SKIP = {"CleanEvict"}
COLS = ["seq", "tick", "line", "cmd_name", "pc", "tensor_idx", "tensor", "role"]


DEMAND_READS = ["ReadSharedReq", "ReadExReq", "ReadCleanReq", "UpgradeReq", "ReadReq"]
FORWARD_WINDOW = 100_000   # ticks (100 ns): an L2 demand miss reaches L3 after the
                           # L2 lookup (8-9 ns here); prefetches have no L2 request


def detect_l2_prefetches(run):
    """L3 reads that are NOT forwards of an L2 demand miss = L2 prefetcher
    requests (gem5 sends them as ReadSharedReq, so the command name does not
    show it). A read is a demand forward if the same line entered L2 as a
    demand read at most FORWARD_WINDOW ticks earlier."""
    import pandas as pd
    l2 = pq.read_table(run / "l2_roi.parquet", columns=["tick", "line", "cmd_name"],
                       read_dictionary=["cmd_name"]).to_pandas()
    d3 = pq.read_table(run / "l3_roi.parquet", columns=["tick", "line", "cmd_name"],
                       read_dictionary=["cmd_name"]).to_pandas()
    rd = d3.cmd_name.isin(DEMAND_READS).to_numpy()
    dem = l2[l2.cmd_name.isin(DEMAND_READS)][["tick", "line"]].rename(columns={"tick": "t2"})
    del l2
    q = pd.DataFrame({"tick": d3.tick.to_numpy(), "line": d3.line.to_numpy(), "i": np.arange(len(d3))})[rd]
    del d3
    m = pd.merge_asof(q.sort_values("tick"), dem.sort_values("t2"), left_on="tick", right_on="t2",
                      by="line", direction="backward")
    is_pf = (m.t2.isna() | ((m.tick - m.t2) > FORWARD_WINDOW)).to_numpy()
    pf_idx = np.sort(m.i.to_numpy()[is_pf])
    del m, q, dem
    l3 = pq.read_table(run / "l3_roi.parquet", columns=COLS)
    return l3.take(pa.array(pf_idx)), int(rd.sum() - is_pf.sum()), int(is_pf.sum())


def load(run, level, pf_fills, pf_detect=False):
    t = pq.read_table(run / f"{level}_roi.parquet", columns=COLS)
    t = t.filter(pa.compute.invert(pa.compute.is_in(t["cmd_name"], pa.array(sorted(SKIP)))))
    if level == "l2" and pf_detect:
        pf, n_dem, n_pf = detect_l2_prefetches(run)
        pf = pf.set_column(pf.schema.get_field_index("cmd_name"), "cmd_name",
                           pa.array(["HardPFReq"] * len(pf), pa.string()))
        t = pa.concat_tables([t, pf])
        is_pf = np.concatenate([np.zeros(len(t) - len(pf), bool), np.ones(len(pf), bool)])
        order = np.lexsort((is_pf, t["tick"].to_numpy()))
        t = t.take(pa.array(order))
        load.counts = {"l3_demand_forwards": n_dem, "l2_prefetch_fills": n_pf}
        return t
    if level == "l2" and pf_fills:
        l3 = pq.read_table(run / "l3_roi.parquet", columns=COLS)
        pf = l3.filter(pa.compute.equal(l3["cmd_name"], "HardPFReq"))
        t = pa.concat_tables([t, pf])
        # order by tick; demand packets first on equal ticks
        is_pf = np.concatenate([np.zeros(len(t) - len(pf), bool), np.ones(len(pf), bool)])
        order = np.lexsort((is_pf, t["tick"].to_numpy()))
        t = t.take(pa.array(order))
    return t


def next_use(lines):
    n = len(lines)
    order = np.lexsort((np.arange(n), lines))
    nxt = np.full(n, -1, dtype=np.int64)
    same = lines[order[1:]] == lines[order[:-1]]
    nxt[order[:-1][same]] = order[1:][same]
    return nxt


def demand_next_use(lines, is_pf):
    """Demand-MIN future (prefetch-aware Belady, Jain & Lin): the next use of
    a line that matters is its next DEMAND access; if a prefetch of the line
    comes first, the prefetcher brings it back anyway, so evicting it is free
    (next use = never). Returns index of that demand access or -1."""
    n = len(lines)
    order = np.lexsort((np.arange(n), lines))
    nd = np.full(n, -1, dtype=np.int64)
    pending = -1          # walking each line's accesses backwards
    ls = lines[order]
    pf = is_pf[order]
    for j in range(n - 1, -1, -1):
        if j == n - 1 or ls[j] != ls[j + 1]:
            pending = -1
        nd[order[j]] = pending
        pending = -1 if pf[j] else order[j]
    return nd


def simulate(lines, sets, ways, nxt):
    n = len(lines)
    INF = np.int64(1 << 62)
    fut = np.where(nxt >= 0, nxt, INF)
    opt_hit = np.zeros(n, np.uint8)
    lru_hit = np.zeros(n, np.uint8)
    evicted_before_reuse = np.zeros(n, np.uint8)  # set on the access whose line OPT drops
    opt_sets = defaultdict(dict)     # set -> {line: (next_use, last_access_idx)}
    opt_heaps = defaultdict(list)    # set -> max-heap of (-next_use, line)
    lru_sets = defaultdict(OrderedDict)
    setidx = lines % sets
    for i in range(n):
        ln = int(lines[i])
        s = int(setidx[i])
        f = int(fut[i])
        # ---- OPT (Belady MIN) ----
        cs = opt_sets[s]
        hp = opt_heaps[s]
        if ln in cs:
            opt_hit[i] = 1
        elif len(cs) >= ways:
            while True:  # lazy deletion of stale heap entries
                negf, vl = heapq.heappop(hp)
                ent = cs.get(vl)
                if ent is not None and ent[0] == -negf:
                    break
            del cs[vl]
            evicted_before_reuse[ent[1]] = 1
        cs[ln] = (f, i)
        heapq.heappush(hp, (-f, ln))
        if len(hp) > 8 * ways:      # drop stale entries (keeps memory bounded)
            hp[:] = [(-v[0], k) for k, v in cs.items()]
            heapq.heapify(hp)
        # ---- LRU ----
        ls = lru_sets[s]
        if ln in ls:
            lru_hit[i] = 1
            ls.move_to_end(ln)
        else:
            if len(ls) >= ways:
                ls.popitem(last=False)
            ls[ln] = True
    opt_keep = ((nxt >= 0) & (evicted_before_reuse == 0)).astype(np.uint8)
    return opt_hit, lru_hit, opt_keep


def summarize(roles, opt_hit, lru_hit):
    out = {"accesses": int(len(roles)), "opt_misses": int((opt_hit == 0).sum()),
           "lru_misses": int((lru_hit == 0).sum()), "per_role": {}}
    for r in sorted(set(roles)):
        m = roles == r
        out["per_role"][r] = {"accesses": int(m.sum()),
                              "opt_misses": int((opt_hit[m] == 0).sum()),
                              "lru_misses": int((lru_hit[m] == 0).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--levels", default="l2,l3")
    ap.add_argument("--l2-prefetch-fills", action="store_true")
    ap.add_argument("--demand-opt", action="store_true",
                    help="with --l2-pf-detect: OPT minimises DEMAND misses (Demand-MIN); "
                         "suffix _pfd; opt_hit/opt_keep refer to demand accesses")
    ap.add_argument("--l2-pf-detect", action="store_true",
                    help="L2 only: add L2 prefetch fills detected in the L3 trace "
                         "(detect_l2_prefetches); outputs get the suffix _pf")
    opt = ap.parse_args()
    for run in map(Path, opt.run_dirs):
        for level in opt.levels.split(","):
            size, ways = GEOMETRY[level]
            sets = size // (LINE * ways)
            sfx = ("_pfd" if opt.demand_opt else "_pf") if opt.l2_pf_detect else ""
            load.counts = {}
            t = load(run, level, opt.l2_prefetch_fills, opt.l2_pf_detect)
            lines = t["line"].to_numpy().astype(np.int64)
            nxt = next_use(lines)
            if opt.demand_opt and level == "l2":
                is_pf = (t["cmd_name"].to_numpy(zero_copy_only=False) == "HardPFReq")
                nd = demand_next_use(lines, is_pf)
                opt_hit, lru_hit, opt_keep = simulate(lines, sets, ways, nd)
                nxt = nd
            else:
                opt_hit, lru_hit, opt_keep = simulate(lines, sets, ways, nxt)
            dist = np.where(nxt >= 0, nxt - np.arange(len(lines)), -1)
            out = t.append_column("set", pa.array((lines % sets).astype(np.uint32)))
            out = out.append_column("next_use", pa.array(dist))
            out = out.append_column("opt_hit", pa.array(opt_hit))
            out = out.append_column("opt_keep", pa.array(opt_keep))
            out = out.append_column("lru_hit", pa.array(lru_hit))
            pq.write_table(out, run / f"{level}_labels{sfx}.parquet", compression="zstd")
            summ = summarize(t["role"].to_numpy(zero_copy_only=False), opt_hit, lru_hit)
            summ.update({"level": level, "sets": sets, "ways": ways,
                         "l2_prefetch_fills": bool(opt.l2_prefetch_fills and level == "l2"),
                         "pf_detect": load.counts})
            (run / f"{level}_belady_summary{sfx}.json").write_text(json.dumps(summ, indent=2))
            print(f"{run.name} {level}{sfx} {load.counts}: accesses={summ['accesses']} "
                  f"LRU misses={summ['lru_misses']} OPT misses={summ['opt_misses']} "
                  f"({100 * (1 - summ['opt_misses'] / max(summ['lru_misses'], 1)):.1f}% fewer)",
                  flush=True)


if __name__ == "__main__":
    main()
