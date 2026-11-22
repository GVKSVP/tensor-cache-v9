#!/usr/bin/env python3
"""offline_slice_sim.py - small offline L3 replay used to design the slice
protection (Steps 4 and 6): DRAM read lines on a traced (LRU-run) L3 stream for
LRU and for the TensorAwareRP RRPV structure with a given per-access keep
decision (keep -> RRPV 0, others age to 6; not keep -> RRPV 7; victim = highest
RRPV, oldest first). Library: load(run_dir) -> (lines, tensor names, is_read);
sim(lines, keep, is_read, policy) -> read misses."""
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

SETS, WAYS = 16384, 16


def load(d):
    t = pq.read_table(f"{d}/l3_roi.parquet", columns=["line", "cmd_name", "tensor"])
    t = t.filter(pc.invert(pc.equal(t["cmd_name"], "CleanEvict"))).to_pandas()
    rd = ~t.cmd_name.isin(["WritebackDirty", "WritebackClean"])
    return t.line.to_numpy(np.int64), t.tensor.astype(str).to_numpy(), rd.to_numpy()


def sim(lines, keep, isrd, policy):
    sets = [dict() for _ in range(SETS)]
    kc = [0] * SETS
    miss = seq = 0
    for i in range(len(lines)):
        ln = int(lines[i]); s = ln & (SETS - 1); cs = sets[s]; seq += 1
        if ln not in cs:
            if isrd[i]:
                miss += 1
            if len(cs) >= WAYS:
                k = kc[s]
                v = max(cs, key=lambda x: ((7 if cs[x][0] < 0 else min(6, k - cs[x][0])), -cs[x][1]))
                del cs[v]
        if policy == "lru" or keep[i]:
            kc[s] += 1
            cs[ln] = [kc[s], seq]
        else:
            cs[ln] = [-1, seq]
    return miss
