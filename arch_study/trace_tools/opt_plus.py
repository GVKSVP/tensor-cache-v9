#!/usr/bin/env python3
"""
opt_plus.py - DRAM-traffic ceilings for the L3 of one traced run (Step 3).

Replays <run>/l3_roi.parquet (the requests that reach L3, CleanEvict skipped)
and counts DRAM traffic = 64 B per L3 read miss + 64 B per dirty eviction
(or bypassed dirty writeback). Variants, each adding one "OPT weak point":

  lru        : LRU, allocate everything            (check against gem5's LRU)
  opt        : Belady MIN, allocate everything     (standard OPT)
  +bypass    : Belady with bypass: a fill whose next use is later than every
               resident line's next use is not inserted (#1)
  +dirty     : on eviction, among lines never used again prefer a clean one,
               so no write is needed (#2)
  +dead      : a dirty line that is never accessed again and belongs to an
               intermediate tensor (activation_mid / scratch: dead after its
               last user layer) is dropped without a DRAM write (#3)
  +coord     : L2 and L3 do not keep the same data twice: the L3 gets the
               L2 capacity on top (16 + 4 ways per L3 set) (#4, approximate)
  opt_plus   : all of the above together

#5 (prefetch-aware) is an L2 matter: the L2 labels use Demand-MIN
(belady.py --demand-opt) already; the L3 stream here contains the L2
prefetches as they happened under LRU.

These are offline estimates on the LRU-run trace (the request stream of
another policy would differ slightly), not proven optima: Belady with
bypass is optimal for miss count; the dirty/dead rules are greedy.
Output: <run>/l3_opt_plus.json and a printed table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

L3_BYTES, L3_WAYS, L2_BYTES = 16 * 2**20, 16, 4 * 2**20
LINE = 64
READS = {"ReadSharedReq", "ReadExReq", "ReadCleanReq", "UpgradeReq", "ReadReq",
         "HardPFReq", "SCUpgradeReq"}
WBS = {"WritebackDirty", "WritebackClean", "WriteClean", "WriteReq"}
DEAD_ROLES = {"activation_mid", "scratch"}
INF = 1 << 62


def next_use(lines):
    n = len(lines)
    order = np.lexsort((np.arange(n), lines))
    nxt = np.full(n, INF, dtype=np.int64)
    same = lines[order[1:]] == lines[order[:-1]]
    nxt[order[:-1][same]] = order[1:][same]
    return nxt


def replay(lines, kind, dead_ok, fut, sets, ways, policy, bypass=False,
           dirty_aware=False, dead_free=False):
    """kind: 0 read, 1 dirty writeback, 2 clean writeback. Returns (rd, wr) lines."""
    cache = [dict() for _ in range(sets)]  # line -> [next_use, dirty, last_seq]
    rd = wr = 0
    setidx = lines % sets
    for i in range(len(lines)):
        ln = int(lines[i]); s = int(setidx[i]); k = int(kind[i]); f = int(fut[i])
        cs = cache[s]
        e = cs.get(ln)
        if e is not None:
            e[0] = f; e[2] = i
            if k == 1:
                e[1] = True
            continue
        if k == 0:
            rd += 1
        dirty = k == 1
        if len(cs) >= ways:
            if policy == "lru":
                victim = min(cs, key=lambda x: cs[x][2])
            else:
                far = max(v[0] for v in cs.values())
                if bypass and f >= far:            # incoming is the farthest: do not insert
                    if dirty and not (dead_free and f == INF and dead_ok[i]):
                        wr += 1
                    continue
                cands = [x for x, v in cs.items() if v[0] == far]
                victim = cands[0]
                if dirty_aware and far == INF:
                    clean = [x for x in cands if not cs[x][1]]
                    if clean:
                        victim = clean[0]
                    elif dead_free:
                        dead = [x for x in cands if cs[x][3]]
                        victim = dead[0] if dead else victim
            v = cs.pop(victim)
            if v[1] and not (dead_free and v[0] == INF and v[3]):
                wr += 1
        cs[ln] = [f, dirty, i, bool(dead_ok[i])]
    return rd, wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--variants", default="lru,opt,+bypass,+dirty,+dead,+coord,opt_plus")
    opt = ap.parse_args()
    for rd_ in opt.run_dirs:
        run = Path(rd_)
        t = pq.read_table(run / "l3_roi.parquet", columns=["line", "cmd_name", "role"],
                          read_dictionary=["cmd_name", "role"])
        t = t.filter(pc.invert(pc.equal(t["cmd_name"], "CleanEvict")))
        lines = t["line"].to_numpy().astype(np.int64)
        # dictionary codes instead of Python strings (memory: large traces)
        t = t.unify_dictionaries()
        cmd = t["cmd_name"].combine_chunks()
        cnames = cmd.dictionary.to_pylist()
        ccode = cmd.indices.to_numpy(zero_copy_only=False)
        kmap = np.array([0 if c in READS else 1 if c in ("WritebackDirty", "WriteReq") else 2
                         if c in WBS else 0 for c in cnames], np.uint8)
        kind = kmap[ccode]
        unknown = sorted(c for c in cnames if c not in READS and c not in WBS)
        rol = t["role"].combine_chunks()
        dmap = np.array([r in DEAD_ROLES for r in rol.dictionary.to_pylist()], bool)
        dead_ok = dmap[rol.indices.to_numpy(zero_copy_only=False)]
        del t, cmd, rol, ccode
        fut = next_use(lines)
        sets = L3_BYTES // LINE // L3_WAYS
        extra = L2_BYTES // LINE // sets          # L2 capacity per L3 set (ways)
        spec = {
            "lru": dict(policy="lru", ways=L3_WAYS),
            "opt": dict(policy="opt", ways=L3_WAYS),
            "+bypass": dict(policy="opt", ways=L3_WAYS, bypass=True),
            "+dirty": dict(policy="opt", ways=L3_WAYS, dirty_aware=True),
            "+dead": dict(policy="opt", ways=L3_WAYS, dirty_aware=True, dead_free=True),
            "+coord": dict(policy="opt", ways=L3_WAYS + extra),
            "opt_plus": dict(policy="opt", ways=L3_WAYS + extra, bypass=True,
                             dirty_aware=True, dead_free=True),
        }
        res = {"run": run.name, "accesses": int(len(lines)),
               "reads": int((kind == 0).sum()), "writebacks": int((kind > 0).sum()),
               "unknown_cmds": unknown, "variants": {}}
        for name in opt.variants.split(","):
            r, w = replay(lines, kind, dead_ok, fut, sets, **spec[name])
            res["variants"][name] = {"dram_read_lines": r, "dram_write_lines": w,
                                     "dram_MB": round((r + w) * LINE / 1e6, 3)}
            print(f"{run.name} {name:<9} reads {r:>9} writes {w:>7}  "
                  f"DRAM {(r + w) * LINE / 1e6:8.2f} MB", flush=True)
        base = res["variants"].get("lru", {}).get("dram_MB")
        if base:
            for v in res["variants"].values():
                v["vs_lru"] = round(v["dram_MB"] / base, 4)
        (run / "l3_opt_plus.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
