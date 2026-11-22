#!/usr/bin/env python3
"""
build_features.py - Step 4a: per-access feature table for learning a
tensor-aware replacement rule from the labelled traces (belady.py,
chain_layers.py). Only features that a real cache + software hint could know
AT THE TIME OF THE ACCESS are produced (nothing from the future):

  role_id         tensor role hint from software (weight, activation_in, ...)
  cmd_id          request kind: 0 read, 1 read-exclusive/upgrade, 2 writeback,
                  3 prefetch, 4 other
  tsize_log2      log2(tensor bytes / cache bytes), clipped to [-8, 8]
                  (-9 = address outside any tensor)
  used_next       1 if the next layer of the chain uses this tensor (hint)
  dead            1 if no later layer uses this tensor (hint)
  first_touch     1 if this line was never accessed before at this level
  prev_dist_log2  log2(1 + accesses since the previous access to this line), -1 if first
  n_prev          number of previous accesses to this line, capped at 7
  t_retouch8      share (eighths, 0..8) of EARLIER accesses of the same tensor that
                  re-touched a line (per-tensor counters, causal)
  t_dist          mean log2(reuse distance / cache lines) of those earlier
                  re-touches, clipped [-12, 8]; 9 = no re-touch seen yet
  prev2_dist_log2 prev_dist_log2 of the previous access to this line (-2 if none)
  t_maxn          highest per-line use count (n_prev) seen in earlier accesses of
                  the same tensor (-1 if none)
  hit             1 if this access hits (taken from the LRU replay of the trace;
                  in a real cache: the actual hit/miss of the access)
Label: opt_keep (Hawkeye-style: Belady OPT keeps the line until its next use).
Also kept for evaluation: opt_hit, lru_hit, line, set.

In single-layer runs the trace contains one layer only, so used_next = 0 and
dead = 1 for every tensor (consistent with what the trace shows).

Output (new dir): <out>/<run>/<level>_features.parquet (all accesses) and
<out>/<run>/info.json (model, layer_id, kind). Existing files are not touched.
Usage: build_features.py --out results/features_v1 [--workers 3] RUN_DIR...
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROLES = ["weight", "activation_in", "activation_out", "activation_mid", "kv_cache",
         "gather_table", "index", "scratch", "flush", "program_image", "stack", "other"]
CACHE = {"l2": (4 * 2**20, 16), "l3": (16 * 2**20, 16)}
LINE = 64


def cmd_id(names):
    out = np.full(len(names), 4, np.int8)
    out[np.char.startswith(names, "ReadShared") | (names == "ReadReq") | (names == "ReadCleanReq")] = 0
    out[np.char.startswith(names, "ReadEx") | np.char.startswith(names, "Upgrade")] = 1
    out[np.char.startswith(names, "Writeback") | (names == "WriteReq") | (names == "WriteClean")] = 2
    out[np.char.find(names, "PF") >= 0] = 3
    return out


def cmd_codes(col):
    d = col.combine_chunks()
    lut = cmd_id(np.array(d.dictionary.to_pylist() or ["x"], dtype=str))
    return lut[d.indices.to_numpy(zero_copy_only=False)]


def prev_index(keys):
    """index of the previous element with the same key (-1 if none), the rank
    of each element within its key group, the stable sort order and the group
    start position (in sorted order) of each sorted element. int32 to save RAM."""
    n = len(keys)
    order = np.argsort(keys, kind="stable").astype(np.int32)
    k = keys[order]
    start = np.empty(n, bool)
    start[0] = True
    np.not_equal(k[1:], k[:-1], out=start[1:])
    del k
    prev = np.full(n, -1, np.int32)
    prev[order[1:][~start[1:]]] = order[:-1][~start[1:]]
    gstart = np.where(start, np.arange(n, dtype=np.int32), 0)
    np.maximum.accumulate(gstart, out=gstart)
    rank = np.empty(n, np.int32)
    rank[order] = np.arange(n, dtype=np.int32) - gstart
    return prev, rank, order, gstart


def one(job):
    run, maps, out, level, sfx = job
    run, maps, out = Path(run), Path(maps), Path(out)
    chain = (run / f"{level}_chain_labels{sfx}.parquet").exists()
    src = run / (f"{level}_chain_labels{sfx}.parquet" if chain else f"{level}_labels{sfx}.parquet")
    cols = ["line", "set", "cmd_name", "tensor_idx", "role", "opt_keep", "opt_hit", "lru_hit"]
    if chain:
        cols += ["used_next", "dead"]
    t = pq.read_table(src, columns=cols, read_dictionary=["role", "cmd_name"])
    n = len(t)
    size, ways = CACHE[level]
    cap = size // LINE
    lines = t["line"].to_numpy().astype(np.int64)
    tidx = t["tensor_idx"].to_numpy().astype(np.int64)
    rd = t["role"].combine_chunks()
    lut = np.array([ROLES.index(r) if r in ROLES else len(ROLES) - 1
                    for r in rd.dictionary.to_pylist()] or [0], np.int8)
    role_id = lut[rd.indices.to_numpy(zero_copy_only=False)]
    tbytes = {int(r["idx"]): int(r["bytes"]) for r in csv.DictReader(open(maps / run.name / "tensor_map.csv"))}
    tsize = np.full(n, -9, np.int8)
    for i, b in tbytes.items():
        m = tidx == i
        if m.any():
            tsize[m] = int(np.clip(np.round(np.log2(b / size)), -8, 8))
    prev, rank, _, _ = prev_index(lines)
    del lines
    dist = np.arange(n, dtype=np.int32) - prev
    dist[prev < 0] = -1
    first = prev < 0
    pdl = np.where(dist >= 0, np.floor(np.log2(1 + np.maximum(dist, 0))), -1).astype(np.int8)
    prev2 = np.where(prev >= 0, pdl[np.maximum(prev, 0)], -2).astype(np.int8)   # -2 = no previous access
    del prev
    # causal per-tensor running statistics over the EARLIER accesses of the tensor
    # (a cache can keep them with a few counters per tensor id):
    #   t_retouch8 share (eighths) of earlier accesses that re-touched a line
    #   t_dist     mean log2(reuse distance / cache lines) of those re-touches
    _, trank, torder, gbase = prev_index(tidx + 1)    # -1 (no tensor) -> group 0

    def before(x):
        xs = x[torder].astype(np.float64)
        cs = np.cumsum(xs)
        cs -= xs
        cs -= cs[gbase]          # minus the running sum at the group start
        out = np.empty(n)
        out[torder] = cs
        return out
    re = dist >= 0
    n_re = before(re)
    sum_ld = before(np.where(re, np.log2(np.maximum(dist, 1).astype(np.float64) / cap), 0))
    # t_maxn: highest per-line use count (n_prev, capped 7) seen in EARLIER
    # accesses of the tensor (-1 if none): "how often are this tensor's lines
    # usually used"; n_prev >= t_maxn hints at a last use
    rk = np.minimum(rank, 7)[torder].astype(np.int64)
    gid = np.cumsum(gbase == np.arange(n))                # group number in sorted order
    cm = np.maximum.accumulate(gid * 16 + rk)
    exc = np.empty(n, np.int64)
    exc[0] = -1
    exc[1:] = cm[:-1]
    exc = np.where(gbase == np.arange(n), -1, exc - gid * 16)
    t_maxn = np.empty(n, np.int8)
    t_maxn[torder] = exc
    del torder, gbase, rk, gid, cm, exc
    t_retouch8 = np.where(trank > 0, np.round(8 * n_re / np.maximum(trank, 1)), 0).astype(np.int8)
    t_dist = np.where(n_re > 0, np.clip(np.round(sum_ld / np.maximum(n_re, 1)), -12, 8), 9).astype(np.int8)
    del n_re, sum_ld, trank
    if chain:
        used_next = t["used_next"].to_numpy().astype(np.int8)
        dead = t["dead"].to_numpy().astype(np.int8)
    else:
        used_next = np.zeros(n, np.int8)
        dead = (tidx >= 0).astype(np.int8)
    feats = pa.table({
        "role_id": role_id,
        "cmd_id": cmd_codes(t["cmd_name"]),
        "tsize_log2": tsize,
        "used_next": used_next,
        "dead": dead,
        "first_touch": first.astype(np.int8),
        "prev_dist_log2": pdl,
        "prev2_dist_log2": prev2,
        "n_prev": np.minimum(rank, 7).astype(np.int8),
        "t_retouch8": t_retouch8,
        "t_dist": t_dist,
        "t_maxn": t_maxn,
        "hit": t["lru_hit"],
        "opt_keep": t["opt_keep"], "opt_hit": t["opt_hit"], "lru_hit": t["lru_hit"],
        "line": t["line"], "set": t["set"],
    })
    od = out / run.name
    od.mkdir(parents=True, exist_ok=True)
    pq.write_table(feats, od / f"{level}_features.parquet", compression="zstd")
    plan = json.loads((run / "meta.json").read_text())["plan"]
    (od / "info.json").write_text(json.dumps({"model": plan["model"], "layer_id": plan["layer_id"],
                                             "kind": "chain" if chain else "layer"}))
    return f"{run.name} {level}: {n} accesses, keep rate {t['opt_keep'].to_numpy().mean():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--levels", default="l2,l3")
    ap.add_argument("--suffix", default="", help="label file suffix, e.g. _pf")
    opt = ap.parse_args()
    jobs = []
    for r in map(Path, opt.run_dirs):
        maps = r.parent.parent / (r.parent.name + "_maps")
        for level in opt.levels.split(","):
            if not (Path(opt.out) / r.name / f"{level}_features.parquet").exists():
                jobs.append((str(r), str(maps), opt.out, level, opt.suffix))
    # biggest first so the long ones do not end last
    jobs.sort(key=lambda j: -(Path(j[0]) / f"{j[3]}_labels{j[4]}.parquet").stat().st_size)
    with ProcessPoolExecutor(opt.workers) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
