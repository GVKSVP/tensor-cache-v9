#!/usr/bin/env python3
"""
l2_offline.py - Step 7: offline L2 test bench for L2 rules (new file).

Stream : <run>/l2_labels_pfd.parquet of an LRU-recorded run (belady.py
         --l2-pf-detect --demand-opt): L2 demand/writeback requests plus the
         L2 prefetch fills, in tick order; opt_hit = Demand-MIN.
Hints  : results/tensor_hints_v3/<layer_id>.txt (same physical layout).
Replay : exactly the gem5 TensorAwareRP victim logic
         keep   -> keepStamp = ++setKeep[set]; rrpv = min(6, setKeep - stamp)
         averse -> rrpv 7 ; invalid -> 8 ; ties -> oldest lastSeq
Metric : demand misses = L2 demand reads that miss (gem5 demand MSHR misses).
Policies (keep decision per access):
  lru    keep everything                    (gem5 LRU check)
  hint   current rule: weight tensors of layers > L2: keep the first wsFrac
         of the tensor (contiguous if >= 0.35, else hashed); prefetch -> keep
  opt    Demand-MIN demand misses (from the labels)
Caveat : the prefetch stream is the one LRU produced; a rule that keeps other
         lines changes what the prefetcher issues. Checked against gem5 below.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
DEMAND = {"ReadSharedReq", "ReadExReq", "ReadCleanReq", "UpgradeReq", "ReadReq"}
L2_BYTES, WAYS, LINE = 4 * 2**20, 16, 64
SETS = L2_BYTES // LINE // WAYS


def load_hints(path):
    T, R = {}, []
    for ln in open(path):
        p = ln.split()
        if not p or p[0] == "#":
            continue
        if p[0] == "T":
            T[int(p[1])] = (int(p[2]), int(p[3]))          # mask, bytes
        elif p[0] == "R":
            R.append((int(p[1], 16), int(p[2], 16), int(p[3]), int(p[4])))
    R.sort()
    lo = np.array([r[0] for r in R], np.int64)
    hi = np.array([r[1] for r in R], np.int64)
    ten = np.array([r[2] for r in R], np.int64)
    rol = np.array([r[3] for r in R], np.int64)
    n = max(T) + 1 if T else 0
    tlo = np.full(n, np.iinfo(np.int64).max, np.int64)
    thi = np.zeros(n, np.int64)
    trole = np.full(n, -1, np.int64)
    for a, b, t, r in R:
        if 0 <= t < n:
            tlo[t] = min(tlo[t], a); thi[t] = max(thi[t], b); trole[t] = r
    mask = np.array([T.get(t, (0, 0))[0] for t in range(n)], np.int64)
    tb = np.array([T.get(t, (0, 0))[1] for t in range(n)], np.float64)
    return dict(lo=lo, hi=hi, ten=ten, rol=rol, tlo=tlo, thi=thi, trole=trole,
                mask=mask, tbytes=tb, n=n)


def lookup(h, addr):
    i = np.searchsorted(h["lo"], addr, side="right") - 1
    ok = (i >= 0) & (addr < h["hi"][np.clip(i, 0, None)])
    ten = np.where(ok, h["ten"][np.clip(i, 0, None)], -1)
    rol = np.where(ok, h["rol"][np.clip(i, 0, None)], 11)
    return ten, rol


def layer_weights(h):
    W = np.zeros(32)
    for t in range(h["n"]):
        if h["trole"][t] == 0:
            for k in range(32):
                if h["mask"][t] >> k & 1:
                    W[k] += h["tbytes"][t]
    return W


def plan_even(h, cap, alpha):
    """gem5 planFrac (even): per tensor fraction kept (0 = not planned)."""
    W = layer_weights(h)
    f = np.zeros(h["n"])
    for t in range(h["n"]):
        if h["trole"][t] != 0:
            continue
        w = max([W[k] for k in range(32) if h["mask"][t] >> k & 1] or [0])
        if w > cap:
            f[t] = alpha * cap / w
    return f


def pos_in_tensor(h, ten, addr):
    tt = np.clip(ten, 0, None)
    span = (h["thi"][tt] - h["tlo"][tt]).astype(np.float64)
    pos = (addr - h["tlo"][tt]) / np.where(span > 0, span, 1)
    return np.where(ten >= 0, pos, -1.0)


def hint_keep(h, ten, addr, lines, ispf, alpha=0.9, contig_min=0.35):
    f = plan_even(h, L2_BYTES, alpha)
    tt = np.clip(ten, 0, None)
    fr = np.where(ten >= 0, f[tt], 0.0)
    pos = pos_in_tensor(h, ten, addr)
    hsh = ((lines.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(16)) & np.uint64(0xFF)
    k_c = pos < fr
    k_h = hsh.astype(np.float64) < fr * 256.0
    keep = np.where(fr > 0, np.where(fr >= contig_min, k_c, k_h), True)
    return keep | ispf


def sim(lines, keep, isdem):
    """gem5 TensorAwareRP replay. Returns demand misses."""
    return sim2(lines, keep, isdem)[0]


def sim2(lines, keep, isdem):
    """gem5 TensorAwareRP replay. Returns (demand misses, all fills), where
    fills = every miss incl. prefetch fills = the L2 -> L3 read traffic."""
    sets = {}
    setkeep = np.zeros(SETS, np.int64)
    miss = fills = 0
    for i, (ln, k, d) in enumerate(zip(lines.tolist(), keep.tolist(), isdem.tolist())):
        s = ln % SETS
        c = sets.get(s)
        if c is None:
            c = sets[s] = {}
        e = c.get(ln)
        if e is None:
            fills += 1
            if d:
                miss += 1
            if len(c) >= WAYS:
                K = setkeep[s]
                best, bseq, vic = -1, 0, None
                for x, (av, st, sq) in c.items():
                    r = 7 if av else min(6, K - st)
                    if r > best or (r == best and sq < bseq):
                        best, bseq, vic = r, sq, x
                del c[vic]
        if k:
            setkeep[s] += 1
            c[ln] = (False, setkeep[s], i)
        else:
            c[ln] = (True, e[1] if e else 0, i)
    return miss, fills


def load_stream(run):
    t = pq.read_table(run / "l2_labels_pfd.parquet",
                      columns=["line", "cmd_name", "opt_hit"]).to_pandas()
    lines = t.line.to_numpy().astype(np.int64)
    cmd = t.cmd_name.astype(str).to_numpy()
    isdem = np.isin(cmd, list(DEMAND))
    ispf = cmd == "HardPFReq"
    return lines, isdem, ispf, t.opt_hit.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--policies", default="lru,hint,opt")
    opt = ap.parse_args()
    for r in opt.runs:
        run = Path(r)
        lid = json.loads((run / "meta.json").read_text())["plan"]["layer_id"]
        h = load_hints(RESULTS / "tensor_hints_v3" / f"{lid}.txt")
        lines, isdem, ispf, opt_hit = load_stream(run)
        addr = lines * LINE
        ten, _ = lookup(h, addr)
        res = {}
        for p in opt.policies.split(","):
            if p == "lru":
                res[p] = sim(lines, np.ones(len(lines), bool), isdem)
            elif p == "hint":
                res[p] = sim(lines, hint_keep(h, ten, addr, lines, ispf), isdem)
            elif p == "opt":
                res[p] = int((isdem & (opt_hit == 0)).sum())
            print(run.name, p, res[p], flush=True)


if __name__ == "__main__":
    main()
