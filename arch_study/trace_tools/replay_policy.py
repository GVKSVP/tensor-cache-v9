#!/usr/bin/env python3
"""
replay_policy.py - Step 4c: offline cache replay of the TEST runs with the
learned rule, next to classic rules, on the same access stream.

Policies (same geometry as belady.py: L2 4 MiB / L3 16 MiB, 16 ways, 64 B):
  lru    true LRU (self-check: must equal the lru_hit labels of belady.py)
  srrip  2-bit RRPV, insert 2, hit -> 0
  brrip  2-bit RRPV, insert 3 (every 32nd insert 2), hit -> 0
  ours   learned tree (train_policy.py) predicts keep / not for each access
         (Hawkeye-style): keep -> RRPV 0 and older kept lines age by 1
         (max 6); not -> RRPV 7 (evicted first). Victim = highest RRPV,
         oldest first on ties.
  ours_duel  'ours' with set dueling against LRU (safety net, as in DRRIP)
  opt    Belady (from the labels; the best any rule can do)
--thr: 'not keep' is used only when the model's P(keep) is below thr.
Misses are counted on every access in the stream (as belady.py does).
Caveat: the stream was recorded with LRU caches; under a different L2 rule the
L3 stream would change a little. The real test is Step 5 in gem5.
Output: <out>/replay.csv (one row per run x level), printed table.
"""

import argparse
import csv
import json
import pickle
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from train_policy import FEATURES, TEST_MODELS, runs

WAYS = 16
RESULTS = str(Path(__file__).resolve().parent.parent / "results")
LABEL_SUFFIX = ""
BETAS = [0.5, 0.75, 0.9]


def sim_lru(lines, sets):
    st = {}
    miss = 0
    for ln in lines.tolist():
        s = st.setdefault(ln % sets, OrderedDict())
        if ln in s:
            s.move_to_end(ln)
        else:
            miss += 1
            if len(s) >= WAYS:
                s.popitem(last=False)
            s[ln] = True
    return miss


def sim_rrip(lines, sets, mode):
    st = {}
    miss = 0
    nins = 0
    for ln in lines.tolist():
        s = st.setdefault(ln % sets, {})      # line -> rrpv (dict keeps insertion order)
        if ln in s:
            s[ln] = 0
            continue
        miss += 1
        if len(s) >= WAYS:
            while True:
                v = next((k for k, r in s.items() if r >= 3), None)
                if v is not None:
                    break
                for k in s:
                    s[k] += 1
            del s[v]
        if mode == "srrip":
            s[ln] = 2
        else:
            nins += 1
            s[ln] = 2 if nins % 32 == 0 else 3
    return miss


def sim_ours(lines, sets, keep, duel=False):
    """keep: model decision per access. With duel=True, set dueling against
    LRU: sets with (set % 64 == 0) always behave as LRU (all accesses treated
    as 'keep', which makes this structure exact LRU), sets with (set % 64 == 1)
    always follow the model, and all other sets follow whichever leader group
    misses less (10-bit saturating counter PSEL, start 512)."""
    st = {}
    miss = 0
    psel = 512
    for ln, k in zip(lines.tolist(), keep.tolist()):
        si = ln % sets
        s = st.setdefault(si, {})
        if duel:
            g = si % 64
            if g == 0:
                k = 1
            elif g != 1 and psel < 512:
                k = 1
        hit = ln in s
        if hit:
            del s[ln]                          # re-insert = most recent position
        else:
            miss += 1
            if duel:
                g = si % 64
                if g == 0:
                    psel = min(psel + 1, 1023)   # LRU leader missed -> favour model
                elif g == 1:
                    psel = max(psel - 1, 0)      # model leader missed -> favour LRU
            if len(s) >= WAYS:
                top = max(s.values())
                v = next(x for x, r in s.items() if r == top)   # oldest with max RRPV
                del s[v]
        if k:
            for x, r in s.items():
                if r < 6:
                    s[x] = r + 1
            s[ln] = 0
        else:
            s[ln] = 7
    return miss


def protect_mask(lines, t_dist, beta, tsize=None):
    """tensor-aware thrash protection: if the tensor's mean reuse distance is
    2^t_dist >= 1 cache capacities (t_dist in 0..8; 9 = unknown), keep a fixed
    slice of its lines (chosen by an address hash, the same lines every pass)
    of size beta / 2^t_dist of the tensor, so the slice fits in the cache."""
    h = ((lines * 2654435761) >> 16) & 0xFF                 # 0..255
    frac = np.where((t_dist >= 0) & (t_dist <= 8), beta / np.exp2(np.clip(t_dist, 0, 8)), 0.0)
    if tsize is not None:   # reuse not seen yet: size hint (tensor > cache)
        early = (t_dist == 9) & (tsize >= 1) & (tsize <= 8)
        frac = np.where(early, beta / np.exp2(np.clip(tsize, 0, 8)), frac)
    return (h < frac * 256).astype(np.uint8)


def slice_keep(keep, lines, t_dist, beta, tsize=None):
    """thrashing tensor - measured (t_dist in 0..8) or, before any reuse is
    seen, announced by the size hint (tensor larger than the cache): the slice
    decides (keep inside it, stream the rest); otherwise the tree decides"""
    thrash = (t_dist >= 0) & (t_dist <= 8)
    if tsize is not None:
        thrash |= (t_dist == 9) & (tsize >= 1) & (tsize <= 8)
    return np.where(thrash, protect_mask(lines, t_dist, beta, tsize), keep).astype(np.uint8)


def tensor_ratio(fdir, level, sfx):
    """per access: tensor bytes / cache bytes (0 = not a tensor), from the
    labels (row-aligned with the features) and the run's tensor map"""
    import csv
    import glob
    run = Path(fdir).name
    lab = (glob.glob(f"{RESULTS}/*/{run}/{level}_chain_labels{sfx}.parquet")
           or glob.glob(f"{RESULTS}/*/{run}/{level}_labels{sfx}.parquet"))[0]
    tidx = pq.read_table(lab, columns=["tensor_idx"])["tensor_idx"].to_numpy().astype(np.int64)
    rd = Path(lab).parent
    tmap = rd.parent.parent / (rd.parent.name + "_maps") / run / "tensor_map.csv"
    nb = {int(r["idx"]): int(r["bytes"]) for r in csv.DictReader(open(tmap))}
    lut = np.zeros(max(nb) + 2, np.float64)
    for i, b in nb.items():
        lut[i + 1] = b
    cache = (4 if level == "l2" else 16) * 2**20
    return lut[tidx + 1] / cache


KV_CACHE = 4   # role_id of kv_cache (build_features.ROLES)


def l2_slice_keep(keep, lines, ratio, cmd, beta, role=None):
    """L2 rule for a tensor larger than the cache (size hint): prefetch fills
    are kept until their demand use; demand accesses keep only a fixed slice
    (address hash) of size beta / ratio; other tensors: the tree decides"""
    big = ratio > 1.0
    if role is not None:      # KV cache: reused at once by the query heads of
        big &= role != KV_CACHE   # its group (GQA) -> leave it to the tree
    h = ((lines * 2654435761) >> 16) & 0xFF
    in_slice = h < (beta / np.maximum(ratio, 1.0)) * 256
    return np.where(big, in_slice | (cmd == 3), keep).astype(np.uint8)


def one(job):
    fdir, level, model_path, thrs = job
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    meta = json.loads((Path(model_path).parent / "metrics.json").read_text())
    FEATURES = meta.get(f"features_{level}", meta["features"])
    if f"thr_{level}" in meta:              # per-level threshold of a final policy
        thrs = [meta[f"thr_{level}"]]
    t = pq.read_table(Path(fdir) / f"{level}_features.parquet",
                      columns=sorted(set(FEATURES + ["t_dist", "tsize_log2", "cmd_id", "role_id"])) + ["line", "opt_hit", "lru_hit", "opt_keep"])
    lines = t["line"].to_numpy().astype(np.int64)
    sets = (4 if level == "l2" else 16) * 2**20 // (64 * WAYS)
    X = np.column_stack([t[c].to_numpy() for c in FEATURES]).astype(np.float32)
    tdist = t["t_dist"].to_numpy().astype(np.int64)
    tsz = t["tsize_log2"].to_numpy().astype(np.int64)
    ratio = tensor_ratio(fdir, level, LABEL_SUFFIX) if level == "l2" else None
    cmdv = t["cmd_id"].to_numpy().astype(np.int64) if "cmd_id" in t.column_names else None
    rolev = t["role_id"].to_numpy().astype(np.int64)
    pk = np.concatenate([model.predict_proba(X[i:i + 1_000_000])[:, 1].astype(np.float32)
                         for i in range(0, len(X), 1_000_000)])
    del X
    opt_keep = t["opt_keep"].to_numpy()
    base = {"run": Path(fdir).name, "level": level, "accesses": len(t),
            "lru_label": int((t["lru_hit"].to_numpy() == 0).sum()),
            "opt": int((t["opt_hit"].to_numpy() == 0).sum()),
            "lru": sim_lru(lines, sets), "srrip": sim_rrip(lines, sets, "srrip"),
            "brrip": sim_rrip(lines, sets, "brrip")}
    out = []
    for thr in thrs:
        keep = (pk >= thr).astype(np.uint8)             # 'throw out' only when P(keep) < thr
        out.append(dict(base, thr=thr,
                        pred_acc=round(float((keep == opt_keep).mean()), 4),
                        ours=sim_ours(lines, sets, keep),
                        ours_duel=sim_ours(lines, sets, keep, duel=True),
                        **{f"slice{b}_duel": sim_ours(lines, sets, slice_keep(keep, lines, tdist, b), duel=True)
                           for b in BETAS},
                        **{f"sizeslice{b}_duel": sim_ours(lines, sets, slice_keep(keep, lines, tdist, b, tsz), duel=True)
                           for b in BETAS},
                        **({f"l2slice{b}_duel": sim_ours(lines, sets, l2_slice_keep(keep, lines, ratio, cmdv, b, rolev), duel=True)
                            for b in BETAS} if ratio is not None else {})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features_v1")
    ap.add_argument("--policy", required=True, help="dir from train_policy.py")
    ap.add_argument("--learner", default="tree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="comma list of run names (default: all runs of --split)")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--thr", default="0.5", help="comma list of P(keep) thresholds")
    ap.add_argument("--label-suffix", default="", help="labels used for the features, e.g. _pf")
    opt = ap.parse_args()
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "replay.csv"
    if dest.exists():
        raise SystemExit(f"{dest} exists (outputs are never overwritten)")
    global LABEL_SUFFIX
    LABEL_SUFFIX = opt.label_suffix
    only = set(filter(None, opt.only.split(",")))
    jobs = []
    for d, info in runs(opt.features):
        if (info["model"] in TEST_MODELS) == (opt.split == "test") and (not only or d.name in only):
            for level in [lv for lv in ("l2", "l3") if (Path(opt.policy) / f"{lv}_{opt.learner}.pkl").exists()]:
                jobs.append((str(d), level, str(Path(opt.policy) / f"{level}_{opt.learner}.pkl"),
                             [float(x) for x in opt.thr.split(",")]))
    jobs.sort(key=lambda j: -(Path(j[0]) / f"{j[1]}_features.parquet").stat().st_size)
    rows = []
    with ProcessPoolExecutor(opt.workers) as ex:
        for res in ex.map(one, jobs):
            for r in res:
                rows.append(r)
                base = max(r["lru"], 1)
                print(f"{r['run']:<32}{r['level']} thr {r['thr']:.2f} LRU {r['lru']:>9} "
                      f"(label {r['lru_label']:>9})  " + "  ".join(
                          f"{p} {r[p] / base:.3f}" for p in ["srrip", "brrip", "ours", "ours_duel"] + [f"{k}{b}_duel" for k in ("slice", "sizeslice", "l2slice") for b in BETAS if f"{k}{b}_duel" in r] + ["opt"]),
                      flush=True)
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["level"], r["run"], r["thr"])))


if __name__ == "__main__":
    main()
