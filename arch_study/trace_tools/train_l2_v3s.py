#!/usr/bin/env python3
"""
train_l2_v3s.py - Step 7, round 2 of the learned L2 rule ("stable" model).

Round 1 (train_l2_v3.py) found weak point #5: Demand-MIN labels treat a
prefetch fill as free, so the tree learned to drop lines the prefetcher will
fetch again -> L2->L3 traffic back at LRU level (ViT +53 % vs the hint rule,
checked in gem5). Fixes here:
  labels   plain Belady OPT on the same stream, EVERY fill counts as a miss
           (demand and prefetch), so re-fetch traffic is a cost
  features STATIC per region only (no per-access history): the decision for
           a part of a tensor is the same on every pass = a stable, learned
           slice like the hint rule's, which the prefetcher can work with
             role_id tsize_log2 used_next dead pos16 plan16 over_log2 inplan
  choice   a model is accepted only if it is no worse than the hint rule on
           BOTH demand misses and L2 fills (bench l2_offline.sim2, matched to
           gem5 for both numbers)
Split: leave-one-workload-out (ViT, BERT, DLRM, RNN-T); RN50/MBV2 always train.
Output (new dir): tree pkl/txt per fold, summary.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

import belady
import l2_offline as B
import train_l2_v3 as T

ALL = T.OLD + T.NEW
STATIC = ["role_id", "tsize_log2", "used_next", "dead", "pos16", "plan16", "over_log2", "inplan"]
COLS = [ALL.index(f) for f in STATIC]


def all_opt(name, cache):
    fn = cache / f"{name}_allopt.npz"
    if not fn.exists():
        z = np.load(cache / f"{name}.npz")
        lines = z["lines"]
        _, _, keep = belady.simulate(lines, B.SETS, B.WAYS, belady.next_use(lines))
        np.savez(fn, y=keep.astype(np.int8))
    return np.load(fn)["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="4,6,8")
    ap.add_argument("--thrs", default="0.3,0.5,0.7")
    ap.add_argument("--per-run", type=int, default=300_000)
    o = ap.parse_args()
    out = Path(o.out)
    if out.exists():
        raise SystemExit(f"{out} exists (outputs are never overwritten)")
    out.mkdir(parents=True)
    cache = T.R / "policy_l2_v3_cache"
    for n in T.RUNS:
        all_opt(n, cache)
        print("labels", n, flush=True)
    rng = np.random.default_rng(0)
    summary = {"features": STATIC, "folds": {}}
    for test in T.EVAL:
        Xs, ys = [], []
        for n in T.RUNS:
            if n == test:
                continue
            z = np.load(cache / f"{n}.npz")
            y = all_opt(n, cache)
            idx = np.flatnonzero(z["w"] > 0)
            if len(idx) > o.per_run:
                idx = np.sort(rng.choice(idx, o.per_run, replace=False))
            Xs.append(z["X"][idx][:, COLS]); ys.append(y[idx])
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        z = np.load(cache / f"{test}.npz")
        L, D = z["lines"], z["isdem"]
        res = {"lru": B.sim2(L, np.ones(len(L), bool), D), "hint": B.sim2(L, z["hint"], D)}
        Xt = z["X"][:, COLS]
        for depth in map(int, o.depths.split(",")):
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=500,
                                       class_weight="balanced", random_state=0).fit(Xtr, ytr)
            pickle.dump(m, open(out / f"tree_{test}_d{depth}.pkl", "wb"))
            (out / f"tree_{test}_d{depth}.txt").write_text(export_text(m, feature_names=STATIC))
            p = m.predict_proba(Xt)[:, 1]
            for thr in map(float, o.thrs.split(",")):
                res[f"s_d{depth}_t{thr}"] = B.sim2(L, (p >= thr) | z["ispf"], D)
                print(test, depth, thr, res[f"s_d{depth}_t{thr}"], "hint", res["hint"], flush=True)
        summary["folds"][test] = {k: {"demand_misses": v[0], "fills": v[1]} for k, v in res.items()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
