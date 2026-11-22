#!/usr/bin/env python3
"""
train_l2_v2.py - Step 5: L2 tree trained with the prefetcher in view.

Data  : per-access L2 features with Demand-MIN labels (belady.py --l2-pf-detect
        --demand-opt, suffix _pfd) from several feature dirs (single layers,
        old chains, the multi-layer blocks).
New feature (computed here, causal):
  prev_pf   1 if the previous access to this line at L2 was a prefetch fill,
            i.e. the line is sitting in L2 because the prefetcher brought it
            (on a demand access: the prefetch is being used now; on a
            prefetch: re-prefetch of a line not yet used)
Split : by model as train_policy.py (test models never seen in training).
Learn : DecisionTree depth <= 6 (hardware-sized) for feature sets with and
        without prev_pf; GBT as an upper bound for the full set.
Output (new dir): l2_tree_<set>.pkl, l2_tree_<set>.txt, metrics.json
Usage : train_l2_v2.py --out results/policy_l2_v2 DIR [DIR ...] [--exclude RUN,...]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

from train_policy import FEATURE_SETS, TEST_MODELS

BASE = FEATURE_SETS["v3"]
SETS = {"v3": BASE, "v3pf": BASE + ["prev_pf"]}


def load_run(d, per_run, rng):
    cols = sorted(set(BASE) | {"cmd_id", "line", "opt_keep"})
    t = pq.read_table(d / "l2_features.parquet", columns=cols).to_pandas()
    prev = t.groupby("line", sort=False)["cmd_id"].shift(1)
    t["prev_pf"] = (prev == 3).astype(np.int8)
    if per_run and len(t) > per_run:
        t = t.iloc[np.sort(rng.choice(len(t), per_run, replace=False))]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-run", type=int, default=200_000)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--exclude", default="")
    opt = ap.parse_args()
    out = Path(opt.out)
    if out.exists():
        raise SystemExit(f"{out} exists (outputs are never overwritten)")
    excl = set(filter(None, opt.exclude.split(",")))
    rng = np.random.default_rng(0)
    parts = {True: [], False: []}
    for fd in opt.dirs:
        for d in sorted(Path(fd).iterdir()):
            if not (d / "info.json").exists() or not (d / "l2_features.parquet").exists() or d.name in excl or str(d) in excl:
                continue
            info = json.loads((d / "info.json").read_text())
            t = load_run(d, opt.per_run, rng)
            t["run"] = d.name
            parts[info["model"] in TEST_MODELS].append(t)
    tr, te = pd.concat(parts[False], ignore_index=True), pd.concat(parts[True], ignore_index=True)
    out.mkdir(parents=True)
    metrics = {"sets": SETS, "depth": opt.depth, "dirs": opt.dirs, "excluded": sorted(excl),
               "train_runs": sorted(tr.run.unique()), "test_runs": sorted(te.run.unique()),
               "train_rows": len(tr), "test_rows": len(te), "results": {}}
    print(f"train {tr.run.nunique()} runs / {len(tr)} rows; test {te.run.nunique()} runs / {len(te)} rows")
    for name, feats in SETS.items():
        X, y = tr[feats].to_numpy(), tr["opt_keep"].to_numpy()
        Xt, yt = te[feats].to_numpy(), te["opt_keep"].to_numpy()
        m = DecisionTreeClassifier(max_depth=opt.depth, min_samples_leaf=200, random_state=0).fit(X, y)
        p = m.predict(Xt)
        r = {"train_acc": round(float(accuracy_score(y, m.predict(X))), 4),
             "test_acc": round(float(accuracy_score(yt, p)), 4),
             "test_bal_acc": round(float(balanced_accuracy_score(yt, p)), 4),
             "leaves": int(m.get_n_leaves()),
             "per_test_run_acc": {k: round(float((p[te.run == k] == yt[te.run == k]).mean()), 3)
                                  for k in sorted(te.run.unique())}}
        metrics["results"][f"tree_{name}"] = r
        with open(out / f"l2_tree_{name}.pkl", "wb") as f:
            pickle.dump(m, f)
        (out / f"l2_tree_{name}.txt").write_text(export_text(m, feature_names=feats))
        print(name, {k: v for k, v in r.items() if k != "per_test_run_acc"}, flush=True)
    feats = SETS["v3pf"]
    g = HistGradientBoostingClassifier(max_iter=200, max_depth=6, categorical_features=[0, 1],
                                       random_state=0).fit(tr[feats].to_numpy(), tr["opt_keep"].to_numpy())
    p = g.predict(te[feats].to_numpy())
    metrics["results"]["gbt_v3pf"] = {"test_acc": round(float(accuracy_score(te["opt_keep"], p)), 4),
                                      "test_bal_acc": round(float(balanced_accuracy_score(te["opt_keep"], p)), 4)}
    print("gbt v3pf", metrics["results"]["gbt_v3pf"], flush=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
