#!/usr/bin/env python3
"""
train_policy.py - Step 4b: learn "keep or not" (Belady opt_keep label) from the
per-access features of build_features.py, one model per cache level.

Split BY MODEL (the test models are never seen in training):
  test  = Stable Diffusion XL, ViT-B/16, MobileNetV2, Llama-2-70B
          (their single layers AND the ViT / MobileNetV2 chains)
  train = all other models (incl. the BERT, ResNet, RNN-T, Llama-8B chains)
Each run contributes at most --per-run sampled accesses (so huge runs do not
dominate). Two learners:
  tree  DecisionTree, depth <= 6 (<= 64 leaves: small enough for a hardware table)
  gbt   HistGradientBoosting (upper bound on what these features can give)
Output (new dir --out): <level>_<learner>.pkl, metrics.json, and the tree rules
as text (<level>_tree.txt).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = ["role_id", "cmd_id", "tsize_log2", "used_next", "dead", "first_touch",
            "prev_dist_log2", "n_prev", "t_retouch8", "t_dist"]
FEATURE_SETS = {"v1": FEATURES,
                "v2": FEATURES + ["prev2_dist_log2", "t_maxn", "hit"],
                # chosen by leave-one-model-out CV on the TRAIN models only
                "v3": [f for f in FEATURES if f not in ("tsize_log2", "n_prev")]}
TEST_MODELS = {"Stable Diffusion XL", "ViT-B/16", "MobileNetV2", "Llama-2-70B"}


def runs(feat_dir):
    for d in sorted(Path(feat_dir).iterdir()):
        if (d / "info.json").exists():
            yield d, json.loads((d / "info.json").read_text())


def load(feat_dir, level, per_run, test, seed=0, feats=FEATURES):
    rng = np.random.default_rng(seed)
    parts = []
    for d, info in runs(feat_dir):
        if (info["model"] in TEST_MODELS) != test:
            continue
        t = pq.read_table(d / f"{level}_features.parquet", columns=feats + ["opt_keep"]).to_pandas()
        if per_run and len(t) > per_run:
            t = t.iloc[np.sort(rng.choice(len(t), per_run, replace=False))]
        t["run"] = d.name
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-run", type=int, default=200_000)
    ap.add_argument("--feature-set", default="v1", choices=sorted(FEATURE_SETS))
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--levels", default="l2,l3")
    opt = ap.parse_args()
    out = Path(opt.out)
    if out.exists():
        raise SystemExit(f"{out} exists (outputs are never overwritten)")
    out.mkdir(parents=True)
    feats = FEATURE_SETS[opt.feature_set]
    metrics = {"features": feats, "feature_set": opt.feature_set, "depth": opt.depth, "test_models": sorted(TEST_MODELS), "levels": {}}
    for level in opt.levels.split(","):
        tr = load(opt.features, level, opt.per_run, test=False, feats=feats)
        te = load(opt.features, level, opt.per_run, test=True, feats=feats)
        X, y = tr[feats].to_numpy(), tr["opt_keep"].to_numpy()
        Xt, yt = te[feats].to_numpy(), te["opt_keep"].to_numpy()
        lv = {"train_rows": len(tr), "test_rows": len(te), "train_runs": tr.run.nunique(),
              "test_runs": te.run.nunique(), "test_keep_rate": round(float(yt.mean()), 4),
              "majority_acc": round(float(max(yt.mean(), 1 - yt.mean())), 4)}
        learners = {"tree": DecisionTreeClassifier(max_depth=opt.depth, min_samples_leaf=200, random_state=0),
                    "gbt": HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                                          categorical_features=[0, 1], random_state=0)}
        for name, m in learners.items():
            m.fit(X, y)
            p = m.predict(Xt)
            lv[name] = {"train_acc": round(float(accuracy_score(y, m.predict(X))), 4),
                        "test_acc": round(float(accuracy_score(yt, p)), 4),
                        "test_bal_acc": round(float(balanced_accuracy_score(yt, p)), 4),
                        "per_test_run_acc": {r: round(float((p[te.run == r] == yt[te.run == r]).mean()), 3)
                                             for r in sorted(te.run.unique())}}
            with open(out / f"{level}_{name}.pkl", "wb") as f:
                pickle.dump(m, f)
            if name == "tree":
                (out / f"{level}_tree.txt").write_text(export_text(m, feature_names=feats))
                lv["tree"]["leaves"] = int(m.get_n_leaves())
            print(level, name, {k: v for k, v in lv[name].items() if k != "per_test_run_acc"}, flush=True)
        metrics["levels"][level] = lv
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
