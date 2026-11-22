#!/usr/bin/env python3
"""
lomo_replay.py - choose between feature sets / thresholds using TRAINING
models only: for every training model M, train the tree on the other training
models and replay M's traces (offline cache, set dueling vs LRU). The test
models are never touched. Output: <out>.csv
Usage: lomo_replay.py --sets v1,v3 --thr 0.3,0.5 --runs RUN,RUN,... --out FILE
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.tree import DecisionTreeClassifier

import replay_policy
from replay_policy import l2_slice_keep, sim_lru, sim_ours, slice_keep, tensor_ratio
from train_policy import FEATURE_SETS, load, runs

FEAT = "../results/features_v2"
BETAS = []
SUFFIX = ""


def replay(job):
    run, level, feats, model, thrs = job
    t = pq.read_table(Path(FEAT) / run / f"{level}_features.parquet",
                      columns=sorted(set(feats + ["t_dist", "tsize_log2", "cmd_id", "role_id"])) + ["line", "lru_hit", "opt_hit"])
    tdist = t["t_dist"].to_numpy().astype(np.int64)
    tsz = t["tsize_log2"].to_numpy().astype(np.int64)
    ratio = tensor_ratio(Path(FEAT) / run, level, SUFFIX) if level == "l2" and SUFFIX else None
    cmdv = t["cmd_id"].to_numpy().astype(np.int64)
    rolev = t["role_id"].to_numpy().astype(np.int64)
    lines = t["line"].to_numpy().astype(np.int64)
    sets = (4 if level == "l2" else 16) * 2**20 // (64 * 16)
    X = np.column_stack([t[c].to_numpy() for c in feats]).astype(np.float32)
    pk = model.predict_proba(X)[:, 1]
    lru = int((t["lru_hit"].to_numpy() == 0).sum())
    opt = int((t["opt_hit"].to_numpy() == 0).sum())
    out = []
    for thr in thrs:
        keep = (pk >= thr).astype(np.uint8)
        out.append((run, level, thr, "none", lru, opt, sim_ours(lines, sets, keep, duel=True)))
        for b in BETAS:
            k2 = (l2_slice_keep(keep, lines, ratio, cmdv, b, rolev) if ratio is not None
                  else slice_keep(keep, lines, tdist, b, tsz))
            out.append((run, level, thr, b, lru, opt, sim_ours(lines, sets, k2, duel=True)))
    return out


def main():
    global FEAT, BETAS, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="v1,v3")
    ap.add_argument("--thr", default="0.3,0.5")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--features", default=FEAT)
    ap.add_argument("--levels", default="l2,l3")
    ap.add_argument("--betas", default="", help="slice sizes to try (with size hint), e.g. 0.5,0.75")
    ap.add_argument("--label-suffix", default="", help="_pf: L2 uses the size-ratio slice (l2_slice_keep)")
    opt = ap.parse_args()
    FEAT = opt.features
    SUFFIX = opt.label_suffix
    BETAS = [float(x) for x in opt.betas.split(",") if x]
    thrs = [float(x) for x in opt.thr.split(",")]
    model_of = {d.name: i["model"] for d, i in runs(FEAT)}
    targets = opt.runs.split(",")
    allf = sorted({f for s in opt.sets.split(",") for f in FEATURE_SETS[s]})
    out_path = Path(opt.out)
    fields = ["set", "run", "level", "thr", "slice", "lru", "opt", "ours_duel"]
    done = set()
    if out_path.exists():                      # resume: skip finished (set, run, level)
        with open(out_path) as f:
            for r in csv.DictReader(f):
                done.add((r["set"], r["run"], r["level"]))
    else:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    for level in opt.levels.split(","):
        tr = load(FEAT, level, 200_000, test=False, feats=allf)
        tr["model"] = tr.run.map(model_of)
        jobs = []
        for fs in opt.sets.split(","):
            feats = FEATURE_SETS[fs]
            for m in sorted({model_of[r] for r in targets}):
                a = tr[tr.model != m]
                clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, random_state=0)
                clf.fit(a[feats].values, a.opt_keep.values)
                for r in targets:
                    if model_of[r] == m and (fs, r, level) not in done:
                        jobs.append((fs, (r, level, feats, clf, thrs)))
        with ProcessPoolExecutor(opt.workers) as ex:
            for fs, res in zip([j[0] for j in jobs], ex.map(replay, [j[1] for j in jobs])):
                with open(out_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    for run, lv, thr, beta, lru, optm, ours in res:
                        w.writerow({"set": fs, "run": run, "level": lv, "thr": thr, "slice": beta,
                                    "lru": lru, "opt": optm, "ours_duel": ours})
                        print(fs, run, lv, thr, beta, f"{ours / max(lru, 1):.3f}", flush=True)


if __name__ == "__main__":
    main()
