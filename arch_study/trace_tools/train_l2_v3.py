#!/usr/bin/env python3
"""
train_l2_v3.py - Step 7: learned L2 rule that fixes the old model's weak points.

Weak point -> fix
  #1 offline tests on a stream recorded under another rule
       -> LRU-recorded streams only; bench l2_offline.sim (= gem5 victim
          logic, LRU checked against gem5 to < 0.5 %)
  #3 no "where in the tensor" feature
       -> new features from the next-layer hint file:
          pos16      position of the line inside its tensor (16 buckets)
          plan16     L2 hint plan: kept fraction of this tensor x 16 (0 = none)
          over_log2  log2(layer weight bytes / L2 size), clamped -4..8
          inplan     1 if pos < plan fraction (the hint rule's own decision)
          prev_pf    previous access to this line at L2 was a prefetch fill
  #4 trained for accuracy, not misses
       -> model chosen by DEMAND MISSES in the replay, never by accuracy;
          labels = Demand-MIN (only demand reuse counts); sample weight 0
          for decisions that cannot matter (last access of a line)
  #2 (prefetcher) is fixed in gem5, not here.
Split: leave-one-workload-out over the 4 workloads with L2 headroom
       (ViT, BERT, DLRM, RNN-T); ResNet-50 / MobileNetV2 always in training.
Output (new dir): per fold tree + replay results; summary.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.tree import DecisionTreeClassifier, export_text

import l2_offline as B

R = B.RESULTS
RUNS = {
    "vit": ("block_trace_v3/BTR004_BLOCK_vit", "features_blocks_l2pfd_v1/BTR004_BLOCK_vit"),
    "bert": ("block_trace_v4/BTR001_BLOCK_bert", "features_blocks_l2pfd_v2/BTR001_BLOCK_bert"),
    "dlrm": ("block_trace_v3/BTR002_BLOCK_dlrm", "features_blocks_l2pfd_v1/BTR002_BLOCK_dlrm"),
    "rnnt": ("block_trace_v3/BTR003_CHAIN_rnnt_encoder", "features_blocks_l2pfd_v1/BTR003_CHAIN_rnnt_encoder"),
    "rn50": ("block_trace_v3/BTR005_BLOCK_rn50", "features_blocks_l2pfd_v1/BTR005_BLOCK_rn50"),
    "mbv2": ("block_trace_v3/BTR006_BLOCK_mbv2", "features_blocks_l2pfd_v1/BTR006_BLOCK_mbv2"),
}
EVAL = ["vit", "bert", "dlrm", "rnnt"]
OLD = ["role_id", "cmd_id", "tsize_log2", "used_next", "dead", "first_touch",
       "prev_dist_log2", "n_prev", "t_retouch8", "t_dist"]
NEW = ["pos16", "plan16", "over_log2", "inplan", "prev_pf"]


def load(name):
    run, feat = R / RUNS[name][0], R / RUNS[name][1]
    lid = json.loads((run / "meta.json").read_text())["plan"]["layer_id"]
    h = B.load_hints(R / "tensor_hints_v3" / f"{lid}.txt")
    lines, isdem, ispf, opt_hit = B.load_stream(run)
    f = pq.read_table(feat / "l2_features.parquet", columns=OLD + ["opt_keep", "line"]).to_pandas()
    assert (f.line.to_numpy() == lines).all()
    addr = lines * B.LINE
    ten, _ = B.lookup(h, addr)
    pos = B.pos_in_tensor(h, ten, addr)
    plan = B.plan_even(h, B.L2_BYTES, 0.9)
    tt = np.clip(ten, 0, None)
    fr = np.where(ten >= 0, plan[tt], 0.0)
    W = B.layer_weights(h)
    over = np.full(h["n"], -4.0)
    for t in range(h["n"]):
        ws = [W[k] for k in range(32) if h["mask"][t] >> k & 1]
        if ws and max(ws) > 0:
            over[t] = np.clip(np.round(np.log2(max(ws) / B.L2_BYTES)), -4, 8)
    f["pos16"] = np.where(ten >= 0, np.clip((pos * 16).astype(int), 0, 15), -1)
    f["plan16"] = np.round(fr * 16).astype(int)
    f["over_log2"] = np.where(ten >= 0, over[tt], -4)
    f["inplan"] = ((fr > 0) & (pos < fr)).astype(int)
    prev = pd.Series(f.cmd_id.to_numpy()).groupby(lines).shift(1)
    f["prev_pf"] = (prev == 3).astype(int).to_numpy()
    # a line's last access: its decision cannot change any later hit
    nxt_same = pd.Series(lines).duplicated(keep="last").to_numpy()
    f["w"] = nxt_same.astype(float)
    hk = B.hint_keep(h, ten, addr, lines, ispf)
    # compact (memory: 6 runs x ~5 M rows next to running gem5 jobs)
    f = f.drop(columns=["line"])
    for c in OLD + NEW + ["opt_keep"]:
        f[c] = f[c].astype(np.int8)
    f["w"] = f["w"].astype(np.float32)
    del addr, ten, pos, tt, fr, prev
    return dict(f=f, lines=lines, isdem=isdem, ispf=ispf,
                opt_hit=opt_hit.astype(np.int8), hint=hk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="6,8")
    ap.add_argument("--thrs", default="0.3,0.5,0.7")
    ap.add_argument("--per-run", type=int, default=300_000)
    o = ap.parse_args()
    out = Path(o.out)
    if out.exists():
        raise SystemExit(f"{out} exists (outputs are never overwritten)")
    out.mkdir(parents=True)
    rng = np.random.default_rng(0)
    cache = R / "policy_l2_v3_cache"
    cache.mkdir(exist_ok=True)
    for n in RUNS:                      # one run in memory at a time
        fn = cache / f"{n}.npz"
        if not fn.exists():
            d = load(n)
            np.savez(fn, X=d["f"][OLD + NEW].to_numpy(np.int8),
                     y=d["f"].opt_keep.to_numpy(np.int8), w=d["f"].w.to_numpy(np.float32),
                     plan16=d["f"].plan16.to_numpy(np.int8), lines=d["lines"],
                     isdem=d["isdem"], ispf=d["ispf"], opt_hit=d["opt_hit"], hint=d["hint"])
            del d
        print("cached", n, flush=True)

    def get(n):
        z = np.load(cache / f"{n}.npz")
        return {k: z[k] for k in z.files}
    feats = OLD + NEW
    summary = {"features": feats, "folds": {}}
    for test in EVAL:
        Xs, ys = [], []
        for n in RUNS:
            if n == test:
                continue
            z = get(n)
            idx = np.flatnonzero(z["w"] > 0)
            if len(idx) > o.per_run:
                idx = np.sort(rng.choice(idx, o.per_run, replace=False))
            Xs.append(z["X"][idx]); ys.append(z["y"][idx])
            del z
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        del Xs, ys
        d = get(test)
        base = {"lru": B.sim(d["lines"], np.ones(len(d["lines"]), bool), d["isdem"]),
                "hint": B.sim(d["lines"], d["hint"], d["isdem"]),
                "opt": int((d["isdem"] & (d["opt_hit"] == 0)).sum())}
        res = dict(base)
        for depth in map(int, o.depths.split(",")):
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=500,
                                       class_weight="balanced", random_state=0)
            m.fit(Xtr, ytr)
            with open(out / f"tree_{test}_d{depth}.pkl", "wb") as fh:
                pickle.dump(m, fh)
            (out / f"tree_{test}_d{depth}.txt").write_text(export_text(m, feature_names=feats))
            p = m.predict_proba(d["X"])[:, 1]
            for thr in map(float, o.thrs.split(",")):
                keep = (p >= thr) | d["ispf"]
                res[f"tree_d{depth}_t{thr}"] = B.sim(d["lines"], keep, d["isdem"])
                # hybrid: hint plan decides inside planned tensors unless the
                # model is confident, model decides everywhere else
                hyb = np.where(d["plan16"] > 0, d["hint"], keep)
                res[f"hyb_d{depth}_t{thr}"] = B.sim(d["lines"], hyb, d["isdem"])
                print(test, depth, thr, res[f"tree_d{depth}_t{thr}"], res[f"hyb_d{depth}_t{thr}"], flush=True)
        lru, opt = res["lru"], res["opt"]
        summary["folds"][test] = {k: {"misses": v, "room_captured_pct":
                                      round(100 * (lru - v) / (lru - opt), 1) if lru > opt else None}
                                  for k, v in res.items()}
        print(test, json.dumps(summary["folds"][test]), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
