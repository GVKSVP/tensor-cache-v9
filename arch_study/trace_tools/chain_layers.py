#!/usr/bin/env python3
"""
chain_layers.py - add layer and next-layer features to the labelled traces of
the layer-chain runs.

1. Layer boundaries: every chain layer is its own function layer_<p>_<k>_<name>
   in bin/riscv/chain. The ELF symbol table gives each function's PC range; the
   first ROI tick at which an L2 *data* request (address inside a tensor)
   carries a PC of layer k marks the start of layer k (requests without a PC, e.g. writebacks, are placed by tick).
2. Features per access (from tensor_map.csv column "layers"):
     layer            layer (1..n) running when the request was made
     used_later       1 if a later layer still uses this tensor
     used_next        1 if the very next layer uses it (the "next-layer hint")
     dead             1 if no later layer uses it (tensor is finished)
Input : <run>/<level>_labels.parquet (belady.py), tensor map (--maps/<run>/)
Output: <run>/<level>_chain_labels.parquet, <run>/chain_layers.json (boundaries
        and a per-layer x per-role summary of LRU / OPT misses and OPT keep rate)
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CHAIN_BIN = HERE.parent / "ai_kernels" / "bin" / "riscv" / "chain"


def layer_ranges(prefix):
    out = subprocess.run(["riscv64-linux-gnu-nm", "-S", "--defined-only", str(CHAIN_BIN)],
                         capture_output=True, text=True, check=True).stdout
    ranges = []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4 and p[3].startswith(f"layer_{prefix}_"):
            k = int(p[3].split("_")[2])
            start, size = int(p[0], 16), int(p[1], 16)
            ranges.append((start, start + size, k, p[3]))
    return sorted(ranges)


def boundaries(l2, ranges, nlayers):
    # data requests only: an early instruction fetch of the next layer's code
    # (wrong-path fetch) must not mark that layer as started
    data = l2["tensor_idx"].to_numpy() >= 0
    pcs = l2["pc"].to_numpy()[data]
    ticks = l2["tick"].to_numpy()[data]
    layer_of_pc = np.zeros(len(pcs), np.int8)
    for lo, hi, k, _ in ranges:
        layer_of_pc[(pcs >= lo) & (pcs < hi)] = k
    starts = {}
    for k in range(1, nlayers + 1):
        idx = np.nonzero(layer_of_pc == k)[0]
        if len(idx):
            starts[k] = int(ticks[idx[0]])
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--maps", required=True)
    ap.add_argument("--levels", default="l2,l3")
    ap.add_argument("--suffix", default="", help="label file suffix, e.g. _pf (belady.py --l2-pf-detect)")
    opt = ap.parse_args()
    import sys
    sys.path.insert(0, str(HERE))
    from tensor_roles import CHAINS

    for run in map(Path, opt.run_dirs):
        plan = json.loads((run / "meta.json").read_text())["plan"]
        chain = plan["args"].split(":")[0]
        prefix, names, _ = CHAINS[chain]
        n = len(names)
        tmap = {int(r["idx"]): [int(x) for x in r["layers"].split()]
                for r in csv.DictReader(open(Path(opt.maps) / run.name / "tensor_map.csv"))}
        l2 = pq.read_table(run / "l2_labels.parquet", columns=["tick", "pc", "tensor_idx"])
        starts = boundaries(l2, layer_ranges(prefix), n)
        bticks = np.array([starts.get(k, np.iinfo(np.int64).max) for k in range(1, n + 1)])
        summary = {"chain": chain, "layers": names,
                   "layer_start_tick": {names[k - 1]: starts.get(k) for k in range(1, n + 1)},
                   "levels": {}}
        sfx = opt.suffix
        for level in opt.levels.split(","):
            pf = pq.ParquetFile(run / f"{level}_labels{sfx}.parquet")
            writer = None
            acc = {}      # key -> [accesses, lru_miss, opt_miss, keep]
            for b in pf.iter_batches(batch_size=1_000_000):
                t = pa.Table.from_batches([b])
                ticks = t["tick"].to_numpy()
                layer = np.searchsorted(bticks, ticks, side="right").astype(np.int8)
                layer[layer == 0] = 1
                tidx = t["tensor_idx"].to_numpy()
                used_later = np.zeros(len(t), np.uint8)
                used_next = np.zeros(len(t), np.uint8)
                dead = np.zeros(len(t), np.uint8)
                for i, users in tmap.items():
                    m = tidx == i
                    if not m.any() or not users:
                        continue
                    lay = layer[m]
                    last = max(users)
                    used_later[m] = lay < last
                    used_next[m] = np.isin(lay + 1, users)
                    dead[m] = lay >= last
                t = t.append_column("layer", pa.array(layer))
                t = t.append_column("used_later", pa.array(used_later))
                t = t.append_column("used_next", pa.array(used_next))
                t = t.append_column("dead", pa.array(dead))
                if writer is None:
                    writer = pq.ParquetWriter(run / f"{level}_chain_labels{sfx}.parquet", t.schema,
                                              compression="zstd")
                writer.write_table(t)
                roles = t["role"].to_numpy(zero_copy_only=False).astype(str)
                opt_hit, lru_hit, keep = (t[c].to_numpy() for c in ("opt_hit", "lru_hit", "opt_keep"))
                groups = []
                for k in range(1, n + 1):
                    for r in set(roles):
                        groups.append((f"{k}:{names[k - 1]}:{r}", (layer == k) & (roles == r)))
                for flag, arr in (("used_next", used_next), ("dead", dead)):
                    for v in (0, 1):
                        groups.append((f"{flag}={v}", (arr == v) & (tidx >= 0)))
                for key, m in groups:
                    c = int(m.sum())
                    if not c:
                        continue
                    a = acc.setdefault(key, [0, 0, 0, 0])
                    a[0] += c
                    a[1] += int((lru_hit[m] == 0).sum())
                    a[2] += int((opt_hit[m] == 0).sum())
                    a[3] += int(keep[m].sum())
            if writer is not None:
                writer.close()
            lv = {}
            for key in sorted(acc, key=lambda x: (not x[0].isdigit(), x)):
                a = acc[key]
                if a[0] < 50:
                    continue
                lv[key] = {"accesses": a[0], "lru_misses": a[1], "opt_misses": a[2],
                           "opt_keep_pct": round(100 * a[3] / a[0], 1)}
            summary["levels"][level] = lv
        (run / f"chain_layers{opt.suffix}.json").write_text(json.dumps(summary, indent=2))
        print(run.name, "layer starts:", summary["layer_start_tick"], flush=True)


if __name__ == "__main__":
    main()
