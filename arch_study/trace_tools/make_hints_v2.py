#!/usr/bin/env python3
"""
make_hints_v2.py - COPY of make_hints.py generalised for Step 4 (make_hints.py
is unchanged): tensor hint files for the multi-layer workloads run on the
ROI_FLUSH setup.
  * kernels "chain" (src/chain.c) and "block" (src/block.c) both get layer
    PC ranges (L lines) and per-tensor layer masks (T lines);
  * PCs come from the binary that actually ran (ai_kernels/bin/riscv_flush),
    and the program break from the same binary.
Same file format as make_hints.py:
  L <k> <pc_lo hex> <pc_hi hex>; T <tensor idx> <layer mask> <bytes>;
  R <pa_lo hex> <pa_hi hex> <tensor> <role_id>
Usage: make_hints_v2.py --out results/tensor_hints_v2 RUN_DIR...
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from build_features import ROLES  # noqa: E402,F401
from extract_roi import PAGE_SHIFT, load_map, load_ptable  # noqa: E402
from make_hints import page_segments  # noqa: E402
from run_map import initial_brk  # noqa: E402
from tensor_roles import BLOCKS, CHAINS  # noqa: E402
import run_ai_kernels as R  # noqa: E402

BIN = R.KDIR / "bin" / "riscv_flush"


def layer_ranges(binary, prefix):
    out = subprocess.run(["riscv64-linux-gnu-nm", "-S", "--defined-only", str(binary)],
                         capture_output=True, text=True, check=True).stdout
    rng = []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4 and p[3].startswith(f"layer_{prefix}_"):
            k = int(p[3].split("_")[2])
            start, size = int(p[0], 16), int(p[1], 16)
            rng.append((start, start + size, k, p[3]))
    return sorted(rng)


def make(run, out_dir):
    plan = json.loads((run / "meta.json").read_text())["plan"]
    maps = run.parent.parent / (run.parent.name + "_maps") / run.name
    kern = plan["kernel"]
    binary = BIN / kern
    brk0 = initial_brk(binary)
    p2v = load_ptable(maps / "ptable_ckpt" / "m5.cpt")
    rows, starts = load_map(maps / "tensor_map.csv", brk0)
    allrows = list(csv.DictReader(open(maps / "tensor_map.csv")))
    lines = [f"# tensor hints v2 for {plan['layer_id']} ({run.name}, {binary}); roles: "
             + " ".join(f"{i}={r}" for i, r in enumerate(ROLES))]
    multi = kern in ("chain", "block")
    if multi:
        table = CHAINS if kern == "chain" else BLOCKS
        prefix = table[plan["args"].split(":")[0]][0]
        lr = layer_ranges(binary, prefix)
        assert lr, f"no layer_{prefix}_* symbols in {binary}"
        for lo, hi, k, _ in lr:
            lines.append(f"L {k} {lo:x} {hi:x}")
    for r in allrows:
        users = [int(x) for x in r.get("layers", "").split()] if multi else [1]
        mask = sum(1 << (k - 1) for k in users)
        lines.append(f"T {r['idx']} {mask} {r['bytes']}")
    out_ranges = []
    for ppage in sorted(p2v):
        for lo, hi, t, rid in page_segments(p2v[ppage], rows, starts, brk0):
            a, b = (ppage << PAGE_SHIFT) + lo, (ppage << PAGE_SHIFT) + hi
            if out_ranges and out_ranges[-1][1] == a and out_ranges[-1][2:] == (t, rid):
                out_ranges[-1][1] = b
            else:
                out_ranges.append([a, b, t, rid])
    lines += [f"R {a:x} {b:x} {t} {rid}" for a, b, t, rid in out_ranges]
    dest = out_dir / f"{plan['layer_id']}.txt"
    dest.write_text("\n".join(lines) + "\n")
    return f"{plan['layer_id']}: {len(lr) if multi else 0} layers, {len(p2v)} pages -> {len(out_ranges)} ranges"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    opt = ap.parse_args()
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in map(Path, opt.run_dirs):
        print(make(run, out), flush=True)


if __name__ == "__main__":
    main()
