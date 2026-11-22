#!/usr/bin/env python3
"""
make_hints.py - Step 5: tensor hint table for the gem5 TensorAwareRP policy,
one text file per layer / chain, built from what the trace pipeline already
produced (no new simulation):
  tensor_map.csv   (run_map.py)    virtual ranges, role, layers using each tensor
  ptable_ckpt/m5.cpt (run_ptable.py) physical page -> virtual page
  ELF symbols of bin/riscv/chain   PC ranges of the layer functions (chains)
Every mapped physical page is classified exactly as extract_roi.py does
(tensor / program_image / stack / other), so the policy sees the same role
and tensor ids as the training features. Lines:
  L <k> <pc_lo hex> <pc_hi hex>        layer k code range (chains only)
  T <tensor idx> <layer mask> <bytes>  bit k-1 set = layer k uses the tensor
  R <pa_lo hex> <pa_hi hex> <tensor> <role_id>
Usage: make_hints.py --out results/tensor_hints_v1 RUN_DIR...
       (maps are taken from <results>/<tag>_maps/<run>/)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from build_features import ROLES  # noqa: E402
from extract_roi import PAGE_SHIFT, classify, load_map, load_ptable  # noqa: E402
from run_map import initial_brk  # noqa: E402
import run_ai_kernels as R  # noqa: E402

PAGE = 1 << PAGE_SHIFT
LINE = 64


def role_id(role):
    return ROLES.index(role) if role in ROLES else len(ROLES) - 1


def page_segments(vpage, rows, starts, brk0):
    """split one virtual page into (offset_lo, offset_hi, tensor, role_id) pieces"""
    v0 = vpage << PAGE_SHIFT
    cuts = {0, PAGE}
    for r in rows:
        for a in (int(r["start"]), int(r["end"])):
            if v0 < a < v0 + PAGE:
                cuts.add((a - v0) // LINE * LINE)    # tensors are line aligned in use
    cuts = sorted(cuts)
    segs = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        t, _, role = classify(v0 + lo, rows, starts, brk0)
        if segs and segs[-1][2:] == (t, role_id(role)) and segs[-1][1] == lo:
            segs[-1] = (segs[-1][0], hi, t, role_id(role))
        else:
            segs.append((lo, hi, t, role_id(role)))
    return segs


def make(run, out_dir):
    plan = json.loads((run / "meta.json").read_text())["plan"]
    maps = run.parent.parent / (run.parent.name + "_maps") / run.name
    brk0 = initial_brk(R.RV_BIN / plan["kernel"])
    p2v = load_ptable(maps / "ptable_ckpt" / "m5.cpt")
    rows, starts = load_map(maps / "tensor_map.csv", brk0)
    allrows = list(csv.DictReader(open(maps / "tensor_map.csv")))
    lines = [f"# tensor hints for {plan['layer_id']} ({run.name}); roles: "
             + " ".join(f"{i}={r}" for i, r in enumerate(ROLES))]
    if plan["kernel"] == "chain":
        from chain_layers import layer_ranges
        from tensor_roles import CHAINS
        prefix = CHAINS[plan["args"].split(":")[0]][0]
        for lo, hi, k, name in layer_ranges(prefix):
            lines.append(f"L {k} {lo:x} {hi:x}")
    for r in allrows:
        users = [int(x) for x in r.get("layers", "").split()] if plan["kernel"] == "chain" else [1]
        mask = sum(1 << (k - 1) for k in users)
        lines.append(f"T {r['idx']} {mask} {r['bytes']}")
    # physical ranges, merged across consecutive pages
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
    return f"{plan['layer_id']}: {len(p2v)} pages -> {len(out_ranges)} ranges"


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
