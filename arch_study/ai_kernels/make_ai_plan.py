#!/usr/bin/env python3
"""
make_ai_plan.py - build the P-C pilot plan from layers.csv.

For every layer, choose the number of work units so that the computed work
is close to a MAC budget (default 20 M). Layers smaller than the budget run
in full. Pooling and embedding layers are cheap and always run in full.

Output: a NEW plan CSV (refuses to overwrite an existing file).

Usage:
    python make_ai_plan.py [--budget 20e6] [--mode cold]
                           [--out experiment_plan_ai_pilot_v1.csv]
"""

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYERS = HERE / "layers.csv"

PLAN_FIELDS = [
    "run_id", "layer_id", "model", "source", "op", "kernel", "args",
    "units", "units_total", "work_unit", "macs_per_unit", "macs_planned",
    "work_fraction_planned", "impl", "mode", "config_id", "cache_args",
    "mem_size", "sim_flags", "operand_roles", "citation_url", "notes",
]


def units_and_cost(kernel, args):
    """Return (units_total, macs_per_unit) for a layer."""
    a = args.split()
    if kernel == "gemm":
        M, N, K = map(int, a[:3])
        return M, K * N
    if kernel == "conv2d_nchw":
        C, H, W, K, R, S, st, p = map(int, a[:8])
        OH = (H + 2 * p - R) // st + 1
        OW = (W + 2 * p - S) // st + 1
        return K, C * R * S * OH * OW
    if kernel == "attention":
        S, D, heads = map(int, a[:3])
        return heads, 2 * S * S * D
    if kernel == "pool":
        C, H, W, R, st, p = map(int, a[1:7])
        OH = (H + 2 * p - R) // st + 1
        OW = (W + 2 * p - R) // st + 1
        return C, OH * OW * R * R
    if kernel == "embedding":
        if a[0] == "gather":
            V, D, L = map(int, a[1:4])
            return L, D
        B = int(a[1])
        return B, 214 * 128
    if kernel == "mlp":
        dims = [int(x) for x in a[0].split(",")]
        B = int(a[1])
        return B, sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    raise ValueError(f"unknown kernel {kernel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=20e6)
    ap.add_argument("--mode", default="cold", choices=["cold", "warm", "none"])
    ap.add_argument("--out", default=str(HERE / "experiment_plan_ai_pilot_v1.csv"))
    ap.add_argument("--layers", default="",
                    help="comma-separated layer_ids to include (default: all); "
                         "run_ids keep their full-plan numbering")
    ap.add_argument("--config-id", default="B1_default")
    ap.add_argument("--mem-size", default="512MiB",
                    help="simulated memory size recorded in the plan (and passed to sim_config.py)")
    ap.add_argument("--cache-args", default="",
                    help='"l1_size l1_assoc l2_size l2_assoc l3_size l3_assoc" '
                         "(default: sim_config.py defaults)")
    ap.add_argument("--sim-flags", default="",
                    help='extra sim_config.py flags, e.g. "--freq 2GHz --mem ddr5_4400"')
    opt = ap.parse_args()

    out = Path(opt.out)
    if out.exists():
        sys.exit(f"REFUSED: {out} already exists (new plans need a new filename)")
    if out.name.startswith("tensor_cache_dataset"):
        sys.exit("REFUSED: plan name must not start with tensor_cache_dataset")

    wanted = {x.strip() for x in opt.layers.split(",") if x.strip()}

    rows = []
    with LAYERS.open(newline="") as f:
        all_layers = list(csv.DictReader(f))
    unknown = wanted - {l["layer_id"] for l in all_layers}
    if unknown:
        sys.exit(f"ERROR: unknown layer_id(s): {', '.join(sorted(unknown))}")

    for n, layer in enumerate(all_layers, start=1):
        if wanted and layer["layer_id"] not in wanted:
            continue
        kernel = layer["kernel"]
        total, per_unit = units_and_cost(kernel, layer["args"])
        if kernel in ("pool", "embedding"):
            units = total
        else:
            units = max(1, min(total, math.ceil(opt.budget / per_unit)))
        rows.append({
            "run_id": f"AI{n:04d}",
            "layer_id": layer["layer_id"],
            "model": layer["model"],
            "source": layer["source"],
            "op": layer["op"],
            "kernel": kernel,
            "args": layer["args"],
            "units": units,
            "units_total": total,
            "work_unit": layer["work_unit"],
            "macs_per_unit": per_unit,
            "macs_planned": units * per_unit,
            "work_fraction_planned": f"{units / total:.6f}",
            "impl": "naive" if kernel == "gemm" else "",
            "mode": opt.mode,
            "config_id": opt.config_id,
            "cache_args": opt.cache_args,  # empty = sim_config.py defaults (Baseline-v1)
            "mem_size": opt.mem_size,
            "sim_flags": opt.sim_flags,
            "operand_roles": layer["operand_roles"],
            "citation_url": layer["citation_url"],
            "notes": layer["notes"],
        })

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{'run':<7}{'layer':<15}{'units':>12}{'fraction':>10}{'MACs planned':>14}")
    for r in rows:
        print(f"{r['run_id']:<7}{r['layer_id']:<15}{str(r['units']) + '/' + str(r['units_total']):>12}"
              f"{float(r['work_fraction_planned']):>10.4f}{r['macs_planned']:>14.3g}")
    print(f"\n{len(rows)} runs written to {out}")


if __name__ == "__main__":
    main()
