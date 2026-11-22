#!/usr/bin/env python3
"""
make_chain_plan.py - plan for the layer chains (src/chain.c) on config F
(P870-D-inspired + L2 stride prefetcher), with the hardware settings copied
from the existing LRU plans so the configuration is identical.

Work metadata (units, MACs) is read from the chain binary's own META line
(native run, mode none). Source/citation fields are copied from the
single-layer rows of the same model. Output must not exist yet.

Usage: make_chain_plan.py --out PLAN.csv [--small] [--extra-flags "--trace l2l3"]
                          [--prefix CHN]
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from make_repl_plan import DEFAULT_SOURCES

KDIR = Path(__file__).resolve().parent

# chain -> (model, single layer whose source/citation is reused, mem size)
CHAINS = [
    ("bert_ffn", "BERT-Large", "BERT_ffn1", "1GiB"),
    ("vit_mlp", "ViT-B/16", "VIT_mlp_up", "1GiB"),
    ("rn50_bottleneck", "ResNet-50", "RN50_c2_3x3", "1GiB"),
    ("mbv2_block", "MobileNetV2", "MBV2_expand_56", "1GiB"),
    ("rnnt_encoder", "RNN-T", "RNNT_enc_pre1", "1GiB"),
    ("llama_decode", "Llama-3.1-8B", "LLAMA8B_q_proj", "2GiB"),
]


def meta(arg):
    p = subprocess.run([str(KDIR / "bin" / "native" / "chain"), arg, "0", "none"],
                       capture_output=True, text=True, check=True)
    line = next(l for l in p.stdout.splitlines() if l.startswith("META "))
    return json.loads(line[5:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--small", action="store_true", help="tiny shapes (smoke test)")
    ap.add_argument("--extra-flags", default="")
    ap.add_argument("--prefix", default="CHN")
    opt = ap.parse_args()
    out = Path(opt.out)
    if not out.is_absolute():
        out = KDIR / out
    if out.exists():
        sys.exit(f"ERROR: {out} already exists (plans are never overwritten)")

    base, fields = {}, None
    for src in DEFAULT_SOURCES:
        with open(KDIR / src, newline="") as f:
            reader = csv.DictReader(f)
            fields = fields or reader.fieldnames
            for r in reader:
                base[r["layer_id"]] = r

    plan = []
    for i, (chain, model, ref_layer, mem) in enumerate(CHAINS, 1):
        arg = chain + (":small" if opt.small else "")
        m = meta(arg)
        ref = base[ref_layer]
        row = {k: "" for k in fields}
        row.update({
            "run_id": f"{opt.prefix}{i:03d}",
            "layer_id": f"CHAIN_{chain}" + ("_SMOKE" if opt.small else ""),
            "model": model, "source": ref["source"], "op": "layer_chain",
            "kernel": "chain", "args": arg,
            "units": m["units_executed"], "units_total": m["units_total"],
            "work_unit": m["work_unit"],
            "macs_per_unit": round(m["macs_executed"] / max(m["units_executed"], 1)),
            "macs_planned": round(m["macs_executed"]),
            "work_fraction_planned": f"{m['work_fraction']:.6f}",
            "impl": "", "mode": "cold", "config_id": ref["config_id"],
            "cache_args": ref["cache_args"], "mem_size": mem,
            "sim_flags": (ref["sim_flags"] + " " + opt.extra_flags).strip(),
            "operand_roles": "see trace_tools/tensor_roles.py CHAINS",
            "citation_url": ref["citation_url"],
            "notes": f"layer chain ({m['dims']}); consecutive layers share tensors"
                     + ("; SMOKE TEST (tiny shapes)" if opt.small else ""),
        })
        plan.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(plan)
    for r in plan:
        print(f"{r['run_id']} {r['layer_id']:<28} units {r['units']}/{r['units_total']} "
              f"MACs {int(r['macs_planned']):>12,} mem {r['mem_size']}")
    print(f"wrote {out}: {len(plan)} chains")


if __name__ == "__main__":
    main()
