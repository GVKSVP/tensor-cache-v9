#!/usr/bin/env python3
"""make_block_plan.py - plan for the whole-block workloads (src/block.c, Step 2)
on config F, with the hardware settings copied from the chain plan
(experiment_plan_chain_v1.csv) so the configuration is identical. One row per
(block, replacement rule). Run with run_ai_kernels_flush.py (true cache flush
at ROI begin). Output must not exist.
Usage: make_block_plan.py --out PLAN.csv --blocks bert_block --rules lru,brrip
       [--units N] [--prefix BLK] [--extra-flags "..."]"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
K = Path(__file__).resolve().parent
# block -> (layer_id, model, chain row whose source/citation/hardware is reused, mem)
BLOCKS = {
    "bert_block": ("BLOCK_bert", "BERT-Large", "CHAIN_bert_ffn", "1GiB"),
    "vit_block": ("BLOCK_vit", "ViT-B/16", "CHAIN_vit_mlp", "1GiB"),
    "rn50_block2": ("BLOCK_rn50", "ResNet-50", "CHAIN_rn50_bottleneck", "1GiB"),
    "mbv2_block2": ("BLOCK_mbv2", "MobileNetV2", "CHAIN_mbv2_block", "1GiB"),
    "dlrm_block": ("BLOCK_dlrm", "DLRM-v2", "DLRM_top_mlp", "1GiB"),
    # existing multi-layer chains (src/chain.c), same flush setup
    "chain:rnnt_encoder": ("CHAIN_rnnt_encoder", "RNN-T", "CHAIN_rnnt_encoder", "1GiB"),
    "chain:llama_decode": ("CHAIN_llama_decode", "Llama-3.1-8B", "CHAIN_llama_decode", "2GiB"),
}
HW = "CHAIN_bert_ffn"   # hardware columns (cache_args, sim_flags) come from this row
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--blocks", required=True)
ap.add_argument("--rules", default="lru")
ap.add_argument("--units", type=int, default=0)
ap.add_argument("--prefix", default="BLK")
ap.add_argument("--extra-flags", default="")
ap.add_argument("--hints-dir", default="", help="tensor hint dir (for tensor/tensor_l3 rules)")
o = ap.parse_args()
out = K / o.out
if out.exists():
    sys.exit(f"ERROR: {out} exists (plans are never overwritten)")
with open(K / "experiment_plan_chain_v1.csv", newline="") as f:
    r = csv.DictReader(f); fields = r.fieldnames; chain = {x["layer_id"]: x for x in r}
with open(K / "experiment_plan_trace_v1.csv", newline="") as f:
    for x in csv.DictReader(f):
        chain.setdefault(x["layer_id"], x)
rows, n = [], 0
for b in o.blocks.split(","):
    lid, model, cref, mem = BLOCKS[b]
    kern, arg = b.split(":") if ":" in b else ("block", b)
    p = subprocess.run([str(K / "bin/native" / kern), arg, str(o.units), "none"],
                       capture_output=True, text=True, check=True)
    meta = json.loads(next(l for l in p.stdout.splitlines() if l.startswith("META "))[5:])
    for rule in o.rules.split(","):
        n += 1
        x = {k: chain[cref].get(k, "") for k in fields}
        for k in ("cache_args", "sim_flags", "operand_roles"):
            x[k] = chain[HW][k]
        flags = x["sim_flags"] + f" --repl {rule}" + (f" {o.extra_flags}" if o.extra_flags else "")
        if rule in ("tensor", "tensor_l3"):
            flags += f" --tensor-hints {o.hints_dir}/{lid}.txt"
        x.update(run_id=f"{o.prefix}{n:03d}", layer_id=lid, model=model, op="model_block",
                 kernel=kern, args=arg, units=str(meta["units_executed"]),
                 units_total=str(meta["units_total"]), work_unit=meta["work_unit"],
                 macs_per_unit=str(int(meta["macs_total"] / meta["units_total"])),
                 macs_planned=str(int(meta["macs_executed"])),
                 work_fraction_planned=f"{meta['work_fraction']:.6f}", mode="cold",
                 config_id=f"F_P870D_full_L2stride_repl-{rule}_flush", mem_size=mem,
                 sim_flags=flags,
                 notes=f"whole block ({meta['dims']}), Step 2; true cache flush at ROI begin")
        rows.append(x)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"wrote {out} ({len(rows)} runs)")
