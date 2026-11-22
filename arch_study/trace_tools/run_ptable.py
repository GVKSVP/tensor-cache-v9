#!/usr/bin/env python3
"""
run_ptable.py - page table (virtual page -> physical page) for each layer of a
trace plan.

The caches (and so the L2/L3 traces) see physical addresses. In gem5 SE mode a
physical page is assigned when the program first touches a virtual page, in
program order, so the mapping does not depend on the CPU model. It is
therefore taken from a run of the ORIGINAL binary with the SAME arguments on
the simple timing CPU, which (unlike the O3 CPU) can be checkpointed after the
program exits. The checkpoint's m5.cpt holds the page table; the memory image
(*.pmem) is deleted. Correctness is checked later: every ROI access of the O3
trace must hit a page in this table (extract_roi.py reports the coverage).

Output: <results>/<run_id>_<layer_id>/ptable_ckpt/m5.cpt
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_ai_kernels as R  # noqa: E402

WRAPPER = HERE / "sim_config_ptable.py"


def ptable_one(plan, results):
    run_dir = results / f"{plan['run_id']}_{plan['layer_id']}"
    cpt = run_dir / "ptable_ckpt" / "m5.cpt"
    if cpt.exists():
        return plan["layer_id"], "exists"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(R.GEM5), "-d", str(run_dir), str(WRAPPER), "--cmd",
           " ".join([str(R.RV_BIN / plan["kernel"])] + R.kernel_args(plan)),
           "--mem-size", plan["mem_size"], "--cpu", "timing"]
    with (run_dir / "ptable_run.log").open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    for pmem in (run_dir / "ptable_ckpt").glob("*.pmem"):
        pmem.unlink()
    if rc != 0 or not cpt.exists():
        return plan["layer_id"], f"FAILED rc={rc}"
    n = sum(1 for line in cpt.open() if line.startswith("paddr="))
    return plan["layer_id"], f"OK {n} pages"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--workers", type=int, default=4)
    opt = ap.parse_args()
    results = Path(opt.results)
    R.check_output_path(results)
    with open(opt.plan, newline="") as f:
        rows = list(csv.DictReader(f))
    with ThreadPoolExecutor(opt.workers) as ex:
        for layer, msg in ex.map(lambda r: ptable_one(r, results), rows):
            print(f"{layer:<28} {msg}", flush=True)


if __name__ == "__main__":
    main()
