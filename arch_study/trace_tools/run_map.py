#!/usr/bin/env python3
"""
run_map.py - tensor address map for each layer of a trace plan.

For each plan row the "map" build of the kernel (bin/riscv_map, same source,
plus tensor_map_hook.h) runs in gem5 with the same tensor shapes (1 work
unit, mode none, timing CPU: only the allocations matter). It prints one
TENSORMAP line per allocation. Addresses are converted to the ORIGINAL
binary's address space:
  * mmap'ed tensors (large): identical in both binaries (checked);
  * heap tensors (small): shifted by the difference of the initial program
    break, computed from the two ELF files (checked against gem5's brk log).
Output: <results>/<run_id>_<layer_id>/tensor_map.csv
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(HERE))

import run_ai_kernels as R  # noqa: E402  (only its helpers are used)
from tensor_roles import label, tensor_users  # noqa: E402

MAP_BIN = R.KDIR / "bin" / "riscv_map"
MMAP_BASE = 0x4000000000000000
PAGE = 4096


def initial_brk(binary):
    out = subprocess.run(["riscv64-linux-gnu-readelf", "-lW", str(binary)],
                         capture_output=True, text=True, check=True).stdout
    end = 0
    for line in out.splitlines():
        p = line.split()
        if p and p[0] == "LOAD":
            end = max(end, int(p[2], 16) + int(p[5], 16))
    return (end + PAGE - 1) // PAGE * PAGE


def map_one(plan, results_dir):
    run_dir = results_dir / f"{plan['run_id']}_{plan['layer_id']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "tensor_map.csv"
    if out_csv.exists():
        return plan["layer_id"], "exists"
    p = dict(plan, units="1", mode="none")
    kargs = R.kernel_args(p)
    mdir = run_dir / "map_run"
    cmd = [str(R.GEM5), "-d", str(mdir), str(R.PROFILE["sim"]), "--cmd",
           " ".join([str(MAP_BIN / plan["kernel"])] + kargs),
           "--mem-size", plan["mem_size"], "--cpu", "timing"]
    mdir.mkdir(exist_ok=True)
    with (mdir / "simulation.log").open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    allocs = []
    for line in (mdir / "simulation.log").read_text(errors="ignore").splitlines():
        if line.startswith("TENSORMAP "):
            _, i, addr, size = line.split()
            allocs.append((int(i), int(addr, 16), int(size)))
    if rc != 0 or not allocs:
        return plan["layer_id"], f"FAILED rc={rc} allocs={len(allocs)}"
    shift = initial_brk(MAP_BIN / plan["kernel"]) - initial_brk(R.RV_BIN / plan["kernel"])
    names = label(plan["kernel"], plan["args"], len(allocs))
    users = tensor_users(plan["kernel"], plan["args"])
    users += [[]] * (len(allocs) - len(users))
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "name", "role", "start", "end", "bytes", "region", "layers"])
        for (i, addr, size), (name, role), lay in zip(allocs, names, users):
            region = "mmap" if addr >= MMAP_BASE else "heap"
            start = addr if region == "mmap" else addr - shift
            w.writerow([i, name, role, start, start + size, size, region,
                        " ".join(map(str, lay))])
    return plan["layer_id"], f"OK {len(allocs)} tensors, heap shift {shift}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--workers", type=int, default=4)
    opt = ap.parse_args()
    results = Path(opt.results)
    R.check_output_path(results)
    with open(opt.plan, newline="") as f:
        plan_rows = list(csv.DictReader(f))
    with ThreadPoolExecutor(opt.workers) as ex:
        for layer, msg in ex.map(lambda r: map_one(r, results), plan_rows):
            print(f"{layer:<28} {msg}", flush=True)


if __name__ == "__main__":
    main()
