1#!/usr/bin/env python3
"""
run_ai_kernels.py - run the model-derived AI kernel plan in gem5 and build a
NEW dataset.

Safety rules:
  * Never writes the 56k baseline: any output whose name starts with
    "tensor_cache_dataset" is refused, and outputs must lie inside arch_study/.
  * The dataset is append-only: completed run_ids are skipped and existing
    rows are never rewritten (so an interrupted run can be resumed).
  * A leftover result directory of an unfinished run is renamed, not deleted.

Per run: gem5 + arch_study/sim_config.py --cmd "<riscv kernel> <args>".
Statistics come from the FIRST stats block (the kernel ROI, delimited by
m5_reset_stats / m5_dump_stats in the kernel).

Usage:
    python run_ai_kernels.py [--plan PLAN] [--out DATASET] [--results DIR]
                             [--workers 4] [--limit N]
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path.home() / "gem5-project"
STUDY = PROJECT / "arch_study"
GEM5_DIR = Path.home() / "gem5"
GEM5 = GEM5_DIR / "build" / "RISCV" / "gem5.opt"
KDIR = STUDY / "ai_kernels"
RV_BIN = KDIR / "bin" / "riscv"
NATIVE_BIN = KDIR / "bin" / "native"

COMPILE_FLAGS = ("riscv64-linux-gnu-gcc -O2 -ffp-contract=off -std=c11 -static "
                 "-march=rv64gc -mabi=lp64d -DGEM5 + m5op.S")
CHECKSUM_RTOL = 1e-5

CACHES = ("l1i", "l1d", "l2", "l3")

# Simulation script and statistic names per system model.
#   classic : arch_study/sim_config.py (Baseline-v1 and P870-D-inspired configs)
#   hifive  : arch_study/hifive_unmatched_sim.py (gem5 RISCVMatchedBoard, no L3)
PROFILES = {
    "classic": {
        "sim": STUDY / "sim_config.py",
        "cpu": "system.cpu",
        "mem": "system.mem_ctrl",
        "caches": {"l1i": "system.cpu.icache", "l1d": "system.cpu.dcache",
                   "l2": "system.l2", "l3": "system.l3"},
        "takes_config_args": True,
    },
    "hifive": {
        "sim": STUDY / "hifive_unmatched_sim.py",
        "cpu": "board.processor.cores.core",
        "mem": "board.memory.mem_ctrl",
        "caches": {"l1i": "board.cache_hierarchy.l1icaches",
                   "l1d": "board.cache_hierarchy.l1dcaches",
                   "l2": "board.cache_hierarchy.l2cache"},
        "takes_config_args": False,
    },
}
PROFILE = PROFILES["classic"]  # set from --profile in main()

PLAN_META = ["run_id", "layer_id", "model", "source", "op", "kernel", "args",
             "impl", "mode", "config_id", "cache_args", "mem_size", "sim_flags",
             "operand_roles", "citation_url", "notes"]

FIELDS = (
    PLAN_META
    + ["label", "dims", "dtype", "work_unit", "units_executed", "units_total",
       "work_fraction", "macs_executed", "macs_total", "input_bytes",
       "weight_bytes", "output_bytes", "footprint_bytes",
       "binary_sha256", "compile_flags", "sim_config_sha256", "gem5_version",
       "status", "checksum_gem5", "checksum_native", "checksum_ok",
       "sim_ticks", "sim_seconds", "committed_insts", "num_cycles", "ipc"]
    + [f"{c}_{m}" for c in CACHES for m in
       ("accesses", "hits", "misses", "miss_rate", "mpki", "mshr_misses",
        "avg_miss_latency_ticks", "writebacks", "replacements")]
    + ["dram_bytes_read", "dram_bytes_written", "dram_read_reqs",
       "dram_write_reqs", "dram_avg_rd_bw_MBps", "dram_avg_wr_bw_MBps",
       "dram_avg_mem_acc_lat_ticks", "l3_to_dram_bytes",
       "host_runtime_s", "peak_rss_mb"]
    + [f"{c}_pf_{m}" for c in CACHES for m in
       ("issued", "useful", "accuracy", "coverage")]
)

PF_STATS = {"issued": "pfIssued", "useful": "pfUseful",
            "accuracy": "accuracy", "coverage": "coverage"}


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gem5_version():
    try:
        return subprocess.run(["git", "-C", str(GEM5_DIR), "describe", "--tags"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def check_output_path(path):
    path = path.resolve()
    if path.name.startswith("tensor_cache_dataset"):
        sys.exit(f"REFUSED: {path.name} is a protected baseline name")
    if STUDY.resolve() not in path.parents:
        sys.exit(f"REFUSED: {path} is outside {STUDY}")
    return path


def kernel_args(row):
    """Positional arguments for the kernel binary, in plan order."""
    a = row["args"].split()
    units = str(row["units"])
    if row["kernel"] == "gemm":
        return a + [units, row["impl"] or "naive", "64", row["mode"]]
    return a + [units, row["mode"]]


def parse_first_block(stats_file):
    """name -> float for the first 'Begin Simulation Statistics' block."""
    values, block = {}, 0
    with open(stats_file, errors="ignore") as f:
        for line in f:
            if "Begin Simulation Statistics" in line:
                block += 1
                if block > 1:
                    break
                continue
            if block != 1 or not line.strip() or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return values, block


def parse_output(text):
    meta, checksum = None, None
    for line in text.splitlines():
        if line.startswith("META ") and meta is None:
            meta = json.loads(line[5:])
        elif line.startswith("CHECKSUM ") and checksum is None:
            checksum = line.split()[1:]
    return meta, checksum


def fmt_rate(value):
    """Format a miss rate; empty when the cache level does not exist."""
    return "n/a" if value == "" else f"{value:.3f}"


def checksums_match(a, b):
    if not a or not b or a[2] != b[2]:
        return False
    s1, a1, s2, a2 = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    tol = CHECKSUM_RTOL * max(abs(a2), 1e-12)
    return abs(s1 - s2) <= tol and abs(a1 - a2) <= tol


def build_row(plan, stats, meta, ck_g, ck_n, host_s, rss_kb, ctx):
    row = {k: plan.get(k, "") for k in PLAN_META}
    for k in ("label", "dims", "dtype", "work_unit", "units_executed",
              "units_total", "work_fraction", "macs_executed", "macs_total",
              "input_bytes", "weight_bytes", "output_bytes", "footprint_bytes"):
        row[k] = meta.get(k, "") if meta else ""
    row.update(ctx)
    g = lambda name: stats.get(name, 0.0)

    cpu, mem = PROFILE["cpu"], PROFILE["mem"]
    insts = g(f"{cpu}.commitStats0.numInsts")
    row["sim_ticks"] = int(g("simTicks"))
    row["sim_seconds"] = g("simSeconds")
    row["committed_insts"] = int(insts)
    row["num_cycles"] = int(g(f"{cpu}.numCycles"))
    row["ipc"] = g(f"{cpu}.ipc")

    for c in CACHES:
        p = PROFILE["caches"].get(c)
        if p is None:  # level not present in this system model
            for m in ("accesses", "hits", "misses", "miss_rate", "mpki", "mshr_misses",
                      "avg_miss_latency_ticks", "writebacks", "replacements",
                      "pf_issued", "pf_useful", "pf_accuracy", "pf_coverage"):
                row[f"{c}_{m}"] = ""
            continue
        acc = g(f"{p}.demandAccesses::total")
        miss = g(f"{p}.demandMisses::total")
        hits = stats.get(f"{p}.demandHits::total", acc - miss)  # omitted when 0
        row[f"{c}_accesses"] = int(acc)
        row[f"{c}_hits"] = int(hits)
        row[f"{c}_misses"] = int(miss)
        row[f"{c}_miss_rate"] = miss / acc if acc else 0.0
        row[f"{c}_mpki"] = 1000.0 * miss / insts if insts else 0.0
        row[f"{c}_mshr_misses"] = int(g(f"{p}.demandMshrMisses::total"))
        row[f"{c}_avg_miss_latency_ticks"] = g(f"{p}.demandAvgMissLatency::total")
        row[f"{c}_writebacks"] = int(g(f"{p}.writebacks::total"))
        row[f"{c}_replacements"] = int(g(f"{p}.replacements"))
        # prefetcher statistics; empty when this cache has no prefetcher
        for m, s in PF_STATS.items():
            v = stats.get(f"{p}.prefetcher.{s}")
            row[f"{c}_pf_{m}"] = "" if v is None else v

    row["dram_bytes_read"] = int(g(f"{mem}.dram.bytesRead::total"))
    row["dram_bytes_written"] = int(g(f"{mem}.bytesWrittenSys"))
    row["dram_read_reqs"] = int(g(f"{mem}.readReqs"))
    row["dram_write_reqs"] = int(g(f"{mem}.writeReqs"))
    row["dram_avg_rd_bw_MBps"] = g(f"{mem}.dram.avgRdBW")
    row["dram_avg_wr_bw_MBps"] = g(f"{mem}.avgWrBWSys")
    row["dram_avg_mem_acc_lat_ticks"] = g(f"{mem}.dram.avgMemAccLat")
    row["l3_to_dram_bytes"] = row["l3_misses"] * 64 if row["l3_misses"] != "" else ""

    row["checksum_gem5"] = " ".join(ck_g) if ck_g else ""
    row["checksum_native"] = " ".join(ck_n) if ck_n else ""
    row["checksum_ok"] = checksums_match(ck_g, ck_n)
    row["host_runtime_s"] = round(host_s, 2)
    row["peak_rss_mb"] = round(rss_kb / 1024.0, 1)
    return row


# ------------------------------------------------------------------
# one experiment
# ------------------------------------------------------------------

def run_one(plan, results_dir, ctx):
    run_dir = results_dir / f"{plan['run_id']}_{plan['layer_id']}"
    if run_dir.exists():  # leftover of an unfinished run: keep it, renamed
        run_dir.rename(run_dir.with_name(run_dir.name + f".stale-{int(time.time())}"))
    run_dir.mkdir(parents=True)

    kbin = RV_BIN / plan["kernel"]
    kargs = kernel_args(plan)
    cmd = [str(GEM5), "-d", str(run_dir), str(PROFILE["sim"]),
           "--cmd", " ".join([str(kbin)] + kargs)]
    if PROFILE["takes_config_args"]:
        cmd += ["--mem-size", plan["mem_size"]]
        if plan.get("sim_flags"):
            cmd += plan["sim_flags"].split()
        if plan["cache_args"]:
            cmd += plan["cache_args"].split()
    elif plan.get("sim_flags") or plan["cache_args"]:
        raise RuntimeError("this profile has a fixed hardware model; "
                           "plan must not set sim_flags or cache_args")

    native = subprocess.run([str(NATIVE_BIN / plan["kernel"])] + kargs,
                            capture_output=True, text=True)
    _, ck_native = parse_output(native.stdout)

    start = time.perf_counter()
    log_path = run_dir / "simulation.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        _, status, rusage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
    host_s = time.perf_counter() - start

    text = log_path.read_text(errors="ignore")
    meta, ck_gem5 = parse_output(text)
    stats_file = run_dir / "stats.txt"
    stats, blocks = parse_first_block(stats_file) if stats_file.exists() else ({}, 0)

    row = build_row(plan, stats, meta, ck_gem5, ck_native, host_s,
                    rusage.ru_maxrss, ctx)
    if proc.returncode != 0 or "Simulation completed." not in text:
        row["status"] = f"FAILED(rc={proc.returncode})"
    elif blocks < 1 or row["committed_insts"] <= 0 or meta is None:
        row["status"] = "FAILED(no ROI stats)"
    elif not row["checksum_ok"]:
        row["status"] = "CHECKSUM_MISMATCH"
    else:
        row["status"] = "SUCCESS"

    (run_dir / "meta.json").write_text(json.dumps(
        {"plan": plan, "kernel_meta": meta, "command": cmd,
         "status": row["status"]}, indent=2))
    return row


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(KDIR / "experiment_plan_ai_pilot_v1.csv"))
    ap.add_argument("--out", default=str(STUDY / "datasets" / "ai_kernels_pilot_v1.csv"))
    ap.add_argument("--results", default=str(STUDY / "results" / "ai_pilot_v1"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="run at most N new experiments")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="classic",
                    help="system model: classic (sim_config.py) or hifive (RISCVMatchedBoard)")
    opt = ap.parse_args()

    global PROFILE
    PROFILE = PROFILES[opt.profile]

    out = check_output_path(Path(opt.out))
    results_dir = check_output_path(Path(opt.results))
    out.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(opt.plan, newline="") as f:
        plan_rows = list(csv.DictReader(f))

    done = set()
    if out.exists():
        with out.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                sys.exit(f"REFUSED: {out} has a different column layout; use a new --out")
            done = {r["run_id"] for r in reader if r["status"] == "SUCCESS"}

    todo = [r for r in plan_rows if r["run_id"] not in done]
    if opt.limit > 0:
        todo = todo[:opt.limit]

    ctx = {
        "binary_sha256": "",
        "compile_flags": COMPILE_FLAGS,
        "sim_config_sha256": sha256(PROFILE["sim"]),
        "gem5_version": gem5_version(),
    }
    bin_hash = {k: sha256(RV_BIN / k) for k in {r["kernel"] for r in todo}}

    print("=" * 72)
    print("MODEL-DERIVED AI KERNELS - gem5 runner")
    print("=" * 72)
    print(f"Profile     : {opt.profile} ({PROFILE['sim'].name})")
    print(f"Plan        : {opt.plan} ({len(plan_rows)} runs)")
    print(f"Dataset     : {out}")
    print(f"Already done: {len(done)}   To run now: {len(todo)}   Workers: {opt.workers}")
    print()

    new_file = not out.exists()
    lock = threading.Lock()
    counts = {"SUCCESS": 0, "other": 0}
    t0 = time.perf_counter()

    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
            f.flush()

        def task(plan):
            c = dict(ctx, binary_sha256=bin_hash[plan["kernel"]])
            return run_one(plan, results_dir, c)

        with ThreadPoolExecutor(max_workers=opt.workers) as ex:
            futures = {ex.submit(task, p): p for p in todo}
            for fut in as_completed(futures):
                plan = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # keep going; record nothing for this run
                    print(f"{plan['run_id']} {plan['layer_id']:<15} ERROR {exc!r}", flush=True)
                    counts["other"] += 1
                    continue
                with lock:
                    writer.writerow(row)
                    f.flush()
                key = "SUCCESS" if row["status"] == "SUCCESS" else "other"
                counts[key] += 1
                print(f"{row['run_id']} {row['layer_id']:<15} {row['status']:<18} "
                      f"insts={row['committed_insts']:>11} ipc={row['ipc']:.3f} "
                      f"L1D_mr={row['l1d_miss_rate']:.4f} L2_mr={row['l2_miss_rate']:.3f} "
                      f"L3_mr={fmt_rate(row['l3_miss_rate'])} host={row['host_runtime_s']:.0f}s "
                      f"rss={row['peak_rss_mb']:.0f}MB", flush=True)

    print()
    print(f"Finished in {(time.perf_counter() - t0) / 60:.1f} min: "
          f"{counts['SUCCESS']} SUCCESS, {counts['other']} not successful")
    print(f"Dataset: {out}")


if __name__ == "__main__":
    main()
