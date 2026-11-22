#!/usr/bin/env python3
"""Make the test plans point at THIS computer's folders.
Writes NEW files only (never overwrites):
  ai_kernels/experiment_plan_smoke_test.csv   (9 runs: LRU, DRRIP, v9 on 3 layers)
  ai_kernels/experiment_plan_v9_full_local.csv (all 72 v9 runs)"""
from pathlib import Path
import sys
STUDY = Path(__file__).resolve().parent.parent
PROJ = STUDY.parent
if PROJ != Path.home() / "gem5-project":
    sys.exit(f"Please put the project at {Path.home()/'gem5-project'} (it is at {PROJ}). The run scripts expect that place.")
# Unzipping on Windows ("Extract All") or with Python drops the Unix execute
# permission, so gem5 could not start the workload programs. Restore it.
import stat
nx = 0
for d in ("ai_kernels/bin", "results"):
    for f in (STUDY / d).rglob("*"):
        if f.is_file() and (f.parent.parent.name == "bin" or f.suffix == ".sh"):
            m = f.stat().st_mode
            if not m & stat.S_IXUSR:
                f.chmod(m | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH); nx += 1
print("made executable:", nx, "files")
jobs = [(STUDY/"testing/experiment_plan_smoke_test_TEMPLATE.csv", STUDY/"ai_kernels/experiment_plan_smoke_test.csv", "__PROJECT__"),
        (STUDY/"ai_kernels/experiment_plan_v9_full.csv", STUDY/"ai_kernels/experiment_plan_v9_full_local.csv", "/home/lord/gem5-project")]
for src, dst, old in jobs:
    if dst.exists():
        print("exists, kept:", dst); continue
    dst.write_text(src.read_text().replace(old, str(PROJ)))
    print("written:", dst)
missing = [p for p in (STUDY/"results/tensor_hints_v3", STUDY/"results/tensor_hints_layers", Path.home()/"gem5/build/RISCV/gem5.opt") if not p.exists()]
print("MISSING:", *missing, sep="\n  ") if missing else print("all needed files found - ready to run")
