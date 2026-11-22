#!/usr/bin/env python3
"""flush_env_u.py - as flush_env.py, but for the Step 8 UNSEEN workloads:
binaries from ai_kernels/bin/riscv_flush_u and riscv_map_flush_u
(block_unseen.c), so the original block binary and its hints stay untouched.
Usage: flush_env_u.py MODULE [args...]"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import run_ai_kernels as R  # noqa: E402

R.RV_BIN = R.KDIR / "bin" / "riscv_flush_u"
R.PROFILES["classic"]["sim"] = R.STUDY / "sim_config_flush6.py"
R.PROFILE = R.PROFILES["classic"]

name = sys.argv[1]
mod = importlib.import_module(name)
if hasattr(mod, "MAP_BIN"):
    mod.MAP_BIN = R.KDIR / "bin" / "riscv_map_flush_u"
if hasattr(mod, "WRAPPER"):
    mod.WRAPPER = HERE / "sim_config_ptable_flush.py"
if hasattr(mod, "BIN"):
    mod.BIN = R.KDIR / "bin" / "riscv_flush_u"
sys.argv = [name] + sys.argv[2:]
sys.exit(mod.main())
