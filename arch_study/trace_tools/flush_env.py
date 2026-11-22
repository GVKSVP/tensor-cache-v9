#!/usr/bin/env python3
"""flush_env.py - run a trace_tools script against the ROI_FLUSH setup
(ai_kernels/bin/riscv_flush kernels + sim_config_flush.py, Step 1/2) without
changing the script. Usage: flush_env.py MODULE [args...]
  e.g. flush_env.py run_map --plan P --results R
The module's main() is called after run_ai_kernels (R) is patched:
  R.RV_BIN -> bin/riscv_flush, classic profile sim -> sim_config_flush.py,
  run_map MAP_BIN -> bin/riscv_map_flush, run_ptable WRAPPER -> sim_config_ptable_flush.py."""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import run_ai_kernels as R  # noqa: E402

R.RV_BIN = R.KDIR / "bin" / "riscv_flush"
R.PROFILES["classic"]["sim"] = R.STUDY / "sim_config_flush.py"
R.PROFILE = R.PROFILES["classic"]

name = sys.argv[1]
mod = importlib.import_module(name)
if hasattr(mod, "MAP_BIN"):
    mod.MAP_BIN = R.KDIR / "bin" / "riscv_map_flush"
if hasattr(mod, "WRAPPER"):
    mod.WRAPPER = HERE / "sim_config_ptable_flush.py"
sys.argv = [name] + sys.argv[2:]
sys.exit(mod.main())
