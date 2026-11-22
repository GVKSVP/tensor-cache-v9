#!/usr/bin/env python3
"""run_ai_kernels_flush6u.py - Step 8 UNSEEN workloads: binaries from
bin/riscv_flush_u (block_unseen.c) + sim_config_flush6.py."""
import run_ai_kernels as R

R.RV_BIN = R.KDIR / "bin" / "riscv_flush_u"
R.PROFILES["classic"]["sim"] = R.STUDY / "sim_config_flush6.py"
R.PROFILE = R.PROFILES["classic"]
R.COMPILE_FLAGS += " -Iinclude_flush (ROI_FLUSH) block_unseen.c"

if __name__ == "__main__":
    R.main()
