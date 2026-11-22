#!/usr/bin/env python3
"""run_ai_kernels_flush2.py - as run_ai_kernels_flush.py, but with
sim_config_flush9.py (Step 9b: + L2 plans activations)."""
import run_ai_kernels as R

R.RV_BIN = R.KDIR / "bin" / "riscv_flush"
R.PROFILES["classic"]["sim"] = R.STUDY / "sim_config_flush9.py"
R.PROFILE = R.PROFILES["classic"]
R.COMPILE_FLAGS += " -Iinclude_flush (ROI_FLUSH)"

if __name__ == "__main__":
    R.main()
