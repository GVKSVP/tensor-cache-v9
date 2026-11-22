#!/usr/bin/env python3
"""run_ai_kernels_flush.py - same runner as run_ai_kernels.py, but with the
ROI_FLUSH kernels (ai_kernels/bin/riscv_flush) and sim_config_flush.py, which
write back and invalidate every cache right before the ROI (Step 1 fair re-test).
run_ai_kernels.py itself is not changed."""
import run_ai_kernels as R

R.RV_BIN = R.KDIR / "bin" / "riscv_flush"
R.PROFILES["classic"]["sim"] = R.STUDY / "sim_config_flush.py"
R.PROFILE = R.PROFILES["classic"]
R.COMPILE_FLAGS += " -Iinclude_flush (ROI_FLUSH)"

if __name__ == "__main__":
    R.main()
