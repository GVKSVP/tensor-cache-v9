#!/usr/bin/env python3
"""run_ai_kernels_flush14u.py - UNSEEN workloads with the FINAL v9 setup:
binaries from bin/riscv_flush_u (block_unseen.c) + sim_config_flush14.py
(the same config as every final result). New file; nothing else changed."""
import run_ai_kernels as R

R.RV_BIN = R.KDIR / "bin" / "riscv_flush_u"
R.NATIVE_BIN = R.KDIR / "bin" / "native_u"   # reference checksums: native build of block_unseen.c
R.PROFILES["classic"]["sim"] = R.STUDY / "sim_config_flush14.py"
R.PROFILE = R.PROFILES["classic"]
R.COMPILE_FLAGS += " -Iinclude_flush (ROI_FLUSH) block_unseen.c"

if __name__ == "__main__":
    R.main()
