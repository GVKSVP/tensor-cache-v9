"""
sim_config_ptable.py - gem5 config wrapper for Step-3 trace runs.

Runs sim_config.py unchanged (same arguments), then writes a gem5 checkpoint
after the program has exited, only to obtain the SE-mode page table
(virtual page -> physical page) from m5.cpt. The caches see physical
addresses, the tensor map has virtual ones; this table links the two.
"""
import os
import runpy
import sys

import m5

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "sim_config.py")
sys.argv[0] = SIM
runpy.run_path(SIM, run_name="__m5_main__")
ckpt = os.path.join(m5.options.outdir, "ptable_ckpt")
m5.checkpoint(ckpt)
print(f"page-table checkpoint written to {ckpt}")
