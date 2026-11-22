"""
hifive_unmatched_sim.py - run a static RISC-V binary on gem5's built-in
model of the SiFive HiFive Unmatched board (RISCVMatchedBoard).

The board's hardware model is used UNMODIFIED from the gem5 standard library
(src/python/gem5/prebuilt/riscvmatched/). It was calibrated against the real
board by the gem5 developers ("Validating Hardware and SimPoints with gem5:
A RISC-V Board Case Study", UC Davis, 2023).

Model contents (from the gem5 source, citing the FU740-C000 datasheet):
    CPU    : SiFive U74 core as an in-order MinorCPU (dual issue), 1.2 GHz,
             1 core in SE mode (the real FU740 has 4x U74 + 1x S7)
    Caches : L1I 32 KiB 4-way, L1D 32 KiB 8-way, shared L2 2 MiB 16-way
    Memory : single-channel DDR4-2400, 16 GiB

Usage:
    gem5.opt -d OUT hifive_unmatched_sim.py --cmd "<binary> <args...>"
"""

import argparse
import os
import shlex

from gem5.prebuilt.riscvmatched.riscvmatched_board import RISCVMatchedBoard
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument("--cmd", required=True, help='"<binary> <args...>"')
opt = parser.parse_args()

parts = shlex.split(opt.cmd)
binary = os.path.abspath(parts[0])
if not os.path.isfile(binary):
    raise SystemExit(f"ERROR: binary not found: {binary}")

board = RISCVMatchedBoard(clk_freq="1.2GHz", l2_size="2MiB", is_fs=False)

# Host-side setting only (does not change the simulated hardware): back the
# board's 16 GiB of simulated memory lazily, because the host PC has 7 GB RAM.
board.mmap_using_noreserve = True
board.set_se_binary_workload(
    BinaryResource(local_path=binary),
    arguments=parts[1:],
)

print("=" * 60)
print("HiFive Unmatched model (gem5 RISCVMatchedBoard, unmodified)")
print("=" * 60)
print("CPU             : SiFive U74 (MinorCPU) @ 1.2GHz, 1 core (SE mode)")
print("Caches          : L1I 32KiB/4, L1D 32KiB/8, L2 2MiB/16")
print("Memory          : DDR4-2400, 16GiB")
print("Binary          :", binary)
print("Arguments       :", parts[1:])
print("=" * 60)

simulator = Simulator(board=board)
simulator.run()

print("=" * 60)
print("Simulation completed.")
print("=" * 60)
