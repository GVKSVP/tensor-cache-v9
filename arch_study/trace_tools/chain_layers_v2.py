#!/usr/bin/env python3
"""chain_layers_v2.py - run chain_layers.py (unchanged) on multi-layer runs of
the ROI_FLUSH setup (Step 5): blocks (src/block.c) as well as chains, with the
layer PC ranges taken from the binary that ran (ai_kernels/bin/riscv_flush).
Usage: chain_layers_v2.py --maps MAPS [--levels l2] [--suffix _pfd] RUN_DIR"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chain_layers  # noqa: E402
import tensor_roles  # noqa: E402

run = Path(sys.argv[-1])
plan = json.loads((run / "meta.json").read_text())["plan"]
chain_layers.CHAIN_BIN = HERE.parent / "ai_kernels" / "bin" / "riscv_flush" / plan["kernel"]
tensor_roles.CHAINS = {**tensor_roles.CHAINS, **tensor_roles.BLOCKS}
sys.argv = ["chain_layers.py"] + sys.argv[1:]
chain_layers.main()
