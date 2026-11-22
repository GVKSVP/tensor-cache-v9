#!/usr/bin/env python3
"""
make_ai_plan_v3.py - build experiment plans from layers_v3.csv
(full 32-head Llama-3.1-8B 4k decode attention + Stable Diffusion XL,
the unseen test model).

Same as make_ai_plan_v2.py (v2 kernel costs, same options and checks);
only the layer list changes.
"""

import make_ai_plan_v2  # noqa: F401  (installs the v2 kernel costs)
import make_ai_plan as base

base.LAYERS = base.HERE / "layers_v3.csv"

if __name__ == "__main__":
    base.main()
