#!/usr/bin/env python3
"""
make_ai_plan_v4.py - build experiment plans from layers_v4.csv
(RNN-T, R-GAT, ViT-B/16). Reuses make_ai_plan.py / make_ai_plan_v2.py
unchanged and adds the work-unit costs of lstm and gat_aggregate.
"""

import make_ai_plan_v2
import make_ai_plan as base

base.LAYERS = base.HERE / "layers_v4.csv"
_v2_units_and_cost = make_ai_plan_v2.units_and_cost


def units_and_cost(kernel, args):
    """Return (units_total, macs_per_unit), including the v4 kernels."""
    if kernel == "lstm":
        I, H, T = [int(x) for x in args.split()[:3]]
        return T, 4 * H * (I + H)
    if kernel == "gat_aggregate":
        N, F, heads, fan, targets = [int(x) for x in args.split()[:5]]
        return targets, heads * (fan * 2 * (F // heads) + F // heads)
    if kernel in ("attention_decode", "depthwise_conv", "conv3d"):
        return _v2_units_and_cost(kernel, args)
    return make_ai_plan_v2._v1_units_and_cost(kernel, args)   # v1 kernels (incl. "embedding gather ...")


base.units_and_cost = units_and_cost

if __name__ == "__main__":
    base.main()
