#!/usr/bin/env python3
"""
make_ai_plan_v2.py - build experiment plans from layers_v2.csv
(Llama-3.1-8B, Llama-2-70B, MobileNetV2, 3D-UNet).

Reuses make_ai_plan.py unchanged: same options, same unit-budget rule,
same refuse-to-overwrite checks. It only points the generator at
layers_v2.csv and adds the work-unit costs of the v2 kernels.

Usage: same options as make_ai_plan.py, e.g.
    python make_ai_plan_v2.py --out experiment_plan_ai_v2_p870d_pf_v1.csv \
        --config-id F_P870D_full_L2stride --mem-size 1GiB \
        --cache-args "64KiB 8 4MiB 16 16MiB 16" --sim-flags="--cpu o3 ..."
"""

import make_ai_plan as base

base.LAYERS = base.HERE / "layers_v2.csv"
_v1_units_and_cost = base.units_and_cost


def units_and_cost(kernel, args):
    """Return (units_total, macs_per_unit), including the v2 kernels."""
    a = [int(x) for x in args.split()]
    if kernel == "attention_decode":
        L, D, qh, _kvh = a[:4]
        return qh, 2 * L * D
    if kernel == "depthwise_conv":
        C, H, W, R, st, p = a[:6]
        OH = (H + 2 * p - R) // st + 1
        OW = (W + 2 * p - R) // st + 1
        return C, R * R * OH * OW
    if kernel == "conv3d":
        C, D, H, W, K, R, st, p = a[:8]
        OD = (D + 2 * p - R) // st + 1
        OH = (H + 2 * p - R) // st + 1
        OW = (W + 2 * p - R) // st + 1
        return K * OD, C * R ** 3 * OH * OW
    return _v1_units_and_cost(kernel, args)


base.units_and_cost = units_and_cost

if __name__ == "__main__":
    base.main()
