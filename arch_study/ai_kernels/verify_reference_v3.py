#!/usr/bin/env python3
"""
verify_reference_v3.py - correctness check of the v3 kernels (lstm,
gat_aggregate). Same method and pass rule as verify_reference.py, whose LCG
helpers and run_binary() it reuses (that file is not modified).
"""

import sys

import numpy as np

from verify_reference import ATOL, RTOL, lcg_floats, lcg_ints, run_binary


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def ref_lstm(I, H, T, U):
    x = lcg_floats(T * I, 1).reshape(T, I)
    wih = lcg_floats(4 * H * I, 2).reshape(4 * H, I)
    whh = lcg_floats(4 * H * H, 3).reshape(4 * H, H)
    b = lcg_floats(4 * H, 4)
    h = np.zeros(H)
    c = np.zeros(H)
    ys = []
    for t in range(U):
        z = wih @ x[t] + whh @ h + b
        i, f, g, o = sigmoid(z[:H]), sigmoid(z[H:2 * H]), np.tanh(z[2 * H:3 * H]), sigmoid(z[3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)
        ys.append(h.copy())
    return np.stack(ys)


def ref_gat(N, F, heads, fan, targets, U):
    feat = lcg_floats(N * F, 1).reshape(N, F)
    asrc = lcg_floats(F, 2)
    adst = lcg_floats(F, 3)
    nbr = (lcg_ints(targets * fan, 7) % N).reshape(targets, fan)
    dh = F // heads
    out = np.zeros((U, F))
    for t in range(U):
        for hd in range(heads):
            sl = slice(hd * dh, (hd + 1) * dh)
            sd = adst[sl] @ feat[t, sl]
            e = feat[nbr[t], sl] @ asrc[sl] + sd
            e = np.where(e > 0, e, 0.2 * e)
            a = np.exp(e - e.max())
            a /= a.sum()
            out[t, sl] = a @ feat[nbr[t], sl]
    return out


CASES = [
    ("lstm_small", "lstm", [5, 4, 6, 6], lambda: ref_lstm(5, 4, 6, 6)),
    ("lstm_sampled", "lstm", [7, 3, 9, 4], lambda: ref_lstm(7, 3, 9, 4)),
    ("lstm_I_lt_H", "lstm", [2, 8, 5, 0], lambda: ref_lstm(2, 8, 5, 5)),
    ("gat_1head", "gat_aggregate", [20, 6, 1, 4, 7, 7], lambda: ref_gat(20, 6, 1, 4, 7, 7)),
    ("gat_4heads", "gat_aggregate", [30, 8, 4, 5, 10, 6], lambda: ref_gat(30, 8, 4, 5, 10, 6)),
    ("gat_fan1", "gat_aggregate", [9, 4, 2, 1, 3, 0], lambda: ref_gat(9, 4, 2, 1, 3, 3)),
]

MODE_CASES = {"lstm_sampled", "gat_4heads"}
MODE_POS = {"lstm": 4, "gat_aggregate": 6}


def with_mode(binary, args, mode):
    return list(args)[:MODE_POS[binary]] + [mode]


def main():
    failures = 0
    print(f"{'case':<24}{'checksum (C)':>18}{'reference':>18}{'abs err':>12}  result")
    for name, binary, args, ref in CASES:
        out = np.asarray(ref())
        ref_sum, ref_abs, ref_n = out.sum(), np.abs(out).sum(), out.size
        try:
            s, a, n = run_binary(binary, with_mode(binary, args, "none"))
        except RuntimeError as exc:
            print(f"{name:<24} ERROR {exc}")
            failures += 1
            continue
        tol = RTOL * ref_abs + ATOL
        ok = abs(s - ref_sum) <= tol and abs(a - ref_abs) <= tol and n == ref_n
        if ok and name in MODE_CASES:
            for mode in ("cold", "warm"):
                if run_binary(binary, with_mode(binary, args, mode)) != (s, a, n):
                    print(f"{name:<24} mode '{mode}' checksum differs from 'none'")
                    ok = False
        failures += 0 if ok else 1
        print(f"{name:<24}{s:>18.9e}{ref_sum:>18.9e}{abs(s - ref_sum):>12.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print()
    print(f"{len(CASES) - failures}/{len(CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
