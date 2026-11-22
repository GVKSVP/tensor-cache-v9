#!/usr/bin/env python3
"""
verify_reference_v2.py - correctness check of the v2 kernels
(attention_decode, depthwise_conv, conv3d).

Same method and pass rule as verify_reference.py, whose LCG and
run_binary() it reuses (that file is not modified): run the NATIVE binary
on tiny shapes, compare its CHECKSUM line with an independent numpy
(float64) reference, and check that cold / warm / none modes agree.
"""

import sys

import numpy as np

from verify_reference import ATOL, RTOL, lcg_floats, run_binary


def ref_attention_decode(L, D, QH, KVH, U):
    q = lcg_floats(QH * D, 1).reshape(QH, D)
    K = lcg_floats(KVH * L * D, 2).reshape(KVH, L, D)
    V = lcg_floats(KVH * L * D, 3).reshape(KVH, L, D)
    group = QH // KVH
    outs = []
    for h in range(U):
        g = h // group
        sc = K[g] @ q[h] / np.sqrt(D)
        sc = np.exp(sc - sc.max())
        sc /= sc.sum()
        outs.append(sc @ V[g])
    return np.stack(outs)


def ref_depthwise(C, H, W, R, st, p, U):
    x = lcg_floats(C * H * W, 1).reshape(C, H, W)
    w = lcg_floats(C * R * R, 2).reshape(C, R, R)
    OH = (H + 2 * p - R) // st + 1
    OW = (W + 2 * p - R) // st + 1
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    out = np.zeros((U, OH, OW))
    for c in range(U):
        for r in range(R):
            for s in range(R):
                out[c] += w[c, r, s] * xp[c, r:r + st * OH:st, s:s + st * OW:st]
    return out


def ref_conv3d(C, D, H, W, K, R, st, p, U):
    x = lcg_floats(C * D * H * W, 1).reshape(C, D, H, W)
    w = lcg_floats(K * C * R ** 3, 2).reshape(K, C, R, R, R)
    OD = (D + 2 * p - R) // st + 1
    OH = (H + 2 * p - R) // st + 1
    OW = (W + 2 * p - R) // st + 1
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (p, p)))
    full = np.zeros((K, OD, OH, OW))
    for k in range(K):
        for c in range(C):
            for a in range(R):
                for r in range(R):
                    for s in range(R):
                        full[k] += w[k, c, a, r, s] * xp[c, a:a + st * OD:st,
                                                         r:r + st * OH:st, s:s + st * OW:st]
    # units are (k, od) planes in memory order
    return full.reshape(K * OD, OH, OW)[:U]


CASES = [
    ("attn_decode_mha", "attention_decode", [9, 4, 3, 3, 3], lambda: ref_attention_decode(9, 4, 3, 3, 3)),
    ("attn_decode_gqa", "attention_decode", [11, 5, 8, 2, 6], lambda: ref_attention_decode(11, 5, 8, 2, 6)),
    ("attn_decode_L1", "attention_decode", [1, 4, 4, 1, 0], lambda: ref_attention_decode(1, 4, 4, 1, 4)),
    ("dw_3x3_p1", "depthwise_conv", [4, 9, 8, 3, 1, 1, 3], lambda: ref_depthwise(4, 9, 8, 3, 1, 1, 3)),
    ("dw_3x3_s2_p1", "depthwise_conv", [3, 11, 10, 3, 2, 1, 0], lambda: ref_depthwise(3, 11, 10, 3, 2, 1, 3)),
    ("dw_5x5_s2_p0", "depthwise_conv", [2, 12, 13, 5, 2, 0, 0], lambda: ref_depthwise(2, 12, 13, 5, 2, 0, 2)),
    ("conv3d_3_p1", "conv3d", [2, 5, 6, 7, 3, 3, 1, 1, 7], lambda: ref_conv3d(2, 5, 6, 7, 3, 3, 1, 1, 7)),
    ("conv3d_3_s2_p1", "conv3d", [3, 7, 8, 6, 2, 3, 2, 1, 0], lambda: ref_conv3d(3, 7, 8, 6, 2, 3, 2, 1, 8)),
    ("conv3d_1x1x1", "conv3d", [4, 3, 4, 5, 3, 1, 1, 0, 5], lambda: ref_conv3d(4, 3, 4, 5, 3, 1, 1, 0, 5)),
]

MODE_CASES = {"attn_decode_gqa", "dw_3x3_p1", "conv3d_3_p1"}

# Number of positional args before [mode] for each kernel.
MODE_POS = {"attention_decode": 5, "depthwise_conv": 7, "conv3d": 9}


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
