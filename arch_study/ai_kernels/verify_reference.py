#!/usr/bin/env python3
"""
verify_reference.py - correctness check of the model-derived AI kernels.

Runs the NATIVE binaries (bin/native/*) on tiny shapes and compares their
CHECKSUM line with an independent numpy (float64) reference that rebuilds
the identical input data from the same LCG sequence used in ai_common.h.

Pass rule per case:  |sum_c - sum_ref|       <= RTOL * abssum_ref + ATOL
                     |abssum_c - abssum_ref| <= RTOL * abssum_ref + ATOL
                     element counts equal
Also checks that cold / warm / none modes give identical checksums.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BIN = HERE / "bin" / "native"
RTOL = 1e-4
ATOL = 1e-6


# ------------------------------------------------------------------
# LCG identical to ai_common.h
# ------------------------------------------------------------------

def lcg_floats(n, seed):
    s = seed & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (s >> 8) * (1.0 / 8388608.0) - 1.0
    return out


def lcg_ints(n, seed):
    s = seed & 0xFFFFFFFF
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = s
    return out


# ------------------------------------------------------------------
# References (return the computed output elements)
# ------------------------------------------------------------------

def ref_gemm(M, N, K, U):
    A = lcg_floats(M * K, 1).reshape(M, K)
    B = lcg_floats(K * N, 2).reshape(K, N)
    return A[:U] @ B


def ref_conv(C, H, W, K, R, S, st, p, U):
    x = lcg_floats(C * H * W, 1).reshape(C, H, W)
    w = lcg_floats(K * C * R * S, 2).reshape(K, C, R, S)
    OH = (H + 2 * p - R) // st + 1
    OW = (W + 2 * p - S) // st + 1
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    out = np.zeros((U, OH, OW))
    for k in range(U):
        for c in range(C):
            for r in range(R):
                for s in range(S):
                    out[k] += w[k, c, r, s] * xp[c, r:r + st * OH:st, s:s + st * OW:st]
    return out


def ref_attention(S, D, heads, U):
    n = heads * S * D
    Q = lcg_floats(n, 1).reshape(heads, S, D)
    Kt = lcg_floats(n, 2).reshape(heads, S, D)
    V = lcg_floats(n, 3).reshape(heads, S, D)
    outs = []
    for h in range(U):
        sc = Q[h] @ Kt[h].T / np.sqrt(D)
        sc = np.exp(sc - sc.max(axis=1, keepdims=True))
        sc /= sc.sum(axis=1, keepdims=True)
        outs.append(sc @ V[h])
    return np.stack(outs)


def ref_pool(kind, C, H, W, R, st, p, U):
    x = lcg_floats(C * H * W, 1).reshape(C, H, W)
    OH = (H + 2 * p - R) // st + 1
    OW = (W + 2 * p - R) // st + 1
    fill = -np.inf if kind == "max" else 0.0
    xp = np.pad(x, ((0, 0), (p, p), (p, p)), constant_values=fill)
    out = np.zeros((U, OH, OW))
    for c in range(U):
        for oh in range(OH):
            for ow in range(OW):
                win = xp[c, oh * st:oh * st + R, ow * st:ow * st + R]
                out[c, oh, ow] = win.max() if kind == "max" else win.sum() / (R * R)
    return out


def ref_gather(V, D, L, U):
    table = lcg_floats(V * D, 1).reshape(V, D)
    idx = [(i * 2654435761) % V for i in range(U)]
    return table[idx]


DLRM_ROWS = [40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
             3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
             40000000, 40000000, 590152, 12973, 108, 36]
DLRM_HOT = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]


def ref_bag(B, rcap, U):
    rows = [min(r, rcap) for r in DLRM_ROWS]
    tables = [lcg_floats(rows[t] * 128, 100 + t).reshape(rows[t], 128) for t in range(26)]
    raw = lcg_ints(B * sum(DLRM_HOT), 7)
    out = np.zeros((U, 26, 128))
    pos = 0
    for b in range(B):
        for t in range(26):
            for _ in range(DLRM_HOT[t]):
                if b < U:
                    out[b, t] += tables[t][raw[pos] % rows[t]]
                pos += 1
    return out


def ref_mlp(dims, B, U):
    X = lcg_floats(B * dims[0], 1).reshape(B, dims[0])
    y = X[:U]
    nl = len(dims) - 1
    for l in range(nl):
        W = lcg_floats(dims[l] * dims[l + 1], 10 + l).reshape(dims[l], dims[l + 1])
        b = lcg_floats(dims[l + 1], 50 + l)
        y = y @ W + b
        if l != nl - 1:
            y = np.maximum(y, 0.0)
    return y


# ------------------------------------------------------------------
# Test cases: (name, binary, args, reference callable)
# ------------------------------------------------------------------

CASES = [
    ("gemm_naive", "gemm", [7, 5, 9, 4, "naive", 64], lambda: ref_gemm(7, 5, 9, 4)),
    ("gemm_blocked", "gemm", [7, 5, 9, 4, "blocked", 3], lambda: ref_gemm(7, 5, 9, 4)),
    ("gemv", "gemm", [6, 1, 8, 0, "naive", 64], lambda: ref_gemm(6, 1, 8, 6)),
    ("conv_3x3_p1", "conv2d_nchw", [3, 9, 8, 4, 3, 3, 1, 1, 3],
     lambda: ref_conv(3, 9, 8, 4, 3, 3, 1, 1, 3)),
    ("conv_5x3_s2_p2", "conv2d_nchw", [2, 11, 10, 3, 5, 3, 2, 2, 0],
     lambda: ref_conv(2, 11, 10, 3, 5, 3, 2, 2, 3)),
    ("conv_nonsquare_s2_p0", "conv2d_nchw", [1, 8, 20, 2, 2, 5, 2, 0, 0],
     lambda: ref_conv(1, 8, 20, 2, 2, 5, 2, 0, 2)),
    ("conv_1x1", "conv2d_nchw", [5, 4, 4, 6, 1, 1, 1, 0, 0],
     lambda: ref_conv(5, 4, 4, 6, 1, 1, 1, 0, 6)),
    ("attention", "attention", [5, 4, 3, 2], lambda: ref_attention(5, 4, 3, 2)),
    ("maxpool_3x3_s2_p1", "pool", ["max", 2, 7, 7, 3, 2, 1, 0],
     lambda: ref_pool("max", 2, 7, 7, 3, 2, 1, 2)),
    ("avgpool_global", "pool", ["avg", 3, 4, 4, 4, 1, 0, 0],
     lambda: ref_pool("avg", 3, 4, 4, 4, 1, 0, 3)),
    ("avgpool_pad", "pool", ["avg", 2, 5, 5, 3, 2, 1, 1],
     lambda: ref_pool("avg", 2, 5, 5, 3, 2, 1, 1)),
    ("emb_gather", "embedding", ["gather", 50, 8, 13, 10], lambda: ref_gather(50, 8, 13, 10)),
    ("emb_bag", "embedding", ["bag", 3, 16, 2], lambda: ref_bag(3, 16, 2)),
    ("mlp", "mlp", ["13,16,8,4", 5, 3], lambda: ref_mlp([13, 16, 8, 4], 5, 3)),
]

# Cases re-run in every mode to check that cold/warm/none agree.
MODE_CASES = {"gemm_naive", "conv_3x3_p1", "attention", "emb_bag", "mlp"}

# Number of positional args before [mode] for each kernel.
MODE_POS = {"gemm": 6, "conv2d_nchw": 9, "attention": 4, "pool": 8, "embedding": None, "mlp": 3}


def run_binary(binary, args):
    proc = subprocess.run([str(BIN / binary)] + [str(a) for a in args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} exit {proc.returncode}: {proc.stderr.strip()}")
    for line in proc.stdout.splitlines():
        if line.startswith("CHECKSUM"):
            _, s, a, n = line.split()
            return float(s), float(a), int(n)
    raise RuntimeError(f"{binary}: no CHECKSUM line")


def with_mode(binary, args, mode):
    pos = MODE_POS[binary]
    if binary == "embedding":
        pos = 5 if args[0] == "gather" else 4
    args = list(args)
    return args[:pos] + [mode]


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
