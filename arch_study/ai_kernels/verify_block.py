#!/usr/bin/env python3
"""
verify_block.py - correctness check of the whole blocks (src/block.c, "small"
shapes) against numpy. Same method and pass rule as verify_chain.py; reuses
verify_reference.py helpers (neither file is modified). Also checks that
cold/warm modes give the same checksum as none.
"""
import sys

import numpy as np

from verify_reference import ATOL, RTOL, lcg_floats, run_binary


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def add_ln(a, b):
    y = a + b
    m = y.mean(axis=1, keepdims=True)
    v = ((y - m) ** 2).mean(axis=1, keepdims=True)
    return (y - m) / np.sqrt(v + 1e-5)


def ref_eb(S, D, H, F, U):
    dh = D // H
    X = lcg_floats(S * D, 1).reshape(S, D)
    Wq, Wk, Wv = (lcg_floats(D * D, s).reshape(D, D) for s in (2, 3, 4))
    K = lcg_floats(S * D, 5).reshape(S, D)
    V = lcg_floats(S * D, 6).reshape(S, D)
    Wo = lcg_floats(D * D, 7).reshape(D, D)
    W1 = lcg_floats(D * F, 8).reshape(D, F)
    W2 = lcg_floats(F * D, 9).reshape(F, D)
    Q = X[:U] @ Wq
    K[:U] = X[:U] @ Wk
    V[:U] = X[:U] @ Wv
    C = np.zeros((U, D))
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        s = Q[:, sl] @ K[:, sl].T / np.sqrt(dh)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        C[:, sl] = p @ V[:, sl]
    X1 = add_ln(X[:U], C @ Wo)
    Y = gelu(X1 @ W1) @ W2
    Out = np.zeros((S, D))
    Out[:U] = add_ln(X1, Y)
    return Out


def conv3x3(A, Wb, rows, dw):
    C, H, W = A.shape
    K = Wb.shape[0]
    P = np.zeros((C, H + 2, W + 2))
    P[:, 1:H + 1, 1:W + 1] = A
    B = np.zeros((K, H, W))
    for k in range(K):
        for oh in range(rows):
            for ow in range(W):
                win = P[:, oh:oh + 3, ow:ow + 3]
                B[k, oh, ow] = (win[k] * Wb[k]).sum() if dw else (win * Wb[k]).sum()
    return B


def ref_cb(C, M, H, dw, clip, ract, U):
    act = (lambda v: np.clip(v, 0, clip)) if clip else (lambda v: np.maximum(v, 0))
    nb = M * 9 if dw else M * M * 9
    shp = (M, 3, 3) if dw else (M, M, 3, 3)
    X = lcg_floats(C * H * H, 1).reshape(C, H, H)

    def blk(Xin, sa, sb, sc, rows_out):
        Wa = lcg_floats(M * C, sa).reshape(M, C)
        Wb = lcg_floats(nb, sb).reshape(shp)
        Wc = lcg_floats(C * M, sc).reshape(C, M)
        ra = min(rows_out + 1, H)
        A = np.zeros((M, H, H))
        A[:, :ra] = act(np.einsum("kc,chw->khw", Wa, Xin[:, :ra]))
        B = conv3x3(A, Wb, rows_out, dw)
        B[:, :rows_out] = act(B[:, :rows_out])
        Y = np.zeros((C, H, H))
        v = np.einsum("km,mhw->khw", Wc, B[:, :rows_out]) + Xin[:, :rows_out]
        Y[:, :rows_out] = np.maximum(v, 0) if ract else v
        return Y
    Y1 = blk(X, 2, 3, 4, min(U + 1, H))
    return blk(Y1, 5, 6, 7, U)


DROWS = [40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
         3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
         40000000, 40000000, 590152, 12973, 108, 36]
DHOT = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]


def ref_dl(Bsz, rcap, U):
    bot, top = [13, 512, 256, 128], [3456, 1024, 1024, 512, 256, 1]
    dense = lcg_floats(Bsz * 13, 1).reshape(Bsz, 13)
    bw = [lcg_floats(bot[l] * bot[l + 1], 10 + l).reshape(bot[l], bot[l + 1]) for l in range(3)]
    rows = [min(r, rcap) for r in DROWS]
    tabs = [lcg_floats(r * 128, 100 + t).reshape(r, 128) for t, r in enumerate(rows)]
    s = 7
    idx = []
    for b in range(Bsz):
        ib = []
        for t in range(26):
            for _ in range(DHOT[t]):
                s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
                ib.append((t, s % rows[t]))
        idx.append(ib)
    tw = [lcg_floats(top[l] * top[l + 1], 20 + l).reshape(top[l], top[l + 1]) for l in range(5)]
    out = np.zeros(Bsz)
    for i in range(U):
        h = dense[i]
        for l in range(3):
            h = np.maximum(h @ bw[l], 0)
        pool = np.zeros((26, 128))
        for t, r in idx[i]:
            pool[t] += tabs[t][r]
        z = np.concatenate([h, pool.ravel()])
        for l in range(5):
            z = z @ tw[l]
            if l < 4:
                z = np.maximum(z, 0)
        out[i] = z[0]
    return out


CASES = [
    ("bert_block", 3, lambda: ref_eb(8, 16, 4, 32, 3)),
    ("bert_block", 8, lambda: ref_eb(8, 16, 4, 32, 8)),
    ("vit_block", 2, lambda: ref_eb(7, 12, 3, 24, 2)),
    ("vit_block", 5, lambda: ref_eb(7, 12, 3, 24, 5)),
    ("rn50_block2", 3, lambda: ref_cb(8, 4, 6, 0, 0.0, 1, 3)),
    ("rn50_block2", 6, lambda: ref_cb(8, 4, 6, 0, 0.0, 1, 6)),
    ("mbv2_block2", 2, lambda: ref_cb(4, 8, 6, 1, 6.0, 0, 2)),
    ("mbv2_block2", 6, lambda: ref_cb(4, 8, 6, 1, 6.0, 0, 6)),
    ("dlrm_block", 2, lambda: ref_dl(3, 50, 2)),
]


def main():
    failures = 0
    print(f"{'case':<24}{'checksum (C)':>18}{'reference':>18}{'abs err':>12}  result")
    for blk, units, ref in CASES:
        name = f"{blk}/u{units}"
        out = np.asarray(ref())
        ref_sum, ref_abs, ref_n = out.sum(), np.abs(out).sum(), out.size
        try:
            s, a, n = run_binary("block", [blk + ":small", str(units), "none"])
        except RuntimeError as exc:
            print(f"{name:<24} ERROR {exc}")
            failures += 1
            continue
        tol = RTOL * ref_abs + ATOL
        ok = abs(s - ref_sum) <= tol and abs(a - ref_abs) <= tol and n == ref_n
        if ok:
            for mode in ("cold", "warm"):
                if run_binary("block", [blk + ":small", str(units), mode]) != (s, a, n):
                    print(f"{name:<24} mode '{mode}' checksum differs from 'none'")
                    ok = False
        failures += 0 if ok else 1
        print(f"{name:<24}{s:>18.9e}{ref_sum:>18.9e}{abs(s - ref_sum):>12.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
