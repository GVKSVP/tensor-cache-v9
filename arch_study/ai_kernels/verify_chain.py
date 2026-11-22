#!/usr/bin/env python3
"""
verify_chain.py - correctness check of the layer chains (src/chain.c, "small"
shapes) against numpy. Same method and pass rule as verify_reference.py,
whose LCG helpers and run_binary() it reuses (that file is not modified).
Also checks that cold/warm modes give the same checksum as none.
"""

import sys

import numpy as np

from verify_reference import ATOL, RTOL, lcg_floats, run_binary


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def sigm(v):
    return 1.0 / (1.0 + np.exp(-v))


def ref_tf(S, D, F, U):
    O = lcg_floats(S * D, 1).reshape(S, D)
    Wo = lcg_floats(D * D, 2).reshape(D, D)
    W1 = lcg_floats(D * F, 3).reshape(D, F)
    W2 = lcg_floats(F * D, 4).reshape(F, D)
    Y = np.zeros((S, D))
    Y[:U] = gelu((O[:U] @ Wo) @ W1) @ W2
    return Y


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


def ref_cv(cin, cmid, cout, H, dw, clip, U):
    act = (lambda v: np.clip(v, 0, clip)) if clip else (lambda v: np.maximum(v, 0))
    X = lcg_floats(cin * H * H, 1).reshape(cin, H, H)
    Wa = lcg_floats(cmid * cin, 2).reshape(cmid, cin)
    nb = cmid * 9 if dw else cmid * cmid * 9
    Wb = lcg_floats(nb, 3).reshape((cmid, 3, 3) if dw else (cmid, cmid, 3, 3))
    Wc = lcg_floats(cout * cmid, 4).reshape(cout, cmid)
    ra = min(U + 1, H)
    A = np.zeros((cmid, H, H))
    A[:, :ra] = act(np.einsum("kc,chw->khw", Wa, X[:, :ra]))
    B = conv3x3(A, Wb, U, dw)
    B[:, :U] = act(B[:, :U])
    Y = np.zeros((cout, H, H))
    Y[:, :U] = np.einsum("kc,chw->khw", Wc, B[:, :U])
    return Y


def lstm(x, I, H, seed, steps):
    wih = lcg_floats(4 * H * I, seed).reshape(4 * H, I)
    whh = lcg_floats(4 * H * H, seed + 1).reshape(4 * H, H)
    b = lcg_floats(4 * H, seed + 2)
    h = np.zeros(H)
    c = np.zeros(H)
    ys = []
    for t in range(steps):
        z = wih @ x[t] + whh @ h + b
        i, f, g, o = sigm(z[:H]), sigm(z[H:2 * H]), np.tanh(z[2 * H:3 * H]), sigm(z[3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)
        ys.append(h.copy())
    return np.array(ys)


def ref_rnnt(I, H, T, U):
    x = lcg_floats(T * I, 1).reshape(T, I)
    y1 = lstm(x, I, H, 10, 2 * U)
    y2 = lstm(y1, H, H, 20, 2 * U)
    y3 = lstm(y2.reshape(U, 2 * H), 2 * H, H, 30, U)
    out = np.zeros((T // 2, H))
    out[:U] = y3
    return out


def ref_llama(D, KVH, QH, DH, L, F):
    kvd = KVH * DH
    x = lcg_floats(D, 1)
    Wq = lcg_floats(D * D, 2).reshape(D, D)
    Wk = lcg_floats(D * kvd, 3).reshape(D, kvd)
    Wv = lcg_floats(D * kvd, 4).reshape(D, kvd)
    Kc = lcg_floats(KVH * (L + 1) * DH, 5).reshape(KVH, L + 1, DH)
    Vc = lcg_floats(KVH * (L + 1) * DH, 6).reshape(KVH, L + 1, DH)
    Wo = lcg_floats(D * D, 7).reshape(D, D)
    Wup = lcg_floats(D * F, 8).reshape(D, F)
    Wdn = lcg_floats(F * D, 9).reshape(F, D)
    q = x @ Wq
    Kc[:, L] = (x @ Wk).reshape(KVH, DH)
    Vc[:, L] = (x @ Wv).reshape(KVH, DH)
    o = np.zeros(D)
    for h in range(QH):
        g = h // (QH // KVH)
        s = Kc[g] @ q[h * DH:(h + 1) * DH] / np.sqrt(DH)
        p = np.exp(s - s.max())
        p /= p.sum()
        o[h * DH:(h + 1) * DH] = p @ Vc[g]
    hdn = x_up = (o @ Wo) @ Wup
    hdn = x_up * sigm(x_up)
    return hdn @ Wdn


CASES = [
    ("bert_ffn", 3, lambda: ref_tf(6, 8, 16, 3)),
    ("vit_mlp", 2, lambda: ref_tf(5, 8, 16, 2)),
    ("rn50_bottleneck", 3, lambda: ref_cv(8, 4, 8, 6, False, 0, 3)),
    ("mbv2_block", 6, lambda: ref_cv(4, 8, 4, 6, True, 6.0, 6)),
    ("rnnt_encoder", 2, lambda: ref_rnnt(5, 4, 8, 2)),
    ("llama_decode", 1, lambda: ref_llama(16, 2, 4, 4, 5, 24)),
    # a second sampling point for the sampled chains
    ("bert_ffn", 1, lambda: ref_tf(6, 8, 16, 1)),
    ("rn50_bottleneck", 6, lambda: ref_cv(8, 4, 8, 6, False, 0, 6)),
    ("rnnt_encoder", 4, lambda: ref_rnnt(5, 4, 8, 4)),
]


def main():
    failures = 0
    print(f"{'case':<24}{'checksum (C)':>18}{'reference':>18}{'abs err':>12}  result")
    for chain, units, ref in CASES:
        name = f"{chain}/u{units}"
        out = np.asarray(ref())
        ref_sum, ref_abs, ref_n = out.sum(), np.abs(out).sum(), out.size
        try:
            s, a, n = run_binary("chain", [chain + ":small", str(units), "none"])
        except RuntimeError as exc:
            print(f"{name:<24} ERROR {exc}")
            failures += 1
            continue
        tol = RTOL * ref_abs + ATOL
        ok = abs(s - ref_sum) <= tol and abs(a - ref_abs) <= tol and n == ref_n
        if ok:
            for mode in ("cold", "warm"):
                if run_binary("chain", [chain + ":small", str(units), mode]) != (s, a, n):
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
