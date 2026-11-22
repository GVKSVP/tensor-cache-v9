/*
 * conv2d_nchw.c - model-derived direct 2D convolution (float32, NCHW, N=1).
 *
 *   out[K][OH][OW] = sum_{c,r,s} in[C][ih][iw] * wt[K][C][R][S]
 *   ih = oh*stride - pad + r,  iw = ow*stride - pad + s
 *
 * R = filter height, S = filter width (DeepBench naming).
 * The full input, weight and output tensors are allocated with their real
 * shapes; only the first `units` output channels are computed
 * (work unit = one output channel = C*R*S*OH*OW MACs).
 *
 * Usage: conv2d_nchw C H W K R S stride pad [units] [mode]
 * Seeds: input = 1, weights = 2.
 */
#include "ai_common.h"

static long C, H, W, K, R, S, ST, P, OH, OW, U;
static float *in, *wt, *out;

static void zero(void)
{
    memset(out, 0, (size_t)K * OH * OW * sizeof(float));
}

static void run(void)
{
    for (long k = 0; k < U; k++) {
        float *o = out + k * OH * OW;
        for (long c = 0; c < C; c++) {
            const float *x = in + c * H * W;
            const float *w = wt + ((k * C + c) * R) * S;
            for (long r = 0; r < R; r++) {
                for (long s = 0; s < S; s++) {
                    const float wv = w[r * S + s];
                    /* valid output columns for this filter column */
                    const long d = P - s;
                    const long lo = d > 0 ? (d + ST - 1) / ST : 0;
                    const long num = W - 1 + P - s;
                    long hi = num < 0 ? 0 : num / ST + 1;
                    if (hi > OW)
                        hi = OW;
                    for (long oh = 0; oh < OH; oh++) {
                        const long ih = oh * ST - P + r;
                        if (ih < 0 || ih >= H)
                            continue;
                        const float *xr = x + ih * W;
                        float *orow = o + oh * OW;
                        for (long ow = lo; ow < hi; ow++)
                            orow[ow] += wv * xr[ow * ST - P + s];
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 9) {
        fprintf(stderr, "Usage: %s C H W K R S stride pad [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    C = atol(argv[1]);
    H = atol(argv[2]);
    W = atol(argv[3]);
    K = atol(argv[4]);
    R = atol(argv[5]);
    S = atol(argv[6]);
    ST = atol(argv[7]);
    P = atol(argv[8]);
    if (C < 1 || H < 1 || W < 1 || K < 1 || R < 1 || S < 1 || ST < 1 || P < 0) {
        fprintf(stderr, "ERROR: invalid dimensions\n");
        return 1;
    }
    OH = (H + 2 * P - R) / ST + 1;
    OW = (W + 2 * P - S) / ST + 1;
    if (OH < 1 || OW < 1) {
        fprintf(stderr, "ERROR: filter larger than padded input\n");
        return 1;
    }
    U = clamp_units(arg_long(argc, argv, 9, 0), K);
    const char *mode = arg_str(argc, argv, 10, "cold");

    in = falloc((size_t)C * H * W);
    wt = falloc((size_t)K * C * R * S);
    out = falloc((size_t)K * OH * OW);
    fill_lcg(in, (size_t)C * H * W, 1);
    fill_lcg(wt, (size_t)K * C * R * S, 2);

    ai_execute(mode, run, zero);

    char dims[128];
    snprintf(dims, sizeof dims, "C=%ld H=%ld W=%ld K=%ld R=%ld S=%ld stride=%ld pad=%ld OH=%ld OW=%ld",
             C, H, W, K, R, S, ST, P, OH, OW);
    const double per_unit = (double)C * R * S * OH * OW;
    report_meta("conv2d_nchw", dims, mode, "direct", "output_channel", U, K,
                per_unit * U, per_unit * K,
                (double)C * H * W * 4, (double)K * C * R * S * 4, (double)K * OH * OW * 4);
    report_checksum(out, (size_t)U * OH * OW);
    return 0;
}
