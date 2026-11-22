/*
 * depthwise_conv.c - model-derived depthwise 2D convolution (float32,
 * NCHW, N=1, square R x R filter, one filter per channel).
 *
 *   out[c][OH][OW] = sum_{r,s} in[c][ih][iw] * wt[c][r][s]
 *   ih = oh*stride - pad + r,  iw = ow*stride - pad + s
 *
 * The full input, weight and output tensors are allocated with their real
 * shapes; only the first `units` channels are computed
 * (work unit = one channel = R*R*OH*OW MACs).
 *
 * Usage: depthwise_conv C H W R stride pad [units] [mode]
 * Seeds: input = 1, weights = 2.
 */
#include "ai_common.h"

static long C, H, W, R, ST, P, OH, OW, U;
static float *in, *wt, *out;

static void zero(void)
{
    memset(out, 0, (size_t)C * OH * OW * sizeof(float));
}

static void run(void)
{
    for (long c = 0; c < U; c++) {
        float *o = out + c * OH * OW;
        const float *x = in + c * H * W;
        const float *w = wt + c * R * R;
        for (long r = 0; r < R; r++) {
            for (long s = 0; s < R; s++) {
                const float wv = w[r * R + s];
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

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr, "Usage: %s C H W R stride pad [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    C = atol(argv[1]);
    H = atol(argv[2]);
    W = atol(argv[3]);
    R = atol(argv[4]);
    ST = atol(argv[5]);
    P = atol(argv[6]);
    if (C < 1 || H < 1 || W < 1 || R < 1 || ST < 1 || P < 0) {
        fprintf(stderr, "ERROR: invalid dimensions\n");
        return 1;
    }
    OH = (H + 2 * P - R) / ST + 1;
    OW = (W + 2 * P - R) / ST + 1;
    if (OH < 1 || OW < 1) {
        fprintf(stderr, "ERROR: filter larger than padded input\n");
        return 1;
    }
    U = clamp_units(arg_long(argc, argv, 7, 0), C);
    const char *mode = arg_str(argc, argv, 8, "cold");

    in = falloc((size_t)C * H * W);
    wt = falloc((size_t)C * R * R);
    out = falloc((size_t)C * OH * OW);
    fill_lcg(in, (size_t)C * H * W, 1);
    fill_lcg(wt, (size_t)C * R * R, 2);

    ai_execute(mode, run, zero);

    char dims[128];
    snprintf(dims, sizeof dims, "C=%ld H=%ld W=%ld R=%ld stride=%ld pad=%ld OH=%ld OW=%ld",
             C, H, W, R, ST, P, OH, OW);
    const double per_unit = (double)R * R * OH * OW;
    report_meta("depthwise_conv", dims, mode, "direct", "channel", U, C,
                per_unit * U, per_unit * C,
                (double)C * H * W * 4, (double)C * R * R * 4, (double)C * OH * OW * 4);
    report_checksum(out, (size_t)U * OH * OW);
    return 0;
}
