/*
 * conv3d.c - model-derived direct 3D convolution (float32, NCDHW, N=1,
 * cubic R x R x R filter).
 *
 *   out[K][OD][OH][OW] = sum_{c,a,r,s} in[C][id][ih][iw] * wt[K][C][a][r][s]
 *   id = od*stride - pad + a,  ih = oh*stride - pad + r,  iw = ow*stride - pad + s
 *
 * The full input, weight and output tensors are allocated with their real
 * shapes. A single output channel of a large 3D layer can exceed 4e8 MACs,
 * so the work unit is one output depth-plane of one output channel
 * (C*R^3*OH*OW MACs). Units are taken in memory order (k-major, then od),
 * so the computed outputs are the first units*OH*OW elements of `out`.
 *
 * Usage: conv3d C D H W K R stride pad [units] [mode]
 * Seeds: input = 1, weights = 2.
 */
#include "ai_common.h"

static long C, D, H, W, K, R, ST, P, OD, OH, OW, U;
static float *in, *wt, *out;

static void zero(void)
{
    memset(out, 0, (size_t)K * OD * OH * OW * sizeof(float));
}

static void run(void)
{
    for (long u = 0; u < U; u++) {
        const long k = u / OD, od = u % OD;
        float *o = out + u * OH * OW;
        for (long c = 0; c < C; c++) {
            for (long a = 0; a < R; a++) {
                const long id = od * ST - P + a;
                if (id < 0 || id >= D)
                    continue;
                const float *xp = in + (c * D + id) * H * W;
                const float *w = wt + ((k * C + c) * R + a) * R * R;
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
                            const float *xr = xp + ih * W;
                            float *orow = o + oh * OW;
                            for (long ow = lo; ow < hi; ow++)
                                orow[ow] += wv * xr[ow * ST - P + s];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 9) {
        fprintf(stderr, "Usage: %s C D H W K R stride pad [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    C = atol(argv[1]);
    D = atol(argv[2]);
    H = atol(argv[3]);
    W = atol(argv[4]);
    K = atol(argv[5]);
    R = atol(argv[6]);
    ST = atol(argv[7]);
    P = atol(argv[8]);
    if (C < 1 || D < 1 || H < 1 || W < 1 || K < 1 || R < 1 || ST < 1 || P < 0) {
        fprintf(stderr, "ERROR: invalid dimensions\n");
        return 1;
    }
    OD = (D + 2 * P - R) / ST + 1;
    OH = (H + 2 * P - R) / ST + 1;
    OW = (W + 2 * P - R) / ST + 1;
    if (OD < 1 || OH < 1 || OW < 1) {
        fprintf(stderr, "ERROR: filter larger than padded input\n");
        return 1;
    }
    U = clamp_units(arg_long(argc, argv, 9, 0), K * OD);
    const char *mode = arg_str(argc, argv, 10, "cold");

    const size_t nin = (size_t)C * D * H * W, nwt = (size_t)K * C * R * R * R;
    const size_t nout = (size_t)K * OD * OH * OW;
    in = falloc(nin);
    wt = falloc(nwt);
    out = falloc(nout);
    fill_lcg(in, nin, 1);
    fill_lcg(wt, nwt, 2);

    ai_execute(mode, run, zero);

    char dims[160];
    snprintf(dims, sizeof dims,
             "C=%ld D=%ld H=%ld W=%ld K=%ld R=%ld stride=%ld pad=%ld OD=%ld OH=%ld OW=%ld",
             C, D, H, W, K, R, ST, P, OD, OH, OW);
    const double per_unit = (double)C * R * R * R * OH * OW;
    report_meta("conv3d", dims, mode, "direct", "output_channel_depth_plane", U, K * OD,
                per_unit * U, per_unit * K * OD,
                (double)nin * 4, (double)nwt * 4, (double)nout * 4);
    report_checksum(out, (size_t)U * OH * OW);
    return 0;
}
