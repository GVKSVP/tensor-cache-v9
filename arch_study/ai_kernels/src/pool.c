/*
 * pool.c - model-derived 2D pooling (float32, NCHW, N=1, square window).
 *
 *   max: out = max over the R x R window (padding positions ignored)
 *   avg: out = sum over the window / (R*R)   (padding counted as zero,
 *        i.e. count_include_pad = True, the PyTorch default)
 *
 * Global average pooling = avg with R = H = W, stride 1, pad 0.
 * Work unit = one channel = OH*OW*R*R operations.
 *
 * Usage: pool max|avg C H W R stride pad [units] [mode]
 * Seed: input = 1.
 */
#include "ai_common.h"

static long C, H, W, R, ST, P, OH, OW, U;
static int is_max;
static float *in, *out;

static void zero(void)
{
    memset(out, 0, (size_t)C * OH * OW * sizeof(float));
}

static void run(void)
{
    const float inv = 1.0f / (float)(R * R);
    for (long c = 0; c < U; c++) {
        const float *x = in + c * H * W;
        float *o = out + c * OH * OW;
        for (long oh = 0; oh < OH; oh++) {
            for (long ow = 0; ow < OW; ow++) {
                float acc = is_max ? -INFINITY : 0.0f;
                for (long r = 0; r < R; r++) {
                    const long ih = oh * ST - P + r;
                    if (ih < 0 || ih >= H)
                        continue;
                    for (long s = 0; s < R; s++) {
                        const long iw = ow * ST - P + s;
                        if (iw < 0 || iw >= W)
                            continue;
                        const float v = x[ih * W + iw];
                        if (is_max)
                            acc = v > acc ? v : acc;
                        else
                            acc += v;
                    }
                }
                o[oh * OW + ow] = is_max ? acc : acc * inv;
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 8) {
        fprintf(stderr, "Usage: %s max|avg C H W R stride pad [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "max") == 0)
        is_max = 1;
    else if (strcmp(argv[1], "avg") == 0)
        is_max = 0;
    else {
        fprintf(stderr, "ERROR: type must be max|avg\n");
        return 1;
    }
    C = atol(argv[2]);
    H = atol(argv[3]);
    W = atol(argv[4]);
    R = atol(argv[5]);
    ST = atol(argv[6]);
    P = atol(argv[7]);
    if (C < 1 || H < 1 || W < 1 || R < 1 || ST < 1 || P < 0) {
        fprintf(stderr, "ERROR: invalid dimensions\n");
        return 1;
    }
    OH = (H + 2 * P - R) / ST + 1;
    OW = (W + 2 * P - R) / ST + 1;
    if (OH < 1 || OW < 1) {
        fprintf(stderr, "ERROR: window larger than padded input\n");
        return 1;
    }
    U = clamp_units(arg_long(argc, argv, 8, 0), C);
    const char *mode = arg_str(argc, argv, 9, "cold");

    in = falloc((size_t)C * H * W);
    out = falloc((size_t)C * OH * OW);
    fill_lcg(in, (size_t)C * H * W, 1);

    ai_execute(mode, run, zero);

    char dims[128];
    snprintf(dims, sizeof dims, "type=%s C=%ld H=%ld W=%ld R=%ld stride=%ld pad=%ld OH=%ld OW=%ld",
             argv[1], C, H, W, R, ST, P, OH, OW);
    const double per_unit = (double)OH * OW * R * R;
    report_meta("pool", dims, mode, is_max ? "max" : "avg", "channel", U, C,
                per_unit * U, per_unit * C,
                (double)C * H * W * 4, 0.0, (double)C * OH * OW * 4);
    report_checksum(out, (size_t)U * OH * OW);
    return 0;
}
