/*
 * mlp.c - model-derived multi-layer perceptron (float32), DLRM-style.
 *
 * For each sample b and layer l (d_l -> d_{l+1}):
 *   y = x * W_l + bias_l ;  ReLU on every layer except the last.
 * W_l is [d_l][d_{l+1}] row-major. Weights are reused across samples.
 * Work unit = one sample = sum_l d_l * d_{l+1} MACs.
 *
 * Usage: mlp d0,d1,...,dn B [units] [mode]
 *   e.g. DLRM-v2 bottom MLP: mlp 13,512,256,128 64
 * Seeds: input = 1, W_l = 10 + l, bias_l = 50 + l.
 */
#include "ai_common.h"

#define MAX_LAYERS 16

static long NL, Bsz, U, DMAX;
static long dim[MAX_LAYERS + 1];
static float *Wl[MAX_LAYERS], *bl[MAX_LAYERS];
static float *X, *out, *act0, *act1;

static void zero(void)
{
    memset(out, 0, (size_t)Bsz * dim[NL] * sizeof(float));
}

static void run(void)
{
    for (long b = 0; b < U; b++) {
        const float *x = X + b * dim[0];
        float *y = act0;
        for (long l = 0; l < NL; l++) {
            const long din = dim[l], dout = dim[l + 1];
            if (l == NL - 1)
                y = out + b * dout;
            memcpy(y, bl[l], (size_t)dout * sizeof(float));
            const float *w = Wl[l];
            for (long i = 0; i < din; i++) {
                const float xv = x[i];
                const float *wr = w + i * dout;
                for (long j = 0; j < dout; j++)
                    y[j] += xv * wr[j];
            }
            if (l != NL - 1) {
                for (long j = 0; j < dout; j++)
                    y[j] = y[j] > 0.0f ? y[j] : 0.0f;
                x = y;
                y = (y == act0) ? act1 : act0;
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s d0,d1,...,dn B [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    char spec[256];
    snprintf(spec, sizeof spec, "%s", argv[1]);
    long nd = 0;
    for (char *tok = strtok(spec, ","); tok; tok = strtok(NULL, ",")) {
        if (nd > MAX_LAYERS) {
            fprintf(stderr, "ERROR: at most %d layers\n", MAX_LAYERS);
            return 1;
        }
        dim[nd++] = atol(tok);
    }
    NL = nd - 1;
    Bsz = atol(argv[2]);
    if (NL < 1 || Bsz < 1) {
        fprintf(stderr, "ERROR: need at least two dims and B >= 1\n");
        return 1;
    }
    DMAX = 0;
    for (long i = 0; i <= NL; i++) {
        if (dim[i] < 1) {
            fprintf(stderr, "ERROR: invalid dimension\n");
            return 1;
        }
        if (dim[i] > DMAX)
            DMAX = dim[i];
    }
    U = clamp_units(arg_long(argc, argv, 3, 0), Bsz);
    const char *mode = arg_str(argc, argv, 4, "cold");

    double w_bytes = 0.0, per_unit = 0.0;
    for (long l = 0; l < NL; l++) {
        Wl[l] = falloc((size_t)dim[l] * dim[l + 1]);
        bl[l] = falloc((size_t)dim[l + 1]);
        fill_lcg(Wl[l], (size_t)dim[l] * dim[l + 1], 10u + (uint32_t)l);
        fill_lcg(bl[l], (size_t)dim[l + 1], 50u + (uint32_t)l);
        w_bytes += ((double)dim[l] * dim[l + 1] + dim[l + 1]) * 4;
        per_unit += (double)dim[l] * dim[l + 1];
    }
    X = falloc((size_t)Bsz * dim[0]);
    fill_lcg(X, (size_t)Bsz * dim[0], 1);
    out = falloc((size_t)Bsz * dim[NL]);
    act0 = falloc((size_t)DMAX);
    act1 = falloc((size_t)DMAX);

    ai_execute(mode, run, zero);

    char dims[300];
    snprintf(dims, sizeof dims, "layers=%s B=%ld", argv[1], Bsz);
    report_meta("mlp", dims, mode, "gemv_chain_relu", "sample", U, Bsz,
                per_unit * U, per_unit * Bsz,
                (double)Bsz * dim[0] * 4, w_bytes, (double)Bsz * dim[NL] * 4);
    report_checksum(out, (size_t)U * dim[NL]);
    return 0;
}
