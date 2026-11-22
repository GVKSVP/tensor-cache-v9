/*
 * attention.c - model-derived scaled dot-product attention (float32).
 *
 * Per head h:
 *   P   = softmax_row( Q_h * K_h^T / sqrt(D) )      (S x S)
 *   O_h = P * V_h                                   (S x D)
 *
 * Q, K, V, O are [heads][S][D], allocated for all heads with their real
 * shapes; only the first `units` heads are computed
 * (work unit = one head = 2*S*S*D MACs). The S x S score buffer is reused
 * across heads, as in an unfused per-head implementation.
 *
 * Usage: attention S D heads [units] [mode]
 * Seeds: Q = 1, K = 2, V = 3.
 */
#include "ai_common.h"

static long S, D, HEADS, U;
static float *Q, *Kt, *V, *O, *Pm;

static void zero(void)
{
    memset(O, 0, (size_t)HEADS * S * D * sizeof(float));
}

static void run(void)
{
    const float scale = 1.0f / sqrtf((float)D);
    for (long h = 0; h < U; h++) {
        const float *q = Q + h * S * D;
        const float *k = Kt + h * S * D;
        const float *v = V + h * S * D;
        float *o = O + h * S * D;

        /* scores = Q K^T * scale, then row softmax */
        for (long i = 0; i < S; i++) {
            float *p = Pm + i * S;
            const float *qi = q + i * D;
            float mx = -INFINITY;
            for (long j = 0; j < S; j++) {
                const float *kj = k + j * D;
                float acc = 0.0f;
                for (long d = 0; d < D; d++)
                    acc += qi[d] * kj[d];
                acc *= scale;
                p[j] = acc;
                if (acc > mx)
                    mx = acc;
            }
            float sum = 0.0f;
            for (long j = 0; j < S; j++) {
                p[j] = expf(p[j] - mx);
                sum += p[j];
            }
            const float inv = 1.0f / sum;
            for (long j = 0; j < S; j++)
                p[j] *= inv;
        }

        /* O = P V */
        for (long i = 0; i < S; i++) {
            const float *p = Pm + i * S;
            float *oi = o + i * D;
            for (long j = 0; j < S; j++) {
                const float pv = p[j];
                const float *vj = v + j * D;
                for (long d = 0; d < D; d++)
                    oi[d] += pv * vj[d];
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s S D heads [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    S = atol(argv[1]);
    D = atol(argv[2]);
    HEADS = atol(argv[3]);
    if (S < 1 || D < 1 || HEADS < 1) {
        fprintf(stderr, "ERROR: invalid dimensions\n");
        return 1;
    }
    U = clamp_units(arg_long(argc, argv, 4, 0), HEADS);
    const char *mode = arg_str(argc, argv, 5, "cold");

    const size_t n = (size_t)HEADS * S * D;
    Q = falloc(n);
    Kt = falloc(n);
    V = falloc(n);
    O = falloc(n);
    Pm = falloc((size_t)S * S);
    fill_lcg(Q, n, 1);
    fill_lcg(Kt, n, 2);
    fill_lcg(V, n, 3);

    ai_execute(mode, run, zero);

    char dims[96];
    snprintf(dims, sizeof dims, "S=%ld D=%ld heads=%ld", S, D, HEADS);
    const double per_unit = 2.0 * S * S * D;
    report_meta("attention", dims, mode, "unfused_per_head", "head", U, HEADS,
                per_unit * U, per_unit * HEADS,
                3.0 * n * 4 + (double)S * S * 4, 0.0, (double)n * 4);
    report_checksum(O, (size_t)U * S * D);
    return 0;
}
