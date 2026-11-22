/*
 * gat_aggregate.c - model-derived graph-attention (GAT) neighbour
 * aggregation for one layer of a sampled subgraph (float32), as in R-GAT.
 *
 * feat is [N][F] with F = heads * Dh (features already projected).
 * Target t is node t; its `fanout` sampled neighbours are nbr[t][k].
 * Per target t and head h (slice of Dh features):
 *   e_k    = LeakyReLU_0.2( a_src_h . feat[nbr_k]_h + a_dst_h . feat[t]_h )
 *   alpha  = softmax_k(e)
 *   out[t]_h = sum_k alpha_k * feat[nbr_k]_h
 *
 * All tensors are allocated with their real shapes; neighbour indices are
 * generated before the ROI. Work unit = one target node
 * (heads*fanout*2*Dh + heads*Dh MACs); the first `units` targets are computed.
 *
 * Usage: gat_aggregate N F heads fanout targets [units] [mode]
 * Seeds: feat = 1, a_src = 2, a_dst = 3, neighbour indices = 7 (uniform over N).
 */
#include "ai_common.h"

static long N, F, HEADS, DH, FAN, TG, U;
static float *feat, *asrc, *adst, *out, *e;
static int32_t *nbr;

static void zero(void)
{
    memset(out, 0, (size_t)TG * F * sizeof(float));
}

static void run(void)
{
    for (long t = 0; t < U; t++) {
        const int32_t *nb = nbr + t * FAN;
        const float *ft = feat + t * F;
        float *ot = out + t * F;
        for (long hd = 0; hd < HEADS; hd++) {
            const long off = hd * DH;
            float sd = 0.0f;
            for (long d = 0; d < DH; d++)
                sd += adst[off + d] * ft[off + d];
            float mx = -INFINITY;
            for (long k = 0; k < FAN; k++) {
                const float *fn = feat + (long)nb[k] * F + off;
                float s = sd;
                for (long d = 0; d < DH; d++)
                    s += asrc[off + d] * fn[d];
                s = s > 0.0f ? s : 0.2f * s;
                e[k] = s;
                if (s > mx)
                    mx = s;
            }
            float sum = 0.0f;
            for (long k = 0; k < FAN; k++) {
                e[k] = expf(e[k] - mx);
                sum += e[k];
            }
            const float inv = 1.0f / sum;
            for (long k = 0; k < FAN; k++) {
                const float a = e[k] * inv;
                const float *fn = feat + (long)nb[k] * F + off;
                for (long d = 0; d < DH; d++)
                    ot[off + d] += a * fn[d];
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "Usage: %s N F heads fanout targets [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    N = atol(argv[1]);
    F = atol(argv[2]);
    HEADS = atol(argv[3]);
    FAN = atol(argv[4]);
    TG = atol(argv[5]);
    if (N < 1 || F < 1 || HEADS < 1 || F % HEADS != 0 || FAN < 1 || TG < 1 || TG > N) {
        fprintf(stderr, "ERROR: invalid dimensions (F must be a multiple of heads, targets <= N)\n");
        return 1;
    }
    DH = F / HEADS;
    U = clamp_units(arg_long(argc, argv, 6, 0), TG);
    const char *mode = arg_str(argc, argv, 7, "cold");

    feat = falloc((size_t)N * F);
    asrc = falloc((size_t)F);
    adst = falloc((size_t)F);
    out = falloc((size_t)TG * F);
    e = falloc((size_t)FAN);
    nbr = (int32_t *)xalloc((size_t)TG * FAN * sizeof(int32_t));
    fill_lcg(feat, (size_t)N * F, 1);
    fill_lcg(asrc, (size_t)F, 2);
    fill_lcg(adst, (size_t)F, 3);
    uint32_t s = 7;
    for (long i = 0; i < TG * FAN; i++)
        nbr[i] = (int32_t)(lcg_next(&s) % (uint32_t)N);

    ai_execute(mode, run, zero);

    char dims[128];
    snprintf(dims, sizeof dims, "N=%ld F=%ld heads=%ld fanout=%ld targets=%ld", N, F, HEADS, FAN, TG);
    const double per_unit = (double)HEADS * (FAN * 2.0 * DH + DH);
    report_meta("gat_aggregate", dims, mode, "per_head_softmax", "target_node", U, TG,
                per_unit * U, per_unit * TG,
                (double)N * F * 4 + (double)TG * FAN * 4, 2.0 * F * 4, (double)TG * F * 4);
    report_checksum(out, (size_t)U * F);
    return 0;
}
