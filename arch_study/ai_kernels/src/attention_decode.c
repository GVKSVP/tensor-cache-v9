/*
 * attention_decode.c - model-derived LLM decode attention with a KV cache
 * and grouped-query attention (GQA), float32, batch 1, one new token.
 *
 * For query head h (kv head g = h / (QH / KVH)):
 *   p   = softmax( q_h . K_g^T / sqrt(D) )     (1 x L)
 *   o_h = p * V_g                              (1 x D)
 *
 * q and o are [QH][D]; the KV cache K, V is [KVH][L][D] (the layout used by
 * Hugging Face Llama). All tensors are allocated with their real shapes;
 * only the first `units` query heads are computed
 * (work unit = one query head = 2*L*D MACs). The L-long score buffer is
 * reused across heads, as in an unfused per-head implementation.
 *
 * Usage: attention_decode L D q_heads kv_heads [units] [mode]
 * Seeds: q = 1, K = 2, V = 3.
 */
#include "ai_common.h"

static long L, D, QH, KVH, U;
static float *q, *Kc, *Vc, *o, *p;

static void zero(void)
{
    memset(o, 0, (size_t)QH * D * sizeof(float));
}

static void run(void)
{
    const float scale = 1.0f / sqrtf((float)D);
    const long group = QH / KVH;
    for (long h = 0; h < U; h++) {
        const float *qh = q + h * D;
        const float *k = Kc + (h / group) * L * D;
        const float *v = Vc + (h / group) * L * D;
        float *oh = o + h * D;

        /* scores = q K^T * scale, then softmax */
        float mx = -INFINITY;
        for (long j = 0; j < L; j++) {
            const float *kj = k + j * D;
            float acc = 0.0f;
            for (long d = 0; d < D; d++)
                acc += qh[d] * kj[d];
            acc *= scale;
            p[j] = acc;
            if (acc > mx)
                mx = acc;
        }
        float sum = 0.0f;
        for (long j = 0; j < L; j++) {
            p[j] = expf(p[j] - mx);
            sum += p[j];
        }
        const float inv = 1.0f / sum;
        for (long j = 0; j < L; j++)
            p[j] *= inv;

        /* o = p V */
        for (long j = 0; j < L; j++) {
            const float pv = p[j];
            const float *vj = v + j * D;
            for (long d = 0; d < D; d++)
                oh[d] += pv * vj[d];
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s L D q_heads kv_heads [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    L = atol(argv[1]);
    D = atol(argv[2]);
    QH = atol(argv[3]);
    KVH = atol(argv[4]);
    if (L < 1 || D < 1 || QH < 1 || KVH < 1 || QH % KVH != 0) {
        fprintf(stderr, "ERROR: invalid dimensions (q_heads must be a multiple of kv_heads)\n");
        return 1;
    }
    U = clamp_units(arg_long(argc, argv, 5, 0), QH);
    const char *mode = arg_str(argc, argv, 6, "cold");

    const size_t nq = (size_t)QH * D, nkv = (size_t)KVH * L * D;
    q = falloc(nq);
    Kc = falloc(nkv);
    Vc = falloc(nkv);
    o = falloc(nq);
    p = falloc((size_t)L);
    fill_lcg(q, nq, 1);
    fill_lcg(Kc, nkv, 2);
    fill_lcg(Vc, nkv, 3);

    ai_execute(mode, run, zero);

    char dims[96];
    snprintf(dims, sizeof dims, "L=%ld D=%ld q_heads=%ld kv_heads=%ld", L, D, QH, KVH);
    const double per_unit = 2.0 * L * D;
    /* input = query + KV cache (+ score buffer); no weights in this op */
    report_meta("attention_decode", dims, mode, "unfused_per_head_gqa", "query_head", U, QH,
                per_unit * U, per_unit * QH,
                (double)nq * 4 + 2.0 * nkv * 4 + (double)L * 4, 0.0, (double)nq * 4);
    report_checksum(o, (size_t)U * D);
    return 0;
}
