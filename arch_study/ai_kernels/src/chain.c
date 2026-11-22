/*
 * chain.c - model-derived LAYER CHAINS (float32): consecutive layers of one
 * model block run back to back inside one ROI, and the output tensor of
 * layer k is the real input tensor of layer k+1 (shared memory). Used to
 * study cross-layer cache behaviour (next-layer reuse, dead tensors).
 *
 * Each layer is its own non-inlined function (layer_<chain>_<k>_<name>), so
 * the PC of a memory request tells which layer issued it (ELF symbols).
 * Maths follows the single-layer kernels (gemm.c ikj order, conv2d_nchw.c,
 * depthwise_conv.c, lstm.c, attention_decode.c); those files are unchanged.
 *
 * Sampling is CONSISTENT along the chain, so data really flows:
 *   bert_ffn, vit_mlp  : unit = token row; every layer computes rows 0..U-1
 *   rn50_bottleneck,
 *   mbv2_block         : unit = output image row; earlier layers compute the
 *                        rows (plus 3x3 halo) the later layers need
 *   rnnt_encoder       : unit = post-stack time step; pre layers run 2U steps
 *   llama_decode       : one full decode step (all layers complete)
 *
 * Usage: chain NAME[:small] [units] [mode]
 *   NAME:small = tiny shapes for the numpy check (verify_chain.py) and the
 *   gem5 smoke test.
 * The checksum covers the whole final output tensor (unsampled parts are 0).
 */
#include "ai_common.h"

#define NOINL __attribute__((noinline))

static const char *CHAIN;
static long U, UTOT;
static int SMALL;
static double MACS_EXEC, MACS_TOTAL, IN_B, W_B, OUT_B;
static float *OUT;
static size_t OUT_N;
static char DIMS[160];

static float *tensor(size_t n, uint32_t seed)
{
    float *p = falloc(n);
    fill_lcg(p, n, seed);
    return p;
}

static float *buffer(size_t n)
{
    float *p = falloc(n);
    memset(p, 0, n * sizeof(float));
    return p;
}

/* c[0..N) += a[0..K) * B[K][N]   (one GEMM row, ikj order as gemm.c) */
static inline __attribute__((always_inline))
void gemm_row(float *c, const float *a, const float *B, long K, long N)
{
    for (long k = 0; k < K; k++) {
        const float av = a[k];
        const float *b = B + k * N;
        for (long j = 0; j < N; j++)
            c[j] += av * b[j];
    }
}

static inline float gelu(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline float sigm(float v)
{
    return 1.0f / (1.0f + expf(-v));
}

/* ================================================================== */
/* bert_ffn / vit_mlp: attention-output projection -> FFN up -> down   */
/* ================================================================== */
static long T_S, T_D, T_F;
static float *t_O, *t_Wo, *t_X1, *t_W1, *t_H, *t_W2, *t_Y;

static NOINL void layer_tf_1_oproj(void)
{
    for (long i = 0; i < U; i++)
        gemm_row(t_X1 + i * T_D, t_O + i * T_D, t_Wo, T_D, T_D);
}

static NOINL void layer_tf_2_ffn_up(void)
{
    for (long i = 0; i < U; i++) {
        float *h = t_H + i * T_F;
        gemm_row(h, t_X1 + i * T_D, t_W1, T_D, T_F);
        for (long j = 0; j < T_F; j++)
            h[j] = gelu(h[j]);
    }
}

static NOINL void layer_tf_3_ffn_down(void)
{
    for (long i = 0; i < U; i++)
        gemm_row(t_Y + i * T_D, t_H + i * T_F, t_W2, T_F, T_D);
}

static void tf_run(void)
{
    layer_tf_1_oproj();
    layer_tf_2_ffn_up();
    layer_tf_3_ffn_down();
}

static void tf_zero(void)
{
    memset(t_X1, 0, (size_t)T_S * T_D * 4);
    memset(t_H, 0, (size_t)T_S * T_F * 4);
    memset(t_Y, 0, (size_t)T_S * T_D * 4);
}

static void tf_setup(long S, long D, long F, long def_units)
{
    T_S = S; T_D = D; T_F = F;
    t_O = tensor((size_t)S * D, 1);
    t_Wo = tensor((size_t)D * D, 2);
    t_X1 = buffer((size_t)S * D);
    t_W1 = tensor((size_t)D * F, 3);
    t_H = buffer((size_t)S * F);
    t_W2 = tensor((size_t)F * D, 4);
    t_Y = buffer((size_t)S * D);
    UTOT = S;
    U = clamp_units(U ? U : def_units, S);
    const double per = (double)D * D + 2.0 * D * F;
    MACS_EXEC = per * U; MACS_TOTAL = per * S;
    IN_B = (double)S * D * 4; W_B = (D * (double)D + 2.0 * D * F) * 4; OUT_B = (double)S * D * 4;
    OUT = t_Y; OUT_N = (size_t)S * D;
    snprintf(DIMS, sizeof DIMS, "S=%ld D=%ld F=%ld", S, D, F);
}

/* ================================================================== */
/* rn50_bottleneck / mbv2_block: 1x1 -> 3x3 -> 1x1 (NCHW, stride 1)    */
/* ================================================================== */
static long C_IN, C_MID, C_OUT, C_H, C_W, C_DW; /* C_DW: 1 = depthwise 3x3 */
static float *c_X, *c_Wa, *c_A, *c_Wb, *c_B, *c_Wc, *c_Y;
static float C_CLIP; /* ReLU (0 = no upper clip) or ReLU6 */

static inline float act(float v)
{
    v = v > 0.0f ? v : 0.0f;
    return (C_CLIP > 0.0f && v > C_CLIP) ? C_CLIP : v;
}

/* 1x1 conv on rows [0, rows): o[k] += sum_c w[k][c] * x[c] */
static inline __attribute__((always_inline))
void conv1x1_rows(float *o, const float *x, const float *w, long K, long C,
                  long rows, long HW, long W)
{
    for (long k = 0; k < K; k++) {
        float *ok = o + k * HW;
        for (long c = 0; c < C; c++) {
            const float wv = w[k * C + c];
            const float *xc = x + c * HW;
            for (long i = 0; i < rows * W; i++)
                ok[i] += wv * xc[i];
        }
    }
}

static NOINL void layer_cv_1_expand(void)
{
    const long rows = U + 1 < C_H ? U + 1 : C_H; /* +1 halo row for the 3x3 */
    const long HW = C_H * C_W;
    conv1x1_rows(c_A, c_X, c_Wa, C_MID, C_IN, rows, HW, C_W);
    for (long k = 0; k < C_MID; k++)
        for (long i = 0; i < rows * C_W; i++)
            c_A[k * HW + i] = act(c_A[k * HW + i]);
}

static NOINL void layer_cv_2_conv3x3(void)
{
    const long HW = C_H * C_W;
    for (long k = 0; k < C_MID; k++) {
        float *o = c_B + k * HW;
        const long c0 = C_DW ? k : 0, c1 = C_DW ? k + 1 : C_MID;
        for (long c = c0; c < c1; c++) {
            const float *x = c_A + c * HW;
            const float *w = C_DW ? c_Wb + k * 9 : c_Wb + (k * C_MID + c) * 9;
            for (long r = 0; r < 3; r++) {
                for (long s = 0; s < 3; s++) {
                    const float wv = w[r * 3 + s];
                    const long lo = s == 0 ? 1 : 0;
                    const long hi = s == 2 ? C_W - 1 : C_W;
                    for (long oh = 0; oh < U; oh++) {
                        const long ih = oh - 1 + r;
                        if (ih < 0 || ih >= C_H)
                            continue;
                        const float *xr = x + ih * C_W;
                        float *orow = o + oh * C_W;
                        for (long ow = lo; ow < hi; ow++)
                            orow[ow] += wv * xr[ow - 1 + s];
                    }
                }
            }
        }
        for (long i = 0; i < U * C_W; i++)
            o[i] = act(o[i]);
    }
}

static NOINL void layer_cv_3_project(void)
{
    conv1x1_rows(c_Y, c_B, c_Wc, C_OUT, C_MID, U, C_H * C_W, C_W);
}

static void cv_run(void)
{
    layer_cv_1_expand();
    layer_cv_2_conv3x3();
    layer_cv_3_project();
}

static void cv_zero(void)
{
    const size_t HW = (size_t)C_H * C_W;
    memset(c_A, 0, C_MID * HW * 4);
    memset(c_B, 0, C_MID * HW * 4);
    memset(c_Y, 0, C_OUT * HW * 4);
}

static void cv_setup(long cin, long cmid, long cout, long H, long dw, float clip, long def_units)
{
    C_IN = cin; C_MID = cmid; C_OUT = cout; C_H = H; C_W = H; C_DW = dw; C_CLIP = clip;
    const size_t HW = (size_t)H * H;
    const size_t nb = dw ? (size_t)cmid * 9 : (size_t)cmid * cmid * 9;
    c_X = tensor(cin * HW, 1);
    c_Wa = tensor((size_t)cmid * cin, 2);
    c_A = buffer(cmid * HW);
    c_Wb = tensor(nb, 3);
    c_B = buffer(cmid * HW);
    c_Wc = tensor((size_t)cout * cmid, 4);
    c_Y = buffer(cout * HW);
    UTOT = H;
    U = clamp_units(U ? U : def_units, H);
    const long rows_a = U + 1 < H ? U + 1 : H;
    MACS_EXEC = (double)rows_a * H * cmid * cin + (double)U * H * nb + (double)U * H * cout * cmid;
    MACS_TOTAL = (double)HW * (cmid * cin + nb + cout * cmid);
    IN_B = cin * HW * 4.0; W_B = ((double)cmid * cin + nb + (double)cout * cmid) * 4;
    OUT_B = cout * HW * 4.0;
    OUT = c_Y; OUT_N = cout * HW;
    snprintf(DIMS, sizeof DIMS, "Cin=%ld Cmid=%ld Cout=%ld H=W=%ld %s", cin, cmid, cout, H,
             dw ? "depthwise" : "full3x3");
}

/* ================================================================== */
/* rnnt_encoder: LSTM pre1 -> LSTM pre2 -> stack 2 frames -> LSTM post1 */
/* ================================================================== */
typedef struct {
    long I, H, T;
    float *wih, *whh, *b, *y, *h, *c, *z;
    const float *x;
} Lstm;
static Lstm R1, R2, R3;

static inline __attribute__((always_inline)) void lstm_steps(Lstm *L, long steps)
{
    const long H = L->H, I = L->I;
    for (long t = 0; t < steps; t++) {
        const float *xt = L->x + t * I;
        for (long r = 0; r < 4 * H; r++) {
            const float *wi = L->wih + r * I;
            const float *wh = L->whh + r * H;
            float acc = L->b[r];
            for (long k = 0; k < I; k++)
                acc += wi[k] * xt[k];
            for (long k = 0; k < H; k++)
                acc += wh[k] * L->h[k];
            L->z[r] = acc;
        }
        float *yt = L->y + t * H;
        for (long j = 0; j < H; j++) {
            const float ig = sigm(L->z[j]);
            const float fg = sigm(L->z[H + j]);
            const float gg = tanhf(L->z[2 * H + j]);
            const float og = sigm(L->z[3 * H + j]);
            L->c[j] = fg * L->c[j] + ig * gg;
            L->h[j] = og * tanhf(L->c[j]);
            yt[j] = L->h[j];
        }
    }
}

static NOINL void layer_rn_1_pre1(void) { lstm_steps(&R1, 2 * U); }
static NOINL void layer_rn_2_pre2(void) { lstm_steps(&R2, 2 * U); }
static NOINL void layer_rn_3_post1(void) { lstm_steps(&R3, U); }

static void rn_run(void)
{
    layer_rn_1_pre1();
    layer_rn_2_pre2();
    layer_rn_3_post1();
}

static void rn_zero_one(Lstm *L)
{
    memset(L->y, 0, (size_t)L->T * L->H * 4);
    memset(L->h, 0, (size_t)L->H * 4);
    memset(L->c, 0, (size_t)L->H * 4);
}

static void rn_zero(void)
{
    rn_zero_one(&R1);
    rn_zero_one(&R2);
    rn_zero_one(&R3);
}

static void rn_alloc(Lstm *L, long I, long H, long T, const float *x, uint32_t seed)
{
    L->I = I; L->H = H; L->T = T; L->x = x;
    L->wih = tensor((size_t)4 * H * I, seed);
    L->whh = tensor((size_t)4 * H * H, seed + 1);
    L->b = tensor((size_t)4 * H, seed + 2);
    L->y = buffer((size_t)T * H);
    L->h = buffer((size_t)H);
    L->c = buffer((size_t)H);
    L->z = buffer((size_t)4 * H);
}

static void rn_setup(long I, long H, long T, long def_units)
{
    float *x = tensor((size_t)T * I, 1);
    rn_alloc(&R1, I, H, T, x, 10);
    rn_alloc(&R2, H, H, T, R1.y, 20);
    rn_alloc(&R3, 2 * H, H, T / 2, R2.y, 30); /* stacked: row t = y2[2t] ++ y2[2t+1] */
    UTOT = T / 2;
    U = clamp_units(U ? U : def_units, T / 2);
    const double p1 = 4.0 * H * (I + H), p2 = 4.0 * H * (2.0 * H), p3 = 4.0 * H * (3.0 * H);
    MACS_EXEC = 2.0 * U * (p1 + p2) + U * p3;
    MACS_TOTAL = (double)T * (p1 + p2) + (double)(T / 2) * p3;
    IN_B = (double)T * I * 4;
    W_B = (p1 + p2 + p3 + 12.0 * H) * 4;
    OUT_B = (double)(T / 2) * H * 4;
    OUT = R3.y; OUT_N = (size_t)(T / 2) * H;
    snprintf(DIMS, sizeof DIMS, "I=%ld H=%ld T=%ld (pre1,pre2,stack2,post1)", I, H, T);
}

/* ================================================================== */
/* llama_decode: one decode step of a Llama block (GQA attention)      */
/* ================================================================== */
static long L_D, L_KVH, L_QH, L_DH, L_L, L_F;
static float *l_x, *l_Wq, *l_q, *l_Wk, *l_Wv, *l_kn, *l_vn, *l_Kc, *l_Vc, *l_p, *l_o,
             *l_Wo, *l_y, *l_Wup, *l_h, *l_Wdn, *l_out;

static NOINL void layer_ll_1_q_proj(void)
{
    gemm_row(l_q, l_x, l_Wq, L_D, L_D);
}

static NOINL void layer_ll_2_kv_proj(void)
{
    const long kvd = L_KVH * L_DH;
    gemm_row(l_kn, l_x, l_Wk, L_D, kvd);
    gemm_row(l_vn, l_x, l_Wv, L_D, kvd);
    for (long g = 0; g < L_KVH; g++) { /* append to the cache at position L */
        memcpy(l_Kc + (g * (L_L + 1) + L_L) * L_DH, l_kn + g * L_DH, L_DH * 4);
        memcpy(l_Vc + (g * (L_L + 1) + L_L) * L_DH, l_vn + g * L_DH, L_DH * 4);
    }
}

static NOINL void layer_ll_3_attn(void)
{
    const long D = L_DH, Lt = L_L + 1, group = L_QH / L_KVH;
    const float scale = 1.0f / sqrtf((float)D);
    for (long hh = 0; hh < L_QH; hh++) {
        const float *qh = l_q + hh * D;
        const float *k = l_Kc + (hh / group) * Lt * D;
        const float *v = l_Vc + (hh / group) * Lt * D;
        float *oh = l_o + hh * D;
        float mx = -INFINITY;
        for (long j = 0; j < Lt; j++) {
            const float *kj = k + j * D;
            float acc = 0.0f;
            for (long d = 0; d < D; d++)
                acc += qh[d] * kj[d];
            acc *= scale;
            l_p[j] = acc;
            if (acc > mx)
                mx = acc;
        }
        float sum = 0.0f;
        for (long j = 0; j < Lt; j++) {
            l_p[j] = expf(l_p[j] - mx);
            sum += l_p[j];
        }
        const float inv = 1.0f / sum;
        for (long j = 0; j < Lt; j++)
            l_p[j] *= inv;
        for (long j = 0; j < Lt; j++) {
            const float pv = l_p[j];
            const float *vj = v + j * D;
            for (long d = 0; d < D; d++)
                oh[d] += pv * vj[d];
        }
    }
}

static NOINL void layer_ll_4_o_proj(void)
{
    gemm_row(l_y, l_o, l_Wo, L_D, L_D);
}

static NOINL void layer_ll_5_ffn_up(void)
{
    gemm_row(l_h, l_y, l_Wup, L_D, L_F);
    for (long j = 0; j < L_F; j++)
        l_h[j] = l_h[j] * sigm(l_h[j]); /* SiLU */
}

static NOINL void layer_ll_6_ffn_down(void)
{
    gemm_row(l_out, l_h, l_Wdn, L_F, L_D);
}

static void ll_run(void)
{
    layer_ll_1_q_proj();
    layer_ll_2_kv_proj();
    layer_ll_3_attn();
    layer_ll_4_o_proj();
    layer_ll_5_ffn_up();
    layer_ll_6_ffn_down();
}

static void ll_zero(void)
{
    const long kvd = L_KVH * L_DH;
    memset(l_q, 0, L_D * 4);
    memset(l_kn, 0, kvd * 4);
    memset(l_vn, 0, kvd * 4);
    memset(l_o, 0, L_D * 4);
    memset(l_y, 0, L_D * 4);
    memset(l_h, 0, L_F * 4);
    memset(l_out, 0, L_D * 4);
}

static void ll_setup(long D, long KVH, long QH, long DH, long L, long F)
{
    L_D = D; L_KVH = KVH; L_QH = QH; L_DH = DH; L_L = L; L_F = F;
    const long kvd = KVH * DH;
    const size_t ncache = (size_t)KVH * (L + 1) * DH;
    l_x = tensor(D, 1);
    l_Wq = tensor((size_t)D * D, 2);
    l_q = buffer(D);
    l_Wk = tensor((size_t)D * kvd, 3);
    l_Wv = tensor((size_t)D * kvd, 4);
    l_kn = buffer(kvd);
    l_vn = buffer(kvd);
    l_Kc = tensor(ncache, 5); /* positions 0..L-1 = earlier tokens; L = this token */
    l_Vc = tensor(ncache, 6);
    l_p = buffer(L + 1);
    l_o = buffer(D);
    l_Wo = tensor((size_t)D * D, 7);
    l_y = buffer(D);
    l_Wup = tensor((size_t)D * F, 8);
    l_h = buffer(F);
    l_Wdn = tensor((size_t)F * D, 9);
    l_out = buffer(D);
    UTOT = 1;
    U = 1;
    MACS_EXEC = MACS_TOTAL = (double)D * D * 2 + 2.0 * D * kvd + 2.0 * QH * (L + 1) * DH
                             + 2.0 * D * F;
    IN_B = D * 4.0;
    W_B = ((double)D * D * 2 + 2.0 * D * kvd + 2.0 * D * F) * 4 + 2.0 * ncache * 4;
    OUT_B = D * 4.0;
    OUT = l_out; OUT_N = D;
    snprintf(DIMS, sizeof DIMS, "D=%ld KVH=%ld QH=%ld Dh=%ld L=%ld F=%ld", D, KVH, QH, DH, L, F);
}

/* ================================================================== */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s bert_ffn|vit_mlp|rn50_bottleneck|mbv2_block|"
                        "rnnt_encoder|llama_decode[:small] [units] [cold|warm|none]\n",
                argv[0]);
        return 1;
    }
    static char cname[48];
    snprintf(cname, sizeof cname, "%s", argv[1]);
    char *suffix = strchr(cname, ':');
    SMALL = suffix && strcmp(suffix, ":small") == 0;
    if (suffix)
        *suffix = '\0';
    CHAIN = cname;
    U = arg_long(argc, argv, 2, 0);
    const char *mode = arg_str(argc, argv, 3, "cold");
    void (*run)(void), (*zero)(void);
    const char *unit;

    if (!strcmp(CHAIN, "bert_ffn")) {
        SMALL ? tf_setup(6, 8, 16, 3) : tf_setup(384, 1024, 4096, 5);
        run = tf_run; zero = tf_zero; unit = "token_row";
    } else if (!strcmp(CHAIN, "vit_mlp")) {
        SMALL ? tf_setup(5, 8, 16, 2) : tf_setup(197, 768, 3072, 8);
        run = tf_run; zero = tf_zero; unit = "token_row";
    } else if (!strcmp(CHAIN, "rn50_bottleneck")) {
        SMALL ? cv_setup(8, 4, 8, 6, 0, 0.0f, 3) : cv_setup(256, 64, 256, 56, 0, 0.0f, 6);
        run = cv_run; zero = cv_zero; unit = "output_row";
    } else if (!strcmp(CHAIN, "mbv2_block")) {
        SMALL ? cv_setup(4, 8, 4, 6, 1, 6.0f, 6) : cv_setup(24, 144, 24, 56, 1, 6.0f, 56);
        run = cv_run; zero = cv_zero; unit = "output_row";
    } else if (!strcmp(CHAIN, "rnnt_encoder")) {
        SMALL ? rn_setup(5, 4, 8, 2) : rn_setup(240, 1024, 557, 1);
        run = rn_run; zero = rn_zero; unit = "post_time_step";
    } else if (!strcmp(CHAIN, "llama_decode")) {
        SMALL ? ll_setup(16, 2, 4, 4, 5, 24) : ll_setup(4096, 8, 32, 128, 1024, 14336);
        run = ll_run; zero = ll_zero; unit = "decode_step";
    } else {
        fprintf(stderr, "ERROR: unknown chain %s\n", CHAIN);
        return 1;
    }

    ai_execute(mode, run, zero);

    char kname[64];
    snprintf(kname, sizeof kname, "chain:%s", CHAIN);
    report_meta(kname, DIMS, mode, SMALL ? "small" : "full", unit, U, UTOT,
                MACS_EXEC, MACS_TOTAL, IN_B, W_B, OUT_B);
    report_checksum(OUT, OUT_N);
    return 0;
}
