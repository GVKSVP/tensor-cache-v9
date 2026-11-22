/*
 * block.c - model-derived WHOLE BLOCKS (float32) for the multi-layer study
 * (Step 2). One ROI runs every layer of one model block back to back; the
 * output tensor of each layer is the real input of the next (shared memory),
 * and residual connections re-read earlier tensors, as in real inference.
 *
 * Each layer is its own non-inlined function (layer_<blk>_<k>_<name>), so the
 * PC of a memory request tells which layer issued it (ELF symbols), exactly
 * as in chain.c (chain.c is unchanged).
 *
 * enc_block (bert_block, vit_block): post-LN Transformer encoder layer
 *   1 qkv      Q,K,V = X Wq, X Wk, X Wv           (rows 0..U-1)
 *   2 attn     per head: softmax(Q K^T / sqrt(dh)) V over ALL S keys
 *   3 oproj    A = Ctx Wo
 *   4 add_ln1  X1 = LayerNorm(X + A)               (re-reads X: residual)
 *   5 ffn_up   H = GELU(X1 W1)
 *   6 ffn_down Y = H W2
 *   7 add_ln2  Out = LayerNorm(X1 + Y)             (re-reads X1: residual)
 * Sampling: unit = token row; every layer computes rows 0..U-1, so data
 * really flows through the block. Attention needs K,V of all S tokens:
 * rows 0..U-1 are produced by layer 1 inside the ROI, rows U..S-1 are
 * "computed earlier by other row tiles" and are initialised in setup
 * (deterministic LCG values, outside the ROI, so they come from DRAM after
 * the cold flush). Every weight matrix is therefore streamed U times per
 * layer (row-by-row ikj GEMM, as gemm.c), and K,V are re-read by every
 * query row and head. LayerNorm has no affine parameters (gamma=1, beta=0)
 * and there are no biases; this does not change the memory traffic shape
 * of the weight/activation tensors noticeably and keeps the reference simple.
 *
 * Further blocks (below): rn50_block2 / mbv2_block2 (two residual conv blocks)
 * and dlrm_block (bottom MLP -> embedding bags -> interaction -> top MLP).
 *
 * Usage: block NAME[:small] [units] [mode]
 * The checksum covers the whole final output tensor (unsampled rows are 0).
 */
#include "ai_common.h"

#define NOINL __attribute__((noinline))

static const char *BLOCK;
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

/* y[0..D) = LayerNorm(a + b), no affine parameters, eps 1e-5 */
static inline __attribute__((always_inline))
void add_ln_row(float *y, const float *a, const float *b, long D)
{
    float mean = 0.0f;
    for (long j = 0; j < D; j++) {
        y[j] = a[j] + b[j];
        mean += y[j];
    }
    mean /= (float)D;
    float var = 0.0f;
    for (long j = 0; j < D; j++) {
        const float d = y[j] - mean;
        var += d * d;
    }
    var /= (float)D;
    const float inv = 1.0f / sqrtf(var + 1e-5f);
    for (long j = 0; j < D; j++)
        y[j] = (y[j] - mean) * inv;
}

/* ================================================================== */
/* enc_block: post-LN Transformer encoder layer (BERT, ViT)            */
/* ================================================================== */
static long E_S, E_D, E_H, E_DH, E_F;
static float *e_X, *e_Wq, *e_Wk, *e_Wv, *e_Q, *e_K, *e_V, *e_P, *e_C, *e_Wo, *e_A,
             *e_X1, *e_W1, *e_H, *e_W2, *e_Y, *e_Out;

static NOINL void layer_eb_1_qkv(void)
{
    for (long i = 0; i < U; i++) {
        const float *x = e_X + i * E_D;
        gemm_row(e_Q + i * E_D, x, e_Wq, E_D, E_D);
        gemm_row(e_K + i * E_D, x, e_Wk, E_D, E_D);
        gemm_row(e_V + i * E_D, x, e_Wv, E_D, E_D);
    }
}

static NOINL void layer_eb_2_attn(void)
{
    const long D = E_D, DH = E_DH, S = E_S;
    const float scale = 1.0f / sqrtf((float)DH);
    for (long i = 0; i < U; i++) {
        for (long h = 0; h < E_H; h++) {
            const float *q = e_Q + i * D + h * DH;
            float mx = -INFINITY;
            for (long j = 0; j < S; j++) {
                const float *k = e_K + j * D + h * DH;
                float acc = 0.0f;
                for (long d = 0; d < DH; d++)
                    acc += q[d] * k[d];
                acc *= scale;
                e_P[j] = acc;
                if (acc > mx)
                    mx = acc;
            }
            float sum = 0.0f;
            for (long j = 0; j < S; j++) {
                e_P[j] = expf(e_P[j] - mx);
                sum += e_P[j];
            }
            const float inv = 1.0f / sum;
            float *c = e_C + i * D + h * DH;
            for (long j = 0; j < S; j++) {
                const float pv = e_P[j] * inv;
                const float *v = e_V + j * D + h * DH;
                for (long d = 0; d < DH; d++)
                    c[d] += pv * v[d];
            }
        }
    }
}

static NOINL void layer_eb_3_oproj(void)
{
    for (long i = 0; i < U; i++)
        gemm_row(e_A + i * E_D, e_C + i * E_D, e_Wo, E_D, E_D);
}

static NOINL void layer_eb_4_add_ln1(void)
{
    for (long i = 0; i < U; i++)
        add_ln_row(e_X1 + i * E_D, e_X + i * E_D, e_A + i * E_D, E_D);
}

static NOINL void layer_eb_5_ffn_up(void)
{
    for (long i = 0; i < U; i++) {
        float *h = e_H + i * E_F;
        gemm_row(h, e_X1 + i * E_D, e_W1, E_D, E_F);
        for (long j = 0; j < E_F; j++)
            h[j] = gelu(h[j]);
    }
}

static NOINL void layer_eb_6_ffn_down(void)
{
    for (long i = 0; i < U; i++)
        gemm_row(e_Y + i * E_D, e_H + i * E_F, e_W2, E_F, E_D);
}

static NOINL void layer_eb_7_add_ln2(void)
{
    for (long i = 0; i < U; i++)
        add_ln_row(e_Out + i * E_D, e_X1 + i * E_D, e_Y + i * E_D, E_D);
}

static void eb_run(void)
{
    layer_eb_1_qkv();
    layer_eb_2_attn();
    layer_eb_3_oproj();
    layer_eb_4_add_ln1();
    layer_eb_5_ffn_up();
    layer_eb_6_ffn_down();
    layer_eb_7_add_ln2();
}

static void eb_zero(void)
{
    const size_t sd = (size_t)E_S * E_D;
    /* rows 0..U-1 of Q,K,V are produced in the ROI; rows U.. keep setup values */
    memset(e_Q, 0, sd * 4);
    memset(e_K, 0, (size_t)U * E_D * 4);
    memset(e_V, 0, (size_t)U * E_D * 4);
    memset(e_C, 0, sd * 4);
    memset(e_A, 0, sd * 4);
    memset(e_X1, 0, sd * 4);
    memset(e_H, 0, (size_t)E_S * E_F * 4);
    memset(e_Y, 0, sd * 4);
    memset(e_Out, 0, sd * 4);
}

static void eb_setup(long S, long D, long H, long F, long def_units)
{
    E_S = S; E_D = D; E_H = H; E_DH = D / H; E_F = F;
    const size_t sd = (size_t)S * D;
    e_X = tensor(sd, 1);
    e_Wq = tensor((size_t)D * D, 2);
    e_Wk = tensor((size_t)D * D, 3);
    e_Wv = tensor((size_t)D * D, 4);
    e_Q = buffer(sd);
    e_K = tensor(sd, 5); /* rows U..S-1: K of the other row tiles */
    e_V = tensor(sd, 6); /* rows U..S-1: V of the other row tiles */
    e_P = buffer(S);
    e_C = buffer(sd);
    e_Wo = tensor((size_t)D * D, 7);
    e_A = buffer(sd);
    e_X1 = buffer(sd);
    e_W1 = tensor((size_t)D * F, 8);
    e_H = buffer((size_t)S * F);
    e_W2 = tensor((size_t)F * D, 9);
    e_Y = buffer(sd);
    e_Out = buffer(sd);
    UTOT = S;
    U = clamp_units(U ? U : def_units, S);
    /* per token row: QKV 3 D^2, attention 2 S D, O D^2, FFN 2 D F */
    const double per = 4.0 * D * D + 2.0 * S * D + 2.0 * D * F;
    MACS_EXEC = per * U; MACS_TOTAL = per * S;
    IN_B = sd * 4.0;
    W_B = (4.0 * D * D + 2.0 * D * F) * 4;
    OUT_B = sd * 4.0;
    OUT = e_Out; OUT_N = sd;
    snprintf(DIMS, sizeof DIMS, "S=%ld D=%ld H=%ld F=%ld", S, D, H, F);
}

/* ================================================================== */
/* conv_block2 (rn50_block2, mbv2_block2): TWO residual blocks         */
/*   block b: expand 1x1 -> 3x3 (full or depthwise) -> project 1x1     */
/*            + residual add of the block input (stride 1, Cin = Cout) */
/*   rn50 : ReLU after expand / 3x3 / add  (ResNet-50 v1.5 stage 2)    */
/*   mbv2 : ReLU6 after expand / dw, linear project, add, no act       */
/* Sampling: unit = final output image row. Block 2 outputs rows       */
/* 0..U-1, so its 3x3 needs block-2 expand rows 0..U, i.e. block-1     */
/* output rows 0..U; block 1 computes U+1 rows and its expand U+2.     */
/* ================================================================== */
static long B_C, B_M, B_H, B_DW, B_RACT;     /* channels in/out, mid, H=W, dw, act after add */
static float B_CLIP;
static float *b_X, *b_Wa1, *b_A1, *b_Wb1, *b_B1, *b_Wc1, *b_Y1,
             *b_Wa2, *b_A2, *b_Wb2, *b_B2, *b_Wc2, *b_Y2;

static inline float bact(float v)
{
    v = v > 0.0f ? v : 0.0f;
    return (B_CLIP > 0.0f && v > B_CLIP) ? B_CLIP : v;
}

static inline long rmin(long a, long b) { return a < b ? a : b; }

/* o[k][0..rows*W) += sum_c w[k][c] * x[c][0..rows*W)  (as chain.c) */
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

static inline __attribute__((always_inline))
void expand_rows(float *A, const float *X, const float *Wa, long rows)
{
    const long HW = B_H * B_H;
    conv1x1_rows(A, X, Wa, B_M, B_C, rows, HW, B_H);
    for (long k = 0; k < B_M; k++)
        for (long i = 0; i < rows * B_H; i++)
            A[k * HW + i] = bact(A[k * HW + i]);
}

/* 3x3, stride 1, pad 1, output rows 0..rows-1 (as chain.c layer_cv_2) */
static inline __attribute__((always_inline))
void conv3x3_rows(float *Bo, const float *A, const float *Wb, long rows)
{
    const long HW = B_H * B_H, W = B_H;
    for (long k = 0; k < B_M; k++) {
        float *o = Bo + k * HW;
        const long c0 = B_DW ? k : 0, c1 = B_DW ? k + 1 : B_M;
        for (long c = c0; c < c1; c++) {
            const float *x = A + c * HW;
            const float *w = B_DW ? Wb + k * 9 : Wb + (k * B_M + c) * 9;
            for (long r = 0; r < 3; r++) {
                for (long s = 0; s < 3; s++) {
                    const float wv = w[r * 3 + s];
                    const long lo = s == 0 ? 1 : 0;
                    const long hi = s == 2 ? W - 1 : W;
                    for (long oh = 0; oh < rows; oh++) {
                        const long ih = oh - 1 + r;
                        if (ih < 0 || ih >= B_H)
                            continue;
                        const float *xr = x + ih * W;
                        float *orow = o + oh * W;
                        for (long ow = lo; ow < hi; ow++)
                            orow[ow] += wv * xr[ow - 1 + s];
                    }
                }
            }
        }
        for (long i = 0; i < rows * W; i++)
            o[i] = bact(o[i]);
    }
}

/* Y = project(B) + Xres  (+ ReLU for ResNet) on rows 0..rows-1 */
static inline __attribute__((always_inline))
void project_add_rows(float *Y, const float *Bm, const float *Wc, const float *Xres, long rows)
{
    const long HW = B_H * B_H;
    conv1x1_rows(Y, Bm, Wc, B_C, B_M, rows, HW, B_H);
    for (long k = 0; k < B_C; k++)
        for (long i = 0; i < rows * B_H; i++) {
            float v = Y[k * HW + i] + Xres[k * HW + i];
            Y[k * HW + i] = B_RACT ? (v > 0.0f ? v : 0.0f) : v;
        }
}

static NOINL void layer_cb_1_b1_expand(void) { expand_rows(b_A1, b_X, b_Wa1, rmin(U + 2, B_H)); }
static NOINL void layer_cb_2_b1_conv3x3(void) { conv3x3_rows(b_B1, b_A1, b_Wb1, rmin(U + 1, B_H)); }
static NOINL void layer_cb_3_b1_project(void)
{
    project_add_rows(b_Y1, b_B1, b_Wc1, b_X, rmin(U + 1, B_H));
}
static NOINL void layer_cb_4_b2_expand(void) { expand_rows(b_A2, b_Y1, b_Wa2, rmin(U + 1, B_H)); }
static NOINL void layer_cb_5_b2_conv3x3(void) { conv3x3_rows(b_B2, b_A2, b_Wb2, U); }
static NOINL void layer_cb_6_b2_project(void) { project_add_rows(b_Y2, b_B2, b_Wc2, b_Y1, U); }

static void cb_run(void)
{
    layer_cb_1_b1_expand();
    layer_cb_2_b1_conv3x3();
    layer_cb_3_b1_project();
    layer_cb_4_b2_expand();
    layer_cb_5_b2_conv3x3();
    layer_cb_6_b2_project();
}

static void cb_zero(void)
{
    const size_t HW = (size_t)B_H * B_H;
    memset(b_A1, 0, B_M * HW * 4);
    memset(b_B1, 0, B_M * HW * 4);
    memset(b_Y1, 0, B_C * HW * 4);
    memset(b_A2, 0, B_M * HW * 4);
    memset(b_B2, 0, B_M * HW * 4);
    memset(b_Y2, 0, B_C * HW * 4);
}

static void cb_setup(long C, long M, long H, long dw, float clip, long ract, long def_units)
{
    B_C = C; B_M = M; B_H = H; B_DW = dw; B_CLIP = clip; B_RACT = ract;
    const size_t HW = (size_t)H * H;
    const size_t nb = dw ? (size_t)M * 9 : (size_t)M * M * 9;
    b_X = tensor(C * HW, 1);
    b_Wa1 = tensor((size_t)M * C, 2);
    b_A1 = buffer(M * HW);
    b_Wb1 = tensor(nb, 3);
    b_B1 = buffer(M * HW);
    b_Wc1 = tensor((size_t)C * M, 4);
    b_Y1 = buffer(C * HW);
    b_Wa2 = tensor((size_t)M * C, 5);
    b_A2 = buffer(M * HW);
    b_Wb2 = tensor(nb, 6);
    b_B2 = buffer(M * HW);
    b_Wc2 = tensor((size_t)C * M, 7);
    b_Y2 = buffer(C * HW);
    UTOT = H;
    U = clamp_units(U ? U : def_units, H);
    const double pe = (double)M * C, p3 = (double)nb, pp = (double)C * M;  /* MACs per pixel */
    MACS_EXEC = H * (rmin(U + 2, H) * pe + rmin(U + 1, H) * (p3 + pp)
                     + rmin(U + 1, H) * pe + U * (p3 + pp));
    MACS_TOTAL = 2.0 * HW * (pe + p3 + pp);
    IN_B = C * HW * 4.0; W_B = 2.0 * (pe + p3 + pp) * 4; OUT_B = C * HW * 4.0;
    OUT = b_Y2; OUT_N = C * HW;
    snprintf(DIMS, sizeof DIMS, "2 blocks C=%ld M=%ld H=W=%ld %s", C, M, H,
             dw ? "depthwise" : "full3x3");
}

/* ================================================================== */
/* dlrm_block: DLRM-v2 inference, U samples of a batch of B            */
/*   1 bot_mlp   13 -> 512 -> 256 -> 128 (ReLU)                        */
/*   2 embbag    26 sum-pooled bags, dim 128 (MLPerf rows, capped at   */
/*               R_cap rows; multi-hot sizes as embedding.c)           */
/*   3 interact  concat [bottom(128), 26 x pooled(128)] = 3456 [A]     */
/*   4 top_mlp   3456 -> 1024 -> 1024 -> 512 -> 256 -> 1 (ReLU, last   */
/*               linear)                                                */
/* Tables, multi-hot sizes, seeds (tables 100+t, indices 7) and the    */
/* row cap follow embedding.c; MLP shapes follow the mlp.c plan rows.  */
/* ================================================================== */
static const long DLRM_ROWS[26] = {
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36};
static const long DLRM_HOT[26] = {
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1};
#define NT 26
#define ED 128
static const long BOT[4] = {13, 512, 256, 128};
static const long TOP[6] = {3456, 1024, 1024, 512, 256, 1};
static long D_B, D_HOT;
static float *d_dense, *d_bw[3], *d_ba[3], *d_tab[NT], *d_pool, *d_z, *d_tw[5], *d_ta[5];
static uint32_t *d_idx;

/* out[N] = act(in[K] W[K][N]) for one sample row; act: ReLU unless last */
static inline __attribute__((always_inline))
void dense_row(float *out, const float *in, const float *W, long K, long N, int relu)
{
    gemm_row(out, in, W, K, N);
    if (relu)
        for (long j = 0; j < N; j++)
            out[j] = out[j] > 0.0f ? out[j] : 0.0f;
}

static NOINL void layer_dl_1_bot_mlp(void)
{
    for (long i = 0; i < U; i++) {
        const float *in = d_dense + i * BOT[0];
        for (int l = 0; l < 3; l++) {
            float *o = d_ba[l] + i * BOT[l + 1];
            dense_row(o, in, d_bw[l], BOT[l], BOT[l + 1], 1);
            in = o;
        }
    }
}

static NOINL void layer_dl_2_embbag(void)
{
    for (long i = 0; i < U; i++) {
        const uint32_t *ib = d_idx + i * D_HOT;
        for (int t = 0; t < NT; t++) {
            float *o = d_pool + (i * NT + t) * ED;
            for (long h = 0; h < DLRM_HOT[t]; h++) {
                const float *row = d_tab[t] + (size_t)(*ib++) * ED;
                for (long d = 0; d < ED; d++)
                    o[d] += row[d];
            }
        }
    }
}

static NOINL void layer_dl_3_interact(void)
{
    for (long i = 0; i < U; i++) {
        float *z = d_z + i * TOP[0];
        memcpy(z, d_ba[2] + i * ED, ED * 4);
        memcpy(z + ED, d_pool + i * NT * ED, NT * ED * 4);
    }
}

static NOINL void layer_dl_4_top_mlp(void)
{
    for (long i = 0; i < U; i++) {
        const float *in = d_z + i * TOP[0];
        for (int l = 0; l < 5; l++) {
            float *o = d_ta[l] + i * TOP[l + 1];
            dense_row(o, in, d_tw[l], TOP[l], TOP[l + 1], l < 4);
            in = o;
        }
    }
}

static void dl_run(void)
{
    layer_dl_1_bot_mlp();
    layer_dl_2_embbag();
    layer_dl_3_interact();
    layer_dl_4_top_mlp();
}

static void dl_zero(void)
{
    for (int l = 0; l < 3; l++)
        memset(d_ba[l], 0, (size_t)D_B * BOT[l + 1] * 4);
    memset(d_pool, 0, (size_t)D_B * NT * ED * 4);
    memset(d_z, 0, (size_t)D_B * TOP[0] * 4);
    for (int l = 0; l < 5; l++)
        memset(d_ta[l], 0, (size_t)D_B * TOP[l + 1] * 4);
}

static void dl_setup(long Bsz, long rcap, long def_units)
{
    D_B = Bsz;
    double wb = 0.0, tb = 0.0, macs = 0.0;
    d_dense = tensor((size_t)Bsz * BOT[0], 1);
    for (int l = 0; l < 3; l++) {
        d_bw[l] = tensor((size_t)BOT[l] * BOT[l + 1], 10u + l);
        d_ba[l] = buffer((size_t)Bsz * BOT[l + 1]);
        wb += (double)BOT[l] * BOT[l + 1] * 4; macs += (double)BOT[l] * BOT[l + 1];
    }
    D_HOT = 0;
    for (int t = 0; t < NT; t++) {
        const long r = DLRM_ROWS[t] < rcap ? DLRM_ROWS[t] : rcap;
        d_tab[t] = tensor((size_t)r * ED, 100u + (uint32_t)t);
        tb += (double)r * ED * 4;
        D_HOT += DLRM_HOT[t];
    }
    d_idx = (uint32_t *)xalloc((size_t)Bsz * D_HOT * sizeof(uint32_t));
    uint32_t s = 7;
    for (long b = 0; b < Bsz; b++) {
        uint32_t *ib = d_idx + b * D_HOT;
        for (int t = 0; t < NT; t++) {
            const long r = DLRM_ROWS[t] < rcap ? DLRM_ROWS[t] : rcap;
            for (long h = 0; h < DLRM_HOT[t]; h++)
                *ib++ = lcg_next(&s) % (uint32_t)r;
        }
    }
    d_pool = buffer((size_t)Bsz * NT * ED);
    d_z = buffer((size_t)Bsz * TOP[0]);
    for (int l = 0; l < 5; l++) {
        d_tw[l] = tensor((size_t)TOP[l] * TOP[l + 1], 20u + l);
        d_ta[l] = buffer((size_t)Bsz * TOP[l + 1]);
        wb += (double)TOP[l] * TOP[l + 1] * 4; macs += (double)TOP[l] * TOP[l + 1];
    }
    macs += (double)D_HOT * ED;
    UTOT = Bsz;
    U = clamp_units(U ? U : def_units, Bsz);
    MACS_EXEC = macs * U; MACS_TOTAL = macs * Bsz;
    IN_B = (double)Bsz * (BOT[0] * 4.0 + D_HOT * 4.0); W_B = wb + tb; OUT_B = Bsz * 4.0;
    OUT = d_ta[4]; OUT_N = (size_t)Bsz;
    snprintf(DIMS, sizeof DIMS, "B=%ld R_cap=%ld lookups=%ld bot=13-512-256-128 "
             "top=3456-1024-1024-512-256-1", Bsz, rcap, D_HOT);
}

/* ================================================================== */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s bert_block|vit_block|rn50_block2|mbv2_block2|dlrm_block[:small] "
                        "[units] [cold|warm|none]\n",
                argv[0]);
        return 1;
    }
    static char bname[48];
    snprintf(bname, sizeof bname, "%s", argv[1]);
    char *suffix = strchr(bname, ':');
    SMALL = suffix && strcmp(suffix, ":small") == 0;
    if (suffix)
        *suffix = '\0';
    BLOCK = bname;
    U = arg_long(argc, argv, 2, 0);
    const char *mode = arg_str(argc, argv, 3, "cold");
    void (*run)(void), (*zero)(void);
    const char *unit;

    if (!strcmp(BLOCK, "bert_block")) {
        /* BERT-Large (MLPerf bert_config.json): hidden 1024, 16 heads,
         * intermediate 4096, max_seq_length 384 */
        SMALL ? eb_setup(8, 16, 4, 32, 3) : eb_setup(384, 1024, 16, 4096, 4);
        run = eb_run; zero = eb_zero; unit = "token_row";
    } else if (!strcmp(BLOCK, "vit_block")) {
        /* ViT-B/16: hidden 768, 12 heads, intermediate 3072, 197 tokens */
        SMALL ? eb_setup(7, 12, 3, 24, 2) : eb_setup(197, 768, 12, 3072, 6);
        run = eb_run; zero = eb_zero; unit = "token_row";
    } else if (!strcmp(BLOCK, "rn50_block2")) {
        /* ResNet-50 v1.5 stage 2 identity bottlenecks: 256 -> 64 -> 64 -> 256, 56x56 */
        SMALL ? cb_setup(8, 4, 6, 0, 0.0f, 1, 3) : cb_setup(256, 64, 56, 0, 0.0f, 1, 6);
        run = cb_run; zero = cb_zero; unit = "output_row";
    } else if (!strcmp(BLOCK, "mbv2_block2")) {
        /* MobileNetV2 stride-1 inverted residuals: 24 -> 144 (t=6) -> dw -> 24, 56x56 */
        SMALL ? cb_setup(4, 8, 6, 1, 6.0f, 0, 6) : cb_setup(24, 144, 56, 1, 6.0f, 0, 56);
        run = cb_run; zero = cb_zero; unit = "output_row";
    } else if (!strcmp(BLOCK, "dlrm_block")) {
        /* DLRM-v2 (MLPerf): batch 64, tables capped at 65536 rows, as the single layers */
        SMALL ? dl_setup(3, 50, 2) : dl_setup(64, 65536, 4);
        run = dl_run; zero = dl_zero; unit = "sample";
    } else {
        fprintf(stderr, "ERROR: unknown block %s\n", BLOCK);
        return 1;
    }

    ai_execute(mode, run, zero);

    char kname[64];
    snprintf(kname, sizeof kname, "block:%s", BLOCK);
    report_meta(kname, DIMS, mode, SMALL ? "small" : "full", unit, U, UTOT,
                MACS_EXEC, MACS_TOTAL, IN_B, W_B, OUT_B);
    report_checksum(OUT, OUT_N);
    return 0;
}
