/*
 * embedding.c - model-derived embedding kernels (float32).
 *
 * gather : BERT-style token embedding lookup.
 *          out[i][:] = table[idx[i]][:],  idx[i] = (i * 2654435761) mod V
 *          Usage: embedding gather V D L [units] [mode]
 *          Work unit = one lookup (D floats copied).
 *
 * bag    : DLRM-v2 sum-pooled multi-hot embedding bag.
 *          26 tables, dim 128, per-table row counts and multi-hot sizes from
 *          the MLPerf DLRM-v2 reference (see README.md for sources).
 *          Tables larger than R_cap rows are capped to R_cap rows [assumption].
 *          out[b][t][:] = sum_{h < hot[t]} table_t[idx[b][t][h]][:]
 *          Indices are uniform over each (capped) table.
 *          Usage: embedding bag B R_cap [units] [mode]
 *          Work unit = one sample (214 lookups).
 *
 * Seeds: gather table = 1; bag table t = 100 + t; bag indices = 7.
 */
#include "ai_common.h"

/* MLPerf DLRM-v2: num_embeddings_per_feature (inference main.py, multihot-criteo). */
static const long DLRM_ROWS[26] = {
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36};

/* MLPerf DLRM-v2: multi_hot_sizes (facebookresearch torchrec_dlrm README). */
static const long DLRM_HOT[26] = {
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1};

#define DLRM_TABLES 26
#define DLRM_DIM 128

static int is_bag;
static long V, D, L, U;          /* gather */
static long B, RCAP, HOTSUM;     /* bag */
static long rows_eff[DLRM_TABLES];
static float *table, *out;
static float *tables[DLRM_TABLES];
static uint32_t *idx;

static void zero(void)
{
    if (is_bag)
        memset(out, 0, (size_t)B * DLRM_TABLES * DLRM_DIM * sizeof(float));
    else
        memset(out, 0, (size_t)L * D * sizeof(float));
}

static void run_gather(void)
{
    for (long i = 0; i < U; i++) {
        const float *src = table + (size_t)idx[i] * D;
        float *dst = out + i * D;
        for (long d = 0; d < D; d++)
            dst[d] = src[d];
    }
}

static void run_bag(void)
{
    for (long b = 0; b < U; b++) {
        const uint32_t *ib = idx + b * HOTSUM;
        for (long t = 0; t < DLRM_TABLES; t++) {
            float *o = out + (b * DLRM_TABLES + t) * DLRM_DIM;
            const float *tab = tables[t];
            for (long h = 0; h < DLRM_HOT[t]; h++) {
                const float *row = tab + (size_t)(*ib++) * DLRM_DIM;
                for (long d = 0; d < DLRM_DIM; d++)
                    o[d] += row[d];
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s gather V D L [units] [mode]\n"
                        "       %s bag B R_cap [units] [mode]\n", argv[0], argv[0]);
        return 1;
    }
    char dims[160];
    const char *mode;

    if (strcmp(argv[1], "gather") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s gather V D L [units] [mode]\n", argv[0]);
            return 1;
        }
        V = atol(argv[2]);
        D = atol(argv[3]);
        L = atol(argv[4]);
        if (V < 1 || D < 1 || L < 1) {
            fprintf(stderr, "ERROR: invalid dimensions\n");
            return 1;
        }
        U = clamp_units(arg_long(argc, argv, 5, 0), L);
        mode = arg_str(argc, argv, 6, "cold");

        table = falloc((size_t)V * D);
        out = falloc((size_t)L * D);
        idx = (uint32_t *)xalloc((size_t)L * sizeof(uint32_t));
        fill_lcg(table, (size_t)V * D, 1);
        for (long i = 0; i < L; i++)
            idx[i] = (uint32_t)(((uint64_t)i * 2654435761u) % (uint64_t)V);

        ai_execute(mode, run_gather, zero);

        snprintf(dims, sizeof dims, "V=%ld D=%ld L=%ld", V, D, L);
        report_meta("embedding", dims, mode, "gather", "lookup", U, L,
                    (double)U * D, (double)L * D,
                    (double)L * 4, (double)V * D * 4, (double)L * D * 4);
        report_checksum(out, (size_t)U * D);
    } else if (strcmp(argv[1], "bag") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s bag B R_cap [units] [mode]\n", argv[0]);
            return 1;
        }
        is_bag = 1;
        B = atol(argv[2]);
        RCAP = atol(argv[3]);
        if (B < 1 || RCAP < 1) {
            fprintf(stderr, "ERROR: invalid dimensions\n");
            return 1;
        }
        U = clamp_units(arg_long(argc, argv, 4, 0), B);
        mode = arg_str(argc, argv, 5, "cold");

        double table_bytes = 0.0;
        HOTSUM = 0;
        for (int t = 0; t < DLRM_TABLES; t++) {
            rows_eff[t] = DLRM_ROWS[t] < RCAP ? DLRM_ROWS[t] : RCAP;
            tables[t] = falloc((size_t)rows_eff[t] * DLRM_DIM);
            fill_lcg(tables[t], (size_t)rows_eff[t] * DLRM_DIM, 100u + (uint32_t)t);
            table_bytes += (double)rows_eff[t] * DLRM_DIM * 4;
            HOTSUM += DLRM_HOT[t];
        }
        idx = (uint32_t *)xalloc((size_t)B * HOTSUM * sizeof(uint32_t));
        uint32_t s = 7;
        for (long b = 0; b < B; b++) {
            uint32_t *ib = idx + b * HOTSUM;
            for (int t = 0; t < DLRM_TABLES; t++)
                for (long h = 0; h < DLRM_HOT[t]; h++)
                    *ib++ = lcg_next(&s) % (uint32_t)rows_eff[t];
        }
        out = falloc((size_t)B * DLRM_TABLES * DLRM_DIM);

        ai_execute(mode, run_bag, zero);

        snprintf(dims, sizeof dims, "tables=26 dim=128 lookups_per_sample=%ld B=%ld R_cap=%ld",
                 HOTSUM, B, RCAP);
        report_meta("embedding", dims, mode, "bag_sum", "sample", U, B,
                    (double)U * HOTSUM * DLRM_DIM, (double)B * HOTSUM * DLRM_DIM,
                    (double)B * HOTSUM * 4, table_bytes,
                    (double)B * DLRM_TABLES * DLRM_DIM * 4);
        report_checksum(out, (size_t)U * DLRM_TABLES * DLRM_DIM);
    } else {
        fprintf(stderr, "ERROR: first argument must be gather|bag\n");
        return 1;
    }
    return 0;
}
