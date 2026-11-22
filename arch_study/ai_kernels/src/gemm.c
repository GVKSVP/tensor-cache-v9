/*
 * gemm.c - model-derived GEMM / GEMV kernel (float32, row-major).
 *
 *   C[M][N] = A[M][K] * B[K][N]
 *
 * A = activations, B = weights, C = output. The full A, B and C tensors are
 * allocated with their real shapes; only the first `units` output rows are
 * computed (work unit = one output row = K*N MACs).
 *
 * Usage: gemm M N K [units] [impl] [tile] [mode]
 *   units : output rows to compute (0 or > M = all M)
 *   impl  : naive (i-k-j loop, default) | blocked (k/j tiled)
 *   tile  : tile size for blocked (default 64)
 *   mode  : cold (default) | warm | none
 *
 * Seeds: A = 1, B = 2.
 */
#include "ai_common.h"

static long M, N, K, U, T;
static const char *impl;
static float *A, *B, *C;

static void zero(void)
{
    memset(C, 0, (size_t)M * N * sizeof(float));
}

static void run_naive(void)
{
    for (long i = 0; i < U; i++) {
        float *c = C + i * N;
        const float *a = A + i * K;
        for (long k = 0; k < K; k++) {
            const float av = a[k];
            const float *b = B + k * N;
            for (long j = 0; j < N; j++)
                c[j] += av * b[j];
        }
    }
}

static void run_blocked(void)
{
    for (long kk = 0; kk < K; kk += T) {
        const long ke = kk + T < K ? kk + T : K;
        for (long jj = 0; jj < N; jj += T) {
            const long je = jj + T < N ? jj + T : N;
            for (long i = 0; i < U; i++) {
                float *c = C + i * N;
                const float *a = A + i * K;
                for (long k = kk; k < ke; k++) {
                    const float av = a[k];
                    const float *b = B + k * N;
                    for (long j = jj; j < je; j++)
                        c[j] += av * b[j];
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s M N K [units] [naive|blocked] [tile] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    M = atol(argv[1]);
    N = atol(argv[2]);
    K = atol(argv[3]);
    U = clamp_units(arg_long(argc, argv, 4, 0), M);
    impl = arg_str(argc, argv, 5, "naive");
    T = arg_long(argc, argv, 6, 64);
    const char *mode = arg_str(argc, argv, 7, "cold");

    if (M < 1 || N < 1 || K < 1 || T < 1) {
        fprintf(stderr, "ERROR: invalid dimensions\n");
        return 1;
    }

    void (*run)(void);
    if (strcmp(impl, "naive") == 0)
        run = run_naive;
    else if (strcmp(impl, "blocked") == 0)
        run = run_blocked;
    else {
        fprintf(stderr, "ERROR: impl must be naive|blocked\n");
        return 1;
    }

    A = falloc((size_t)M * K);
    B = falloc((size_t)K * N);
    C = falloc((size_t)M * N);
    fill_lcg(A, (size_t)M * K, 1);
    fill_lcg(B, (size_t)K * N, 2);

    ai_execute(mode, run, zero);

    char dims[96];
    snprintf(dims, sizeof dims, "M=%ld N=%ld K=%ld tile=%ld", M, N, K, T);
    report_meta("gemm", dims, mode, impl, "output_row", U, M,
                (double)U * K * N, (double)M * K * N,
                (double)M * K * 4, (double)K * N * 4, (double)M * N * 4);
    report_checksum(C, (size_t)U * N);
    return 0;
}
