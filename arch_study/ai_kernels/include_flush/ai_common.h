/*
 * ai_common.h - shared helpers for the model-derived AI kernels.
 *
 * These kernels are MODEL-DERIVED: they reproduce documented layer shapes
 * from MLPerf reference models and DeepBench. They are NOT full benchmark
 * implementations.
 *
 * Execution order of every kernel:
 *   init tensors -> zero outputs -> [warm: run once, zero outputs]
 *   -> [cold: stream flush buffer] -> ROI begin -> run -> ROI end -> checksum
 *
 * Under gem5 (-DGEM5) the ROI markers reset and dump the gem5 statistics,
 * so the FIRST stats block in stats.txt covers the kernel only.
 */
/* COPY of include/ai_common.h with a true cache flush at ROI begin (Step 1 fair re-test). */
#ifndef AI_COMMON_H
#define AI_COMMON_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GEM5
#include <gem5/m5ops.h>
#endif

/* ------------------------------------------------------------------ */
/* Deterministic data: 32-bit LCG (Numerical Recipes constants).       */
/* value = (state >> 8) / 2^23 - 1  ->  float in [-1, 1)               */
/* verify_reference.py reproduces this sequence exactly.               */
/* ------------------------------------------------------------------ */

static inline uint32_t lcg_next(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static inline float lcg_float(uint32_t *s)
{
    return (float)(lcg_next(s) >> 8) * (1.0f / 8388608.0f) - 1.0f;
}

static void fill_lcg(float *a, size_t n, uint32_t seed)
{
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++)
        a[i] = lcg_float(&s);
}

static void *xalloc(size_t bytes)
{
    void *p = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    if (!p) {
        fprintf(stderr, "ERROR: allocation of %zu bytes failed\n", bytes);
        exit(2);
    }
    return p;
}

static float *falloc(size_t n)
{
    return (float *)xalloc(n * sizeof(float));
}

static long arg_long(int argc, char **argv, int i, long dflt)
{
    return (i < argc) ? atol(argv[i]) : dflt;
}

static const char *arg_str(int argc, char **argv, int i, const char *dflt)
{
    return (i < argc) ? argv[i] : dflt;
}

/* ------------------------------------------------------------------ */
/* Region of interest                                                  */
/* ------------------------------------------------------------------ */

static inline void roi_begin(void)
{
#ifdef GEM5
    /* ROI_FLUSH variant: stop gem5 here so sim_config_flush.py can write back
     * and invalidate every cache (true cold start for every policy), then
     * the simulation resumes and the statistics are reset as before. */
    m5_exit(0);
    m5_reset_stats(0, 0);
#endif
}

static inline void roi_end(void)
{
#ifdef GEM5
    m5_dump_stats(0, 0);
#endif
}

/* Stream through a buffer larger than any simulated cache so that the
 * ROI starts without initialisation residue (dirty lines are written
 * back here, outside the ROI). */
#define FLUSH_BYTES (64u << 20)

static void cache_flush(void)
{
    volatile unsigned char *buf = (volatile unsigned char *)xalloc(FLUSH_BYTES);
    for (size_t i = 0; i < FLUSH_BYTES; i += 64)
        buf[i] = (unsigned char)i;
    unsigned long sink = 0;
    for (size_t i = 0; i < FLUSH_BYTES; i += 64)
        sink += buf[i];
    if (sink == 1)
        printf("#");
    free((void *)buf);
}

/* mode: "cold" (flush before ROI), "warm" (run once before ROI),
 *       "none" (no preparation). */
static void ai_execute(const char *mode, void (*run)(void), void (*zero)(void))
{
    zero();
    if (strcmp(mode, "warm") == 0) {
        run();
        zero();
    } else if (strcmp(mode, "cold") == 0) {
        cache_flush();
    } else if (strcmp(mode, "none") != 0) {
        fprintf(stderr, "ERROR: mode must be cold|warm|none\n");
        exit(1);
    }
    roi_begin();
    run();
    roi_end();
}

/* ------------------------------------------------------------------ */
/* Reporting                                                           */
/* ------------------------------------------------------------------ */

static void report_checksum(const float *out, size_t n)
{
    double sum = 0.0, abssum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += out[i];
        abssum += fabs(out[i]);
    }
    printf("CHECKSUM %.9e %.9e %zu\n", sum, abssum, n);
}

/* One JSON line with the self-reported workload metadata. */
static void report_meta(const char *kernel, const char *dims, const char *mode,
                        const char *impl, const char *unit,
                        long units_exec, long units_total,
                        double macs_exec, double macs_total,
                        double in_bytes, double w_bytes, double out_bytes)
{
    printf("META {\"kernel\":\"%s\",\"dims\":\"%s\",\"mode\":\"%s\","
           "\"impl\":\"%s\",\"work_unit\":\"%s\",\"units_executed\":%ld,"
           "\"units_total\":%ld,\"work_fraction\":%.6f,"
           "\"macs_executed\":%.0f,\"macs_total\":%.0f,"
           "\"input_bytes\":%.0f,\"weight_bytes\":%.0f,\"output_bytes\":%.0f,"
           "\"footprint_bytes\":%.0f,\"dtype\":\"float32\","
           "\"label\":\"model-derived kernel\"}\n",
           kernel, dims, mode, impl, unit, units_exec, units_total,
           units_total ? (double)units_exec / units_total : 0.0,
           macs_exec, macs_total, in_bytes, w_bytes, out_bytes,
           in_bytes + w_bytes + out_bytes);
}

static long clamp_units(long units, long total)
{
    if (units <= 0 || units > total)
        return total;
    return units;
}

#endif /* AI_COMMON_H */
