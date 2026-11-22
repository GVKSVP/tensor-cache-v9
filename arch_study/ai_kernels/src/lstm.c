/*
 * lstm.c - model-derived single-layer LSTM over a sequence (float32,
 * batch 1), as in the RNN-T encoder and prediction networks.
 *
 * For each time step t (gate order i, f, g, o in rows of the weights):
 *   z   = W_ih * x_t + W_hh * h + b             (4H)
 *   i,f,o = sigmoid(z_i, z_f, z_o),  g = tanh(z_g)
 *   c   = f*c + i*g,   h = o*tanh(c),   y_t = h
 *
 * x is [T][I], W_ih [4H][I], W_hh [4H][H], b [4H], y [T][H]. All tensors are
 * allocated with their real shapes. Time steps depend on each other, so the
 * work unit is one time step and the first `units` steps are computed in
 * order (4H*(I+H) MACs per step). The same weights are re-read every step.
 *
 * Usage: lstm I H T [units] [mode]
 * Seeds: x = 1, W_ih = 2, W_hh = 3, b = 4; h0 = c0 = 0.
 */
#include "ai_common.h"

static long I, H, T, U;
static float *x, *wih, *whh, *bias, *y, *h, *c, *z;

static void zero(void)
{
    memset(y, 0, (size_t)T * H * sizeof(float));
    memset(h, 0, (size_t)H * sizeof(float));
    memset(c, 0, (size_t)H * sizeof(float));
}

static inline float sigmoidf(float v)
{
    return 1.0f / (1.0f + expf(-v));
}

static void run(void)
{
    for (long t = 0; t < U; t++) {
        const float *xt = x + t * I;
        for (long r = 0; r < 4 * H; r++) {
            const float *wi = wih + r * I;
            const float *wh = whh + r * H;
            float acc = bias[r];
            for (long k = 0; k < I; k++)
                acc += wi[k] * xt[k];
            for (long k = 0; k < H; k++)
                acc += wh[k] * h[k];
            z[r] = acc;
        }
        float *yt = y + t * H;
        for (long j = 0; j < H; j++) {
            const float ig = sigmoidf(z[j]);
            const float fg = sigmoidf(z[H + j]);
            const float gg = tanhf(z[2 * H + j]);
            const float og = sigmoidf(z[3 * H + j]);
            c[j] = fg * c[j] + ig * gg;
            h[j] = og * tanhf(c[j]);
            yt[j] = h[j];
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s I H T [units] [cold|warm|none]\n", argv[0]);
        return 1;
    }
    I = atol(argv[1]);
    H = atol(argv[2]);
    T = atol(argv[3]);
    if (I < 1 || H < 1 || T < 1) {
        fprintf(stderr, "ERROR: invalid dimensions\n");
        return 1;
    }
    U = clamp_units(arg_long(argc, argv, 4, 0), T);
    const char *mode = arg_str(argc, argv, 5, "cold");

    x = falloc((size_t)T * I);
    wih = falloc((size_t)4 * H * I);
    whh = falloc((size_t)4 * H * H);
    bias = falloc((size_t)4 * H);
    y = falloc((size_t)T * H);
    h = falloc((size_t)H);
    c = falloc((size_t)H);
    z = falloc((size_t)4 * H);
    fill_lcg(x, (size_t)T * I, 1);
    fill_lcg(wih, (size_t)4 * H * I, 2);
    fill_lcg(whh, (size_t)4 * H * H, 3);
    fill_lcg(bias, (size_t)4 * H, 4);

    ai_execute(mode, run, zero);

    char dims[96];
    snprintf(dims, sizeof dims, "I=%ld H=%ld T=%ld", I, H, T);
    const double per_unit = 4.0 * H * (I + H);
    report_meta("lstm", dims, mode, "unfused_gates", "time_step", U, T,
                per_unit * U, per_unit * T,
                (double)T * I * 4, (4.0 * H * (I + H) + 4.0 * H) * 4, (double)T * H * 4);
    report_checksum(y, (size_t)U * H);
    return 0;
}
