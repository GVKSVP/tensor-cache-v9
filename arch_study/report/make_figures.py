#!/usr/bin/env python3
"""
make_figures.py - figures for the UROP report, generated from the study CSVs.

Reads (read-only):
    tensor_cache_dataset_56000.csv            (Baseline-v1, 56k runs)
    arch_study/datasets/ai_kernels_*.csv      (model-derived AI kernel runs)
Writes:
    arch_study/report/figures/fig*.png

Style: fixed-order categorical palette (validated reference palette), one
y-axis per panel, recessive grid, legend whenever >= 2 series.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

PROJECT = Path.home() / "gem5-project"
DS = PROJECT / "arch_study" / "datasets"
OUT = PROJECT / "arch_study" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (light mode), categorical slots in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": SURFACE,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

SIX = ["RN50_conv1", "RN50_maxpool", "DB_gemv_3072", "BERT_ffn1", "BERT_embed", "DLRM_embbag"]
FAMILY_ORDER = ["ResNet-50", "BERT-Large", "DLRM-v2", "DeepBench"]


def load(name):
    return pd.read_csv(DS / name).set_index("layer_id")


def family(layer):
    return {"RN50": "ResNet-50", "BERT": "BERT-Large", "DLRM": "DLRM-v2", "DB": "DeepBench"}[layer.split("_")[0]]


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------
# Fig 1: simulated P870-D-inspired system (block diagram)
# ------------------------------------------------------------------

def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=color, ec=INK2, lw=0.8, alpha=0.18))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="none", ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.2, color=INK2)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))

    box(2.6, 8.4, 4.8, 1.3, "RISC-V CPU core (RiscvO3CPU)", "6-wide out-of-order, 2.0 GHz, TAGE, RAS 64", SERIES[0])
    box(1.2, 6.4, 3.4, 1.2, "L1 I-cache", "64 KiB, 8-way, 4 cycles", SERIES[1])
    box(5.4, 6.4, 3.4, 1.2, "L1 D-cache", "64 KiB, 8-way, 4 cycles", SERIES[1])
    box(2.6, 4.4, 4.8, 1.2, "L2 cache + stride prefetcher", "4 MiB, 16-way, 16 cycles", SERIES[2])
    box(2.6, 2.4, 4.8, 1.2, "L3 cache (last level)", "16 MiB, 16-way, 40 cycles", SERIES[3])
    box(2.6, 0.4, 4.8, 1.2, "Memory controller + DRAM", "DDR5-4400 (gem5 model), open-adaptive", SERIES[4])

    arrow(2.9, 7.6, 3.6, 8.4)
    arrow(7.1, 7.6, 6.4, 8.4)
    arrow(2.9, 6.4, 4.0, 5.6)
    arrow(7.1, 6.4, 6.0, 5.6)
    arrow(5.0, 4.4, 5.0, 3.6)
    arrow(5.0, 2.4, 5.0, 1.6)
    ax.text(9.9, 5.0, "classic memory system\nMOESI snooping\n(coherent crossbars)", ha="right",
            va="center", fontsize=7, color=INK2, style="italic")
    save(fig, "fig1_architecture.png")


# ------------------------------------------------------------------
# Fig 2: experimental methodology (flow)
# ------------------------------------------------------------------

def fig_methodology():
    steps = [
        ("1. Shapes", "MLPerf,\nDeepBench,\nAI-Matrix (27)"),
        ("2. C kernels", "float32,\nstatic RV64GC,\nm5 ROI markers"),
        ("3. gem5 run", "sim_config.py\nor HiFive\nboard model"),
        ("4. ROI stats", "first stats\nblock only\n(kernel region)"),
        ("5. Dataset", "one CSV row\nper run,\n87+ metrics"),
        ("6. Analysis", "one factor\nchanged at\na time"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    ax.set_xlim(0, len(steps) * 2.3)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    for i, (t, s) in enumerate(steps):
        x = i * 2.3 + 0.08
        ax.add_patch(FancyBboxPatch((x, 0.15), 1.95, 2.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=SERIES[0], ec=INK2, lw=0.7, alpha=0.15))
        ax.text(x + 0.975, 2.05, t, ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)
        ax.text(x + 0.975, 1.05, s, ha="center", va="center", fontsize=6.2, color=INK2, linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.3, 1.3), xytext=(x + 2.05, 1.3),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    save(fig, "fig2_methodology.png")


# ------------------------------------------------------------------
# Fig 3: Baseline-v1 sensitivity to cache configuration (56k dataset)
# ------------------------------------------------------------------

def fig_baseline_sensitivity():
    d = pd.read_csv(PROJECT / "tensor_cache_dataset_56000.csv",
                    usecols=["workload", "problem_args", "sim_ticks"])
    spread = (d.groupby(["workload", "problem_args"]).sim_ticks
                .agg(lambda x: x.max() / x.min()).groupby("workload").median())
    spread = (spread - 1.0) * 100
    spread = spread.sort_values()
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    ax.barh(spread.index, spread.values, color=SERIES[0], height=0.7)
    ax.axvline(3, color=INK2, lw=0.8, ls="--")
    ax.text(3.3, 0.2, "3 %", color=INK2, fontsize=7.5)
    ax.set_xlabel("Runtime spread across the 50 cache configurations (%, median per workload)")
    ax.tick_params(axis="y", labelsize=7.2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_baseline_sensitivity.png")
    return spread


# ------------------------------------------------------------------
# Fig 4: clock + DDR5 speed-up vs L3 MPKI (27 layers, TimingSimpleCPU)
# ------------------------------------------------------------------

def fig_clock_speedup():
    a = load("ai_kernels_pilot_v1.csv")
    c = load("ai_kernels_2ghz_ddr5_4400_v1.csv").reindex(a.index)
    sp = a.sim_seconds / c.sim_seconds
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for i, fam in enumerate(FAMILY_ORDER):
        m = [family(l) == fam for l in a.index]
        ax.scatter(a.l3_mpki[m], sp[m], s=36, color=SERIES[i], edgecolor="white", lw=1.2, label=fam, zorder=3)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.axhline(2.0, color=INK2, lw=0.8, ls="--")
    ax.text(0.045, 2.015, "ideal 2x from the clock alone", color=INK2, fontsize=7.5, va="bottom")
    ax.annotate("BERT_embed", (a.l3_mpki["BERT_embed"], sp["BERT_embed"]), xytext=(-8, 6),
                textcoords="offset points", fontsize=7, color=INK2, ha="right")
    ax.annotate("DLRM_embbag", (a.l3_mpki["DLRM_embbag"], sp["DLRM_embbag"]), xytext=(6, 4),
                textcoords="offset points", fontsize=7, color=INK2)
    ax.annotate("BERT_ffn1/ffn2, DLRM_top_mlp, RN50_fc", (a.l3_mpki["BERT_ffn1"], sp["BERT_ffn1"]),
                xytext=(-8, -12), textcoords="offset points", fontsize=7, color=INK2, ha="right")
    ax.set_xlabel("L3 misses per kilo-instruction (baseline, log scale)")
    ax.set_ylabel("Speed-up: 2 GHz + DDR5-4400\nover 1 GHz + DDR3-1600")
    ax.set_ylim(1.3, 2.1)
    ax.legend(ncol=4, loc="lower left", fontsize=7.5)
    save(fig, "fig4_clock_ddr5_speedup.png")
    return sp


# ------------------------------------------------------------------
# Fig 5: DRAM latency, DDR3 vs DDR5 (close / open page policy)
# ------------------------------------------------------------------

def fig_dram_latency():
    o = load("ai_kernels_2ghz_ddr5_4400_open_v1.csv")
    L = list(o.index)
    a = load("ai_kernels_pilot_v1.csv").reindex(L)
    c = load("ai_kernels_2ghz_ddr5_4400_v1.csv").reindex(L)
    series = [("DDR3-1600 (open-adaptive)", a), ("DDR5-4400 (close-adaptive, gem5 default)", c),
              ("DDR5-4400 (open-adaptive)", o)]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    w = 0.26
    for i, (name, df) in enumerate(series):
        ax.bar([k + (i - 1) * w for k in range(len(L))], df.dram_avg_mem_acc_lat_ticks / 1e3,
               width=w - 0.03, color=SERIES[i], label=name)
    ax.set_xticks(range(len(L)))
    ax.set_xticklabels(L, rotation=35, ha="right", fontsize=7.2)
    ax.set_ylabel("Average DRAM access latency (ns)")
    ax.legend(fontsize=7.2, loc="upper left", ncol=1)
    ax.set_ylim(0, 55)
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_dram_latency.png")


# ------------------------------------------------------------------
# Fig 6: execution time across configurations, normalised to HiFive model
# ------------------------------------------------------------------

def fig_configs():
    h = load("ai_kernels_hifive_unmatched_v1.csv").reindex(SIX)
    cfg = [
        ("In-order TimingSimple, 2 GHz, DDR5", load("ai_kernels_2ghz_ddr5_4400_open_v1.csv").reindex(SIX)),
        ("O3 6-wide, baseline caches", load("ai_kernels_o3_2ghz_ddr5_4400_open_v1.csv").reindex(SIX)),
        ("O3 + P870-D caches", load("ai_kernels_p870d_full_v1.csv").reindex(SIX)),
        ("O3 + P870-D caches + L2 prefetcher", load("ai_kernels_p870d_full_pf_v1.csv").reindex(SIX)),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    w = 0.2
    for i, (name, df) in enumerate(cfg):
        ax.bar([k + (i - 1.5) * w for k in range(len(SIX))], df.sim_seconds / h.sim_seconds,
               width=w - 0.03, color=SERIES[i], label=name)
    ax.axhline(1.0, color=INK2, lw=0.9, ls="--")
    ax.text(-0.45, 1.04, "HiFive Unmatched model = 1.0", ha="left", va="bottom", fontsize=7.2, color=INK2)
    ax.set_xticks(range(len(SIX)))
    ax.set_xticklabels(SIX, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Execution time relative to\nHiFive Unmatched model (lower = faster)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_ylim(0, 4.6)
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_configurations.png")


# ------------------------------------------------------------------
# Fig 7: prefetcher benefit and accuracy (two panels, separate axes)
# ------------------------------------------------------------------

def fig_prefetch():
    n = load("ai_kernels_p870d_full_v1.csv").reindex(SIX)
    f = load("ai_kernels_p870d_full_pf_v1.csv").reindex(SIX)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    a1.barh(SIX, n.sim_seconds / f.sim_seconds, color=SERIES[0], height=0.6)
    a1.axvline(1, color=INK2, lw=0.8, ls="--")
    a1.set_xlabel("Speed-up from L2 stride prefetcher")
    a1.set_title("(a) Benefit", loc="left")
    a2.barh(SIX, f.l2_pf_accuracy * 100, color=SERIES[1], height=0.6)
    a2.set_xlabel("Useful prefetches (% of issued)")
    a2.set_title("(b) Accuracy", loc="left")
    a2.set_xlim(0, 100)
    a2.set_yticklabels([])
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "fig7_prefetcher.png")


# ------------------------------------------------------------------
# Fig 8: final 27-layer characterisation on the P870-D-inspired model
# ------------------------------------------------------------------

def fig_final():
    f = pd.read_csv(DS / "ai_kernels_p870d_full_pf_v1.csv").sort_values("run_id").set_index("layer_id")
    colors = [SERIES[FAMILY_ORDER.index(family(l))] for l in f.index]
    traffic = f.dram_bytes_read / 1024 / (f.macs_executed / 1e6)  # KiB per million MACs
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 6.4), sharey=True)
    a1.barh(f.index, f.ipc, color=colors, height=0.7)
    a1.set_xlabel("Instructions per cycle (IPC)")
    a1.set_title("(a) Core throughput", loc="left")
    a2.barh(f.index, traffic, color=colors, height=0.7)
    a2.set_xscale("log")
    a2.set_xlabel("DRAM read traffic (KiB per million MACs, log)")
    a2.set_title("(b) Memory intensity", loc="left")
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7)
    a1.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[i]) for i in range(len(FAMILY_ORDER))]
    fig.legend(handles, FAMILY_ORDER, ncol=4, loc="lower center", bbox_to_anchor=(0.55, -0.01), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig8_final_27_layers.png")


if __name__ == "__main__":
    fig_architecture()
    fig_methodology()
    fig_baseline_sensitivity()
    fig_clock_speedup()
    fig_dram_latency()
    fig_configs()
    fig_prefetch()
    fig_final()
