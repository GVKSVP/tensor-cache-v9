"""
make_comparison_figures.py - two side-by-side comparison images.

  compare_1_simple_vs_2ghz_ddr5.png   B1 (1 GHz, DDR3, simple core)  vs  C (2 GHz, DDR5-4400)
  compare_2_p870d_vs_p870d_pf.png     E  (P870-D-inspired)            vs  F (E + L2 stride prefetcher)

Settings tables are read from each run's gem5 config.ini (what was really
simulated); metrics come from the existing result CSVs. Read-only on all
inputs; writes only the two new PNGs into arch_study/report/figures/.
"""

import configparser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT = Path.home() / "gem5-project"
DS = PROJECT / "arch_study" / "datasets"
RES = PROJECT / "arch_study" / "results"
OUT = PROJECT / "arch_study" / "report" / "figures"

SERIES = ["#2a78d6", "#eb6834"]          # reference palette slots 1-2 (fixed order)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TINT = "#fdf0e9"                          # light tint marking changed settings rows

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": SURFACE,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

FAMILY = {"RN50": 0, "BERT": 1, "DLRM": 2, "DB": 3}
DRAM_BY_TCK = {"833": "DDR4-2400", "1250": "DDR3-1600", "455": "DDR5-4400", "454": "DDR5-4400", "312": "DDR5-6400"}


# ------------------------------------------------------------------
# settings, read from config.ini
# ------------------------------------------------------------------
def read_config(results_dir):
    run = sorted(d for d in (RES / results_dir).iterdir()
                 if d.is_dir() and d.name.startswith("AI") and ".stale" not in d.name)[0]
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str
    cp.read(run / "config.ini")

    def kib(sec):
        b = int(cp[sec]["size"])
        return f"{b // 1048576} MiB" if b >= 1048576 else f"{b // 1024} KiB"

    def cache(sec):
        c = cp[sec]
        return f"{kib(sec)}, {c['assoc']}-way, {c['tag_latency']} cyc"

    def repl(sec):
        return cp[sec + ".replacement_policy"]["type"].replace("RP", "")

    def pf(sec):
        p = cp[sec]["prefetcher"]
        return "none" if p == "Null" else cp[p]["type"].replace("Prefetcher", " prefetcher")

    if "system.cpu" in cp:            # sim_config.py layout
        cpu_s, clk_s, dram_s, bus_s = "system.cpu", "system.clk_domain", "system.mem_ctrl.dram", "system.membus"
        caches = {"L1 I-cache": "system.cpu.icache", "L1 D-cache": "system.cpu.dcache",
                  "L2 cache": "system.l2", "L3 cache": "system.l3"}
    else:                             # gem5 stdlib board (HiFive Unmatched model)
        cpu_s, clk_s = "board.processor.cores.core", "board.clk_domain"
        dram_s, bus_s = "board.memory.mem_ctrl.dram", "board.cache_hierarchy.membus"
        caches = {"L1 I-cache": "board.cache_hierarchy.l1icaches", "L1 D-cache": "board.cache_hierarchy.l1dcaches",
                  "L2 cache": "board.cache_hierarchy.l2cache", "L3 cache": None}

    cpu = cp[cpu_s]
    if cpu["type"] == "BaseO3CPU":
        core = "O3 out-of-order, %s-wide" % cpu["issueWidth"]
    elif cpu["type"] == "BaseMinorCPU":
        core = "Minor in-order, %s-wide (U74-like)" % cpu["executeInputWidth"]
    else:
        core = "TimingSimple (in-order)"
    ghz = 1e3 / int(cp[clk_s]["clock"])
    dram = cp[dram_s]
    present = [s for s in caches.values() if s]
    pfs = [f"{k.split()[0]}{'-' + k.split()[1][0] if k.startswith('L1') else ''}" for k, s in caches.items()
           if s and pf(s) != "none"]
    pf_type = sorted({pf(s) for s in present if pf(s) != "none"})
    row = {"CPU core": core, "Clock": f"{ghz:.2g} GHz"}
    row.update({k: (cache(s) if s else "none") for k, s in caches.items()})
    row.update({
        "Replacement (all levels)": "/".join(sorted({repl(s) for s in present})),
        "Prefetcher": ", ".join(pfs) + ": " + "/".join(pf_type) if pfs else "none",
        "Coherence": "Classic MOESI snooping (%s)" % cp[bus_s]["type"],
        "Memory (DRAM)": DRAM_BY_TCK.get(dram["tCK"], "tCK=" + dram["tCK"]),
        "DRAM page policy": dram["page_policy"],
    })
    return row


# ------------------------------------------------------------------
# metrics, read from the result CSVs
# ------------------------------------------------------------------
METRICS = [
    # column, title, transform, scale, better
    ("sim_seconds", "Run time (ms)", lambda d: d.sim_seconds * 1e3, "log", "lower"),
    ("ipc", "IPC (instr. per cycle)", lambda d: d.ipc, "linear", "higher"),
    # Miss panels count MSHR misses = misses that start a NEW line fetch. gem5's plain "misses"
    # also counts later accesses to a line that is already being fetched (O3 core, prefetcher),
    # which makes the O3/prefetch configs look worse than they are. For the in-order core both
    # counts are identical.
    ("l1d_mshr", "L1-D miss rate (%)*", lambda d: d.l1d_mshr_misses / d.l1d_accesses * 100, "linear", "lower"),
    ("l2_mshr", "L2 misses / 1000 instr.*", lambda d: d.l2_mshr_misses / d.committed_insts * 1000, "symlog", "lower"),
    ("l3_mshr", "L3 misses / 1000 instr.*", lambda d: d.l3_mshr_misses / d.committed_insts * 1000, "symlog", "lower"),
    ("l1d_avg_miss_latency_ticks", "L1-D miss latency (ns)",
     lambda d: d.l1d_avg_miss_latency_ticks / 1000, "linear", "lower"),
    ("dram", "DRAM traffic (MiB)",
     lambda d: (d.dram_bytes_read + d.dram_bytes_written) / 2**20, "log", "lower"),
]


def load(name):
    d = pd.read_csv(DS / name)
    assert (d.status == "SUCCESS").all()
    return d.set_index("layer_id")


def order(layers):
    return sorted(layers, key=lambda l: (FAMILY[l.split("_")[0]], l), reverse=True)


def settings_table(ax, a_cfg, b_cfg, a_name, b_name):
    ax.axis("off")
    rows = list(a_cfg)
    cells = [[r, a_cfg[r], b_cfg[r], "changed" if a_cfg[r] != b_cfg[r] else "same"] for r in rows]
    t = ax.table(cellText=cells, colLabels=["Setting", a_name, b_name, ""],
                 colWidths=[0.17, 0.32, 0.32, 0.07], cellLoc="left", bbox=[0.06, 0.0, 0.88, 0.88])
    t.auto_set_font_size(False)
    t.set_fontsize(8.2)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.get_text().set_color(INK)
        if r == 0:
            cell.set_facecolor("#efeeea")
            cell.get_text().set_fontweight("bold")
        elif cells[r - 1][3] == "changed":
            cell.set_facecolor(TINT)
            if c in (1, 2, 3):
                cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")
            if c == 3:
                cell.get_text().set_color(INK2)
    # legend swatches above the table so identity is shown once, by name
    handles = [Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=SERIES[0], mew=2, label=a_name + " (ring)"),
               Line2D([], [], marker="o", ls="", ms=7, mfc=SERIES[1], mec="white", mew=1, label=b_name + " (dot)"),
               (Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=SERIES[0], mew=2),
                Line2D([], [], marker="o", ls="", ms=7, mfc=SERIES[1], mec="white", mew=1))]
    labels = [a_name + " (ring)", b_name + " (dot)", "dot inside ring = both values equal"]
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower center", bbox_to_anchor=(0.5, 0.88), ncol=3, fontsize=9)


def dumbbells(axes, a, b, layers, metrics):
    y = np.arange(len(layers))
    for ax, (col, title, fn, scale, better) in zip(axes, metrics):
        va, vb = fn(a.loc[layers]).values, fn(b.loc[layers]).values
        for yi, x0, x1 in zip(y, va, vb):
            ax.plot([x0, x1], [yi, yi], color=GRID if abs(x1 - x0) < 1e-12 else "#b9b8b3",
                    lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(va, y, s=110, facecolor="none", edgecolor=SERIES[0], lw=2, zorder=3)
        ax.scatter(vb, y, s=34, color=SERIES[1], edgecolor="white", lw=0.8, zorder=4)
        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=0.1)
        elif scale == "log":
            ax.set_xscale("log")
        ax.set_title(title, loc="left", pad=15)
        ax.text(0, 1.01, f"{better} is better", transform=ax.transAxes, fontsize=7,
                color=INK2, va="bottom", style="italic")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(layers, fontsize=8)
    axes[0].set_ylim(-0.7, len(layers) - 0.3)


def summary(a, b, layers):
    sp = (a.loc[layers].sim_seconds / b.loc[layers].sim_seconds)
    dram_a = (a.loc[layers].dram_bytes_read + a.loc[layers].dram_bytes_written)
    dram_b = (b.loc[layers].dram_bytes_read + b.loc[layers].dram_bytes_written)
    return sp, dram_b.sum() / dram_a.sum()


def figure(a_csv, b_csv, a_res, b_res, a_name, b_name, title, out, extra_pf=False):
    a, b = load(a_csv), load(b_csv)
    layers = order(sorted(set(a.index) & set(b.index)))
    a_cfg, b_cfg = read_config(a_res), read_config(b_res)
    metrics = list(METRICS)
    n = len(metrics) + (1 if extra_pf else 0)

    row_h = 0.26 * len(layers) + 1.3
    fig = plt.figure(figsize=(17, 3.6 + row_h))
    gs = fig.add_gridspec(2, n, height_ratios=[3.6, row_h], hspace=0.16, wspace=0.12,
                          left=0.08, right=0.99)
    settings_table(fig.add_subplot(gs[0, :]), a_cfg, b_cfg, a_name, b_name)
    axes = [fig.add_subplot(gs[1, 0])]
    axes += [fig.add_subplot(gs[1, i], sharey=axes[0]) for i in range(1, n)]
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    dumbbells(axes, a, b, layers, metrics)

    if extra_pf:
        ax = axes[-1]
        y = np.arange(len(layers))
        acc = b.loc[layers].l2_pf_accuracy.values * 100
        ax.barh(y, acc, height=0.55, color=SERIES[1], edgecolor="white", lw=0)
        for yi, v in zip(y, acc):
            ax.text(v + 1.5, yi, f"{v:.1f}%", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, max(acc) * 1.45)
        ax.set_title("L2 prefetch accuracy (%)", loc="left", pad=15)
        ax.text(0, 1.01, "useful prefetches / issued", transform=ax.transAxes, fontsize=7,
                color=INK2, va="bottom", style="italic")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)

    sp, dram_ratio = summary(a, b, layers)
    faster = (sp > 1.01).sum()
    note = (f"{b_name} vs {a_name}, {len(layers)} AI layers:  speed-up {sp.min():.2f}x to {sp.max():.2f}x "
            f"(median {sp.median():.2f}x);  faster by >1% on {faster} of {len(layers)} layers;  "
            f"total DRAM traffic x{dram_ratio:.2f}.\nAn orange dot sitting inside a blue ring means both setups have the SAME value "
            f"(e.g. all cache-miss counts in comparisons 1 and 3).\n* Misses = new cache-line fetches (gem5 MSHR misses); "
            f"repeat accesses to a line already on its way are not counted again.   Source: arch_study/datasets/{a_csv}, {b_csv}; "
            f"settings from each run's config.ini.")
    fig.suptitle(title, x=0.08, ha="left", y=0.995, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.08, -0.01, note, fontsize=8, color=INK2, ha="left", va="bottom")
    path = OUT / out
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
    print("  speed-up per layer:", ", ".join(f"{l}={v:.2f}" for l, v in sp.sort_values().items()))


FIGURES = [
    dict(a_csv="ai_kernels_pilot_v1.csv", b_csv="ai_kernels_2ghz_ddr5_4400_v1.csv",
         a_res="ai_pilot_v1", b_res="ai_2ghz_ddr5_4400_v1",
         a_name="Simple (1 GHz, DDR3)", b_name="2 GHz + DDR5-4400",
         title="Comparison 1: simple baseline vs 2 GHz + DDR5-4400 (same in-order core, same caches)",
         out="compare_1_simple_vs_2ghz_ddr5.png"),
    dict(a_csv="ai_kernels_p870d_full_v1.csv", b_csv="ai_kernels_p870d_full_pf_v1.csv",
         a_res="ai_p870d_full_v1", b_res="ai_p870d_full_pf_v1",
         a_name="P870-D model", b_name="P870-D + prefetcher",
         title="Comparison 2: P870-D-inspired model without vs with the L2 stride prefetcher",
         out="compare_2_p870d_vs_p870d_pf.png", extra_pf=True),
    dict(a_csv="ai_kernels_pilot_v1.csv", b_csv="ai_kernels_2ghz_ddr5_4400_open_v1.csv",
         a_res="ai_pilot_v1", b_res="ai_2ghz_ddr5_4400_open_v1",
         a_name="Simple (1 GHz, DDR3)", b_name="2 GHz + DDR5-4400 (open_adaptive)",
         title="Comparison 3: simple baseline vs 2 GHz + DDR5-4400 with the open_adaptive page policy (C2)",
         out="compare_3_simple_vs_2ghz_ddr5_open.png"),
    dict(a_csv="ai_kernels_2ghz_ddr5_4400_open_v1.csv", b_csv="ai_kernels_p870d_full_v1.csv",
         a_res="ai_2ghz_ddr5_4400_open_v1", b_res="ai_p870d_full_v1",
         a_name="2 GHz + DDR5-4400 (open_adaptive)", b_name="P870-D model",
         title="Comparison 4: 2 GHz + DDR5-4400 (open_adaptive, simple core) vs the P870-D-inspired model",
         out="compare_4_2ghz_ddr5_open_vs_p870d.png"),
    dict(a_csv="ai_kernels_hifive_unmatched_v1.csv", b_csv="ai_kernels_p870d_full_pf_v1.csv",
         a_res="ai_hifive_unmatched_v1", b_res="ai_p870d_full_pf_v1",
         a_name="HiFive Unmatched (gem5 board model)", b_name="P870-D + prefetcher",
         title="Comparison 5: HiFive Unmatched board model vs the P870-D-inspired model with prefetcher",
         out="compare_5_hifive_vs_p870d_pf.png"),
]


if __name__ == "__main__":
    for f in FIGURES:
        if (OUT / f["out"]).exists():
            print("exists, skipped:", OUT / f["out"])
        else:
            figure(**f)
