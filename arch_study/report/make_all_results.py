"""
make_all_results.py - all 46 AI layers on the final model in one place.

Inputs (read-only): the two datasets that used configuration F
(P870-D-inspired O3 core + L2 stride prefetcher, LRU, classic MOESI):
    datasets/ai_kernels_p870d_full_pf_v1.csv   27 layers (ResNet-50, BERT, DLRM, DeepBench)
    datasets/ai_kernels_v2_p870d_pf_v1.csv     19 layers (Llama-3.1-8B, Llama-2-70B, MobileNetV2, 3D-UNet)

Outputs (new files; refuses to overwrite):
    datasets/ai_kernels_all46_p870d_pf_v1.csv          combined raw data + derived columns
    report/output/All_46_layers_P870D_pf_LRU.xlsx      readable workbook
    report/figures/all46_p870d_pf_overview.png         one overview image
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PROJECT = Path.home() / "gem5-project"
DS = PROJECT / "arch_study" / "datasets"
REPORT = PROJECT / "arch_study" / "report"
INPUTS = ["ai_kernels_p870d_full_pf_v1.csv", "ai_kernels_v2_p870d_pf_v1.csv"]
OUT_CSV = DS / "ai_kernels_all46_p870d_pf_v1.csv"
OUT_XLSX = REPORT / "output" / "All_46_layers_P870D_pf_LRU.xlsx"
OUT_PNG = REPORT / "figures" / "all46_p870d_pf_overview.png"

MODELS = {  # dataset model name -> short name, in display order
    "ResNet-50 v1.5": "ResNet-50",
    "BERT-Large": "BERT-Large",
    "DLRM-v2 (DCNv2)": "DLRM-v2",
    "DeepBench / AI-Matrix micro": "DeepBench",
    "Llama-3.1-8B": "Llama-3.1-8B",
    "Llama-2-70B": "Llama-2-70B",
    "MobileNetV2": "MobileNetV2",
    "3D-UNet": "3D-UNet",
}
ORDER = list(MODELS.values())

# reference palette, categorical slots in fixed order (one per model)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = dict(zip(ORDER, SERIES))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

SETTINGS = [
    ("Reference processor", "SiFive P870-D (P870-D-inspired gem5 model, not an exact copy)"),
    ("CPU core", "gem5 RiscvO3CPU, out-of-order, 6-wide, TAGE, RAS 64, iTLB 32 / DTLB 64"),
    ("Clock", "2 GHz"),
    ("L1 I-cache / L1 D-cache", "64 KiB, 8-way, 4 cycles (each)"),
    ("L2 cache", "4 MiB, 16-way, 16 cycles + stride prefetcher"),
    ("L3 cache", "16 MiB, 16-way, 40 cycles"),
    ("Replacement policy", "LRU at every level (gem5 default LRURP)"),
    ("Coherence", "Classic memory system, MOESI snooping (CoherentXBar)"),
    ("Memory", "DDR5-4400 (gem5 DDR5_4400_4x8), open_adaptive page policy"),
    ("Cores / threads", "1 core, 1 thread (single-threaded kernels)"),
    ("Mode", "gem5 SE mode, RISC-V RV64GC, float32 kernels, cold caches (64 MiB flush before ROI)"),
    ("gem5 config", "arch_study/sim_config.py --cpu o3 --freq 2GHz --mem ddr5_4400 "
                    "--dram-page-policy open_adaptive --l1-lat 4 --l2-lat 16 --l3-lat 40 "
                    "--prefetch stride, caches 64KiB 8 4MiB 16 16MiB 16"),
    ("Simulated memory size", "512 MiB (27 Phase-1 layers) / 1 GiB (19 v2 layers); "
                              "in SE mode this does not affect timing"),
]

NOTES = [
    ("Time on chip", "Simulated time of the measured kernel region on the modelled 2 GHz chip "
                     "(gem5 simSeconds of the first stats block)."),
    ("IPC", "Instructions finished per clock cycle (higher = core kept busier)."),
    ("Miss rates / misses per 1000 instr.", "Counted as gem5 MSHR misses = new cache-line fetches. "
     "gem5's plain 'misses' also counts repeat accesses to a line already being fetched, which "
     "inflates numbers for the out-of-order core and the prefetcher."),
    ("L3 misses", "Include requests caused by the L2 prefetcher, so DRAM traffic is the more "
                  "reliable measure of memory pressure."),
    ("DRAM traffic", "Bytes read + written by the memory controller during the measured region."),
    ("Prefetch accuracy", "Useful prefetches / prefetches issued at L2."),
    ("Work fraction", "Shape-faithful, work-sampled: full tensors are allocated, but large layers "
                      "compute only part of their work units (budget about 20 M MACs)."),
    ("Label", "All kernels are model-derived kernels, not full benchmark implementations."),
]


def load():
    frames = []
    for name in INPUTS:
        d = pd.read_csv(DS / name)
        assert (d.status == "SUCCESS").all() and d.checksum_ok.all(), name
        assert (d.config_id == "F_P870D_full_L2stride").all(), name
        d.insert(0, "source_dataset", name)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    assert len(d) == 46 and d.layer_id.is_unique
    d.insert(1, "model_short", d.model.map(MODELS))
    d["order"] = d.model_short.map(ORDER.index)
    d = d.sort_values(["order", "source_dataset", "run_id"]).drop(columns="order").reset_index(drop=True)
    # derived, comparable metrics
    d["time_ms"] = d.sim_seconds * 1e3
    d["l1d_new_line_miss_rate_pct"] = d.l1d_mshr_misses / d.l1d_accesses * 100
    for c in ("l2", "l3"):
        d[f"{c}_new_line_misses_per_1k_instr"] = d[f"{c}_mshr_misses"] / d.committed_insts * 1000
    d["dram_traffic_mib"] = (d.dram_bytes_read + d.dram_bytes_written) / 2**20
    d["footprint_mib"] = d.footprint_bytes / 2**20
    d["l2_pf_accuracy_pct"] = d.l2_pf_accuracy * 100
    d["host_minutes"] = d.host_runtime_s / 60
    return d


SUMMARY_COLS = [
    ("model_short", "Model"), ("layer_id", "Layer"), ("op", "Operation"), ("kernel", "Kernel"),
    ("dims", "Shape"), ("work_fraction", "Work fraction"), ("time_ms", "Time on chip (ms)"),
    ("committed_insts", "Instructions"), ("ipc", "IPC"),
    ("l1d_new_line_miss_rate_pct", "L1-D miss rate (%)"),
    ("l2_new_line_misses_per_1k_instr", "L2 misses / 1000 instr."),
    ("l3_new_line_misses_per_1k_instr", "L3 misses / 1000 instr."),
    ("dram_traffic_mib", "DRAM traffic (MiB)"), ("footprint_mib", "Data size (MiB)"),
    ("l2_pf_accuracy_pct", "L2 prefetch accuracy (%)"), ("checksum_ok", "Answer correct"),
    ("host_minutes", "Simulation time on PC (min)"),
]
FORMATS = {"Work fraction": "0.0000", "Time on chip (ms)": "0.00", "Instructions": "#,##0",
           "IPC": "0.00", "L1-D miss rate (%)": "0.00", "L2 misses / 1000 instr.": "0.00",
           "L3 misses / 1000 instr.": "0.00", "DRAM traffic (MiB)": "0.0", "Data size (MiB)": "0.0",
           "L2 prefetch accuracy (%)": "0.0", "Simulation time on PC (min)": "0.0"}


def style_sheet(ws, widths, formats=None, wrap_cols=()):
    head = PatternFill("solid", fgColor="EFEEEA")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = head
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "C2" if ws.max_column > 4 else "A2"
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    if formats:
        names = [c.value for c in ws[1]]
        for name, fmt in formats.items():
            if name in names:
                col = names.index(name) + 1
                for row in ws.iter_rows(min_row=2, min_col=col, max_col=col):
                    row[0].number_format = fmt
    for col in wrap_cols:
        for row in ws.iter_rows(min_row=2, min_col=col, max_col=col):
            row[0].alignment = Alignment(wrap_text=True, vertical="top")


def write_xlsx(d):
    summary = d[[c for c, _ in SUMMARY_COLS]].rename(columns=dict(SUMMARY_COLS))
    summary.insert(0, "#", range(1, len(summary) + 1))
    per_model = (d.groupby("model_short", sort=False)
                   .agg(Layers=("layer_id", "count"),
                        **{"Median time (ms)": ("time_ms", "median"),
                           "Median IPC": ("ipc", "median"),
                           "Median L3 misses / 1000 instr.": ("l3_new_line_misses_per_1k_instr", "median"),
                           "Total DRAM traffic (MiB)": ("dram_traffic_mib", "sum"),
                           "Max DRAM traffic (MiB)": ("dram_traffic_mib", "max")})
                   .reset_index().rename(columns={"model_short": "Model"}))
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", mode="w") as xw:
        summary.to_excel(xw, sheet_name="All 46 layers", index=False)
        per_model.to_excel(xw, sheet_name="Per model", index=False)
        pd.DataFrame(SETTINGS, columns=["Setting", "Value"]).to_excel(xw, sheet_name="Settings", index=False)
        pd.DataFrame(NOTES, columns=["Column", "Meaning"]).to_excel(xw, sheet_name="Notes", index=False)
        d.to_excel(xw, sheet_name="Raw data (all columns)", index=False)

        ws = xw.sheets["All 46 layers"]
        style_sheet(ws, [4, 13, 22, 20, 16, 46, 10, 10, 13, 7, 10, 11, 11, 11, 10, 11, 9, 11], FORMATS)
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 45
        style_sheet(xw.sheets["Per model"], [14, 8, 12, 10, 14, 14, 14],
                    {k: "0.00" for k in per_model.columns[2:]})
        style_sheet(xw.sheets["Settings"], [26, 110], wrap_cols=(2,))
        style_sheet(xw.sheets["Notes"], [32, 110], wrap_cols=(2,))
        style_sheet(xw.sheets["Raw data (all columns)"], [26, 13] + [14] * (d.shape[1] - 2))
    print("wrote", OUT_XLSX)


PANELS = [
    ("time_ms", "Time on chip (ms)", "log", "lower is better"),
    ("ipc", "IPC", "linear", "higher is better"),
    ("l1d_new_line_miss_rate_pct", "L1-D miss rate (%)*", "linear", "lower is better"),
    ("l2_new_line_misses_per_1k_instr", "L2 misses / 1000 instr.*", "symlog", "lower is better"),
    ("l3_new_line_misses_per_1k_instr", "L3 misses / 1000 instr.*", "symlog", "lower is better"),
    ("dram_traffic_mib", "DRAM traffic (MiB)", "log", "lower is better"),
    ("l2_pf_accuracy_pct", "L2 prefetch accuracy (%)", "linear", "useful / issued"),
]


def write_png(d):
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8.5, "axes.edgecolor": INK2,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE,
        "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
    })
    # one row per layer, a blank row between models
    ys, y, ticks, labels, rows = [], 0, [], [], []
    for m in ORDER:
        part = d[d.model_short == m]
        for _, r in part.iterrows():
            rows.append((y, r))
            ticks.append(y)
            labels.append(r.layer_id)
            y += 1
        y += 0.8
    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(19, 0.27 * y + 2.2), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(top=0.935, bottom=0.04, left=0.08, right=0.99)
    for ax, (col, title, scale, hint) in zip(axes, PANELS):
        for yi, r in rows:
            ax.barh(yi, r[col], height=0.72, color=COLOR[r.model_short], edgecolor="white", lw=0)
        if scale == "log":
            ax.set_xscale("log")
        elif scale == "symlog":
            ax.set_xscale("symlog", linthresh=0.1)
        ax.set_title(title, loc="left", pad=14)
        ax.text(0, 1.004, hint, transform=ax.transAxes, fontsize=7, color=INK2, style="italic", va="bottom")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels, fontsize=7.6)
    axes[0].set_ylim(y - 0.5, -0.8)
    handles = [Patch(facecolor=COLOR[m], label=f"{m} ({(d.model_short == m).sum()})") for m in ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.977), ncol=8, fontsize=9,
               handlelength=1.2, columnspacing=1.4)
    fig.suptitle("All 46 AI layers on the P870-D-inspired model + L2 stride prefetcher "
                 "(O3 6-wide, 2 GHz, 64K/4M/16M caches, LRU, MOESI, DDR5-4400)",
                 x=0.08, ha="left", y=0.99, fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.08, 0.022,
             "* Misses = new cache-line fetches (gem5 MSHR misses). L3 misses include L2-prefetch requests, "
             "so DRAM traffic is the more reliable memory-pressure measure. Model-derived kernels, cold caches, "
             "work-sampled where large.\nSource: arch_study/datasets/ai_kernels_all46_p870d_pf_v1.csv "
             "(= ai_kernels_p870d_full_pf_v1.csv + ai_kernels_v2_p870d_pf_v1.csv).",
             fontsize=7.8, color=INK2, va="top")
    fig.savefig(OUT_PNG)
    plt.close(fig)
    print("wrote", OUT_PNG)


def main():
    for p in (OUT_CSV, OUT_XLSX, OUT_PNG):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    d = load()
    with OUT_CSV.open("x", newline="") as f:
        d.to_csv(f, index=False)
    print("wrote", OUT_CSV, f"({len(d)} rows)")
    write_xlsx(d)
    write_png(d)


if __name__ == "__main__":
    main()
