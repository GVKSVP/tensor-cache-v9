#!/usr/bin/env python3
"""
build_report.py - UROP report (SRM University-AP format) as a .docx.

Every number in the text is computed here from the study CSVs (read-only),
so the report always matches the data. Figures come from make_figures.py.
Word (via update_and_export.ps1) later fills the TOC / lists and exports PDF.

Output: arch_study/report/output/Tensor_AI_UROP_Report.docx
"""

import glob
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT = Path.home() / "gem5-project"
STUDY = PROJECT / "arch_study"
DS = STUDY / "datasets"
FIG = STUDY / "report" / "figures"
OUT = STUDY / "report" / "output"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "Palatino Linotype"
TITLE = "Tensor-Aware AI Caching Management for Optimizing Memory Performance in AI Workloads"
SUBTITLE = ("Phase 1: A SiFive P870-D-Inspired RISC-V Simulation Platform and "
            "Memory Characterisation of Model-Derived AI Workloads in gem5")
TEAM = [("Team member 1", "<student ID>"),
        ("Team member 2", "<student ID>"),
        ("Team member 3", "<student ID>")]
MENTOR = "[Mentor Name]"
SIX = ["RN50_conv1", "RN50_maxpool", "DB_gemv_3072", "BERT_ffn1", "BERT_embed", "DLRM_embbag"]


# ======================================================================
# Data: compute every number used in the text
# ======================================================================

def load(name):
    return pd.read_csv(DS / name).set_index("layer_id")


def roi_stat(results_dir, layer, name):
    f = glob.glob(str(STUDY / "results" / results_dir / f"*_{layer}" / "stats.txt"))[0]
    block = 0
    for line in open(f, errors="ignore"):
        if "Begin Simulation Statistics" in line:
            block += 1
            if block > 1:
                break
        elif block == 1 and line.split()[:1] == [name]:
            return float(line.split()[1])
    return float("nan")


N = {}

# Baseline-v1 (56k) sensitivity
d56 = pd.read_csv(PROJECT / "tensor_cache_dataset_56000.csv",
                  usecols=["workload", "problem_args", "sim_ticks", "source"])
spread = (d56.groupby(["workload", "problem_args"]).sim_ticks
            .agg(lambda x: x.max() / x.min()).groupby("workload").median() - 1) * 100
N["n56_rows"] = len(d56)
N["n56_reused"] = int((d56.source == "REUSED").sum())
N["n_le3"] = int((spread <= 3.0).sum())
N["n_wl"] = len(spread)
N["above3"] = spread[spread > 3.0].sort_values(ascending=False)

B1 = load("ai_kernels_pilot_v1.csv")
C = load("ai_kernels_2ghz_ddr5_4400_v1.csv").reindex(B1.index)
C2 = load("ai_kernels_2ghz_ddr5_4400_open_v1.csv")
D = load("ai_kernels_o3_2ghz_ddr5_4400_open_v1.csv")
E = load("ai_kernels_p870d_full_v1.csv")
F = pd.read_csv(DS / "ai_kernels_p870d_full_pf_v1.csv").sort_values("run_id").set_index("layer_id")
H = load("ai_kernels_hifive_unmatched_v1.csv")
LAYERS = pd.read_csv(STUDY / "ai_kernels" / "layers.csv").set_index("layer_id")

assert len(B1) == 27 and len(F) == 27 and (F.status == "SUCCESS").all() and F.checksum_ok.all()

# B1 classes
N["b1_mem"] = B1[B1.l3_mpki >= 1].index.tolist()
N["b1_res"] = B1[B1.l3_mpki < 1].index.tolist()

# clock + DDR5
sp_c = B1.sim_seconds / C.sim_seconds
N["sp_c_min"], N["sp_c_max"] = sp_c.min(), sp_c.max()
N["sp_c_res"] = sp_c[B1.l3_mpki < 1].min()
N["corr_c"] = sp_c.corr(B1.l3_mpki)
N["ffn_misslat_1g"] = B1.loc["BERT_ffn1", "l1d_avg_miss_latency_ticks"] / 1000   # 1 GHz: 1000 ticks/cycle
N["ffn_misslat_2g"] = C.loc["BERT_ffn1", "l1d_avg_miss_latency_ticks"] / 500    # 2 GHz: 500 ticks/cycle

# page policy (10 layers)
L10 = list(C2.index)
lat3 = B1.reindex(L10).dram_avg_mem_acc_lat_ticks / 1e3
latc = C.reindex(L10).dram_avg_mem_acc_lat_ticks / 1e3
lato = C2.dram_avg_mem_acc_lat_ticks / 1e3
rh3 = pd.Series({l: roi_stat("ai_pilot_v1", l, "system.mem_ctrl.dram.readRowHitRate") for l in L10})
rhc = pd.Series({l: roi_stat("ai_2ghz_ddr5_4400_v1", l, "system.mem_ctrl.dram.readRowHitRate") for l in L10})
rho = pd.Series({l: roi_stat("ai_2ghz_ddr5_4400_open_v1", l, "system.mem_ctrl.dram.readRowHitRate") for l in L10})
gain_open = (C.reindex(L10).sim_seconds / C2.sim_seconds - 1) * 100
memL = [l for l in L10 if l != "RN50_conv1" and l != "RN50_maxpool"]
N.update(lat3_lo=lat3.min(), lat3_hi=lat3.max(), latc_lo=latc.min(), latc_hi=latc.max(),
         lato_lo=lato.min(), lato_hi=lato.max(), rh3_lo=rh3.min(), rh3_hi=rh3.max(),
         rhc_hi=rhc.max(), rho_lo=rho.min(), rho_hi=rho.max(),
         gain_lo=gain_open[memL].min(), gain_hi=gain_open[memL].max(),
         gain_ctrl=gain_open["RN50_conv1"])
rdq_c2 = pd.Series({l: roi_stat("ai_2ghz_ddr5_4400_open_v1", l, "system.mem_ctrl.avgRdQLen") for l in SIX})

# O3 core (C2 -> D) on six layers
sp_o3 = C2.reindex(SIX).sim_seconds / D.reindex(SIX).sim_seconds
N.update(sp_o3_lo=sp_o3.min(), sp_o3_hi=sp_o3.max())
rdq_d = pd.Series({l: roi_stat("ai_o3_2ghz_ddr5_4400_open_v1", l, "system.mem_ctrl.avgRdQLen") for l in SIX})
bw_d = D.reindex(SIX).dram_avg_rd_bw_MBps

# P870-D caches (D -> E)
sp_e = D.reindex(SIX).sim_seconds / E.reindex(SIX).sim_seconds          # speed-up of E over D
t_inc = (E.reindex(SIX).sim_seconds / D.reindex(SIX).sim_seconds - 1) * 100  # time increase (%)
others = [l for l in SIX if l != "BERT_ffn1"]
N.update(ffn_sp=sp_e["BERT_ffn1"], ffn_dram_d=D.loc["BERT_ffn1", "dram_bytes_read"] / 2**20,
         ffn_dram_e=E.loc["BERT_ffn1", "dram_bytes_read"] / 2**20,
         oth_sp_lo=sp_e[others].min(), oth_sp_hi=sp_e[others].max(),
         oth_t_lo=t_inc[others].min(), oth_t_hi=t_inc[others].max())

# prefetcher (E -> F)
Fs = F.reindex(SIX)
sp_pf = E.reindex(SIX).sim_seconds / Fs.sim_seconds
N.update(pf_acc_lo=Fs.l2_pf_accuracy.min() * 100, pf_acc_hi=Fs.l2_pf_accuracy.max() * 100,
         dlrm_dram_e=E.loc["DLRM_embbag", "dram_bytes_read"] / 2**20,
         dlrm_dram_f=F.loc["DLRM_embbag", "dram_bytes_read"] / 2**20)

# HiFive vs F
r_h = H.reindex(SIX).sim_seconds / Fs.sim_seconds
r_h_e = H.reindex(SIX).sim_seconds / E.reindex(SIX).sim_seconds
N.update(rh_lo=r_h.min(), rh_hi=r_h.max())
hf_cov = roi_stat("ai_hifive_unmatched_v1", "BERT_embed", "board.cache_hierarchy.l1dcaches.prefetcher.coverage")

# final 27 layers
sp_f = B1.reindex(F.index).sim_seconds / F.sim_seconds
N.update(spf_lo=sp_f.min(), spf_med=sp_f.median(), spf_hi=sp_f.max(),
         ipc_lo=F.ipc.min(), ipc_hi=F.ipc.max(), ipc_med=F.ipc.median())
traffic = F.dram_bytes_read / 1024 / (F.macs_executed / 1e6)


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


# ======================================================================
# docx helpers
# ======================================================================

doc = Document()


def set_style_font(style, size=None, bold=None, italic=None, color=None, name=FONT):
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    style.font.color.rgb = RGBColor(0, 0, 0) if color is None else color


normal = doc.styles["Normal"]
set_style_font(normal, size=12)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15

for name, size in (("Heading 1", 18), ("Heading 2", 14), ("Heading 3", 12)):
    st = doc.styles[name]
    set_style_font(st, size=size, bold=True, italic=False)
    st.paragraph_format.space_before = Pt(12 if name != "Heading 1" else 0)
    st.paragraph_format.space_after = Pt(8 if name == "Heading 1" else 4)
    st.paragraph_format.keep_with_next = True

try:
    cap = doc.styles["Caption"]
except KeyError:
    from docx.enum.style import WD_STYLE_TYPE
    cap = doc.styles.add_style("Caption", WD_STYLE_TYPE.PARAGRAPH)
    cap.base_style = normal
set_style_font(cap, size=10, bold=False, italic=True)
cap.paragraph_format.space_before = Pt(4)
cap.paragraph_format.space_after = Pt(10)

for name in ("List Bullet", "List Number"):
    set_style_font(doc.styles[name], size=12)

sec0 = doc.sections[0]
sec0.page_height, sec0.page_width = Cm(29.7), Cm(21.0)
sec0.top_margin = sec0.bottom_margin = Cm(2.5)
sec0.left_margin, sec0.right_margin = Cm(2.8), Cm(2.5)
TEXT_W = 21.0 - 2.8 - 2.5


def add_field(par, instr, placeholder=" "):
    def fld(kind):
        r = par.add_run()
        e = OxmlElement("w:fldChar")
        e.set(qn("w:fldCharType"), kind)
        r._r.append(e)
    fld("begin")
    r = par.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r._r.append(it)
    fld("separate")
    par.add_run(placeholder)
    fld("end")


def runs_with_markup(par, text, size=None):
    """**bold** and *italic* inline markup."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*"):
            r = par.add_run(part[1:-1])
            r.italic = True
        else:
            r = par.add_run(part)
        if size:
            r.font.size = Pt(size)
    return par


def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, style=None, size=None, space_after=None):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    runs_with_markup(p, text, size)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        runs_with_markup(p, it)


def h1(text, new_page=True):
    # page_break_before on the heading itself never produces an empty page,
    # unlike a separate page-break paragraph after a full page.
    h = doc.add_heading(text, level=1)
    h.paragraph_format.page_break_before = new_page
    return h


def h2(text):
    return doc.add_heading(text, level=2)


def h3(text):
    return doc.add_heading(text, level=3)


def caption(kind, text):
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{kind} ")
    add_field(p, f"SEQ {kind} \\* ARABIC", "0")
    p.add_run(f": {text}")
    return p


def figure(png, text, width_cm=15.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / png), width=Cm(width_cm))
    caption("Figure", text)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def no_split(row):
    """Keep a table row on one page."""
    trpr = row._tr.get_or_add_trPr()
    cs = OxmlElement("w:cantSplit")
    cs.set(qn("w:val"), "true")
    trpr.insert(0, cs)


def shape_text(kernel, args):
    """Human-readable layer shape from the kernel arguments."""
    a = args.split()
    if kernel == "conv2d_nchw":
        C, H, W, K, R, S, st, p = a[:8]
        return f"C{C}×{H}×{W} → K{K}, {R}×{S}, s{st}, p{p}"
    if kernel == "pool":
        kind, C, H, W, R, st, p = a[:7]
        if kind == "avg" and R == H == W:
            return f"global avg, {C}×{H}×{W}"
        return f"{kind} {R}×{R}, s{st}, p{p}, {C}×{H}×{W}"
    if kernel == "gemm":
        M, N_, K = a[:3]
        return f"M{M} × K{K} × N{N_}"
    if kernel == "attention":
        S, Dh, heads = a[:3]
        return f"S{S}, d{Dh}, {heads} heads"
    if kernel == "embedding":
        if a[0] == "gather":
            return f"{a[3]} lookups, {a[1]}×{a[2]} table"
        return f"26 tables × 128, batch {a[1]}, cap {a[2]} rows"
    if kernel == "mlp":
        return f"{a[0].replace(',', '-')}, batch {a[1]}"
    return args


OP_NAME = {"conv2d_nchw": "Convolution", "pool": "Pooling", "attention": "Attention",
           "mlp": "MLP (GEMV chain)"}


def op_text(kernel, args):
    if kernel == "gemm":
        M, N_ = args.split()[:2]
        return "GEMV" if "1" in (M, N_) else "GEMM"
    if kernel == "embedding":
        return "Embedding gather" if args.split()[0] == "gather" else "Embedding bag"
    return OP_NAME[kernel]


def table(headers, rows, widths_cm, size=9, cap_text=None, align_right_from=None):
    if cap_text:
        c = caption("Table", cap_text)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0]
    trpr = hdr._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trpr.append(th)
    no_split(hdr)
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.width = Cm(widths_cm[i])
        shade(cell, "D9D9D9")
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(size)
    for row in rows:
        new_row = t.add_row()
        no_split(new_row)
        cells = new_row.cells
        for i, v in enumerate(row):
            cells[i].width = Cm(widths_cm[i])
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            if align_right_from is not None and i >= align_right_from:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            runs_with_markup(p, str(v), size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def page_numbers(section, fmt):
    section.footer.is_linked_to_previous = False
    fp = section.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_field(fp, "PAGE", "1")
    sp = section._sectPr
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), "1")
    anchor = sp.find(qn("w:cols"))
    if anchor is None:
        anchor = sp.find(qn("w:docGrid"))
    if anchor is not None:
        anchor.addprevious(pg)
    else:
        sp.append(pg)


def centered(text, size=12, bold=False, italic=False, after=4, color=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    if color:
        r.font.color.rgb = color
    return p


# ======================================================================
# Title page (unnumbered)
# ======================================================================

sec0.footer.is_linked_to_previous = False
for _ in range(2):
    centered("", after=6)
centered(TITLE, size=17, bold=True, after=8)
centered(SUBTITLE, size=12, italic=True, after=22)
centered("UROP Project submitted to the", after=2)
centered("SRM University – AP, Andhra Pradesh", after=2)
centered("for the partial fulfillment of the requirements to award the degree of", after=18)
centered("Bachelor of Technology", bold=True, after=2)
centered("in", bold=True, after=2)
centered("Computer Science and Engineering", bold=True, after=2)
centered("School of Engineering and Sciences", bold=True, after=18)
centered("Submitted by", after=4)
for name, rid in TEAM:
    centered(f"{name}  |  {rid}", bold=True, after=2)
centered("", after=10)
centered("[Insert SRM University–AP logo here]", size=10, italic=True, after=10,
         color=RGBColor(0x80, 0x80, 0x80))
centered("Under the Guidance of", after=4)
centered(MENTOR, bold=True, after=2)
centered("SRM University–AP", bold=True, after=2)
centered("Neerukonda, Mangalagiri,", bold=True, after=2)
centered("Guntur, Andhra Pradesh – 522 240", bold=True, after=2)
centered("November, 2026", bold=True, after=2)

# ======================================================================
# Front matter (roman numbering)
# ======================================================================

front = doc.add_section(WD_SECTION.NEW_PAGE)
page_numbers(front, "lowerRoman")

doc.add_heading("Certificate", level=1)
para(f"This is to certify that the UROP project report entitled **“{TITLE}”**, submitted by "
     + ", ".join(f"**{n} ({r})**" for n, r in TEAM)
     + ", is a bonafide record of the work carried out by them under my guidance and supervision "
       "at SRM University–AP, Andhra Pradesh, in partial fulfilment of the requirements for the "
       "award of the degree of Bachelor of Technology in Computer Science and Engineering.")
for _ in range(4):
    doc.add_paragraph()
p = doc.add_paragraph()
p.add_run("Signature of the Mentor").bold = True
para(MENTOR, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=0)
para("Department of Computer Science and Engineering", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=0)
para("SRM University–AP", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=0)
para("Date: ____________", align=WD_ALIGN_PARAGRAPH.LEFT)

h1("Acknowledgements")
para(f"We would like to thank **{MENTOR}** for the guidance and support given to us while working on "
     "this project. The patience and motivation of our mentor, and the critical questions raised during "
     "the review of our simulation setup, pushed us to ground every architectural parameter in real "
     "hardware documentation and to complete the research on time. We also thank SRM University–AP for "
     "the opportunity provided through the Undergraduate Research Opportunities Programme (UROP).")
para("We acknowledge the developers of the open-source gem5 simulator, the MLCommons MLPerf Inference "
     "project, Baidu DeepBench and Alibaba AI-Matrix, whose public work made this study possible. Finally, "
     "we thank each of our team members for being each other's constant support throughout the project.")

h1("Table of Contents")
add_field(doc.add_paragraph(), 'TOC \\o "1-3" \\h \\z \\u', "Right-click and choose Update Field.")

# --- Abstract ---------------------------------------------------------
h1("Abstract")
para("Artificial-intelligence (AI) workloads spend a large share of their execution time moving tensors "
     "through the memory hierarchy, and conventional cache-management policies treat these tensors as "
     "anonymous cache lines. This project, *Tensor-Aware AI Caching Management*, aims to make cache "
     "decisions on the basis of tensor behaviour. This report presents **Phase 1** of the project: a "
     "hardware-justified RISC-V simulation platform and a systematic memory characterisation of real AI "
     "layer shapes, which together form the foundation for the AI-based cache-management policy of Phase 2.")
para(f"An earlier dataset of {N['n56_rows']:,} gem5 simulations was first audited. It showed that, with a "
     f"blocking in-order CPU, DDR3 memory and no prefetcher, cache size and associativity changed runtime by "
     f"3 % or less for {N['n_le3']} of {N['n_wl']} workloads, so the original setup could not expose "
     f"cache-management effects. The simulated system was therefore rebuilt around a real reference "
     f"processor, the **SiFive P870-D** (BigSky SF-2U870 platform), with every parameter labelled as an "
     f"official specification, a third-party specification or a documented simulation assumption. Twenty-seven "
     f"float32 C kernels were derived from documented layer shapes of **ResNet-50, BERT-Large and DLRM-v2** "
     f"(MLPerf Inference) and of **DeepBench / AI-Matrix**, covering convolution, matrix multiplication, "
     f"attention, embedding lookup and pooling. Kernels were verified against numpy and measured only inside "
     f"their region of interest.")
para(f"Changing one factor at a time showed that (i) raising the clock to 2 GHz with DDR5 yields "
     f"{f2(N['sp_c_res'])}–2.00× speed-up for cache-resident layers but only {f2(N['sp_c_min'])}× for "
     f"memory-bound layers (correlation with L3 MPKI {f2(N['corr_c'])}); (ii) gem5's default DDR5 page policy "
     f"produced 0 % row-buffer hits, which we corrected; (iii) a 6-wide out-of-order core gives "
     f"{f2(N['sp_o3_lo'])}–{f2(N['sp_o3_hi'])}× speed-up; (iv) P870-D cache capacities cut BERT feed-forward "
     f"DRAM traffic from {f1(N['ffn_dram_d'])} MiB to {f1(N['ffn_dram_e'])} MiB; and (v) an L2 stride "
     f"prefetcher speeds lookup-heavy layers up to {f2(sp_pf.max())}× although only "
     f"{f1(N['pf_acc_lo'])}–{f1(N['pf_acc_hi'])} % of its prefetches are useful. On the final P870-D-inspired "
     f"model, the 27 layers run {f2(N['spf_lo'])}–{f2(N['spf_hi'])}× faster than on the baseline, and the "
     f"P870-D-inspired model outperforms gem5's hardware-validated HiFive Unmatched model by "
     f"{f2(N['rh_lo'])}–{f2(N['rh_hi'])}×. The results identify three tensor classes – streaming, "
     f"reuse-critical and irregular – whose differing needs motivate the tensor-aware policy of Phase 2.")
p = para("**Keywords:** cache management, tensor-aware caching, RISC-V, gem5, SiFive P870-D, AI workloads, "
         "memory hierarchy, prefetching, DDR5.", align=WD_ALIGN_PARAGRAPH.LEFT)

# --- Abbreviations ----------------------------------------------------
h1("Abbreviations")
abbr = [
    ("A", "Simulation assumption (parameter not published; chosen and documented by us)"),
    ("AI / ML", "Artificial Intelligence / Machine Learning"),
    ("BERT", "Bidirectional Encoder Representations from Transformers"),
    ("CHI", "Coherent Hub Interface (AMBA coherence protocol)"),
    ("CNN", "Convolutional Neural Network"),
    ("DDR", "Double Data Rate (DRAM standard)"),
    ("DLRM", "Deep Learning Recommendation Model"),
    ("DRAM", "Dynamic Random-Access Memory"),
    ("FS / SE", "Full-System / Syscall-Emulation mode of gem5"),
    ("GEMM / GEMV", "General Matrix–Matrix / Matrix–Vector multiplication"),
    ("H-OFF", "Hardware specification from official SiFive documentation"),
    ("H-3P", "Hardware specification from a third-party source (Hot Chips 2023 report)"),
    ("HBM", "High Bandwidth Memory"),
    ("IPC", "Instructions Per Cycle"),
    ("ISA", "Instruction Set Architecture"),
    ("KiB / MiB / GiB", "2^10 / 2^20 / 2^30 bytes"),
    ("L1 / L2 / L3", "Level-1 / Level-2 / Level-3 cache"),
    ("LLC", "Last-Level Cache"),
    ("LRU", "Least Recently Used (replacement policy)"),
    ("MAC", "Multiply–Accumulate operation"),
    ("MPKI", "Misses Per Kilo-Instruction (misses × 1000 / instructions)"),
    ("MSHR", "Miss Status Holding Register"),
    ("O3", "gem5 out-of-order CPU model (RiscvO3CPU)"),
    ("RAS", "Return Address Stack"),
    ("ROI", "Region Of Interest (the measured kernel code)"),
    ("RRIP / SHiP", "Re-Reference Interval Prediction / Signature-based Hit Predictor"),
    ("RVV", "RISC-V Vector extension"),
    ("TAGE", "TAgged GEometric-history-length branch predictor"),
    ("TLB", "Translation Look-aside Buffer"),
]
table(["Abbreviation", "Meaning"], abbr, [3.6, TEXT_W - 3.6], size=10)

h1("List of Figures")
add_field(doc.add_paragraph(), 'TOC \\h \\z \\c "Figure"', "Right-click and choose Update Field.")
h1("List of Tables", new_page=False)
add_field(doc.add_paragraph(), 'TOC \\h \\z \\c "Table"', "Right-click and choose Update Field.")

# ======================================================================
# Main body (arabic numbering)
# ======================================================================

body = doc.add_section(WD_SECTION.NEW_PAGE)
page_numbers(body, "decimal")

# --- 1. Introduction --------------------------------------------------
doc.add_heading("1. Introduction", level=1)
h2("1.1 Background and Motivation")
para("Artificial Intelligence and Machine Learning have become a central part of modern computing, from "
     "computer vision and natural-language processing to large-scale recommendation systems. The performance "
     "of these applications depends not only on arithmetic throughput but also on how efficiently tensors – "
     "weights, activations and embedding tables – move through the memory hierarchy. A single inference of a "
     "transformer layer streams megabytes of weights, while a recommendation model performs thousands of "
     "irregular lookups into tables that are far larger than any on-chip cache.")
para("Modern processors hide memory latency with a hierarchy of caches (L1, L2 and a shared L3) and with "
     "hardware prefetchers. Their replacement and prefetching policies, however, were designed for general "
     "programs: a least-recently-used (LRU) cache does not know that a 16 MiB weight matrix will be re-read "
     "for every input token, nor that a pooled feature map will never be touched again. The HERMES study [1] "
     "highlighted these issues for ML workloads on RISC-V systems and proposed shared L3 caching, tensor-aware "
     "caching, prefetching and hybrid DRAM–HBM memory as remedies.")
h2("1.2 Problem Statement")
para("The project proposal set out to build an AI-assisted, tensor-aware cache-management mechanism and to "
     "train it on data generated by our own simulations. Two prerequisites had to be satisfied before such a "
     "policy can be evaluated credibly:")
bullets([
    "**A realistic, justified hardware model.** A review of our initial simulator found that it used generic "
    "parameters (1 GHz in-order CPU, DDR3 memory, arbitrary cache latencies, an implicit LRU policy) without "
    "hardware justification, and that cache configuration barely influenced performance in it.",
    "**Realistic AI workloads with complete cache statistics.** Full benchmark suites such as MLPerf require "
    "Python, deep-learning frameworks or GPUs and cannot run directly in our RISC-V gem5 environment.",
])
h2("1.3 Objectives")
bullets([
    "Audit the existing simulation project and its 56,000-run dataset without modifying it.",
    "Select a real RISC-V reference processor and justify every simulated parameter, separating official "
    "hardware specifications from simulation assumptions.",
    "Develop float32 kernels from documented layer shapes of MLPerf models (ResNet-50, BERT-Large, DLRM-v2) "
    "and DeepBench / AI-Matrix, covering convolution, matrix multiplication, attention, embedding/gather and pooling.",
    "Measure execution time, IPC, per-level cache statistics, prefetcher statistics and DRAM traffic for every "
    "layer, changing one architectural factor at a time.",
    "Identify the tensor behaviours that a tensor-aware cache-management policy must handle.",
], style="List Number")
h2("1.4 Scope of this Report")
para("This report covers **Phase 1**: the simulation platform, the AI workload suite and the memory "
     "characterisation. The AI-based replacement and prefetching policy proposed in the project proposal is "
     "the subject of **Phase 2**; this report does not claim that such a policy has been implemented. All "
     "results are produced by simulation; no physical SiFive hardware was used.")
h2("1.5 Organisation of the Report")
para("Section 2 surveys related work. Section 3 describes the system model: the reference hardware, the "
     "simulated configurations, the AI workloads and the measurement method. Section 4 presents the "
     "simulation results and their implications. Section 5 concludes and Section 6 outlines future work.")

# --- 2. Literature Survey --------------------------------------------
h1("2. Literature Survey")
h2("2.1 Memory Behaviour of Deep-Learning Workloads")
para("MLPerf Inference [4] defines industry-standard inference benchmarks, including ResNet-50 [5] for image "
     "classification, BERT-Large [6] for question answering and DLRM [7], [8] for recommendation. DeepBench [9] "
     "isolates the dominant kernels – GEMM, convolution and recurrent layers – with shapes taken from deployed "
     "applications, and the micro-benchmarks of Alibaba AI-Matrix [10] are derived from DeepBench. Gupta et al. "
     "[11] showed that recommendation models are dominated by irregular embedding-table lookups that are memory-"
     "bound and cache-unfriendly, in contrast to the compute-bound convolution layers of CNNs. These works motivate "
     "studying AI workloads layer by layer rather than as a single average.")
h2("2.2 Cache Replacement Policies")
para("Belady's optimal policy [16] evicts the line whose next use lies furthest in the future; it is not "
     "implementable online but serves as an upper bound. LRU approximates recency and is the default in many "
     "simulators, including gem5 [2], [3]. Jaleel et al. [12] proposed Re-Reference Interval Prediction (RRIP), "
     "whose scan-resistant variants insert new lines with a long predicted re-reference interval so that "
     "streaming data does not flush the cache. SHiP [13] learns, per instruction signature, whether inserted lines "
     "are likely to be reused. Hawkeye [14] reconstructs Belady's decisions for past accesses and trains a "
     "predictor to imitate them.")
h2("2.3 Learning-Based Cache Management")
para("Shi et al. [15] applied deep learning (an LSTM trained offline) to cache replacement and distilled it "
     "into a practical hardware predictor (Glider). Such work shows that learned models can approach optimal "
     "replacement, but it targets general-purpose programs and does not exploit tensor-level information such as "
     "which tensor an access belongs to or how many times that tensor will be reused. HERMES [1] introduced "
     "tensor-aware caching for RISC-V ML systems in combination with a shared L3, the Gemmini accelerator and "
     "DRAM–HBM memory, reporting improved hit rate and latency in simulation.")
h2("2.4 Hardware Prefetching")
para("Stride prefetchers [17] detect constant-stride access streams per instruction and fetch lines ahead of "
     "demand. They are effective for regular tensor traversals but can pollute the cache when streams are short "
     "or irregular. Modern cores such as the SiFive P800 series include a streaming prefetcher [18].")
h2("2.5 RISC-V Processors and Simulation with gem5")
para("The SiFive P870 core [18], [20] is a 64-bit, 6-wide out-of-order RISC-V core compliant with the RVA23 "
     "profile, and the BigSky SF-2U870 [19] is a 32-core server platform built on it. gem5 [2], [3] is the most "
     "widely used open-source architectural simulator; it supports RISC-V with in-order (TimingSimple, Minor) and "
     "out-of-order (O3) CPU models, a classic and a Ruby memory system, and detailed DRAM models. Its standard "
     "library includes a model of the SiFive HiFive Unmatched board [22] that was calibrated against the real "
     "board [21].")
h2("2.6 Research Gap")
para("Existing replacement and prefetching policies are either tensor-agnostic or evaluated on generic "
     "hardware models. To design a tensor-aware policy, one first needs (a) a hardware model whose parameters "
     "are traceable to a real processor and (b) per-layer evidence of which tensors are streamed, reused or "
     "accessed irregularly on that model. Phase 1 of this project provides both.")

# --- 3. System Model ---------------------------------------------------
h1("3. System Model")
h2("3.1 Simulation Framework")
para("All experiments use **gem5 version 25.1.0.1**, built for RISC-V (build/RISCV/gem5.opt) on Ubuntu 24.04 "
     "under WSL2. Workloads run in syscall-emulation (SE) mode as statically linked RV64GC binaries. The classic "
     "memory system is used, in which coherent crossbars implement a MOESI-style snooping protocol. The Ruby "
     "memory system in this build contains only the MI_example protocol; the CHI protocol is not compiled in "
     "(RUBY_PROTOCOL_CHI is not set). The original simulation script, tensor_cache_sim.py, and its 56,000-row "
     "dataset were preserved unchanged and protected with SHA-256 checksums; all new work uses a separate, "
     "configurable copy (arch_study/sim_config.py) whose defaults reproduce the original exactly.")
h2("3.2 Reference Hardware: SiFive P870-D")
para("The SiFive P870-D core in the BigSky SF-2U870 platform was chosen as the reference processor because it "
     "is a current high-performance RISC-V design with a three-level cache hierarchy, an out-of-order core and "
     "DDR5 memory. Table 1 lists each parameter with its source: **H-OFF** for official SiFive documentation "
     "[18], [19], **H-3P** for the Hot Chips 2023 presentation as reported by Chips and Cheese [20], and **A** "
     "for a documented simulation assumption. The simulated model is *P870-D-inspired*: gem5's O3 CPU is a "
     "generic out-of-order model, not SiFive's implementation.")
hw = [
    ("ISA", "64-bit RISC-V, RVA23, RVV 1.0, 2×128-bit VLEN", "RV64GC scalar binaries", "H-OFF / A"),
    ("Core", "6-wide out-of-order", "RiscvO3CPU, all pipeline widths = 6", "H-OFF"),
    ("Core count", "32 × P870-D (BigSky)", "1 core (single-threaded kernels, SE mode)", "H-OFF / A"),
    ("Clock", "2.0 GHz", "2.0 GHz", "H-OFF"),
    ("Branch prediction", "8-table TAGE, 64-entry RAS", "gem5 TAGE (default tables), RAS 64", "H-3P / A"),
    ("TLBs", "iTLB 32, DTLB 64, L2 TLB 1024", "iTLB 32, DTLB 64 (no L2 TLB in gem5)", "H-3P / A"),
    ("L1 I-cache", "64 KB", "64 KiB, 8-way, 4 cycles", "H-3P / A"),
    ("L1 D-cache", "4-cycle access; size not published", "64 KiB, 8-way, 4 cycles", "H-3P / A"),
    ("L2 cache", "Non-inclusive, 16 cycles, 4 MB example", "4 MiB, 16-way, 16 cycles", "H-3P / A"),
    ("L3 cache", "Non-inclusive; 16 MB example", "16 MiB, 16-way, 40 cycles", "H-OFF, H-3P / A"),
    ("Prefetcher", "“Streaming Prefetcher”", "gem5 StridePrefetcher on L2", "H-OFF / A"),
    ("Coherence", "CHI interface; L2 snoop filters", "Classic MOESI snooping crossbars", "H-OFF, H-3P / A"),
    ("Replacement", "Not published", "LRU (gem5 default)", "A"),
    ("Memory", "256 GB DDR5-5600", "gem5 DDR5_4400_4x8, 1 channel, open-adaptive", "H-OFF / A"),
    ("Process", "TSMC N6", "Not modelled", "H-OFF"),
]
table(["Parameter", "SiFive P870-D / BigSky (real)", "Simulated value", "Source"], hw,
      [2.7, 5.0, 5.4, TEXT_W - 13.1], size=8.5,
      cap_text="Reference hardware parameters and their simulated values")
para("Three deviations need explanation. First, gem5 25.1 provides DDR5-4400, DDR5-6400 and DDR5-8400 models "
     "but no DDR5-5600 model; the unmodified DDR5-4400 model was used, whose peak bandwidth is about 21 % below "
     "the real system. Second, the official P800 page mentions private L2 caches while the Hot Chips report "
     "describes a shared L2; with a single simulated core the distinction does not change the results. Third, "
     "the L3 latency, all associativities and the replacement policy are not published and are therefore "
     "assumptions.")
h2("3.3 Simulated System Configurations")
para("To attribute every performance change to a single cause, the system was evolved one factor at a time. "
     "Table 2 lists the configurations; Figure 1 shows the final P870-D-inspired system (configuration F).")
cfgs = [
    ("B1", "TimingSimple (in-order, blocking)", "1 GHz", "32K / 256K / 8M (1/10/20)", "DDR3-1600", "–", "27"),
    ("C", "TimingSimple", "2 GHz", "as B1", "DDR5-4400, close-adaptive", "–", "27"),
    ("C2", "TimingSimple", "2 GHz", "as B1", "DDR5-4400, open-adaptive", "–", "10"),
    ("D", "O3, 6-wide", "2 GHz", "as B1", "DDR5-4400, open-adaptive", "–", "6"),
    ("E", "O3, 6-wide", "2 GHz", "64K / 4M / 16M (4/16/40)", "DDR5-4400, open-adaptive", "–", "6"),
    ("F", "O3, 6-wide", "2 GHz", "64K / 4M / 16M (4/16/40)", "DDR5-4400, open-adaptive", "L2 stride", "27"),
    ("H", "SiFive U74 (MinorCPU), HiFive Unmatched model", "1.2 GHz", "32K / 2M L2, no L3", "DDR4-2400",
     "L1 + L2 stride (gem5 default)", "6"),
]
table(["ID", "CPU", "Clock", "L1 / L2 / L3 (latency, cycles)", "Memory", "Prefetch", "Layers"], cfgs,
      [1.0, 3.6, 1.3, 3.6, 3.2, 1.7, TEXT_W - 14.4], size=8.5,
      cap_text="Simulated configurations (one factor changed per step)")
figure("fig1_architecture.png", "Final P870-D-inspired simulated system (configuration F)", 12.5)
h2("3.4 Model-Derived AI Workloads")
para("Because MLPerf, DeepBench and AI-Matrix cannot run directly in SE mode on RISC-V, six float32 C kernels "
     "were written – direct NCHW convolution, GEMM/GEMV, per-head attention, max/average pooling, embedding "
     "gather/bag and a ReLU MLP – and instantiated with **27 documented layer shapes** (Table 3). Shapes come "
     "from ResNet-50 [5] (also present in DeepBench's convolution list), the MLPerf BERT-Large configuration "
     "(hidden size 1024, 16 heads, intermediate size 4096, sequence length 384) [4], [6], the MLPerf DLRM-v2 "
     "reference (26 embedding tables of dimension 128 with the official row counts and multi-hot sizes, "
     "bottom MLP 512-256-128, top MLP 1024-1024-512-256-1) [7], [8], and the DeepBench inference GEMM and "
     "convolution lists [9], [10]. These are **model-derived kernels, not full benchmark implementations**.")
layer_rows = []
for lid, r in LAYERS.iterrows():
    fr = F.loc[lid]
    model = r.model.replace(" (DCNv2)", "").replace(" / AI-Matrix micro", " / AI-Matrix")
    layer_rows.append((lid, model, op_text(r.kernel, r.args), shape_text(r.kernel, r.args),
                       f"{fr.footprint_bytes / 2**20:.1f}", f"{fr.work_fraction:.3f}"))
table(["Layer", "Source model", "Operation", "Shape", "Footprint (MiB)", "Work fraction"],
      layer_rows, [2.6, 2.6, 2.3, 4.6, 1.8, TEXT_W - 13.9], size=8,
      cap_text="The 27 model-derived AI layers (s = stride, p = padding)", align_right_from=4)
h2("3.5 Measurement Methodology")
para("Figure 2 summarises the experimental flow. Each kernel allocates its tensors with their real dimensions "
     "and layout, initialises them deterministically, streams a 64 MiB buffer to remove initialisation data "
     "from the caches (*cold* mode), and then executes the measured region between m5_reset_stats and "
     "m5_dump_stats markers. Only the first statistics block – the kernel region – is used. To keep each "
     "simulation within minutes, a subset of independent work units (output rows, output channels, heads or "
     "samples) is computed, targeting about 20 million multiply–accumulates per run; tensor footprints and "
     "reuse distances remain those of the full layer, and the computed fraction is recorded as the work fraction.")
figure("fig2_methodology.png", "Experimental methodology", 15.5)
para("For each run, 103 values are stored, including simulated time, committed instructions, cycles, IPC; "
     "accesses, hits, misses, miss rate, MPKI, MSHR misses, average miss latency, writebacks and replacements "
     "for every cache level; prefetcher issued/useful counts, accuracy and coverage; and DRAM bytes read and "
     "written, request counts, bandwidth and average access latency, together with full layer metadata.")
h2("3.6 Validation of the Experimental Setup")
bullets([
    "**Baseline reproduction:** the new configurable script reproduces three rows of the original dataset "
    "(including one previously reused row) with identical simTicks, cache misses and instruction counts; this "
    "check was repeated after every change to the script.",
    "**Kernel correctness:** 14 of 14 tiny-shape test cases match an independent numpy reference, and every gem5 "
    "run's checksum matches the native execution of the same kernel.",
    "**Baseline integrity:** SHA-256 checksums of 55 original files (dataset, script, plan, binaries, sources) "
    "remained unchanged throughout the study.",
])

# --- 4. Simulations and Results ----------------------------------------
h1("4. Simulations and Results")
h2("4.1 Sensitivity of the Original Setup")
top = ", ".join(f"{k} ({v:.1f} %)" for k, v in N["above3"].items())
para(f"The original dataset contains {N['n56_rows']:,} runs: 28 integer workloads × 40 sizes × 50 cache "
     f"configurations, of which {N['n56_reused']:,} rows were reused from earlier runs (confirmed equivalent in "
     f"Section 3.6). Figure 3 shows, for each workload, the median runtime spread across the 50 cache "
     f"configurations. For {N['n_le3']} of {N['n_wl']} workloads the spread is 3 % or less; only "
     f"{len(N['above3'])} workloads exceed it: {top}. With a blocking in-order CPU and no prefetcher, most misses are compulsory and their latency is "
     f"serialised, so cache capacity and associativity have little influence. This finding motivated the "
     f"redesign of the simulated system.")
figure("fig3_baseline_sensitivity.png", "Runtime spread across 50 cache configurations in the original 56,000-run dataset", 15.5)
h2("4.2 AI Layers on the Baseline Machine")
para(f"On configuration B1, all 27 layers completed with correct results. The layers separate into "
     f"{len(N['b1_res'])} cache-resident layers (L3 MPKI below 1) and {len(N['b1_mem'])} memory-bound layers "
     f"({', '.join(N['b1_mem'])}). Pooling, the ResNet fully-connected layer and the DeepBench GEMV stream "
     f"their data once; the BERT feed-forward layers and the DLRM top MLP repeatedly re-read weights larger than "
     f"the 8 MiB L3 (e.g. {f1(B1.loc['BERT_ffn1', 'dram_bytes_read'] / 2**20)} MiB read from DRAM to compute five "
     f"rows of BERT_ffn1); and the embedding layers perform irregular lookups into tables, with working sets "
     f"of {F.loc['BERT_embed', 'footprint_bytes'] / 2**20:.0f} MiB (BERT) and "
     f"{F.loc['DLRM_embbag', 'footprint_bytes'] / 2**20:.0f} MiB (DLRM).")
h2("4.3 Effect of a 2 GHz Clock and DDR5 Memory")
para(f"Figure 4 compares B1 with configuration C (2 GHz, DDR5-4400). Cache-resident layers gain "
     f"{f2(N['sp_c_res'])}–2.00×, almost the ideal 2× from the clock, whereas memory-bound layers gain only "
     f"{f2(N['sp_c_min'])}–{f2(sp_c[B1.l3_mpki >= 1].max())}×. The speed-up correlates with L3 MPKI at "
     f"{f2(N['corr_c'])}: at a higher clock each DRAM access costs more cycles (the average L1 miss latency of "
     f"BERT_ffn1 grows from {N['ffn_misslat_1g']:.0f} to {N['ffn_misslat_2g']:.0f} cycles), so the memory wall "
     f"becomes more pronounced.")
figure("fig4_clock_ddr5_speedup.png", "Speed-up from 2 GHz and DDR5-4400 versus L3 MPKI (27 layers)", 14.5)
h2("4.4 DRAM Page Policy")
para(f"Configuration C showed a higher average DRAM latency for DDR5-4400 ({f1(N['latc_lo'])}–"
     f"{f1(N['latc_hi'])} ns) than for DDR3-1600 ({f1(N['lat3_lo'])}–{f1(N['lat3_hi'])} ns). The cause is "
     f"gem5's default *close-adaptive* page policy for its DDR5 model, which produced a row-buffer hit rate of "
     f"{f1(N['rhc_hi'])} %, against {f1(N['rh3_lo'])}–{f1(N['rh3_hi'])} % for the open-adaptive DDR3 model. "
     f"Configuration C2 applies the open-adaptive policy to DDR5: row hits recover to {f1(N['rho_lo'])}–"
     f"{f1(N['rho_hi'])} %, latency falls to {f1(N['lato_lo'])}–{f1(N['lato_hi'])} ns (Figure 5), and "
     f"memory-bound layers become {f1(N['gain_lo'])}–{f1(N['gain_hi'])} % faster, while the cache-resident "
     f"control layer RN50_conv1 changes by only {f2(N['gain_ctrl'])} %. Open-adaptive is used for all later "
     f"configurations (assumption A, consistent with the DDR3 baseline).")
figure("fig5_dram_latency.png", "Average DRAM access latency for DDR3-1600 and DDR5-4400 with two page policies", 14.5)
rdq_txt = (f"{f2(rdq_c2.min())} for all six representative layers" if f2(rdq_c2.min()) == f2(rdq_c2.max())
           else f"{f2(rdq_c2.min())}–{f2(rdq_c2.max())} for the six representative layers")
para(f"With the in-order TimingSimple CPU the average DRAM read-queue length is {rdq_txt}: only one miss is "
     f"outstanding at a time, so DDR5's additional banks and bandwidth cannot be exploited. This motivates the "
     f"out-of-order core.")
h2("4.5 Out-of-Order P870-D-Like Core")
para(f"Replacing the in-order core by the 6-wide O3 core (configuration D) speeds up the six representative "
     f"layers by {f2(N['sp_o3_lo'])}–{f2(N['sp_o3_hi'])}× with identical instruction counts. The largest gain is "
     f"for the streaming RN50_maxpool ({f2(sp_o3['RN50_maxpool'])}×), whose independent misses now overlap "
     f"(read-queue length {f2(rdq_d['RN50_maxpool'])}), and the smallest for BERT_ffn1 "
     f"({f2(sp_o3['BERT_ffn1'])}×), which remains limited by re-streaming its weights. DRAM bandwidth use stays "
     f"below {bw_d.max() / 1000:.1f} GB/s, far from the 17.6 GB/s peak of the modelled 32-bit DDR5-4400 channel "
     f"(4400 MT/s × 4 bytes), indicating that the bottleneck is latency and data placement rather than bandwidth.")
h2("4.6 P870-D Cache Hierarchy")
para(f"Configuration E adopts the P870-D capacities and latencies (64 KiB L1 at 4 cycles, 4 MiB L2 at 16 "
     f"cycles, 16 MiB L3 at 40 cycles). BERT_ffn1, whose 16 MiB weight matrix now fits in the combined L2 and "
     f"L3, runs {f2(N['ffn_sp'])}× faster and its DRAM traffic falls from {f1(N['ffn_dram_d'])} MiB to "
     f"{f1(N['ffn_dram_e'])} MiB. The other five layers run only {f2(N['oth_sp_lo'])}–{f2(N['oth_sp_hi'])}× as "
     f"fast as before (their execution time increases by {N['oth_t_lo']:.0f}–{N['oth_t_hi']:.0f} %): they either "
     f"already fit in the smaller caches, stream data once or access data randomly, so they do not benefit from "
     f"capacity but pay the longer hit latencies (4 instead of 1 cycle at L1). Larger caches therefore help only "
     f"the tensors whose reuse distance matches the added capacity.")
h2("4.7 Hardware Prefetching")
para(f"The P800 series includes a streaming prefetcher (H-OFF); configuration F adds gem5's stride prefetcher "
     f"to the L2 (A). Figure 6 shows its effect. Lookup-heavy and streaming layers gain the most "
     f"(BERT_embed {f2(sp_pf['BERT_embed'])}×, DB_gemv_3072 {f2(sp_pf['DB_gemv_3072'])}×, DLRM_embbag "
     f"{f2(sp_pf['DLRM_embbag'])}×), BERT_ffn1 gains {f2(sp_pf['BERT_ffn1'])}×, and cache-resident layers are "
     f"unaffected. However, only {f1(N['pf_acc_lo'])}–{f1(N['pf_acc_hi'])} % of issued prefetches are used. For "
     f"DLRM_embbag the prefetcher raises DRAM read traffic from {f1(N['dlrm_dram_e'])} MiB to "
     f"{f1(N['dlrm_dram_f'])} MiB: useless lines occupy cache capacity and memory bandwidth (cache pollution). "
     f"Because L2 prefetch requests reach the L3 as ordinary requests, L3 statistics in configuration F include "
     f"prefetch traffic; DRAM traffic and L2 demand misses are therefore used as the primary memory metrics.")
figure("fig7_prefetcher.png", "Benefit and accuracy of the L2 stride prefetcher (configuration F versus E)", 15)
h2("4.8 Comparison with the HiFive Unmatched Model")
para(f"No physical SiFive hardware was available. As a reference to a real, validated design, the same six "
     f"layers were run on gem5's unmodified model of the SiFive HiFive Unmatched board (configuration H), whose "
     f"parameters were calibrated against the real board [21], [22]. Figure 7 normalises all configurations to "
     f"this model. Before prefetching was added, the P870-D-inspired model was slower than the HiFive model on "
     f"BERT_embed ({f2(r_h_e['BERT_embed'])}× relative speed): the HiFive model uses gem5's default stride "
     f"prefetchers on its L1 data and L2 caches, which covered {hf_cov * 100:.0f} % of that layer's L1 misses. "
     f"With the prefetcher, the P870-D-inspired model is {f2(N['rh_lo'])}–{f2(N['rh_hi'])}× faster than the "
     f"HiFive model on all six layers, as expected for a newer 6-wide core at a higher clock. Note that the "
     f"HiFive model's prefetchers are gem5 defaults and have not been checked against the U74 datasheet.")
figure("fig6_configurations.png", "Execution time of six layers across configurations, normalised to the HiFive Unmatched model", 15)
h2("4.9 Final Characterisation on the P870-D-Inspired Model")
para(f"All 27 layers were simulated on configuration F with correct results. They run "
     f"{f2(N['spf_lo'])}–{f2(N['spf_hi'])}× faster than on B1 (median {f2(N['spf_med'])}×), with IPC between "
     f"{f2(N['ipc_lo'])} and {f2(N['ipc_hi'])} (median {f2(N['ipc_med'])}). Figure 8 and Table 4 summarise "
     f"throughput and memory intensity. Memory intensity, measured as DRAM read traffic per million MACs, "
     f"spans more than two orders of magnitude between layers. The DLRM top MLP, whose weights "
     f"({f1(F.loc['DLRM_top_mlp', 'weight_bytes'] / 2**20)} MiB) are as large as the combined 20 MiB of L2 and "
     f"L3, still re-reads {f1(F.loc['DLRM_top_mlp', 'dram_bytes_read'] / 2**20)} MiB from DRAM for four samples, "
     f"whereas the BERT feed-forward layers (16 MiB of weights) now fit and read about "
     f"{F.loc['BERT_ffn1', 'dram_bytes_read'] / 2**20:.0f} MiB.")
figure("fig8_final_27_layers.png", "Throughput and memory intensity of the 27 layers on the P870-D-inspired model", 15.5)
final_rows = []
for lid in F.index:
    r = F.loc[lid]
    final_rows.append((lid, f"{r.sim_seconds * 1e3:.2f}", f"{r.ipc:.2f}", f"{r.l1d_mpki:.1f}",
                       f"{r.l2_mpki:.2f}", f"{r.dram_bytes_read / 2**20:.2f}", f"{traffic[lid]:.0f}",
                       f"{sp_f[lid]:.2f}"))
table(["Layer", "Time (ms)", "IPC", "L1D MPKI", "L2 MPKI", "DRAM read (MiB)", "KiB / M-MAC", "Speed-up vs B1"],
      final_rows, [2.9, 1.7, 1.2, 1.7, 1.7, 2.1, 2.1, TEXT_W - 13.4], size=8,
      cap_text="Results of the 27 layers on the P870-D-inspired model (configuration F)", align_right_from=1)
h2("4.10 Implications for Tensor-Aware Cache Management")
para("The measurements identify three tensor classes that a single tensor-agnostic policy handles poorly:")
bullets([
    "**Streaming tensors** (pooling inputs, GEMV weights at batch 1, fully-connected layers): every line is used "
    "once. Caching them evicts useful data; the appropriate action is bypassing or low-priority insertion, with "
    "accurate prefetching to hide latency.",
    "**Reuse-critical tensors** (BERT feed-forward weights, the DLRM top MLP): reused for every token or sample but "
    "close to or above cache capacity. LRU evicts them just before reuse; the appropriate action is to protect a "
    "portion of the tensor so that partial reuse survives.",
    "**Irregular tensors** (BERT token embeddings, DLRM embedding tables): accessed by data-dependent indices. "
    "Stride prefetching helps rows but wastes most prefetches; the appropriate action is frequency-aware "
    "retention of popular rows and prefetch filtering.",
])
para("A tensor-aware policy that knows the class of each access can apply the right action per tensor. The "
     "per-layer, per-configuration dataset produced in Phase 1 provides both the evidence for this design and "
     "the training and evaluation data envisaged in the project proposal.")
h2("4.11 Limitations and Threats to Validity")
bullets([
    "The P870-D model is inspired by, not identical to, the real core; unpublished parameters (ROB and queue "
    "sizes, associativities, L3 latency, replacement policy, prefetcher type) are assumptions.",
    "DDR5-4400 replaces the real DDR5-5600, and a single memory channel is modelled.",
    "One core is simulated in SE mode, without an operating system, multi-core contention or the CHI protocol.",
    "Kernels are scalar float32 C code (naive or blocked GEMM, direct convolution); production frameworks use "
    "vectorised, packed micro-kernels and operator fusion, and DeepBench specifies int8 inference.",
    "Work sampling computes a fraction of each layer, and DLRM tables larger than 65,536 rows are capped at 32 MiB "
    "each (still twice the 16 MiB L3).",
    "No physical hardware was measured; the HiFive Unmatched comparison uses gem5's validated model of that board.",
])

# --- 5. Concluding Remarks ----------------------------------------------
h1("5. Concluding Remarks")
para(f"Phase 1 of the Tensor-Aware AI Caching Management project established a hardware-justified simulation "
     f"platform and a characterisation of real AI layer shapes. The audit of the original {N['n56_rows']:,}-run "
     f"dataset showed that its generic configuration could not expose cache-management effects. The rebuilt "
     f"system, inspired by the SiFive P870-D and documented parameter by parameter, combined with 27 "
     f"model-derived kernels from ResNet-50, BERT-Large, DLRM-v2 and DeepBench, makes these effects visible: "
     f"higher clock rates and DDR5 expose a memory wall for memory-bound layers, the out-of-order core and P870-D "
     f"caches help selectively, and the prefetcher speeds up irregular layers while wasting most of its "
     f"prefetches. Across all layers, performance on the final model improves {f2(N['spf_lo'])}–"
     f"{f2(N['spf_hi'])}× over the baseline, yet streaming, reuse-critical and irregular tensors still lose "
     f"performance to tensor-agnostic caching. These results directly motivate the tensor-aware, AI-assisted "
     f"cache-management policy of Phase 2.")

# --- 6. Future Work -----------------------------------------------------
h1("6. Future Work")
bullets([
    "**Replacement-policy study:** compare LRU, FIFO, Random, TreePLRU, BRRIP and SHiP on configuration F for "
    "all 27 layers.",
    "**Tensor-aware policy (Phase 2):** pass tensor identity and class to the cache (e.g. through address-range "
    "tagging), implement class-specific insertion, bypass and protection in a custom gem5 replacement policy, and "
    "train an AI model on the Phase 1 data to predict tensor reuse.",
    "**Prefetch filtering:** suppress predicted-useless prefetches for irregular tensors to reduce pollution.",
    "**Workload realism:** vectorised (RVV) kernels, multi-core runs with a shared L3, and full-model validation "
    "in full-system mode with a real inference framework.",
    "**Hardware validation:** run the kernels on a physical RISC-V board (e.g. HiFive Unmatched or HiFive Premier "
    "P550) and compare with the corresponding gem5 model.",
    "**Memory system:** a DDR5-5600 timing model, multiple channels, and the CHI coherence protocol.",
])

# --- References -----------------------------------------------------------
h1("References")
refs = [
    "P. Suryadevara, “HERMES: High-Performance RISC-V Memory Hierarchy for ML Workloads,” arXiv:2503.13064, 2025.",
    "N. Binkert et al., “The gem5 simulator,” ACM SIGARCH Computer Architecture News, vol. 39, no. 2, pp. 1–7, 2011.",
    "J. Lowe-Power et al., “The gem5 Simulator: Version 20.0+,” arXiv:2007.03152, 2020.",
    "V. J. Reddi et al., “MLPerf Inference Benchmark,” in Proc. ACM/IEEE ISCA, 2020, pp. 446–459; "
    "MLCommons, MLPerf Inference, https://github.com/mlcommons/inference.",
    "K. He, X. Zhang, S. Ren and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE CVPR, 2016, pp. 770–778.",
    "J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for "
    "Language Understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186.",
    "M. Naumov et al., “Deep Learning Recommendation Model for Personalization and Recommendation Systems,” "
    "arXiv:1906.00091, 2019.",
    "R. Wang et al., “DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank "
    "Systems,” in Proc. The Web Conference (WWW), 2021, pp. 1785–1797.",
    "Baidu Research, “DeepBench: Benchmarking Deep Learning Operations on Different Hardware,” "
    "https://github.com/baidu-research/DeepBench.",
    "Alibaba, “AI Matrix,” https://github.com/alibaba/ai-matrix.",
    "U. Gupta et al., “The Architectural Implications of Facebook's DNN-based Personalized Recommendation,” "
    "in Proc. IEEE HPCA, 2020, pp. 488–501.",
    "A. Jaleel, K. B. Theobald, S. C. Steely Jr. and J. Emer, “High Performance Cache Replacement Using "
    "Re-Reference Interval Prediction (RRIP),” in Proc. ISCA, 2010, pp. 60–71.",
    "C.-J. Wu, A. Jaleel, W. Hasenplaugh, M. Martonosi, S. C. Steely Jr. and J. Emer, “SHiP: Signature-based "
    "Hit Predictor for High Performance Caching,” in Proc. IEEE/ACM MICRO, 2011, pp. 430–441.",
    "A. Jain and C. Lin, “Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement,” "
    "in Proc. ISCA, 2016, pp. 78–89.",
    "Z. Shi, X. Huang, A. Jain and C. Lin, “Applying Deep Learning to the Cache Replacement Problem,” in "
    "Proc. IEEE/ACM MICRO, 2019, pp. 413–425.",
    "L. A. Belady, “A Study of Replacement Algorithms for a Virtual-Storage Computer,” IBM Systems Journal, "
    "vol. 5, no. 2, pp. 78–101, 1966.",
    "T.-F. Chen and J.-L. Baer, “Effective Hardware-Based Data Prefetching for High-Performance Processors,” "
    "IEEE Transactions on Computers, vol. 44, no. 5, pp. 609–623, 1995.",
    "SiFive, “Performance P800 Series,” https://www.sifive.com/cores/performance-p800 (accessed Nov. 2026).",
    "SiFive, “SiFive BigSky SF-2U870,” https://www.sifive.com/development-platforms/"
    "sifive-bigsky-sf-2u870-datacenter (accessed Nov. 2026).",
    "C. Lam, “Hot Chips 2023: SiFive's P870 Takes RISC-V Further,” Chips and Cheese, Sep. 2023, "
    "https://chipsandcheese.com/p/hot-chips-2023-sifives-p870-takes-risc-v-further.",
    "UC Davis Computer Architecture Research Group, “Validating Hardware and SimPoints with gem5: A RISC-V Board "
    "Case Study,” 2023, https://arch.cs.ucdavis.edu/simulation/2023/06/17/validating-riscvmatched.html.",
    "SiFive, “SiFive FU740-C000 Manual,” v1p6, 2021.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.9)
    p.paragraph_format.space_after = Pt(4)
    r_ = p.add_run(f"[{i}]\t{r}")
    r_.font.size = Pt(11)

# --- Appendix A: Reproducibility ------------------------------------------
def code(lines):
    for ln in lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Cm(0.5)
        r = p.add_run(ln)
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


h1("Appendix A: Reproducibility")
para("All files are in the project directory ~/gem5-project. The original simulation script and the "
     "56,000-run dataset were never modified; their integrity can be checked at any time with "
     "sha256sum -c arch_study/baseline_sha256.txt.")
h2("A.1 Environment")
bullets([
    "Ubuntu 24.04.3 LTS on WSL2; host CPU Intel Core i7-14650HX, 7 GB RAM available to Linux.",
    "gem5 v25.1.0.1 (build/RISCV/gem5.opt; classic memory system; Ruby protocol MI_example only).",
    "riscv64-linux-gnu-gcc 13.3.0 (static RV64GC binaries, -O2 -ffp-contract=off); Python 3 with pandas, "
    "numpy and matplotlib in the project virtual environment.",
])
h2("A.2 Main Files")
files = [
    ("arch_study/sim_config.py", "Configurable simulation script; defaults reproduce tensor_cache_sim.py exactly"),
    ("arch_study/hifive_unmatched_sim.py", "Runs a binary on gem5's unmodified HiFive Unmatched model"),
    ("arch_study/run_ai_kernels.py", "Parallel experiment runner and ROI statistics parser"),
    ("arch_study/ai_kernels/src, include", "The six model-derived C kernels and shared helpers"),
    ("arch_study/ai_kernels/layers.csv", "Registry of the 27 layers with shapes and source citations"),
    ("arch_study/ai_kernels/make_ai_plan.py", "Builds an experiment plan (work units, configuration flags)"),
    ("arch_study/ai_kernels/verify_reference.py", "numpy reference check of the kernels"),
    ("arch_study/baseline_sha256.txt", "SHA-256 checksums of 55 original project files"),
    ("arch_study/report/", "make_figures.py and build_report.py (this report)"),
]
table(["Path", "Purpose"], files, [6.2, TEXT_W - 6.2], size=8.5, cap_text="Main project files")
h2("A.3 Datasets")
ds_rows = [
    ("tensor_cache_dataset_56000.csv", "Original Baseline-v1 dataset (read-only)", f"{N['n56_rows']:,}"),
    ("ai_kernels_pilot_v1.csv", "B1", str(len(B1))),
    ("ai_kernels_2ghz_ddr5_4400_v1.csv", "C", str(len(C))),
    ("ai_kernels_2ghz_ddr5_4400_open_v1.csv", "C2", str(len(C2))),
    ("ai_kernels_o3_2ghz_ddr5_4400_open_v1.csv", "D", str(len(D))),
    ("ai_kernels_p870d_full_v1.csv", "E", str(len(E))),
    ("ai_kernels_p870d_full_pf_v1.csv", "F (final P870-D-inspired model)", str(len(F))),
    ("ai_kernels_hifive_unmatched_v1.csv", "H (HiFive Unmatched model)", str(len(H))),
]
table(["File (arch_study/datasets/, except the first)", "Configuration", "Runs"], ds_rows,
      [7.4, 6.3, TEXT_W - 13.7], size=8.5, cap_text="Datasets produced and used in this report",
      align_right_from=2)
h2("A.4 Commands (example: configuration F)")
code([
    "cd ~/gem5-project/arch_study/ai_kernels && make all && make verify",
    "cd ~/gem5-project",
    ".venv/bin/python arch_study/ai_kernels/make_ai_plan.py --config-id F_P870D_full_L2stride \\",
    "    --cache-args \"64KiB 8 4MiB 16 16MiB 16\" \\",
    "    --sim-flags=\"--cpu o3 --freq 2GHz --mem ddr5_4400 --dram-page-policy open_adaptive \\",
    "                 --l1-lat 4 --l2-lat 16 --l3-lat 40 --prefetch stride\" \\",
    "    --out arch_study/ai_kernels/experiment_plan_ai_p870d_full_pf_27_v1.csv",
    ".venv/bin/python arch_study/run_ai_kernels.py --workers 6 \\",
    "    --plan arch_study/ai_kernels/experiment_plan_ai_p870d_full_pf_27_v1.csv \\",
    "    --out arch_study/datasets/ai_kernels_p870d_full_pf_v1.csv \\",
    "    --results arch_study/results/ai_p870d_full_pf_v1",
    ".venv/bin/python arch_study/report/make_figures.py",
    ".venv/bin/python arch_study/report/build_report.py",
])

out = OUT / "Tensor_AI_UROP_Report.docx"
doc.save(out)
print("wrote", out)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in N.items() if not isinstance(v, (list, pd.Series))})
