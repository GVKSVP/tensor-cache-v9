"""
COPY of sim_config_flush.py (Step 6): adds packing / coordination switches; sim_config_flush.py is unchanged.

sim_config.py - configurable copy of tensor_cache_sim.py (Baseline-v1).

The original tensor_cache_sim.py is NOT modified. With no optional flags,
this script builds exactly the same system as Baseline-v1.

Usage (legacy, identical to tensor_cache_sim.py):
    gem5.opt -d OUT sim_config.py <workload> <args...>
        [l1_size l1_assoc l2_size l2_assoc l3_size l3_assoc]

Usage (generic binary, for new workloads):
    gem5.opt -d OUT sim_config.py --cmd "<binary> <args...>"
        [l1_size l1_assoc l2_size l2_assoc l3_size l3_assoc]

Optional flags (defaults = Baseline-v1):
    --cmd "<binary> <args...>"   run any static RISC-V binary
    --mem-size <size>            memory range (default 512MiB)
    --freq <freq>                CPU/cache clock (default 1GHz)
    --mem <type>                 ddr3_1600 (default) | ddr5_4400 | ddr5_6400
                                 (gem5 built-in DRAMInterface classes, unmodified)
    --dram-page-policy <policy>  open | open_adaptive | close | close_adaptive
                                 (default: keep the DRAM model's own policy)
    --cpu <type>                 timing (default, RiscvTimingSimpleCPU) |
                                 o3 (RiscvO3CPU, P870-D-inspired, see below)
    --l1-lat / --l2-lat / --l3-lat <cycles>
                                 tag, data and response latency of each level
                                 (defaults 1 / 10 / 20, as in Baseline-v1)
    --prefetch <type>            none (default) | stride
                                 stride = gem5 StridePrefetcher (default params)
                                 on the L2. [H-OFF] the P800 series has a
                                 "Streaming Prefetcher"; [A] its type and
                                 placement are not published.

P870-D-inspired O3 settings (--cpu o3):
    [H-OFF] 6-wide fetch/decode/rename/dispatch/issue/writeback/commit
            (SiFive P800 page: "6-wide, out-of-order core")
    [H-3P]  TAGE conditional predictor, 64-entry return address stack,
            32-entry iTLB, 64-entry DTLB (Hot Chips 2023 via Chips and Cheese)
    [A]     TAGE table geometry, ROB, LQ/SQ, IQ and register files:
            gem5 defaults (not published by SiFive)
"""

import os
import shlex
import sys

import m5
from m5.objects import *


# ============================================================
# Optional flags (removed from argv before legacy parsing)
# ============================================================

OPTIONS = {
    "--cmd": None,
    "--mem-size": "512MiB",
    "--freq": "1GHz",
    "--mem": "ddr3_1600",
    "--dram-page-policy": None,
    "--cpu": "timing",
    "--l1-lat": "1",
    "--l2-lat": "10",
    "--l3-lat": "20",
    "--prefetch": "none",
    "--repl": "lru",
    "--trace": "none",
    "--tensor-hints": None,
    "--tensor-dump": None,
    "--tensor-slice": "0",
    "--tensor-slice-l2": "0",
    "--tensor-l2-tree": "main",
    "--tensor-slice-ws": "0",       # Step 4: working-set slice alpha (L3; 0 = off)
    "--tensor-dirty-aware": "0",    # Step 4: clean-first victim tie-break (1 = on)
    "--tensor-duel-skip-first": "0",  # Step 4: dueling ignores first-touch misses (1 = on)
    "--tensor-duel-lru-default": "0",  # Step 4: LRU until tree proves better (1 = on)
    "--tensor-l2-pf-fix": "0",  # Step 5 fix: L2 sees its prefetch fills as prefetches (1 = on)
    "--tensor-l2-duel-all": "0",  # Step 5b: L2 duel counts every re-fetch (demand + prefetch) (1 = on)
    "--tensor-l2-duel-reads": "0",  # Step 5 fix: L2 duel counts demand read misses only (1 = on)
    "--tensor-l2-no-duel": "0",  # Step 5 diag: L2 tree without set dueling (1 = on)
    "--tensor-l2-slice-ws": "0",   # Step 5: working-set slice alpha at L2 (0 = off)
    "--tensor-l2-tree-off": "0",   # Step 5: L2 does not use the learned tree (1 = on)
    "--tensor-l2-slice-contig": "0",  # Step 5: contiguous L2 slice (1 = on)
    "--tensor-l2-contig-min": "0",  # Step 5: contiguous slice only if kept fraction >= this
    "--tensor-l3-pack": "0",     # Step 6: packing plan at L3 (1 = on)
    "--tensor-l2-pack": "0",     # Step 6: packing plan at L2 (1 = on)
    "--tensor-coord": "0",       # Step 6: L3 coordinates with the L2 slice (1 = on)
    "--l2-pf-on-access": "0",    # Step 7 diag: L2 prefetcher sees every L2 access, hits too (1 = on)
    "--tensor-l2-keep-pf": "0",  # Step 5: L2 keeps prefetch fills (1 = on)
    "--tensor-l2-safe": "0",  # Step 5: L2 tree gets skip-first + LRU-default + demand-only duel (1 = on)
}

PREFETCHERS = ("none", "stride")

CPU_TYPES = ("timing", "o3")

PAGE_POLICIES = ("open", "open_adaptive", "close", "close_adaptive")

# gem5 built-in DRAM models (src/mem/DRAMInterface.py), used unmodified.
DRAM_MODELS = {
    "ddr3_1600": DDR3_1600_8x8,
    "ddr5_4400": DDR5_4400_4x8,
    "ddr5_6400": DDR5_6400_4x8,
}

argv = []
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg in OPTIONS:
        if i + 1 >= len(sys.argv):
            print(f"ERROR: {arg} requires a value")
            sys.exit(1)
        OPTIONS[arg] = sys.argv[i + 1]
        i += 2
    elif arg.startswith("--"):
        print(f"ERROR: unknown option {arg}")
        sys.exit(1)
    else:
        argv.append(arg)
        i += 1

MEM_SIZE = OPTIONS["--mem-size"]
FREQ = OPTIONS["--freq"]
MEM_TYPE = OPTIONS["--mem"]
if MEM_TYPE not in DRAM_MODELS:
    print(f"ERROR: --mem must be one of {', '.join(DRAM_MODELS)}")
    sys.exit(1)
PAGE_POLICY = OPTIONS["--dram-page-policy"]
if PAGE_POLICY is not None and PAGE_POLICY not in PAGE_POLICIES:
    print(f"ERROR: --dram-page-policy must be one of {', '.join(PAGE_POLICIES)}")
    sys.exit(1)
L1_LAT = int(OPTIONS["--l1-lat"])
L2_LAT = int(OPTIONS["--l2-lat"])
L3_LAT = int(OPTIONS["--l3-lat"])
PREFETCH = OPTIONS["--prefetch"]
if PREFETCH not in PREFETCHERS:
    print(f"ERROR: --prefetch must be one of {', '.join(PREFETCHERS)}")
    sys.exit(1)
CPU_TYPE = OPTIONS["--cpu"]
if CPU_TYPE not in CPU_TYPES:
    print(f"ERROR: --cpu must be one of {', '.join(CPU_TYPES)}")
    sys.exit(1)

# gem5 built-in replacement policies (ReplacementPolicies.py), used unmodified.
# Applied to L2 and L3 only; L1-I and L1-D always keep gem5's default LRU.
# With "lru" nothing is set, so the configuration equals earlier runs exactly.
REPL_POLICIES = ("lru", "fifo", "random", "treeplru", "srrip", "brrip",
                 "drrip", "ship", "shippc", "tensor", "tensor_l3")
# "tensor" = our learned tensor-aware policy (TensorAwareRP, added to gem5 on
# branch tensor-aware-rp); --tensor-hints gives its per-run hint table.
# "tensor_l3" = the same policy on L3 only (L2 keeps LRU).
REPL = OPTIONS["--repl"]
if REPL not in REPL_POLICIES:
    print(f"ERROR: --repl must be one of {', '.join(REPL_POLICIES)}")
    sys.exit(1)


def make_repl(name, assoc, size=None, level=None):
    if name in ("tensor", "tensor_l3"):
        from m5.util.convert import toMemorySize
        sets = toMemorySize(size) // (64 * assoc)
        dump = OPTIONS["--tensor-dump"]       # diagnostics: <prefix>_l<level>.csv
        return TensorAwareRP(level=level, num_sets=sets,
                             hint_file=OPTIONS["--tensor-hints"] or "",
                             dump_file=f"{dump}_l{level}.csv" if dump else "",
                             dump_max=3000000 if dump else 0,
                             slice_beta=float(OPTIONS["--tensor-slice"] if level == 3
                                              else OPTIONS["--tensor-slice-l2"]),
                             slice_size_ratio=(level == 2),
                             l2_tree_alt=(OPTIONS["--tensor-l2-tree"] == "demand"),
                             slice_ws_alpha=float(OPTIONS["--tensor-slice-ws"] if level == 3
                                                  else OPTIONS["--tensor-l2-slice-ws"]),
                             tree_off=(level == 2 and OPTIONS["--tensor-l2-tree-off"] == "1"),
                             slice_contig=(level == 2 and OPTIONS["--tensor-l2-slice-contig"] == "1"),
                             slice_contig_min=float(OPTIONS["--tensor-l2-contig-min"]),
                             slice_pack=(OPTIONS["--tensor-l3-pack"] == "1" if level == 3
                                         else OPTIONS["--tensor-l2-pack"] == "1"),
                             coord_l2_bytes=(toMemorySize(L2_SIZE)
                                             if level == 3 and OPTIONS["--tensor-coord"] == "1"
                                             else 0),
                             coord_l2_alpha=float(OPTIONS["--tensor-l2-slice-ws"]),
                             coord_l2_contig_min=float(OPTIONS["--tensor-l2-contig-min"]),
                             coord_l2_pack=(level == 3 and OPTIONS["--tensor-l2-pack"] == "1"),
                             dirty_aware=OPTIONS["--tensor-dirty-aware"] == "1",
                             duel_skip_first=(OPTIONS["--tensor-duel-skip-first"] == "1"
                                              if level == 3 else OPTIONS["--tensor-l2-safe"] == "1"),
                             duel_lru_default=(OPTIONS["--tensor-duel-lru-default"] == "1"
                                               if level == 3 else OPTIONS["--tensor-l2-safe"] == "1"),
                             duel_demand_only=(level == 2 and OPTIONS["--tensor-l2-safe"] == "1"
                                               and OPTIONS["--tensor-l2-duel-all"] != "1"),
                             keep_prefetch=(level == 2 and OPTIONS["--tensor-l2-keep-pf"] == "1"),
                             pf_by_requestor=(level == 2 and OPTIONS["--tensor-l2-pf-fix"] == "1"),
                             duel_reads_only=(level == 2 and OPTIONS["--tensor-l2-duel-reads"] == "1"),
                             dueling=not (level == 2 and OPTIONS["--tensor-l2-no-duel"] == "1"))
    if name == "fifo":
        return FIFORP()
    if name == "random":
        return RandomRP()
    if name == "treeplru":
        return TreePLRURP()
    if name == "srrip":
        return RRIPRP()
    if name == "brrip":
        return BRRIPRP()
    if name == "ship":
        return SHiPMemRP()
    if name == "shippc":
        return SHiPPCRP()
    if name == "drrip":
        # gem5 requires team_size == associativity (one whole set per team);
        # 1 of every 64 sets samples each side, as in Jaleel et al. 2010.
        return DRRIPRP(team_size=assoc, constituency_size=64 * assoc)
    raise ValueError(name)


# Access tracing with gem5's CommMonitor + MemTraceProbe (unmodified gem5 objects).
# "l2l3" records every request entering L2 and L3 (tick, cmd, addr, size, PC)
# into l2_trace.trc.gz / l3_trace.trc.gz in the run directory.
TRACE_MODES = ("none", "l2l3")
TRACE = OPTIONS["--trace"]
if TRACE not in TRACE_MODES:
    print(f"ERROR: --trace must be one of {', '.join(TRACE_MODES)}")
    sys.exit(1)


# ============================================================
# Workload selection
# ============================================================

PROJECT_DIR = "/home/lord/gem5-project"

BINARIES = {
    "matrix_multiplication": os.path.join(PROJECT_DIR, "matmul_linux"),
    "matrix_addition": os.path.join(PROJECT_DIR, "matrix_add"),
    "reduction": os.path.join(PROJECT_DIR, "reduction"),
    "convolution_2d": os.path.join(PROJECT_DIR, "conv2d"),
    "vector_add": os.path.join(PROJECT_DIR, "vector_add"),
    "vector_mul": os.path.join(PROJECT_DIR, "vector_mul"),
    "dot_product": os.path.join(PROJECT_DIR, "dot_product"),
    "prefix_sum": os.path.join(PROJECT_DIR, "prefix_sum"),
    "matrix_transpose": os.path.join(PROJECT_DIR, "matrix_transpose"),
    "tiled_matmul": os.path.join(PROJECT_DIR, "tiled_matmul"),
    "conv1d": os.path.join(PROJECT_DIR, "conv1d"),
    "conv3d": os.path.join(PROJECT_DIR, "conv3d"),
    "relu": os.path.join(PROJECT_DIR, "relu"),
    "sigmoid": os.path.join(PROJECT_DIR, "sigmoid"),
    "tanh": os.path.join(PROJECT_DIR, "tanh"),
    "softmax": os.path.join(PROJECT_DIR, "softmax"),
    "max_pooling": os.path.join(PROJECT_DIR, "max_pooling"),
    "avg_pooling": os.path.join(PROJECT_DIR, "avg_pooling"),
    "batch_norm": os.path.join(PROJECT_DIR, "batch_norm"),
    "attention_like": os.path.join(PROJECT_DIR, "attention_like"),
    "gemm_rect": os.path.join(PROJECT_DIR, "gemm_rect"),
    "batched_matmul": os.path.join(PROJECT_DIR, "batched_matmul"),
    "embedding_gather": os.path.join(PROJECT_DIR, "embedding_gather"),
    "scatter": os.path.join(PROJECT_DIR, "scatter"),
    "layer_norm": os.path.join(PROJECT_DIR, "layer_norm"),
    "fused_add_mul": os.path.join(PROJECT_DIR, "fused_add_mul"),
    "depthwise_conv": os.path.join(PROJECT_DIR, "depthwise_conv"),
    "dilated_conv": os.path.join(PROJECT_DIR, "dilated_conv"),
}

if OPTIONS["--cmd"] is not None:
    # Generic mode: positional args are only the optional cache config.
    cmd_parts = shlex.split(OPTIONS["--cmd"])
    if not cmd_parts:
        print("ERROR: --cmd is empty")
        sys.exit(1)
    binary = os.path.abspath(cmd_parts[0])
    WORKLOAD = os.path.basename(binary)
    WORKLOAD_ARGS = cmd_parts[1:]
    TENSOR_SIZE = "n/a"
    cache_args = argv
else:
    # Legacy mode: same argument rules as tensor_cache_sim.py.
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    WORKLOAD = argv[0]

    if WORKLOAD == "gemm_rect":
        if len(argv) < 4:
            print("ERROR: gemm_rect requires M K N")
            sys.exit(1)
        WORKLOAD_ARGS = [int(argv[1]), int(argv[2]), int(argv[3])]
        TENSOR_SIZE = WORKLOAD_ARGS[0]
        n_problem = 3
    elif WORKLOAD == "tiled_matmul":
        if len(argv) < 3:
            print("ERROR: tiled_matmul requires N TILE")
            sys.exit(1)
        WORKLOAD_ARGS = [int(argv[1]), int(argv[2])]
        TENSOR_SIZE = WORKLOAD_ARGS[0]
        n_problem = 2
    elif WORKLOAD == "batched_matmul":
        if len(argv) < 3:
            print("ERROR: batched_matmul requires B N")
            sys.exit(1)
        WORKLOAD_ARGS = [int(argv[1]), int(argv[2])]
        TENSOR_SIZE = WORKLOAD_ARGS[1]
        n_problem = 2
    else:
        TENSOR_SIZE = int(argv[1])
        WORKLOAD_ARGS = [TENSOR_SIZE]
        n_problem = 1

    if WORKLOAD not in BINARIES:
        print("ERROR: Unknown workload:", WORKLOAD)
        sys.exit(1)

    binary = BINARIES[WORKLOAD]
    cache_args = argv[1 + n_problem:]

if not os.path.isfile(binary):
    print("ERROR: Workload binary not found:")
    print(binary)
    sys.exit(1)


# ============================================================
# Cache arguments (same defaults as tensor_cache_sim.py)
# ============================================================

if len(cache_args) >= 6:
    L1_SIZE = cache_args[0]
    L1_ASSOC = int(cache_args[1])
    L2_SIZE = cache_args[2]
    L2_ASSOC = int(cache_args[3])
    L3_SIZE = cache_args[4]
    L3_ASSOC = int(cache_args[5])
else:
    L1_SIZE = "32KiB"
    L1_ASSOC = 8
    L2_SIZE = "256KiB"
    L2_ASSOC = 8
    L3_SIZE = "8MiB"
    L3_ASSOC = 16


# ============================================================
# System (identical to Baseline-v1)
# ============================================================

system = System()

system.clk_domain = SrcClockDomain()
system.clk_domain.clock = FREQ
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = "timing"
system.mem_ranges = [AddrRange(MEM_SIZE)]

if CPU_TYPE == "timing":
    system.cpu = RiscvTimingSimpleCPU()
else:
    system.cpu = RiscvO3CPU()
    for width in ("fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
                  "issueWidth", "wbWidth", "commitWidth"):
        setattr(system.cpu, width, 6)
    system.cpu.branchPred = BranchPredictor(
        conditionalBranchPred=TAGE(),
        ras=ReturnAddrStack(numEntries=64),
    )
    system.cpu.mmu.itb.size = 32
    system.cpu.mmu.dtb.size = 64

system.cpu.icache = Cache()
system.cpu.icache.size = L1_SIZE
system.cpu.icache.assoc = L1_ASSOC
system.cpu.icache.tag_latency = L1_LAT
system.cpu.icache.data_latency = L1_LAT
system.cpu.icache.response_latency = L1_LAT
system.cpu.icache.mshrs = 4
system.cpu.icache.tgts_per_mshr = 20

system.cpu.dcache = Cache()
system.cpu.dcache.size = L1_SIZE
system.cpu.dcache.assoc = L1_ASSOC
system.cpu.dcache.tag_latency = L1_LAT
system.cpu.dcache.data_latency = L1_LAT
system.cpu.dcache.response_latency = L1_LAT
system.cpu.dcache.mshrs = 4
system.cpu.dcache.tgts_per_mshr = 20

system.l2 = Cache()
system.l2.size = L2_SIZE
system.l2.assoc = L2_ASSOC
system.l2.tag_latency = L2_LAT
system.l2.data_latency = L2_LAT
system.l2.response_latency = L2_LAT
system.l2.mshrs = 20
system.l2.tgts_per_mshr = 12
if PREFETCH == "stride":
    system.l2.prefetcher = StridePrefetcher()
    if OPTIONS["--l2-pf-on-access"] == "1":
        system.l2.prefetcher.prefetch_on_access = True
        # gem5 Base::observeAccess: with prefetch_on_pf_hit set, a hit only
        # notifies when the block was prefetched (prefetch_on_access ignored)
        system.l2.prefetcher.prefetch_on_pf_hit = False
if REPL not in ("lru", "tensor_l3"):
    system.l2.replacement_policy = make_repl(REPL, L2_ASSOC, L2_SIZE, 2)

system.l3 = Cache()
system.l3.size = L3_SIZE
system.l3.assoc = L3_ASSOC
system.l3.tag_latency = L3_LAT
system.l3.data_latency = L3_LAT
system.l3.response_latency = L3_LAT
system.l3.mshrs = 32
system.l3.tgts_per_mshr = 20
if REPL != "lru":
    system.l3.replacement_policy = make_repl(REPL, L3_ASSOC, L3_SIZE, 3)

system.l2bus = L2XBar()
system.l3bus = L2XBar()
system.membus = SystemXBar()
# FLUSH variant: memInvalidate() empties the caches but cannot clear the
# crossbars' snoop filters, so stale entries remain after the ROI flush.
# max_capacity is only a panic threshold (no timing effect); raise it so the
# stale entries cannot trip it. Lookup latency is unchanged.
for _bus in (system.l2bus, system.l3bus, system.membus):
    _bus.snoop_filter.max_capacity = "1GiB"

system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

if TRACE == "l2l3":
    system.l2mon = CommMonitor()
    system.l2mon.cpu_side_port = system.l2bus.mem_side_ports
    system.l2.cpu_side = system.l2mon.mem_side_port
    system.l2mon.tracer = MemTraceProbe(manager=system.l2mon, with_pc=True,
                                        trace_file="l2_trace.trc.gz")
    system.l3mon = CommMonitor()
    system.l3mon.cpu_side_port = system.l3bus.mem_side_ports
    system.l3.cpu_side = system.l3mon.mem_side_port
    system.l3mon.tracer = MemTraceProbe(manager=system.l3mon, with_pc=True,
                                        trace_file="l3_trace.trc.gz")
else:
    system.l2.cpu_side = system.l2bus.mem_side_ports
    system.l3.cpu_side = system.l3bus.mem_side_ports
system.l2.mem_side = system.l3bus.cpu_side_ports
system.l3.mem_side = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DRAM_MODELS[MEM_TYPE]()
if PAGE_POLICY is not None:
    system.mem_ctrl.dram.page_policy = PAGE_POLICY
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports

system.cpu.createInterruptController()


# ============================================================
# Workload
# ============================================================

system.workload = SEWorkload.init_compatible(binary)
process = Process()
process.cmd = [binary] + [str(arg) for arg in WORKLOAD_ARGS]
system.cpu.workload = process
system.cpu.createThreads()


# ============================================================
# Print configuration
# ============================================================

print("=" * 60)
print("Tensor-Aware AI Cache Simulation (arch_study/sim_config.py)")
print("=" * 60)
print("Workload        :", WORKLOAD)
print("Tensor Size     :", TENSOR_SIZE)
print("Workload Args   :", WORKLOAD_ARGS)
print("L1 Size         :", L1_SIZE)
print("L1 Associativity:", L1_ASSOC)
print("L2 Size         :", L2_SIZE)
print("L2 Associativity:", L2_ASSOC)
print("L3 Size         :", L3_SIZE)
print("L3 Associativity:", L3_ASSOC)
print("Latency L1/L2/L3:", L1_LAT, "/", L2_LAT, "/", L3_LAT, "cycles")
print("L2 prefetcher   :", PREFETCH)
print("CPU             :", type(system.cpu).__name__, "@", FREQ)
print("Memory          :", DRAM_MODELS[MEM_TYPE].__name__ + ",", MEM_SIZE)
print("DRAM page policy:", PAGE_POLICY or "model default")
print("Binary          :", binary)
print("Command         :", process.cmd)
print("=" * 60)


# ============================================================
# Run
# ============================================================

root = Root(full_system=False, system=system)
m5.instantiate()

print("Beginning RISC-V tensor cache simulation...")
# FLUSH variant (Step 1 fair re-test): the ROI_FLUSH kernels call m5_exit just
# before m5_reset_stats. At that point every cache is written back and
# invalidated, so each replacement policy starts the ROI with empty caches
# (no pre-ROI data can survive, whatever the policy).
exit_event = m5.simulate()
cause = exit_event.getCause()
flushes = 0
while "m5_exit" in cause and flushes == 0:
    m5.drain()
    m5.memWriteback(root)
    m5.memInvalidate(root)
    flushes += 1
    print(f"ROI_FLUSH @ tick {m5.curTick()}: all caches written back and invalidated")
    exit_event = m5.simulate()
    cause = exit_event.getCause()
print(f"Exiting @ tick {m5.curTick()} because {cause}")

print("=" * 60)
print("Simulation completed.")
print("=" * 60)
