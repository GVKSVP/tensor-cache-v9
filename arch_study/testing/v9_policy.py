"""v9_policy.py - use the FINAL v9 tensor-aware replacement rule in ANY gem5 config.

v9 = gem5's TensorAwareRP (added by testing/tensor_aware_v9.patch) at L2 and L3,
with exactly the settings that produced the project's final results
(identical to what sim_config_flush14.py builds for `--repl tensor` + v9 flags;
checked against config.ini of the real v9 runs by testing/check_v9_in_gem5.py).

Usage inside your own gem5 config script:

    import sys; sys.path.insert(0, "/home/<you>/gem5-project/arch_study/testing")
    from v9_policy import apply_v9
    ...                                   # build system.l2 / system.l3 as usual
    apply_v9(system.l2, system.l3, hint_file="/path/to/<workload>.txt",
             l2_size="4MiB", l2_assoc=16, l3_size="16MiB", l3_assoc=16)

Notes
- L1 stays LRU (v9 is an L2 + L3 rule).
- A hint file per workload is REQUIRED (see PROJECT_GUIDE.md section 4.3);
  hint_file="" makes v9 behave almost like LRU.
- apply_v9 also sets the L2 stride prefetcher to train on every access
  (prefetch_on_access=True, prefetch_on_pf_hit=False). The project used this
  setting for EVERY rule, so use it for the rules you compare against too.
"""
from m5.objects import TensorAwareRP, StridePrefetcher
from m5.util.convert import toMemorySize

LINE = 64

# Settings shared by both levels (from the v9 runs' config.ini)
_COMMON = dict(
    slice_beta=0.0, l2_tree_alt=False, slice_ws_alpha=0.9, slice_contig_min=0.35,
    slice_pack=True, coord_l2_alpha=0.9, coord_l2_contig_min=0.35,
    coord_l2_pack=False, l2_tree_v3=-1, dead_demote=True, coord_fit_layers=False,
    pos_by_bytes=True, out_stream=False, bip_stream=0, duel_brrip=False,
    set_aware_ways=16, bypass_streaming=False, dirty_aware=False,
    duel_demand_only=False, keep_prefetch=False, duel_reads_only=False,
    slice_size_hint=True, reset_history_at_roi=True, dump_file="", dump_max=0,
)


def v9_l2(hint_file, l2_size="4MiB", l2_assoc=16, l3_size="16MiB"):
    """v9 at L2: working-set plan (contiguous / packed for layers <= L3 size),
    no learned tree, no set dueling, own prefetch fills recognised."""
    return TensorAwareRP(
        level=2, num_sets=toMemorySize(l2_size) // (LINE * l2_assoc),
        hint_file=hint_file, slice_size_ratio=True, tree_off=True,
        slice_contig=True, pack_max_layer_bytes=toMemorySize(l3_size),
        coord_l2_bytes=0, plan_activations=False, set_aware=False,
        duel_skip_first=False, duel_lru_default=False, pf_by_requestor=True,
        dueling=False, **_COMMON)


def v9_l3(hint_file, l3_size="16MiB", l3_assoc=16, l2_size="4MiB"):
    """v9 at L3: packing plan + learned tree + coordination with L2 +
    set-aware cap + dead demotion + set dueling vs LRU (LRU by default)."""
    return TensorAwareRP(
        level=3, num_sets=toMemorySize(l3_size) // (LINE * l3_assoc),
        hint_file=hint_file, slice_size_ratio=False, tree_off=False,
        slice_contig=False, pack_max_layer_bytes=0,
        coord_l2_bytes=toMemorySize(l2_size), plan_activations=True,
        set_aware=True, duel_skip_first=True, duel_lru_default=True,
        pf_by_requestor=False, dueling=True, **_COMMON)


def v9_l2_prefetcher():
    """L2 stride prefetcher exactly as in the project (all rules)."""
    pf = StridePrefetcher()
    pf.prefetch_on_access = True
    pf.prefetch_on_pf_hit = False
    return pf


def apply_v9(l2, l3, hint_file, l2_size="4MiB", l2_assoc=16,
             l3_size="16MiB", l3_assoc=16, set_prefetcher=True):
    """Put v9 on an existing L2 and L3 cache object."""
    l2.replacement_policy = v9_l2(hint_file, l2_size, l2_assoc, l3_size)
    l3.replacement_policy = v9_l3(hint_file, l3_size, l3_assoc, l2_size)
    if set_prefetcher:
        l2.prefetcher = v9_l2_prefetcher()
