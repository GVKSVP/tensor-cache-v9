"""check_v9_in_gem5.py - is v9 correctly installed in THIS gem5?
Run:  ~/gem5/build/RISCV/gem5.opt ~/gem5-project/arch_study/testing/check_v9_in_gem5.py
Checks (no simulation):
 1. gem5 knows TensorAwareRP (the v9 patch is compiled in);
 2. v9_policy.py builds L2/L3 policies whose settings equal, value for value,
    those recorded in config.ini of the project's real v9 runs (BERT_ffn1)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from m5.objects import TensorAwareRP  # noqa: F401
except ImportError:
    print("FAIL: this gem5 has no TensorAwareRP -> apply testing/tensor_aware_v9.patch and rebuild")
    sys.exit(1)
from v9_policy import v9_l2, v9_l3, v9_l2_prefetcher

EXPECT = {
 2: dict(assoc='16', bip_stream='0', bypass_streaming='false', coord_fit_layers='false', coord_l2_alpha='0.9', coord_l2_bytes='0', coord_l2_contig_min='0.35', coord_l2_pack='false', dead_demote='true', dirty_aware='false', duel_brrip='false', duel_demand_only='false', duel_lru_default='false', duel_reads_only='false', duel_skip_first='false', dueling='false', dump_file='', dump_max='0', keep_prefetch='false', l2_tree_alt='false', l2_tree_v3='-1', level='2', num_sets='4096', out_stream='false', pack_max_layer_bytes='16777216', pf_by_requestor='true', plan_activations='false', pos_by_bytes='true', reset_history_at_roi='true', set_aware='false', set_aware_ways='16', slice_beta='0.0', slice_contig='true', slice_contig_min='0.35', slice_pack='true', slice_size_hint='true', slice_size_ratio='true', slice_ws_alpha='0.9', tree_off='true'),
 3: dict(assoc='16', bip_stream='0', bypass_streaming='false', coord_fit_layers='false', coord_l2_alpha='0.9', coord_l2_bytes='4194304', coord_l2_contig_min='0.35', coord_l2_pack='false', dead_demote='true', dirty_aware='false', duel_brrip='false', duel_demand_only='false', duel_lru_default='true', duel_reads_only='false', duel_skip_first='true', dueling='true', dump_file='', dump_max='0', keep_prefetch='false', l2_tree_alt='false', l2_tree_v3='-1', level='3', num_sets='16384', out_stream='false', pack_max_layer_bytes='0', pf_by_requestor='false', plan_activations='true', pos_by_bytes='true', reset_history_at_roi='true', set_aware='true', set_aware_ways='16', slice_beta='0.0', slice_contig='false', slice_contig_min='0.35', slice_pack='true', slice_size_hint='true', slice_size_ratio='false', slice_ws_alpha='0.9', tree_off='false'),
}

def norm(v):
    s = str(v)
    if s in ("True", "False"):
        return s.lower()
    try:
        f = float(s)
        if "." in s or "e" in s:
            return repr(f)
    except ValueError:
        pass
    return s

bad = 0
for lvl, pol in ((2, v9_l2("x.txt")), (3, v9_l3("x.txt"))):
    for k, want in EXPECT[lvl].items():
        if k == "assoc":
            continue  # set by gem5 from the cache at run time
        got = norm(getattr(pol, k))
        if got != norm(want) and not (k in ("slice_beta",) and float(got) == float(want)):
            print(f"  L{lvl} {k}: got {got}, expected {want}"); bad += 1
pf = v9_l2_prefetcher()
if not (pf.prefetch_on_access and not pf.prefetch_on_pf_hit):
    print("  prefetcher setting wrong"); bad += 1
print("PASS: v9 (TensorAwareRP) is installed and v9_policy.py matches the project's v9 runs"
      if bad == 0 else f"FAIL: {bad} setting(s) differ")
sys.exit(0 if bad == 0 else 1)
