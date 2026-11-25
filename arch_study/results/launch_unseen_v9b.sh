#!/bin/bash
# UNSEEN test with frozen v9. Copy of launch_unseen_v9.sh that STOPS on any failure.
# Usage: bash results/launch_unseen_v9b.sh hints|rules LAYER [LAYER...]   (resumable)
set -euo pipefail
MODE=$1; shift; W=${WORKERS:-3}
cd ~/gem5-project/arch_study; PY=$HOME/gem5-project/.venv/bin/python
log(){ echo "[$(TZ=Asia/Kolkata date '+%d %b %H:%M')] $*"; }
trap 'log "STOPPED: failure at line $LINENO"' ERR
L="$*"; TAG=${L// /+}; LOG=results/unseen_v9b_logs; mkdir -p $LOG results/unseen_v9b_plans
G=$HOME/gem5/build/RISCV/gem5.opt
cmp -s $G results/gem5_rp_backup_step8/gem5.opt.step12b_v9 || { log "STOPPED: gem5.opt is not the saved v9 binary"; exit 1; }
{ date -u; sha256sum $G sim_config_flush14.py run_ai_kernels_flush14u.py ai_kernels/bin/riscv_flush_u/block ai_kernels/bin/native_u/block; } >> $LOG/sha256.txt
sub(){ $PY - "$1" "$2" $L <<'P'
import sys,pandas as pd
d=pd.read_csv(sys.argv[1]); want=sys.argv[3:]
assert set(want)<=set(d.layer_id),want
d[d.layer_id.isin(want)].to_csv(sys.argv[2],index=False)
P
}
check(){ $PY - "$1" "$2" <<'P'
import sys,pandas as pd
d=pd.read_csv(sys.argv[1]); p=pd.read_csv(sys.argv[2])
ok=set(d[d.status=="SUCCESS"].run_id); bad=[r for r in p.run_id if r not in ok]
if bad: sys.exit(f"NOT SUCCESS: {bad}")
print("all",len(p),"rows SUCCESS")
P
}
T=unseen_v9_lru
if [ $MODE = hints ]; then
  P=results/unseen_v9b_plans/lru_$TAG.csv; sub ai_kernels/experiment_plan_unseen_v9_lru.csv $P
  log "profiling + LRU: $L"
  $PY -u run_ai_kernels_flush14u.py --plan $P --out datasets/ai_kernels_$T.csv --results results/$T --workers 2 > $LOG/lru_$TAG.log 2>&1 & a=$!
  $PY -u trace_tools/flush_env_u.py run_ptable --plan $P --results results/${T}_maps --workers 1 > $LOG/ptable_$TAG.log 2>&1 & b=$!
  $PY -u trace_tools/flush_env_u.py run_map --plan $P --results results/${T}_maps --workers 1 > $LOG/map_$TAG.log 2>&1 & c=$!
  wait $a; wait $b; wait $c
  check datasets/ai_kernels_$T.csv $P
  log "hint files"
  for l in $L; do
    $PY trace_tools/flush_env_u.py make_hints_v2 --out results/tensor_hints_unseen_v9 results/$T/UNV_lru_${l}_$l
    f=results/tensor_hints_unseen_v9/$l.txt; n=$(grep -c '^R' $f); [ $n -ge 10 ]; log "ok $f: $(grep -c '^T' $f) tensors, $n ranges"
  done
elif [ $MODE = rules ]; then
  for l in $L; do test -s results/tensor_hints_unseen_v9/$l.txt; done
  P=results/unseen_v9b_plans/rules_$TAG.csv; sub ai_kernels/experiment_plan_unseen_v9_rules.csv $P
  log "v9 + 8 standard rules: $L"
  $PY -u run_ai_kernels_flush14u.py --plan $P --out datasets/ai_kernels_unseen_v9_rules.csv --results results/unseen_v9_rules --workers $W > $LOG/rules_$TAG.log 2>&1
  check datasets/ai_kernels_unseen_v9_rules.csv $P
else log "unknown mode $MODE"; exit 1; fi
log "DONE $MODE $L"
