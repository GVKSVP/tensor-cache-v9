 #!/bin/bash
# UNSEEN test, final v9 setup, one command. New script (launch_unseen.sh is unchanged).
#  Stage 1  profiling: tensor map run + page-table run (+ the 4 LRU runs)
#  Stage 2  make the 4 hint files            -> results/tensor_hints_unseen_v9/
#  Stage 3  check all 4 hint files exist (stops here if not)
#  Stage 4  v9 + the 8 standard rules (36 runs)
#  Stage 5  summary table                    -> datasets/unseen_v9_summary.csv
# Usage: bash results/launch_unseen_v9.sh [only-layer]   e.g. BLOCK_rn50s3 = pilot on 1 workload
set -u
cd ~/gem5-project/arch_study; PY=../.venv/bin/python; T=unseen_v9_lru; W=${WORKERS:-3}
ONLY=${1:-}
log(){ echo "[$(TZ=Asia/Kolkata date +%H:%M)] $*"; }
$PY ai_kernels/make_unseen_v9_plan.py
PL=ai_kernels/experiment_plan_unseen_v9_lru.csv; PR=ai_kernels/experiment_plan_unseen_v9_rules.csv
if [ -n "$ONLY" ]; then   # pilot: plans restricted to one workload (new files)
  for p in $PL $PR; do q=${p%.csv}_$ONLY.csv; [ -e $q ] || $PY -c "import pandas as pd,sys;d=pd.read_csv('$p');d[d.layer_id=='$ONLY'].to_csv('$q',index=False)"; done
  PL=${PL%.csv}_$ONLY.csv; PR=${PR%.csv}_$ONLY.csv; T=${T}_$ONLY
fi
log "Stage 1: profiling + LRU ($PL)"
$PY -u run_ai_kernels_flush14u.py --plan $PL --out datasets/ai_kernels_$T.csv --results results/$T --workers 2 > results/$T.log 2>&1 &
$PY -u trace_tools/flush_env_u.py run_ptable --plan $PL --results results/${T}_maps --workers 1 > results/${T}_ptable.log 2>&1 &
$PY -u trace_tools/flush_env_u.py run_map --plan $PL --results results/${T}_maps --workers 1 > results/${T}_map.log 2>&1 &
wait
log "Stage 2: making hint files"
$PY trace_tools/flush_env_u.py make_hints_v2 --out results/tensor_hints_unseen_v9 results/$T/UNV_lru_* > results/${T}_hints.log 2>&1
cat results/${T}_hints.log
log "Stage 3: checking hint files"
for L in $($PY -c "import pandas as pd;print(' '.join(pd.read_csv('$PL').layer_id))"); do
  f=results/tensor_hints_unseen_v9/$L.txt
  if [ ! -s $f ] || [ $(grep -c '^R' $f) -lt 10 ]; then log "STOP: hint file missing/empty: $f"; exit 1; fi
  log "  ok $f ($(grep -c '^T' $f) tensors, $(grep -c '^R' $f) pages)"
done
log "Stage 4: v9 + 8 standard rules ($PR)"
R=unseen_v9_rules; [ -n "$ONLY" ] && R=${R}_$ONLY
$PY -u run_ai_kernels_flush14u.py --plan $PR --out datasets/ai_kernels_$R.csv --results results/$R --workers $W > results/$R.log 2>&1
log "Stage 5: summary"
[ -z "$ONLY" ] && $PY trace_tools/unseen_v9_summary.py
log "DONE"
