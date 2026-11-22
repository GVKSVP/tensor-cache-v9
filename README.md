# Tensor-Aware AI Cache Replacement (v9) for gem5

UROP project: a tensor-aware replacement rule (**v9**, gem5 `TensorAwareRP`) for the L2 and L3 caches. It uses hint files saying which memory holds which tensor (weights, inputs, outputs), a decision tree trained on Belady's OPT, and a set-dueling safety net against LRU.

**Result** on 72 AI workloads from 12 models (gem5 RISC-V O3, L2 4 MiB + L3 16 MiB), DRAM traffic relative to LRU (lower is better):

| v9 (ours) | DRRIP | SHiP-PC | Random | SHiP | BRRIP | SRRIP | TreePLRU | FIFO |
|---|---|---|---|---|---|---|---|---|
| **0.938** | 0.963 | 0.972 | 0.978 | 0.979 | 0.984 | 0.986 | 0.995 | 1.000 |

v9 has 9 wins, 62 ties and 1 loss (0.5%), and is never more than 0.01% worse than LRU. On the 7 multi-layer blocks it averages **0.821** (DRRIP 0.943).

## Start here

| File | What it is |
|---|---|
| [`SETUP_GUIDE.md`](SETUP_GUIDE.md) | **Step-by-step setup and testing**: WSL, tools, gem5 + v9 patch, build, quick check, troubleshooting, using v9 in your own gem5 |
| [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | **What was done and why**: full history v1 → v9, how v9 works, where the code is, every file and dataset, results, dos and don'ts |
| [`README_TESTING.md`](README_TESTING.md) | Short version of the testing steps |

## Quick start (details in SETUP_GUIDE.md)

```bash
git clone <this repo> ~/tensor-cache-v9
mkdir -p ~/gem5-project && cp -r ~/tensor-cache-v9/arch_study ~/gem5-project/
git clone https://github.com/gem5/gem5.git ~/gem5 && cd ~/gem5 && git checkout cbc94c1a77
git apply ~/gem5-project/arch_study/testing/tensor_aware_v9.patch
scons build/RISCV/gem5.opt -j2 --linker=gold
~/gem5/build/RISCV/gem5.opt --quiet ~/gem5-project/arch_study/testing/check_v9_in_gem5.py   # must print PASS
cd ~/gem5-project/arch_study && python3 testing/setup_test.py
python3 run_ai_kernels_flush14.py --plan ai_kernels/experiment_plan_smoke_test.csv \
    --out datasets/smoke_test_results.csv --results results/smoke_test --workers 3
python3 testing/check_smoke_test.py datasets/smoke_test_results.csv              # must print PASS
```

## Where the v9 code is

The rule itself is C++ inside gem5. It's delivered here as one patch, [`arch_study/testing/tensor_aware_v9.patch`](arch_study/testing/tensor_aware_v9.patch), against gem5 commit `cbc94c1a77`. The main file is `src/mem/cache/replacement_policies/tensor_aware_rp.cc`. This repository holds everything around it: workloads, configs, hint files, training scripts, datasets and checkers.

## Not in this repository

- Raw simulation traces (about 30 GB) and intermediate gem5 builds: not needed to reproduce the results, only to retrain the model.
- The old report document (2 Nov 2026, written before v9) is not included; `arch_study/report/` holds only the scripts that built it.
