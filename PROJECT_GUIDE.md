
# Tensor-Aware AI Cache Replacement (v9): Complete Project Guide

**Project:** UROP, "Tensor-Aware AI Caching Management for Optimizing Memory Performance in AI Workloads"
**Simulator:** gem5 (RISC-V, syscall-emulation mode)
**Final rule:** **v9**, one combined L2 + L3 replacement policy (`TensorAwareRP`)
**Last updated:** 22 Nov 2026

This guide covers everything the project did, why each decision was made, where every file and piece of code is, how to test it, what comes next, and the dos and don'ts. Read sections 1–4 first. The later sections are reference.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Background you need](#2-background-you-need)
3. [How the project developed (v1 → v9) and why](#3-how-the-project-developed-v1--v9-and-why)
4. [The final rule v9: what it does](#4-the-final-rule-v9-what-it-does)
5. [Where the v9 code is (gem5 vs separate files)](#5-where-the-v9-code-is-gem5-vs-separate-files)
6. [Experimental setup](#6-experimental-setup)
7. [Results](#7-results)
8. [Folder and file guide: why each file exists](#8-folder-and-file-guide-why-each-file-exists)
9. [How to test (teammate)](#9-how-to-test-teammate)
10. [What comes next](#10-what-comes-next)
11. [Dos and don'ts](#11-dos-and-donts)
12. [Known limitations](#12-known-limitations)
13. [FAQ](#13-faq)
14. [Glossary](#14-glossary)
15. [Appendix A: step-by-step log](#appendix-a-detailed-step-by-step-log-dates-numbers-decisions) · [Appendix B: how v9 decides](#appendix-b-how-v9-makes-one-decision-walkthrough) · [Appendix C: key numbers](#appendix-c-key-numbers-at-a-glance) · [Appendix D: every dataset](#appendix-d-every-dataset-and-why-it-was-created-all-112-files-in-datasets) · [Appendix E: config versions](#appendix-e-why-there-are-15-versions-of-the-config-and-runner) · [Appendix F: plans and result folders](#appendix-f-plan-files-and-result-folders)

---

## 1. The project in one page

**Problem.** A processor keeps recently used data in small, fast memories called **caches** (L1, L2, L3). Main memory (**DRAM**) is large but slow and energy-hungry. When a cache is full, a **replacement rule** decides which data to throw out. Standard rules such as **LRU** (throw out the least recently used) know nothing about AI programs. AI programs move huge **tensors** (weights, activations), so LRU often throws out weights that are needed again shortly afterwards, which causes extra DRAM traffic.

**What we built.** A new replacement rule, **v9**, inside the gem5 simulator. It knows which memory belongs to which tensor (weights, inputs, outputs and their sizes) from a **hint file**. It uses that knowledge to keep the data that will be reused and throw out data that is finished. Part of its logic is a **decision tree trained to imitate Belady's OPT**, the perfect rule that knows the future. A **safety net** (set dueling against LRU) makes sure it is never worse than LRU.

**Main metric.** DRAM traffic (bytes read and written to main memory) relative to LRU. Lower is better: LRU = 1.00.

**Result** on 72 AI workloads from 12 real models, compared with 9 other rules under identical settings:

| Rule | DRAM traffic vs LRU (geomean, 72 workloads) |
|---|---|
| **v9 (ours)** | **0.938** |
| DRRIP | 0.963 |
| SHiP-PC | 0.972 |
| Random | 0.978 |
| SHiP | 0.979 |
| BRRIP | 0.984 |
| SRRIP | 0.986 |
| TreePLRU | 0.995 |
| FIFO | 1.000 |

- 9 wins, 62 ties, 1 loss (UNET3D_down2, behind BRRIP by 0.5%).
- **Never more than 2% worse than LRU.** Worst case: +0.01%.
- **At least 5% better than LRU on average.** Achieved: 6.2%.
- **Run time not slower.** Geomean 0.9986. The worst case is 0.07% slower, which is noise level.
- On the 7 multi-layer workloads (model blocks), v9 averages **0.821** against 0.943 for DRRIP, the best other rule.

---

## 2. Background you need

| Term | Meaning |
|---|---|
| **Cache levels** | L1 (64 KiB, private, fastest) → L2 (4 MiB) → L3 (16 MiB, last level) → DRAM |
| **Cache line** | The unit the cache moves: 64 bytes |
| **Set / way** | A 16-way cache: each address can live in one of 16 slots (ways) of its set |
| **Replacement rule** | Chooses which line in a full set to throw out (evict) |
| **Hit / miss** | The data was in the cache / the data had to come from the next level |
| **DRAM traffic** | Bytes read from and written to main memory. Our headline metric. |
| **Prefetcher** | Hardware that guesses the next addresses and loads them early (we use a stride prefetcher at L2) |
| **Belady's OPT (MIN)** | The perfect rule: evict the line whose next use is furthest in the future. Impossible in hardware (it needs the future), but computable offline from a recorded trace. |
| **Tensor** | A multi-dimensional array: weights, input/output activations, KV cache, embedding tables |
| **Hint file** | A text file telling v9 which physical address ranges belong to which tensor, its role and its size |
| **Set dueling** | A few sets always use rule A and a few always use rule B. A counter tracks which misses less, and all other sets follow the winner. |
| **ROI** | Region of interest: only the kernel's work is measured, not program start-up |

**Why replacement rules matter for AI.** AI layers often reuse a weight tensor many times, for example once per output row. If the tensor is bigger than the cache, LRU throws it out just before it is needed again ("thrashing"), and every reuse becomes a DRAM read. A rule that knows "this is a 20 MB weight tensor that will be reread" can keep the part that fits and stop thrashing.

---

## 3. How the project developed (v1 → v9) and why

### Phase 1: baseline cache study (Sep 2026, **files now deleted**)

- About 56,000 gem5 runs of 28 simple kernels (matrix multiply, convolution, …) over many L1/L2/L3 sizes, to learn how cache size affects performance.
- **Why:** to understand the simulator and the basic effects before designing anything.
- **Status:** cleaned up on 21 Nov 2026. The dataset, programs and raw results were deleted on the user's decision. The report's Phase 1 section can no longer be regenerated from data; see [§10](#10-what-comes-next).

### Phase 2: tensor-aware rule (Nov 2026), all under `~/gem5-project/arch_study/`

| Step | What we did | Why |
|---|---|---|
| **Workloads** | Wrote 14 C kernels (GEMM, convolution, attention, LSTM, embedding, …) reproducing **real layer shapes** from 12 models (ResNet-50, BERT-Large, DLRM, Llama-8B/70B, ViT, MobileNet-V2, SDXL, 3D U-Net, RNN-T, R-GAT, DeepBench). 65 single layers, later 7 multi-layer blocks/chains. | Realistic AI memory patterns, small enough to simulate in minutes. Verified against numpy (`verify_reference*.py`). |
| **Hardware model** | "Config F": a P870-D-inspired 2 GHz O3 RISC-V core, 64 KiB L1, 4 MiB L2 + stride prefetcher, 16 MiB L3, DDR5-4400 | A modern server-class core, so results are relevant |
| **Step 2: standard rules** | Added `--repl` to the config and ran 8 standard rules × 65 layers (520 runs) | **Finding:** no single rule wins. BRRIP cut traffic by up to 52% on some layers but raised it 10× on others. This motivated a tensor-aware rule. |
| **Step 3: traces** | Recorded every L2/L3 access (physical addresses), mapped addresses to tensors (`run_map`, `run_ptable`), and replayed Belady's OPT offline (`belady.py`) | To know **how much room exists** (OPT vs LRU) and to get perfect keep/evict labels. OPT: L2 −13.5%, L3 −9.3% overall, almost all from **weights**. |
| **Step 4: learning** | Built 10 per-access features (`build_features.py`) and trained a depth-6 decision tree to predict OPT's "keep" label (`train_policy.py`). Test models never used in training: SDXL, ViT, MobileNet-V2, Llama-70B. | A small tree can be turned into C++ and run inside a cache (`export_tree.py` → `tensor_aware_tree.hh`) |
| **Step 5: into gem5** | Wrote `TensorAwareRP` in gem5 C++, plus the hint-file loader, set dueling vs LRU and slice protection | Offline replay doesn't capture everything (the prefetcher, timing), so the real test is inside gem5 |
| **Bugs found and fixed** | (1) Offline L2 traces missed the L2 prefetcher's fills. (2) gem5 calls the policy on CleanEvict. (3) Leftover data from before the ROI made BRRIP look artificially good, so we added a **true cache flush** at ROI start (`sim_config_flush.py`). | Without these fixes the comparison would be wrong or unfair |
| **Multi-layer workloads** | Whole blocks (BERT, ViT, DLRM, RN50, MBV2) and chains (Llama decode, RNN-T encoder) | Real models run layers back-to-back; tensors pass between layers |
| **L2 rule** | The learned tree **lost** at L2 (it broke the prefetcher). We replaced it with a working-set slice (keep the part of a too-large weight tensor that fits) and contiguous slicing. | Measured in gem5: the tree gave 2.3× more misses on BERT at L2 |
| **Step 6: packing + coordination** | L3 "packing" plan (decide exactly which bytes of which tensor to keep) and L2/L3 coordination (don't keep the same data at both levels) | DLRM 0.48 → 0.38, beating offline OPT's 0.41, because OPT works per level and coordination works across levels |
| **Step 7: prefetcher fix** | `--l2-pf-on-access 1`: the stride prefetcher also learns from hits. Applied to **every** rule for fairness. | Our rule turns misses into hits, which starved the prefetcher of training |
| **Step 8: paper ideas** | Adopted **dead-tensor demotion** (throw out finished tensors first). Rejected L3 bypass and coordination for layers that fit (both hurt). | Each idea was tested and kept only if it helped without hurting others |
| **v7 → v8** | Turned off keep-prefetch (it hurt after the fix). Added position-by-bytes and planning for reused input activations. | Found by an ablation (turning each switch off one at a time) and by checking all 65 single layers |
| **v9** | Added the **set-aware plan** (cap of 16 kept lines per set) | The packing plan could overfill some sets. This fixed UNET3D layers. |
| **Rejected after v9** | Step 13 "first-touch" variant (0.9928, worse), output streaming, bimodal stream, BRRIP duel, L2 plan-act | All tested; none improved on v9 |

**Rules we followed throughout:**
- Every change gets a **new file** (that's why `sim_config_flush.py … flush15.py` and `run_ai_kernels_flush … flush15.py` exist; old versions are never overwritten).
- Pilot on 1–2 workloads first.
- Every rule is compared under **identical settings**.

---

## 4. The final rule v9: what it does

v9 is **one policy** (`TensorAwareRP`) used in **both L2 and L3**, sharing one hint file. L1 stays LRU.

### 4.1 The main components (plain words)

| Component | Level | What it does | Why |
|---|---|---|---|
| **Hint lookup** | L2 + L3 | For every access, look up the address in the hint file to get tensor id, role (weight, input, output, KV cache, …) and size | All later decisions need to know "what is this data?" |
| **Working-set plan / packing** | L3 (and L2 for layers ≤ L3 size) | If a reused tensor doesn't fit, keep a planned fraction of it (alpha 0.9 of capacity), **weights first, then reused inputs** | Stops thrashing: the kept part hits on every reuse instead of everything missing |
| **Learned decision tree** | L3 | For data outside the plan, predict "will OPT keep this?" from 10 features | Copies the perfect rule's decisions using only present information |
| **Dead-tensor demotion** | L2 + L3 | A tensor whose last use has passed is evicted first | Frees space for live data |
| **L2/L3 coordination** | L3 | For layers bigger than L3, L3 avoids keeping what L2 already keeps | No duplicate copies, so more unique data is on chip |
| **Set-aware cap** | L3 | At most 16 planned lines per set | Prevents some sets from overfilling while others sit empty |
| **Prefetcher-aware L2** | L2 | L2 recognises its own prefetch fills; no learned tree at L2; no dueling at L2 | The tree and dueling at L2 broke the stride prefetcher |
| **Safety net (set dueling)** | L3 | A few sets always run plain LRU, a few always run v9; the rest follow the winner, **defaulting to LRU** until v9 proves better | Guarantees v9 is never meaningfully worse than LRU |

### 4.2 The exact v9 settings (`sim_flags`)

v9 = `--repl tensor` plus these switches (all defined in `sim_config_flush14.py` and passed to gem5 parameters):

| Flag | Value | Meaning |
|---|---|---|
| `--tensor-slice` | 0 | Old hash-slice protection at L3: off (replaced by the plan) |
| `--tensor-slice-ws` | 0.9 | L3 working-set plan: keep up to 90% of capacity for planned tensors |
| `--tensor-duel-skip-first` | 1 | Dueling ignores first-touch misses (unavoidable for every rule) |
| `--tensor-duel-lru-default` | 1 | Follower sets use LRU until v9 proves better (safety) |
| `--tensor-slice-l2` | 0 | Old L2 hash slice: off |
| `--tensor-l2-pf-fix` | 1 | L2 sees its own prefetch fills as prefetches |
| `--tensor-l2-no-duel` | 1 | No set dueling at L2 |
| `--tensor-l2-tree-off` | 1 | No learned tree at L2 |
| `--tensor-l2-slice-ws` | 0.9 | L2 working-set plan, 90% |
| `--tensor-l2-keep-pf` | 0 | Don't force-keep prefetch fills (it hurt after the prefetcher fix) |
| `--tensor-l2-slice-contig` | 1 | Keep a contiguous part of a tensor at L2 (the prefetcher likes contiguous data) |
| `--tensor-l2-contig-min` | 0.35 | …only if at least 35% of the tensor can be kept |
| `--tensor-l3-pack` | 1 | L3 packing plan |
| `--tensor-l2-pack` | 2 | L2 packing only for layers ≤ L3 size ("mixed") |
| `--tensor-coord` | 1 | L2/L3 coordination for layers > L3 |
| `--tensor-dead-demote` | 1 | Dead tensors evicted first |
| `--tensor-pos-bytes` | 1 | Position inside a tensor measured in bytes |
| `--tensor-plan-act` | 1 | Reused input activations are planned too |
| `--tensor-set-aware` | 1 | Set-aware plan |
| `--tensor-set-aware-ways` | 16 | Cap: 16 planned lines per set |
| `--l2-pf-on-access` | 1 | Prefetcher trains on hits too (**used for all rules**) |
| `--tensor-hints` | file | Path to the workload's hint file |

### 4.3 The hint file

Plain text, one per workload (e.g. `results/tensor_hints_layers/RN50_c5_3x3.txt`):

```
T 1 1 9437184            tensor 1, used by layer 1, 9,437,184 bytes
R 7f040 80000 1 0        physical range 0x7f040-0x80000 -> tensor 1, role 0 (weight)
L 1 10c40 10f00          (multi-layer only) layer 1's code address range
```

Roles: 0 weight, 1 input, 2 output, 3 intermediate activation, 4 KV cache, 5 gather table, 6 index, 7 scratch, 8 flush buffer, 9 program code, 10 stack, 11 other.

**How it's made (not automatic inside gem5):**
1. `trace_tools/run_map.py`: a **profiling run** of a special build (`bin/riscv_map_flush`) that records each tensor's virtual address.
2. `trace_tools/run_ptable.py`: a run that saves the page table (virtual → physical address).
3. `trace_tools/make_hints_v2.py`: combines both with the roles in `trace_tools/tensor_roles.py` to produce the hint file.

v9 only **reads** the hint file. With no hint file, v9 behaves almost like LRU. With a wrong path, gem5 stops with "cannot open hint file". In a real system, the AI framework (PyTorch etc.) knows where its tensors are and could pass these hints directly, so no profiling run would be needed.

### 4.4 The trained model

- **Where:** `results/policy_final_v2/` (`l3_tree.pkl`, `l2_tree.pkl`, `metrics.json`)
- **Trained by:** `trace_tools/train_policy.py` on Belady OPT labels; features from `trace_tools/build_features.py`
- **Exported to C++ by:** `trace_tools/export_tree.py` → `gem5/src/mem/cache/replacement_policies/tensor_aware_tree.hh`
- **Features (10):** `role_id, cmd_id, tsize_log2, used_next, dead, first_touch, prev_dist_log2, n_prev, t_retouch8, t_dist`
- **Split:** test models (never used in training): SDXL, ViT, MobileNet-V2, Llama-70B
- **In v9 only the L3 tree is used.** The L2 tree is off (`--tensor-l2-tree-off 1`).

---

## 5. Where the v9 code is (gem5 vs separate files)

**Short answer:** the replacement rule itself is **inside gem5** (C++). Everything else (workloads, configs, hint making, training, analysis) is in **separate files** in `~/gem5-project/arch_study/`.

### 5.1 Inside gem5: `~/gem5/`

- **Repository:** `~/gem5`, branch **`tensor-aware-rp`**
- **Based on:** gem5 commit `cbc94c1a77` (stable, just after v25.1.0.1). The branch has 5 commits on top; the latest is `09b27aba01`. The latest v9 changes are **uncommitted working-tree changes** on that branch.

| File (under `src/mem/cache/`) | Lines | What it contains |
|---|---|---|
| `replacement_policies/tensor_aware_rp.cc` | 954 | **The v9 rule:** hint loading, features, plan/packing, set-aware cap, dead demotion, coordination, set dueling, victim choice |
| `replacement_policies/tensor_aware_rp.hh` | 225 | Class definition and state |
| `replacement_policies/tensor_aware_tree.hh` | 683 | **The learned decision trees** as C++ (generated by `export_tree.py`) |
| `replacement_policies/tensor_aware_tree_v3.hh` | 654 | Experimental L2 trees from Step 7 (compiled in, **not used** by v9) |
| `replacement_policies/ReplacementPolicies.py` | +184 | `TensorAwareRP` class and its 36 parameters |
| `replacement_policies/SConscript` | +4 | Adds the new files to the build |
| `replacement_policies/base.hh` | +6 | Optional `bypass()` hook (default off) |
| `tags/base.hh`, `tags/base_set_assoc.hh` | +12 | Pass the bypass request to the policy (default off) |
| `base.cc` | +7 | Cache honours a bypass request (never triggered in v9) |

**Using v9 in your own gem5 scripts:** `testing/v9_policy.py` (`apply_v9(system.l2, system.l3, hint_file=…)`), and check the install with `gem5.opt testing/check_v9_in_gem5.py` (see `SETUP_GUIDE.md`, Parts 5B and 11).

**The whole change as one patch:** `testing/tensor_aware_v9.patch` (in the testing zip), applied with `git apply` on commit `cbc94c1a77`. It also contains the rejected Step-13 `first_touch` switch, which is **off by default**, so results are identical to v9.

**Compiled binaries:**

| Binary | What it is |
|---|---|
| `~/gem5/build/RISCV/gem5.opt` | **Active binary = v9** (checked identical to the backup below) |
| `arch_study/results/gem5_rp_backup_step8/gem5.opt.step12b_v9` | **Saved v9 binary**. Keep it. |
| `~/gem5/build/RISCV/gem5.opt.orig_v25.1.0.1` | Original unmodified gem5, for reference |
| `arch_study/results/gem5_rp_backup_step*/…` | Older intermediate builds (step6 … step13b) and source backups |

### 5.2 Outside gem5: `~/gem5-project/arch_study/`

| What | Where |
|---|---|
| Config that switches v9 on and passes the flags | `sim_config_flush14.py` |
| Runner used for **all final results** | `run_ai_kernels_flush14.py` |
| Workload source | `ai_kernels/src/*.c` |
| Hint making | `trace_tools/run_map.py`, `run_ptable.py`, `make_hints_v2.py`, `tensor_roles.py` |
| Training | `trace_tools/build_features.py`, `train_policy.py`, `export_tree.py` |
| Hint files used in the final results | `results/tensor_hints_layers/` (65 layers), `results/tensor_hints_v3/` (7 blocks/chains) |

**How a run flows:**
```
plan CSV row ──► run_ai_kernels_flush14.py ──► gem5.opt sim_config_flush14.py --repl tensor ... --tensor-hints X.txt
                                                   │
                                                   └► TensorAwareRP (gem5 C++) reads X.txt, decides evictions
results ──► results/<tag>/<run>/stats.txt ──► datasets/<name>.csv (one row per run)
```

---

## 6. Experimental setup

| Item | Setting |
|---|---|
| CPU | O3 out-of-order RISC-V, 2 GHz (P870-D-inspired) |
| L1 | 64 KiB, 8-way, 4 cycles (LRU for every rule) |
| L2 | 4 MiB, 16-way, 16 cycles, **stride prefetcher**, `--l2-pf-on-access 1` |
| L3 | 16 MiB, 16-way, 40 cycles |
| DRAM | DDR5-4400, open-adaptive page policy |
| Mode | **Cold**: a true cache flush at the start of the measured region, so nothing is left over from set-up |
| Measured | Region of interest only (`m5_reset_stats` → kernel → `m5_dump_stats`) |
| Work sampling | Real tensor sizes, but only some rows/channels computed ("shape-faithful, work-sampled") to keep each run to about 5–20 min |

**Workloads (72):** 65 single layers plus 7 multi-layer workloads (BLOCK_bert, BLOCK_vit, BLOCK_dlrm, BLOCK_rn50, BLOCK_mbv2, CHAIN_llama_decode, CHAIN_rnnt_encoder). They come from ResNet-50 (12), Llama 8B/70B (10), DeepBench (8), BERT-Large (6), ViT (6), MobileNet-V2 (6), SDXL (6), RNN-T (5), 3D U-Net (5), R-GAT (4) and DLRM (4). The full list with shapes and sources is in `ai_kernels/experiment_plan_v9_full.csv`.

**Fairness:** every rule uses exactly the same plan row (same binary, shape, cache sizes, prefetcher setting, cold flush). Only `--repl` changes.

---

## 7. Results

### 7.1 Final scoreboard (all 72 workloads, 10 rules)

- **Charts:** `results/final_charts_v2/`. Start with `final_scoreboard_part_differ.png` (the 18 workloads where rules differ), then `final_scoreboard_part1-5.png` (all 72) and `cache_stats_part1-5.png` (tables of DRAM MB, L2 and L3 hit %, L3 misses, time).
- **Data:** `datasets/final_scoreboard_v2_wide.csv` (72 × 10 rules plus verdict), `_long.csv` (every run), `_cache_stats.csv`.

**Where v9 wins clearly (DRAM vs LRU):**

| Workload | v9 | Best other rule |
|---|---|---|
| RNNT_enc_pre1 | **0.305** | BRRIP 0.433 |
| DLRM_top_mlp | **0.312** | BRRIP 0.440 |
| BLOCK_dlrm | **0.341** | BRRIP 0.460 |
| CHAIN_rnnt_encoder | **0.775** | SHiP 0.873 |
| RNNT_enc_post1 | **0.836** | DRRIP 0.854 |
| BERT_ffn1 | **0.886** | BRRIP 0.898 |
| BERT_ffn2 | **0.916** | SHiP-PC 0.922 |
| BLOCK_bert | **0.958** | FIFO 1.001 |
| UNET3D_out | **0.983** | TreePLRU 0.988 |

**Why 54 workloads are ties.** Their DRAM traffic is already at the minimum: every byte is loaded exactly once. That's because the data fits in the 16 MiB L3 (38 workloads), every byte is used only once (5, e.g. the Llama FFN layers), or only a small part of a large table is touched (11). **Even OPT would tie there.** Replacement can only help when data is **bigger than the cache and reused**.

### 7.2 Robustness (rule v8, which is v9 without the set-aware cap)

Run on 4 multi-layer workloads under changed conditions (`datasets/ai_kernels_robust_v8.csv`):

| Condition | BERT | DLRM | RNN-T | ViT | DRRIP for comparison |
|---|---|---|---|---|---|
| L3 = 8 MiB | 0.502 | 0.626 | 0.871 | 0.240 | 0.874 / 0.756 / 0.971 / 0.614 |
| L3 = 32 MiB | 0.999 | 1.000 | 0.993 | 1.000 | 1.086 / 1.000 / 1.278 / 1.000 |
| 2× work | 0.904 | 0.226 | 0.605 | 0.995 | 1.090 / 0.373 / 0.852 / 1.470 |

Best or tied in every condition. The gains are **largest when the cache is under pressure** (8 MiB L3).

---

## 8. Folder and file guide: why each file exists

Root: `~/gem5-project/`. It now contains only `arch_study/` (the project) and `.venv/` (the Python environment the scripts run with: pandas, numpy, pyarrow, scikit-learn, matplotlib).

### 8.1 `arch_study/` top level

| File | Why it exists |
|---|---|
| `PROJECT_GUIDE.md` | This document |
| `SETUP_GUIDE.md` | Complete step-by-step setup guide (from a fresh laptop to PASS) |
| `testing/` | The reproduction kit: `README_TESTING.md`, `setup_test.py`, `check_smoke_test.py`, the smoke-plan template, **`tensor_aware_v9.patch` (all gem5 changes)**, **`v9_policy.py`** (v9 for your own gem5 scripts) and **`check_v9_in_gem5.py`** (install check) |
| `run_ai_kernels.py` | **Core runner**: reads a plan CSV, runs gem5 in parallel, parses `stats.txt`, writes one dataset row per run. Refuses to overwrite datasets. |
| `run_ai_kernels_flush.py … flush15.py` | Thin wrappers around `run_ai_kernels.py`, each pointing to one config version. **`run_ai_kernels_flush14.py` = final v9.** Older ones are kept so that every earlier result can be reproduced exactly. |
| `run_ai_kernels_flush6u.py` / `flush14u.py` | Same, for the **unseen** workloads (binary `bin/riscv_flush_u`). `flush14u` = v9 setup. |
| `sim_config.py` | The gem5 configuration: CPU, caches, DRAM, `--repl`, `--trace`. Copied from the original Phase 1 script and extended. |
| `sim_config_flush.py … flush15.py` | One copy per step (flush = true cache flush at ROI). Each adds new `--tensor-*` options. **`sim_config_flush14.py` = v9.** `flush15` adds the rejected first-touch switch. |
| `hifive_unmatched_sim.py` | An early comparison on gem5's HiFive Unmatched board model |
| `baseline_sha256.txt`, `gem5_binary_sha256_before_step5.txt` | Old checksum records. **The baseline file list is now stale**, because those Phase 1 files were deleted. |

### 8.2 `ai_kernels/`: the workloads

| Item | Why it exists |
|---|---|
| `src/*.c` (14 kernels) | `gemm`, `conv2d_nchw`, `conv3d`, `depthwise_conv`, `attention`, `attention_decode`, `embedding`, `lstm`, `gat_aggregate`, `mlp`, `pool`, `chain` (layer chains), `block` (whole blocks), `block_unseen` (the 4 unseen blocks) |
| `include/`, `include_flush/` | Shared helpers: data init, checksum, ROI markers, and the true cache flush |
| `Makefile` | Builds RISC-V and native binaries |
| `bin/riscv_flush/` | **Binaries used by the final results** |
| `bin/riscv_map_flush/` | Map builds (they record tensor addresses for the hint files) |
| `bin/riscv_flush_u/`, `riscv_map_flush_u/` | Same, for the unseen workloads |
| `bin/native*/`, `bin/riscv/`, `bin/riscv_map/` | Native builds (for correctness checks) and older non-flush builds |
| `layers.csv`, `layers_v2/v3/v4.csv` | Layer registry: model, shape, source and citation, added in 4 batches |
| `make_*_plan.py` | Turn layer lists into **experiment plans** (one CSV row = one gem5 run) |
| `experiment_plan_*.csv` (about 160) | One plan per experiment. **Key ones:** `experiment_plan_v9_full.csv` (v9 on 72), `experiment_plan_fit_allrules.csv`, `experiment_plan_layers_allrules.csv`, `experiment_plan_final_blocks_fair.csv`, `experiment_plan_unseen_v9_*.csv` |
| `verify_reference*.py`, `verify_chain.py`, `verify_block.py` | Check the kernels compute correctly (against numpy) |
| `rob_*.csv` | Robustness test plans |
| `README.md` | Kernel notes and citations |

### 8.3 `trace_tools/`: traces, learning, hints, analysis

| Group | Files | Why |
|---|---|---|
| **Address mapping** | `run_map.py`, `run_ptable.py`, `tensor_map_hook.h`, `tensor_roles.py`, `sim_config_ptable*.py`, `flush_env.py`, `flush_env_u.py` | Find which physical address belongs to which tensor (for traces and hint files) |
| **Traces** | `run_all_traces.sh`, `extract_roi.py`, `summarize_traces.py`, `chain_layers.py`, `chain_layers_v2.py`, `protolib.py`, `packet_pb2.py` | Turn raw gem5 traces into labelled per-access tables |
| **OPT / ceilings** | `belady.py`, `opt_plus.py`, `offline_slice_sim.py`, `l2_offline.py` | Compute the perfect rule, how much room exists, and quick offline design tests |
| **Learning** | `build_features.py`, `train_policy.py`, `lomo_replay.py`, `replay_policy.py`, `export_tree.py`; L2 experiments `train_l2_v2.py`, `train_l2_v3.py`, `train_l2_v3s.py`, `export_tree_v3.py` | Features → tree → C++. The L2 trees were tested and **rejected**. |
| **Hints** | `make_hints.py` (single layers), `make_hints_v2.py` (multi-layer and unseen) | Build the hint files v9 reads |
| **Scoreboards / charts** | `scoreboard.py`, `final_check_v9.py`, `final_scoreboard.py`, `plot_final_scoreboard.py`, `plot_final_tables.py`, `unseen_v9_summary.py` | Final comparison tables and images |

### 8.4 `datasets/`: results as CSV (112 files)

One row per gem5 run (status, instructions, cycles, cache hits/misses per level, DRAM bytes, time, host time, …). **Key files:**

| File | Contents |
|---|---|
| `ai_kernels_v9_full.csv` | **v9 on all 72 workloads** |
| `ai_kernels_layers_allrules.csv`, `ai_kernels_fit_allrules.csv`, `ai_kernels_final_blocks_fair.csv`, `ai_kernels_layers_v7_lru.csv`, … | The other rules' runs used in the final scoreboard |
| `final_scoreboard_v2_*.csv` | **The final comparison** |
| `final_check_v9.csv` | The v9 final check |
| `ai_kernels_robust_v8.csv` | Robustness |
| `ai_kernels_repl_v1.csv` | Step 2: 8 standard rules × 65 layers |
| `trace_v1_summary.csv` | OPT vs LRU room per layer |

### 8.5 `results/`: raw outputs (largest folder, about 45 GB)

| Item | Why it exists |
|---|---|
| `<tag>/<run_id>/` | Per-run gem5 output (`stats.txt`, `config.ini`, `simulation.log`) behind each dataset row |
| `trace_v1/`, `trace_check/`, `chain_trace_v1/`, `block_trace_v*/`, `step5_trace_r*/` | Raw traces and labels (about 30 GB). **Needed only to retrain the model.** |
| `features_*/`, `policy_*/` | Training tables and trained models. **`policy_final_v2/` = the final model.** |
| `tensor_hints_layers/`, `tensor_hints_v3/` | **Hint files used by v9's final results** |
| `final_charts_v2/` | **Final images** |
| `gem5_rp_backup_step*/` | gem5 source backups and binaries per step, **including the v9 binary** |
| `launch_*.sh`, `progress*.sh`, `queue_*.sh` | Launchers used to run long batches. **`launch_unseen_v9.sh` = next test.** |
| `*.log` | Run logs |

### 8.6 `report/`

`build_report.py`, `make_figures.py`, `make_comparison_figures.py`, `make_all_results.py` and `update_and_export.ps1` build the Word/PDF report. The old report (`output/Tensor_AI_UROP_Report.docx/.pdf`, 2 Nov, written **before v9**) is kept only in the team's local copy, **not in this repository**, and must be updated. These scripts read the deleted 56k dataset, so they need changing before they're used again.

---

## 9. How to test (teammate)

Everything needed is in **`tensor_cache_testing_kit_v2.zip`**. For the **complete step-by-step setup** (WSL, tools, gem5 + v9, installing v9 into your own gem5, using v9 in your own scripts, troubleshooting), follow **`SETUP_GUIDE.md`**. `README_TESTING.md` is the short version. In short:

1. Linux or WSL2, ≥ 8 GB RAM, about 40 GB of disk. Install the gem5 build packages, Python with pandas, numpy and pyarrow, and `binutils-riscv64-linux-gnu` (RISC-V `readelf`/`nm`, needed only for making hint files in the unseen test).
2. Put the project at `~/gem5-project/arch_study`.
3. `git clone https://github.com/gem5/gem5.git ~/gem5 && cd ~/gem5 && git checkout cbc94c1a77 && git apply ~/gem5-project/arch_study/testing/tensor_aware_v9.patch`
4. Build: `scons build/RISCV/gem5.opt -j2 --linker=gold` (1–2 hours)
5. `python3 testing/setup_test.py`: fixes the paths for your computer **and restores the execute permission** of the workload programs. Unzipping with Windows "Extract All" removes it, and every run would fail with `Permission denied` without this step.
6. **Quick check** (9 runs, about 30 min): LRU, DRRIP and v9 on BERT_ffn1, DLRM_top_mlp and RNNT_enc_pre1. Then `python3 testing/check_smoke_test.py datasets/smoke_test_results.csv` must print **PASS**.

   Expected v9 DRAM: BERT_ffn1 16.92 MB, DLRM_top_mlp 26.21 MB, RNNT_enc_pre1 25.30 MB.
7. Optional: all 72 v9 runs (several hours, ≤ 3 workers).

---

## 10. What comes next

| # | Task | Status | How |
|---|---|---|---|
| 1 | **Teammate reproduces results** | Kit ready | Section 9 |
| 2 | **Unseen-workload test**: 4 blocks v9 never saw (BERT-Base, GPT-2 XL, DLRM batch 128, ResNet-50 stage 3) × 10 rules | Script ready, **not run yet** | Pilot: `bash results/launch_unseen_v9.sh BLOCK_rn50s3`; full: `bash results/launch_unseen_v9.sh` (about 2–3 h). It runs profiling → hint files → checks the hints exist → v9 and 8 rules → summary in `datasets/unseen_v9_summary.csv`. |
| 3 | Smaller-cache study (optional) | Idea | Rerun all rules with L3 = 8 MiB; robustness already shows big v9 gains there |
| 4 | **Final report** | To do | Rewrite with v9 results; Phase 1 can only be described in words (its data was deleted) |
| 5 | Retraining (optional) | Possible | Needs the trace folders in `results/` (kept) |

**Expectations for the unseen test (guesses, not results):** DLRM-128 should be a big win (like BLOCK_dlrm), BERT-Base a small win, ResNet-50 stage 3 a tie (it fits in L3), and GPT-2 XL is the real unknown.

---

## 11. Dos and don'ts

### ✅ Do
- **Build gem5 with `--linker=gold`.** Without it, linking runs out of memory on 8 GB machines.
- Use **`run_ai_kernels_flush14.py`** (v9 setup) for v9 and all comparison rules.
- Save every new result under a **new file name**. The runner refuses to overwrite, which protects old data.
- Run a **pilot on 1 workload** before any long batch.
- Use **at most 3 workers** on an 8 GB machine (Llama runs need about 1.8 GB each).
- Keep the laptop **awake: lid open, charger in**. Sleep pauses or kills runs.
- Keep **identical settings** for every rule (only `--repl` changes) and always include `--l2-pf-on-access 1`.
- Check the hint file exists before a v9 run (the unseen launcher does this automatically).
- Write down exact commands and errors when something fails.
- Keep the **v9 binary backup** (`results/gem5_rp_backup_step8/gem5.opt.step12b_v9`).

### ❌ Don't
- **Don't** edit `sim_config_flush14.py`, `run_ai_kernels_flush14.py` or the gem5 v9 source without telling the team. Make a new copy (e.g. `flush16`) instead.
- **Don't** change cache sizes, the prefetcher or the flush in the plan files if you want to compare with our numbers.
- **Don't** compare a v9 run against another rule's run made with different settings (e.g. without `--l2-pf-on-access 1`).
- **Don't** run `--repl tensor` without a hint file and then call it "v9".
- **Don't** run the gem5 build while long simulations are running (memory). Pause or finish them first.
- **Don't** use `pkill -f` or `pgrep -f` with text that appears in your own command line (it kills your own shell).
- **Don't** delete `results/tensor_hints_*`, `results/policy_final_v2/`, `results/final_charts_v2/`, `datasets/` or the v9 binary.
- **Don't** zip WSL folders with Windows "Send to → Compressed folder"; it fails with error 0x80070006. Zip from Linux instead.
- **Don't** claim results on the unseen workloads until the test has actually been run.

---

## 12. Known limitations

1. **Simulation only.** gem5, not real hardware.
2. **Model-derived kernels**: real layer shapes, sampled work, plain C (no BLAS, no vector instructions). This is not a full MLPerf run.
3. **v9 needs hint files**, made by profiling runs here. A real system would get them from the AI framework.
4. **Gains appear only when data is bigger than the cache and reused**: 18 of 72 workloads here.
5. **One loss:** UNET3D_down2 (0.957 vs BRRIP 0.952), where L3 misses are already nearly compulsory.
6. The robustness numbers are for v8; v9 adds only the set-aware cap.
7. The unseen-workload test hasn't been run yet.
8. Phase 1 (56k baseline) data was deleted, so it can't be re-analysed.

---

## 13. FAQ

**Why are most bars at 1.00?** Those workloads already have the minimum possible DRAM traffic, so no rule (not even OPT) can improve them.

**Is v9 L2 or L3?** Both: one policy at L2 and L3 with different switches, sharing one hint file.

**Why did we use Belady's OPT?** It's the perfect rule. We used its decisions as training labels, and the tree learned to imitate it using only present information. Well-known learned policies (Hawkeye, Glider) use the same approach.

**Why not a bigger ML model?** It has to run inside a cache in hardware, so it must be tiny. Bigger trees and boosted trees overfitted in our tests.

**Why does v9 sometimes have a lower L3 hit rate but less DRAM traffic?** It keeps more useful data in L2, so fewer (and harder) requests reach L3.

**Does v9 make programs faster?** Only slightly (geomean 0.14%), because the O3 core and the prefetcher hide most latency. The benefit is **less DRAM traffic**, which means less energy and less memory bandwidth used.

---

## 14. Glossary

| Word | Meaning |
|---|---|
| ROI | Region of interest (the measured part of the program) |
| Cold mode | Caches flushed at ROI start |
| Geomean | Geometric mean of ratios (fair averaging of "×" values) |
| Plan (CSV) | List of runs; one row = one gem5 simulation |
| Dataset (CSV) | Results; one row = one finished run |
| Hint file | Address → tensor/role/size map read by v9 |
| Working-set plan / packing | Deciding which bytes of which tensors to keep when they don't all fit |
| Set dueling | Leader sets test two rules; the rest follow the winner |
| Dead tensor | A tensor that won't be read again |
| OPT / MIN | Belady's perfect replacement rule |
| DRRIP, SRRIP, BRRIP, SHiP | Standard modern replacement rules (re-reference prediction) |

---

## Appendix A: Detailed step-by-step log (dates, numbers, decisions)

All times are IST. "DRAM vs LRU" = DRAM bytes of the rule ÷ DRAM bytes of LRU on the same workload (lower is better).

### A.1 Workloads and hardware model (1–8 Nov 2026)
- **27 first layers** (ResNet-50, BERT-Large, DLRM-v2, DeepBench), float32, NCHW, batch 1. Shapes come from MLPerf configs, the ResNet paper and DeepBench `gemm_problems.h` / `conv_problems.h`. Each is labelled *"model-derived kernel, not a full benchmark implementation"*.
- **Work sampling:** full tensors are allocated (so footprint and reuse distances are real), but only part of the work units are computed (rows, channels, heads), budget about 20 M MACs.
- **v2 (6–7 Nov):** +19 layers: Llama-3.1-8B, Llama-2-70B, MobileNet-V2, 3D U-Net. New kernels `attention_decode.c`, `depthwise_conv.c`, `conv3d.c`; verified 9/9.
- **v3 + v4 (8 Nov):** + full 32-head Llama 4k decode attention, 6 SDXL layers, 4 RNN-T (new `lstm.c`), 4 R-GAT (new `gat_aggregate.c`), 5 ViT. **Total 65 layers from 12 models.** Mixtral/MoE was dropped (too big, no new cache pattern).
- **Hardware "config F":** modelled on the SiFive P870-D. DDR5-5600 isn't in gem5, so DDR5-4400 is used. L1/L2/L3 = 64K/4M/16M at 4/16/40 cycles, with an L2 stride prefetcher.
- **Metric lesson:** with an O3 core and a prefetcher, gem5's plain "misses" double-count lines already in flight, so **MSHR misses** and **DRAM bytes** are used.
- **Early finding:** RNNT_enc_pre1 has 20 MiB of weights (more than the 16 MiB L3) reread every step, so LRU thrashes (DRAM ≈ 4 × 20 MiB). That made it the first target for a smarter rule.

### A.2 Step 2: eight standard rules (9–10 Nov)
- Added `--repl` (L2 + L3 only; L1 stays LRU). Verified that LRU through the new option is **identical in all 97 columns** to the old runs.
- 8 rules × 65 layers = **520 runs**, all succeeded.
- **BRRIP** cut DRAM on 32 layers (up to −52%) but **raised** it on 16 (DB_gemm_5124: 5.6 → 58.5 MiB). **SHiP-PC** was worst (+56% geomean).
- Picking the best rule per layer would give 1494 MiB in total vs 1637 for LRU. **No single rule wins**, which motivated a tensor-aware rule.
- Run time changed only 0–2%, so **DRAM traffic became the headline metric.**

### A.3 Step 3: traces and OPT room (10–11 Nov)
- Added `--trace l2l3` (CommMonitor + MemTraceProbe in front of L2 and L3). Results are identical to untraced runs.
- Caches see **physical** addresses: `run_ptable.py` (page table via a checkpoint after exit; timing CPU, because the O3 CPU hangs there) plus `run_map.py` (tensor virtual addresses from map builds).
- `extract_roi.py`: labelled ROI-only tables (L2 coverage 99.5–99.96%). `belady.py`: OPT and LRU replay; offline LRU matches gem5 within ±0.2% overall.
- **OPT vs LRU over 65 layers: L2 −13.5%, L3 −9.3%**, mostly **weights**. Biggest L3 room: DLRM_top_mlp (−60%), RNNT_enc_pre1 (−61%), RNNT_enc_post1, LLAMA8B_prefill_qkv, SDXL_ffn_1280.
- **Layer chains** (6 chains) were traced too. 8 rules on chains: best per chain varied (RN50 BRRIP 0.64, RNN-T SHiP 0.92), and BRRIP was 3.4–4.1× worse on BERT/ViT. Again no single winner.

### A.4 Step 4: learning (11 Nov)
- `build_features.py`: 195 M rows; **10 causal features** (only information known at that moment); label = OPT "keep". Verified 0 mismatches against a slow reference loop.
- `train_policy.py`: split **by model**. Test = SDXL, ViT, MobileNet-V2, Llama-70B. A depth-6 tree reached test accuracy L2 82.4% (majority baseline 73.8%) and L3 81.1% (58.4%). Boosted trees overfitted.
- Plain tree use was 1.44× LRU on the ViT chain, so **set dueling vs LRU** and a keep-probability threshold (0.30, tuned on **training runs only**) were added.
- Offline test (20 unseen runs, L3): ours 0.977 vs OPT 0.966, SRRIP 1.025, BRRIP 1.119, with 0 runs > 2% worse.
- Feature selection by leave-one-model-out on training data (`lomo_replay.py`) → final model `policy_final_v2`.

### A.5 Step 5: inside gem5 (11–13 Nov)
- gem5 branch `tensor-aware-rp`: `tensor_aware_rp.{hh,cc}`, generated `tensor_aware_tree.hh` (100% agreement with Python), parameters in `ReplacementPolicies.py`. The rebuilt binary reproduces old LRU/SRRIP stats exactly.
- **Fidelity fixes:** ignore CleanEvict; reset history at the ROI start. Afterwards, gem5 per-access features were **exactly equal** to the offline ones.
- **Bug:** offline L2 streams lacked the L2 prefetcher's fills. Fixed with `belady.py --l2-pf-detect` (counts equal gem5 exactly).
- **Slice protection** (keep a part of a thrashing tensor) brought RNNT_enc_pre1 to 0.608 (gem5 = offline).
- L3-only full run (71 runs): geomean 0.982, worst 1.016, 0 cases > 2% worse.
- **Fair-test fix (12 Nov):** BRRIP's "wins" on RN50_fc, DB_gemv and others came from data left over from before the ROI. Added a **true cache flush** at ROI start (`sim_config_flush.py`). After it, those BRRIP wins were exactly 1.000. All later evaluation uses the flush.
- **User targets set:** never > 2% worse than LRU, ≥ 5% better on average; focus on multi-layer workloads.
- **Multi-layer blocks** built (`block.c`: BERT, ViT, DLRM, RN50, MBV2). OPT on blocks: geomean 0.845; OPT+ (bypass, dirty, dead, coordination) 0.789. **Coordination** had the biggest potential (DLRM 0.41 → 0.28).
- **L3 rule "B+safe":** working-set slice 0.9, skip-first, LRU-default. Results: ViT 1.007, BERT 0.976, RNN-T 0.816, DLRM 0.500.
- **L2:** the learned tree captured only 9–31% of the room offline (BERT 2.3× worse), and dueling at L2 starved the prefetcher. So the L2 tree and L2 dueling were turned off and the L2 working-set slice adopted (hash slice for some workloads, contiguous for others; contiguous if the kept fraction ≥ 0.35).
- **Final L2 + L3 rule of 13 Nov:** DRAM ViT 0.996, BERT 0.975, DLRM 0.481, RNN-T 0.815; L2 misses ≤ LRU on all 4.

### A.6 Step 6: packing + coordination (13–14 Nov)
- Round-2 traces with our rule showed the room left (DLRM 41.2 MB vs OPT 29.4 MB).
- **L3 packing plan + L2/L3 coordination** → "ours_v2": **DLRM 0.377 (better than offline OPT 0.412)**, RNN-T 0.805, BERT 0.976, ViT 0.999, Llama/RN50/MBV2 1.000. Block geomean 0.840 vs best standard DRRIP 0.944.
- L2 packing for all layers was **not** adopted (BERT/DLRM L2 misses went up).

### A.7 Step 7: new L2 work (14 Nov)
- New offline L2 bench (`l2_offline.py`, matches gem5 within 0.5%).
- **Prefetcher fix `--l2-pf-on-access 1`:** the stride prefetcher also trains on hits. LRU is unchanged by it, and **it is applied to every rule** for fairness.
- New learned L2 trees (`train_l2_v3.py`, `train_l2_v3s.py`) all **lost** to packing (the tree broke coordination: DLRM 0.478). Not adopted; kept as evidence.
- **v5** = v2 + pf-on-access + mixed L2 packing (L2 packing only for layers ≤ L3 size): DLRM 0.377, RNN-T 0.805, BERT 0.961, ViT 0.998.

### A.8 Step 8: research ideas (15 Nov)
- **Adopted:** dead-tensor demotion (DLRM 0.377 → 0.361).
- **Rejected:** L3 bypass (2 versions; hurt DLRM), coordination for layers that fit (BERT 0.978, ViT 1.009).
- **Ablation** (each switch off in turn) showed keep-prefetch hurt after the prefetcher fix → **v7** = v5 + dead-demote + keep-pf off: **DLRM 0.340, RNN-T 0.775, BERT 0.9585, ViT 0.9977**, Llama 0.9998, RN50 1.001, MBV2 1.000.
- **v8** (+ position-by-bytes, + plan reused input activations), found by checking all 65 single layers: Llama prefill 0.873 (DRRIP 0.893), SDXL ffn 0.839 (0.860).
- **Robustness** (v7/v8, identical): best or tied in all 12 conditions (section 7.2).

### A.9 v9 and final checks (16–18 Nov)
- **v9** = v8 + set-aware plan (cap 16 per set): UNET3D_l1_conv 0.886 (DRRIP 0.887), down2 0.957 (0.953).
- Rejected after v9: output streaming (UNET3D_out 2% slower), bimodal stream, BRRIP duel, L2 plan-act, first-touch set-aware (0.9928).
- **Final scoreboard v1** (16–17 Nov): fairness enforced (same args, pf-on-access for every rule); 24 extra fair block runs.
- **Fill-in runs** (18 Nov): 266 runs so **all 9 other rules** were simulated on the 38 fit-in-cache layers, not assumed. All 266 succeeded and all were equal, which confirmed the tie claim. Scoreboard v2 + final charts.
- **Handover (19 Nov):** testing kit, meeting brief, unseen-test launcher prepared.
- **Cleanup (21 Nov):** Phase 1 files deleted; full backup `D:\gem5_project_backup\gem5_project_full_backup_2026-11-21.zip` (35.2 GB, every file verified).

---

## Appendix B: How v9 makes one decision (walkthrough)

Example: L3 is full, and a new line of BERT_ffn1's 16 MB weight tensor must come in. One line of the set has to go.

1. **On every access** (`tensor_aware_rp.cc`, around line 485), v9 looks up the physical address in the hint table → tensor 1, role 0 = weight, 16 MB, used by layer 1. For multi-layer workloads, the program counter tells it which layer is running.
2. **It updates per-line state:** last touch time, how many times this tensor was reused, whether the tensor is now *dead* (its last layer has passed), and the position of this line inside the tensor (by bytes).
3. **The plan check:** the weight tensor (16 MB) plus the other reused data doesn't fit. The working-set plan says "keep up to 0.9 × capacity, weights first", and the set-aware cap allows at most 16 planned lines per set. If this line's position falls inside the kept part, it's marked **keep**; otherwise it's **not planned**.
4. **Data that isn't planned:** in sets following v9, the learned tree (`tensor_aware_tree.hh`) gives a keep/evict prediction from the 10 features. In **LRU leader** sets, plain LRU is used, and the duel counter (PSEL) compares misses between the two kinds of leader sets. Followers use LRU **until v9 proves better** (LRU default).
5. **Victim choice** (`getVictim`, `tensor_aware_rp.cc` line 912) works through the 16 lines of the set in this order: (a) an **empty** slot; (b) a line marked **"don't keep"** *or* belonging to a **dead** tensor (equal priority); (c) otherwise the line whose last "keep" decision is **oldest**. Ties go to the least recently touched line.
6. **Coordination (layers > L3):** L3 avoids planning bytes that L2's plan already keeps, so the two levels hold different data.
7. **At L2** the same class runs with different switches: no tree, no dueling, a contiguous or packed working-set plan, and its own prefetch fills recognised as prefetches.

**Result for BERT_ffn1:** the planned 90% of the weights stays in cache and hits on every reuse, instead of LRU throwing them out just before reuse. DRAM: 16.92 MB vs LRU 19.09 MB (0.886).

---

## Appendix C: Key numbers at a glance

| Item | Value |
|---|---|
| Workloads | 72 (65 layers + 7 blocks/chains) from 12 models |
| Rules compared | 10 (LRU, FIFO, Random, TreePLRU, SRRIP, BRRIP, DRRIP, SHiP, SHiP-PC, v9) |
| gem5 runs in the final scoreboard | 720 (72 × 10) |
| v9 DRAM vs LRU (geomean, all 72) | 0.938 |
| v9 on 7 blocks/chains | 0.821 (DRRIP 0.943) |
| v9 on the 11 layers where rules differ | 0.745 (SHiP-PC 0.806, BRRIP 0.807) |
| Verdicts | 9 win / 62 tie / 1 loss (0.5%) |
| Worst v9 vs LRU | 1.0001 |
| Run time geomean | 0.9986 |
| gem5 code added | about 2,070 lines in 9 files (+ 654-line experimental header) |
| Trained model | depth-6 decision tree, 10 features, labels from Belady OPT |
| Final binary | `results/gem5_rp_backup_step8/gem5.opt.step12b_v9` |
| Final runner / config | `run_ai_kernels_flush14.py` / `sim_config_flush14.py` |

---

## Appendix D: Every dataset, and why it was created (all 112 files in `datasets/`)

**Naming rule used everywhere:** `datasets/ai_kernels_X.csv` (results) ↔ `ai_kernels/experiment_plan_X.csv` (what was run) ↔ `results/X/` (raw gem5 output of each run). Once you know one name, you can find the other two. A dataset row = one gem5 run: status, instructions, cycles, IPC, hits, misses and MSHR misses for L1/L2/L3, DRAM bytes read/written, simulated time, host time, memory used, plus the full metadata (model, layer, shape, citation, flags).

"Runs" = number of rows. "Rules": `lru`, the 8 standard rules, `tensor_l3` (our rule at L3 only, early versions) or `tensor` (our rule at L2 + L3).

### D.1 Choosing the hardware model and building the workloads (Phase 2 start, 1–8 Nov)

| Dataset | Runs | Why it was created |
|---|---|---|
| `ai_kernels_pilot_v1` | 27 | First run of the 27 first layers on the baseline-like setup: do the kernels run correctly in gem5, and how long do they take? |
| `ai_kernels_2ghz_ddr5_4400_v1` | 27 | Step towards a modern chip: 2 GHz clock + DDR5-4400 memory |
| `ai_kernels_2ghz_ddr5_4400_open_v1` | 10 | Same, with the open-page DRAM policy (as real DDR5 controllers use) |
| `ai_kernels_o3_2ghz_ddr5_4400_open_v1` | 6 | Same, with the out-of-order (O3) CPU instead of a simple one |
| `ai_kernels_hifive_unmatched_v1` | 6 | Comparison with gem5's built-in model of a real board (SiFive HiFive Unmatched), as a sanity check |
| `ai_kernels_p870d_full_v1` | 6 | The P870-D-like full model (O3, 64K/4M/16M caches) without a prefetcher |
| `ai_kernels_p870d_full_pf_v1` | 27 | **Final hardware model "config F"** (+ L2 stride prefetcher) on the 27 first layers |
| `ai_kernels_v2_p870d_pf_v1` | 19 | +19 layers: Llama-8B/70B, MobileNet-V2, 3D U-Net |
| `ai_kernels_v3_p870d_pf_v1` | 7 | + full 32-head Llama decode attention, + 6 SDXL layers |
| `ai_kernels_v4_p870d_pf_v1` | 13 | + RNN-T, R-GAT, ViT layers. Together: **65 layers, 12 models** |
| `ai_kernels_all46_p870d_pf_v1` | 47 | One combined table of the first 46 layers (made for the old report) |

### D.2 Step 2: standard rules

| Dataset | Runs | Why |
|---|---|---|
| `repl_smoke` | 9 | Check that each of the 9 rules runs through the new `--repl` option |
| `ai_kernels_repl_pilot_v1`, `_v1b` | 1 + 8 | Proof that LRU via `--repl` is identical to before; first BRRIP/DRRIP look |
| **`ai_kernels_repl_v1`** | **520** | **8 standard rules × 65 layers.** The key finding: no single rule wins. |

### D.3 Step 3: traces (recording every L2/L3 access)

| Dataset | Runs | Why |
|---|---|---|
| `ai_kernels_trace_pilot_v1`, `_v2` | 3 + 3 | Pilot of the trace pipeline (v2 was stopped by the user, rc=-15) |
| **`ai_kernels_trace_v1`** | 69 | **Traces of all 65 layers** (4 stopped runs were rerun). Used for OPT labels and training. |
| `trace_v1_summary` | 65 | Per layer: OPT vs LRU room at L2/L3, and offline-vs-gem5 check |
| `chain_smoke` | 6 | First test of the 6 layer chains |
| `ai_kernels_chain_trace_v1` | 6 | Traces of the 6 chains |
| `ai_kernels_repl_chain_v1` | 48 | 8 standard rules × 6 chains |

### D.4 Steps 4–5: our rule enters gem5 (L3 first)

| Dataset | Runs | Why |
|---|---|---|
| `ai_kernels_tensorl3_smoke_final1` | 2 | First gem5 runs of our L3 rule |
| `ai_kernels_tensorl3_smoke_slice05`, `_slice05b` | 2 + 2 | Testing slice protection (0.5) |
| `ai_kernels_tensor_smoke_v3t30`, `ai_kernels_tensor_smoke_final2` | 2 + 2 | First L2 + L3 tests |
| **`ai_kernels_tensorl3_v1`** | 71 | **Full run of the L3-only rule:** geomean 0.982, 0 cases > 2% worse |
| `scoreboard_tensorl3_v1` | 71 | Its scoreboard vs all rules |
| `ai_kernels_l2cmp_A`, `_BC` | 4 + 8 | Choosing the L2 approach: A = L2 LRU, B = Demand-MIN tree, C = L2 slice |
| `ai_kernels_tensorB_v1` | 71 | Full run of L2+L3 rule B (worst 1.029, broke the 2% rule) → L3-only kept for now |
| `scoreboard_tensorB_v1` | 71 | Its scoreboard |
| `ai_kernels_flush_smoke_v1`, `ai_kernels_flush_v1` | 3 + 29 | **Fair re-test with the true cache flush.** It showed that BRRIP's "wins" were leftover data. (Some flush_v1 rows failed (rc=-6) before the snoop-filter fix, then were rerun.) |
| `ai_kernels_block_pilot_v1` | 3 | First BERT block: LRU / SRRIP / BRRIP |
| `ai_kernels_block_trace_v1`, `_v3`, `_v4` | 1 + 6 + 1 | Traces of the 7 multi-layer workloads (v4 = BERT re-traced) |
| `ai_kernels_step4_base_v1` | 5 | Old rule vs BRRIP on blocks (start of Step 4) |
| `ai_kernels_step4_ws_v1` | 3 | Working-set slice + dirty-aware, first try |
| `ai_kernels_step4_AB_v1` | 6 | Options A vs B. **B chosen by the user.** |
| `ai_kernels_step4_B_safety` | 4 | B on the other blocks: ViT 1.179, a safety problem |
| `ai_kernels_step4_C` | 4 | **"B + safe"** (LRU-default): ViT fixed (1.007) |
| `ai_kernels_step5_r0`, `_r0p`, `_r1test` | 3 each | Learned L2 tree, rounds 0/1 |
| `ai_kernels_step5_F_FK`, `_FKA`, `_pfopt`, `_noduel`, `_reads` | 3–6 each | L2 fixes: prefetch-fill detection (F), keep-prefetch (K), duel variants (`_reads` has 3 failed rows from a bad option, kept as a record) |
| `ai_kernels_step5_l2ws`, `_l2wsk`, `_l2wsc`, `_l2hyb` | 4 each | L2 working-set slice: hash, + keep-pf, contiguous, **hybrid (chosen)** |
| `ai_kernels_step5_trace_r1`, `_r2` | 4 each | Traces **with our rule running**, to measure the room left |

### D.5 Steps 6–8: packing, coordination, prefetcher fix, research ideas

| Dataset | Runs | Why |
|---|---|---|
| `ai_kernels_cmp_v1` | 55 | **First full comparison:** 8 standard rules + ours on the 7 blocks |
| `scoreboard_cmp_v1` | 70 | Its scoreboard |
| `ai_kernels_step6_t1`, `_t12`, `_t3`, `_t3b` | 2–7 each | Packing and coordination tests |
| **`ai_kernels_step6_final`** | 7 | **ours_v2** (packing + coordination): DLRM 0.377, better than OPT |
| `scoreboard_step6_v2` | 84 | Its scoreboard |
| `ai_kernels_step7_pfdiag`, `_pfdiag2` | 0 + 4 | Prefetcher diagnosis (pfdiag was empty, so it was redone as pfdiag2) → `--l2-pf-on-access` |
| `ai_kernels_step7_tree` | 5 | New learned L2 trees in gem5 (lost) |
| `ai_kernels_step7_v3safety`, `_v4`, `_v5` | 10 / 4 / 4 | Candidate versions v3–v5 |
| `ai_kernels_step7_night` | 67 | Overnight batch: v5 safety, the fairness re-run of the 8 standard rules **with pf-on-access**, ablation A51–A58 |
| `ai_kernels_ablation_v1` | 24 | Old ablation on v2. Half failed (the binary broke during an OOM link), so it was **superseded** by step7_night. |
| `ai_kernels_step8` | 13 | Research ideas: dead-tensor demotion, bypass |
| `ai_kernels_step8_coordfit` | 4 | Coordination for fitting layers (rejected) |
| `ai_kernels_step8_v6`, `_finalsafe`, **`_v7`** | 4 / 3 / 7 | v6, safety check, **v7** |
| `ai_kernels_robust_v1` | 36 | Robustness: L3 8/32 MiB, 2× work; v7 vs LRU vs DRRIP |
| `ai_kernels_robust_v8` | 12 | Same conditions for v8 (identical to v7) |

### D.6 Steps 9–13: v8 → v9, and rejected ideas

| Dataset | Runs | Why |
|---|---|---|
| `ai_kernels_layers_v7_lru`, `ai_kernels_layers_v7` | 65 + 130 | **All 65 single layers:** LRU, DRRIP and v7. This found the two v8 fixes. |
| `ai_kernels_layers_diag`, `ai_kernels_layers_planact` | 8 + 12 | Diagnosis: position-by-bytes and planning input activations |
| `ai_kernels_step9_pb` | 20 | Testing those two fixes → **v8** |
| `ai_kernels_step9b_unet` | 4 | L2 also plans activations (UNET3D): rejected |
| `ai_kernels_v8_full` | 72 | **v8 on all 72 workloads** |
| `ai_kernels_trace_unet_lru`, `_drrip`, `_ours` | 1 each | Traces to understand the UNET3D loss vs DRRIP |
| `ai_kernels_step10_os` | 34 | Output streaming: rejected (UNET3D_out 2% slower) |
| `ai_kernels_step10b_bip` | 6 | Bimodal streaming: rejected |
| `ai_kernels_step11_sa` | 4 | **Set-aware plan**: kept |
| `ai_kernels_step12_db` | 4 | Duel against BRRIP: rejected |
| `ai_kernels_step12b_sw`, `_sw2` | 6 + 4 | **Set-aware cap = 16 ways** → **v9** |
| **`ai_kernels_v9_full`** | **72** | **v9 on all 72 workloads: the final rule's results** |
| `ai_kernels_v10_pilot`, `_v10b_down2`, `_v10c_ways` | 6 / 1 / 2 | Step 13 "first-touch" ideas after v9: all rejected |

### D.7 Final scoreboard (fair, all 10 rules)

| Dataset | Rows | Why |
|---|---|---|
| `ai_kernels_layers_allrules` | 189 | 7 standard rules × 27 layers with room to differ, with the fair settings |
| `ai_kernels_final_blocks_fair` | 24 | 8 standard rules × 3 blocks that had been missing a fair run |
| `ai_kernels_fit_allrules` | 266 | 7 standard rules × 38 fit-in-cache layers, to **prove** they tie rather than assume it |
| `final_check_v9` | 72 | v9 final check table |
| `final_scoreboard_v1_*` | 72 / 454 / 454 | Scoreboard before the fill-in runs |
| **`final_scoreboard_v2_wide`** | 72 | **Final: 72 workloads × 10 rules, best other rule, verdict** |
| **`final_scoreboard_v2_long`** | 720 | **Every run behind the final scoreboard** |
| **`final_scoreboard_v2_cache_stats`** | 720 | L2/L3 hits and misses, DRAM MB and time per run (behind the table images) |

### D.8 Prepared for the next step

| Dataset | Rows | Why |
|---|---|---|
| `ai_kernels_unseen_lru` | 0 | Empty: the first unseen start (14 Nov) was stopped on hold. The new launcher writes **new** files (`ai_kernels_unseen_v9_lru.csv`, `ai_kernels_unseen_v9_rules.csv`, `unseen_v9_summary.csv`). |

### D.9 Alphabetical index of all dataset files

Every file in `datasets/`, with the table above that explains it.

| File | Explained in |
|---|---|
| `ai_kernels_2ghz_ddr5_4400_open_v1.csv` | D.1 |
| `ai_kernels_2ghz_ddr5_4400_v1.csv` | D.1 |
| `ai_kernels_ablation_v1.csv` | D.5 |
| `ai_kernels_all46_p870d_pf_v1.csv` | D.1 |
| `ai_kernels_block_pilot_v1.csv` | D.4 |
| `ai_kernels_block_trace_v1.csv` | D.4 |
| `ai_kernels_block_trace_v3.csv` | D.4 |
| `ai_kernels_block_trace_v4.csv` | D.4 |
| `ai_kernels_chain_trace_v1.csv` | D.3 |
| `ai_kernels_cmp_v1.csv` | D.5 |
| `ai_kernels_final_blocks_fair.csv` | D.7 |
| `ai_kernels_fit_allrules.csv` | D.7 |
| `ai_kernels_flush_smoke_v1.csv` | D.4 |
| `ai_kernels_flush_v1.csv` | D.4 |
| `ai_kernels_hifive_unmatched_v1.csv` | D.1 |
| `ai_kernels_l2cmp_A.csv` | D.4 |
| `ai_kernels_l2cmp_BC.csv` | D.4 |
| `ai_kernels_layers_allrules.csv` | D.7 |
| `ai_kernels_layers_diag.csv` | D.6 |
| `ai_kernels_layers_planact.csv` | D.6 |
| `ai_kernels_layers_v7.csv` | D.6 |
| `ai_kernels_layers_v7_lru.csv` | D.6 |
| `ai_kernels_o3_2ghz_ddr5_4400_open_v1.csv` | D.1 |
| `ai_kernels_p870d_full_pf_v1.csv` | D.1 |
| `ai_kernels_p870d_full_v1.csv` | D.1 |
| `ai_kernels_pilot_v1.csv` | D.1 |
| `ai_kernels_repl_chain_v1.csv` | D.3 |
| `ai_kernels_repl_pilot_v1.csv` | D.2 |
| `ai_kernels_repl_pilot_v1b.csv` | D.2 |
| `ai_kernels_repl_v1.csv` | D.2 |
| `ai_kernels_robust_v1.csv` | D.5 |
| `ai_kernels_robust_v8.csv` | D.5 |
| `ai_kernels_step10_os.csv` | D.6 |
| `ai_kernels_step10b_bip.csv` | D.6 |
| `ai_kernels_step11_sa.csv` | D.6 |
| `ai_kernels_step12_db.csv` | D.6 |
| `ai_kernels_step12b_sw.csv` | D.6 |
| `ai_kernels_step12b_sw2.csv` | D.6 |
| `ai_kernels_step4_AB_v1.csv` | D.4 |
| `ai_kernels_step4_B_safety.csv` | D.4 |
| `ai_kernels_step4_C.csv` | D.4 |
| `ai_kernels_step4_base_v1.csv` | D.4 |
| `ai_kernels_step4_ws_v1.csv` | D.4 |
| `ai_kernels_step5_FKA.csv` | D.4 |
| `ai_kernels_step5_F_FK.csv` | D.4 |
| `ai_kernels_step5_l2hyb.csv` | D.4 |
| `ai_kernels_step5_l2ws.csv` | D.4 |
| `ai_kernels_step5_l2wsc.csv` | D.4 |
| `ai_kernels_step5_l2wsk.csv` | D.4 |
| `ai_kernels_step5_noduel.csv` | D.4 |
| `ai_kernels_step5_pfopt.csv` | D.4 |
| `ai_kernels_step5_r0.csv` | D.4 |
| `ai_kernels_step5_r0p.csv` | D.4 |
| `ai_kernels_step5_r1test.csv` | D.4 |
| `ai_kernels_step5_reads.csv` | D.4 |
| `ai_kernels_step5_trace_r1.csv` | D.4 |
| `ai_kernels_step5_trace_r2.csv` | D.4 |
| `ai_kernels_step6_final.csv` | D.5 |
| `ai_kernels_step6_t1.csv` | D.5 |
| `ai_kernels_step6_t12.csv` | D.5 |
| `ai_kernels_step6_t3.csv` | D.5 |
| `ai_kernels_step6_t3b.csv` | D.5 |
| `ai_kernels_step7_night.csv` | D.5 |
| `ai_kernels_step7_pfdiag.csv` | D.5 |
| `ai_kernels_step7_pfdiag2.csv` | D.5 |
| `ai_kernels_step7_tree.csv` | D.5 |
| `ai_kernels_step7_v3safety.csv` | D.5 |
| `ai_kernels_step7_v4.csv` | D.5 |
| `ai_kernels_step7_v5.csv` | D.5 |
| `ai_kernels_step8.csv` | D.5 |
| `ai_kernels_step8_coordfit.csv` | D.5 |
| `ai_kernels_step8_finalsafe.csv` | D.5 |
| `ai_kernels_step8_v6.csv` | D.5 |
| `ai_kernels_step8_v7.csv` | D.5 |
| `ai_kernels_step9_pb.csv` | D.6 |
| `ai_kernels_step9b_unet.csv` | D.6 |
| `ai_kernels_tensorB_v1.csv` | D.4 |
| `ai_kernels_tensor_smoke_final2.csv` | D.4 |
| `ai_kernels_tensor_smoke_v3t30.csv` | D.4 |
| `ai_kernels_tensorl3_smoke_final1.csv` | D.4 |
| `ai_kernels_tensorl3_smoke_slice05.csv` | D.4 |
| `ai_kernels_tensorl3_smoke_slice05b.csv` | D.4 |
| `ai_kernels_tensorl3_v1.csv` | D.4 |
| `ai_kernels_trace_pilot_v1.csv` | D.3 |
| `ai_kernels_trace_pilot_v2.csv` | D.3 |
| `ai_kernels_trace_unet_drrip.csv` | D.6 |
| `ai_kernels_trace_unet_lru.csv` | D.6 |
| `ai_kernels_trace_unet_ours.csv` | D.6 |
| `ai_kernels_trace_v1.csv` | D.3 |
| `ai_kernels_unseen_lru.csv` | D.8 |
| `ai_kernels_v10_pilot.csv` | D.6 |
| `ai_kernels_v10b_down2.csv` | D.6 |
| `ai_kernels_v10c_ways.csv` | D.6 |
| `ai_kernels_v2_p870d_pf_v1.csv` | D.1 |
| `ai_kernels_v3_p870d_pf_v1.csv` | D.1 |
| `ai_kernels_v4_p870d_pf_v1.csv` | D.1 |
| `ai_kernels_v8_full.csv` | D.6 |
| `ai_kernels_v9_full.csv` | D.6 |
| `chain_smoke.csv` | D.3 |
| `final_check_v9.csv` | D.7 |
| `final_scoreboard_v1_cache_stats.csv` | D.7 |
| `final_scoreboard_v1_long.csv` | D.7 |
| `final_scoreboard_v1_wide.csv` | D.7 |
| `final_scoreboard_v2_cache_stats.csv` | D.7 |
| `final_scoreboard_v2_long.csv` | D.7 |
| `final_scoreboard_v2_wide.csv` | D.7 |
| `repl_smoke.csv` | D.2 |
| `scoreboard_cmp_v1.csv` | D.5 |
| `scoreboard_step6_v2.csv` | D.5 |
| `scoreboard_tensorB_v1.csv` | D.4 |
| `scoreboard_tensorl3_v1.csv` | D.4 |
| `trace_v1_summary.csv` | D.3 |

---

## Appendix E: Why there are 15 versions of the config and runner

**Rule:** never overwrite a file that produced results. Each new idea got a **new copy** of the config (`sim_config_flushN.py`) with new `--tensor-*` options (each off by default), plus a small runner (`run_ai_kernels_flushN.py`) pointing to it. Any old result can therefore be reproduced exactly with the file that made it.

| Runner / config | Step | What it added | Status |
|---|---|---|---|
| `run_ai_kernels.py` / `sim_config.py` | Phase 2 start | Base runner; config copied from the Phase 1 script, + `--repl`, `--trace`, `--repl tensor`, hints | Used for Steps 1–5 (no flush) |
| `…_flush.py` | Step 1 (fair test) | True cache flush at ROI start | All later work builds on it |
| `…_flush2.py` | Step 6 | Packing, coordination | |
| `…_flush3.py` | Step 7 | `--l2-pf-on-access` (prefetcher sees hits) | In v9 |
| `…_flush4.py` | Step 7 | Learned L2 tree v3 switch | Tree rejected |
| `…_flush5.py` | Step 8 | Dead-tensor demotion, L3 bypass | Dead-demote in v9; bypass rejected |
| `…_flush6.py` | Step 8 | Coordination for fitting layers | Rejected (used for v7) |
| `…_flush6u.py` | Step 8 | Unseen workloads, old setup | Superseded by `flush14u` |
| `…_flush7.py` | Step 9 | Plan input activations | In v9 |
| `…_flush8.py` | Step 9 | Position by tensor bytes | In v9 (= v8 runner) |
| `…_flush9.py` | Step 9b | L2 plans activations | Rejected |
| `…_flush10.py` | Step 10 | Output streaming | Rejected |
| `…_flush11.py` | Step 10b | Bimodal streaming | Rejected |
| `…_flush12.py` | Step 11 | Set-aware plan | In v9 |
| `…_flush13.py` | Step 12 | Duel against BRRIP | Rejected |
| **`…_flush14.py`** | Step 12b | **Set-aware cap** | **FINAL v9, used for every final result** |
| `…_flush14u.py` | Next step | v9 setup for the unseen binaries | For the unseen test |
| `…_flush15.py` | Step 13 | First-touch set-aware | Rejected |

The gem5 binary follows the same rule: each step's binary and source were backed up in `results/gem5_rp_backup_*`. The **v9 binary is `gem5_rp_backup_step8/gem5.opt.step12b_v9`**. Every new switch is **off by default**, so newer binaries reproduce older results exactly.

---

## Appendix F: Plan files and result folders

### F.1 Plan files (`ai_kernels/experiment_plan_*.csv`, about 160)

One plan = the list of gem5 runs of one experiment. Each row holds run_id, layer, model, shape, source/citation, cache sizes, memory size, cold mode and the full `sim_flags`. **Every dataset in Appendix D has a plan with the same name.** Families with no dataset of their own:

| Plan family | Why |
|---|---|
| `abl_a1 … abl_a6`, `s7_a1 … s7_a8` | **Ablation:** each switch turned off one at a time, to prove each part of the rule helps. The s7 results are inside `step7_night`. |
| `s6_a … s6_f`, `s7_*`, `s8_*`, `step6_reg`, `step6_t1reg` | Sub-plans merged into the step6/step7/step8 batches |
| `cmp_ours_rest`, `cmp_std_all` | Parts of the first full comparison (`cmp_v1`) |
| `l2cmp_B`, `l2cmp_C`, `step4_A`, `step4_B`, `step4_base_brrip/_ours`, `step5_F`, `step5_FK` | Single-option parts merged into the combined datasets |
| `layers_v7_ours`, `layers_v7_drrip`, `layers_v7_phase2` | Parts of `layers_v7` |
| `ai_v2_smoke`, `ai_v4_smoke`, `ai_v2_p870d_pf_pilot_v1`, `ai_p870d_full_pf_27_v1`, `chain_v1`, `block_trace_v2` | Smoke tests and pilots before the full runs |
| `repl_pilot_lru_v1`, `trace_unet_*`, `v10_rest`, `v10b_pilot5` | Small pilots and checks |
| `*.UNUSED_*`, `robust_v1.before_*`, `step4_base.UNUSED_missing_slice` | Earlier versions kept (marked UNUSED / before_) instead of being overwritten |
| `v9_full` | **The 72 final v9 runs**, also the template for the testing kit |
| `smoke_test` (created by the kit) | The 9-run reproduction check |
| `unseen_lru`, `unseen_v9_lru`, `unseen_v9_rules` | Unseen test: old (unused), and the new v9 versions (generated by `make_unseen_v9_plan.py`) |

### F.2 Result folders (`results/`, 469 items)

| Folder pattern | What's inside / why |
|---|---|
| `results/<dataset name>/<run_id>_<layer>/` | Raw gem5 output per run: `stats.txt`, `config.ini`, `config.json`, `simulation.log`, `meta.json` |
| `*_maps/` | Tensor address maps and page tables from the profiling runs (inputs to hint files) |
| `trace_v1/`, `trace_check/`, `trace_pilot_*`, `chain_trace_v1/`, `block_trace_v*/`, `step5_trace_r*/` | Raw L2/L3 traces + labels (`*_labels*.parquet`), used for OPT and training |
| `features_*` | Feature tables for training (`features_v1/v2` main; `l2pf`/`l2pfd` = L2 data with the prefetcher handled) |
| `policy_*` | Trained models. **`policy_final_v2` = final.** Others are candidates (v1–v3, the L2 trees l2pf/l2pfd/l2_v2/l2_v3/v3s). `policy_l2_v2_r0.INVALID_old_bert_included` was marked invalid (it included old BERT data). |
| `replay_*`, `lomo_replay_*.csv` | Offline replay tests and leave-one-model-out selection results |
| `tensor_hints_v1/v2/v3/layers` | Hint files. **v3 (blocks/chains) and layers (65 layers) are used by v9.** v1/v2 are older. |
| `gem5_rp_backup_*` | gem5 source and binary backups per step (**v9 binary in `step8`**) |
| `final_charts_v1/v2` | Scoreboard images (**v2 = final**) |
| `step1_validation*`, `step3_validation`, `step5_validation`, `step_pf_validation`, `ai_smoke`, `block_smoke`, `hifive_smoke`, `pffix_smoke` | Checks that each new option left LRU results unchanged |
| `launch_*.sh`, `queue_*.sh`, `progress*.sh` | Launchers and progress watchers for long batches |
| `progress_*.txt`, `ps10_*.txt`, `pv8_*.txt` | Progress snapshots sent to the user during long runs |
| `*.log`, `*.out` | Run logs |
| `*.bak_*`, `*.before_*` | Backups made before editing a script |
| `keep_awake.ps1` | Windows helper that keeps the PC awake during long runs |
