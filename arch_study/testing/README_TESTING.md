# Testing guide: Tensor-Aware cache replacement rule (v9)

> **New to this? Follow `SETUP_GUIDE.md`**: the complete step-by-step setup from a fresh laptop, including how to put v9 into your own gem5 and use it in your own scripts. This file is the short version.
>
> **Read `PROJECT_GUIDE.md` first** (at the top of the zip and in `arch_study/`). It explains the whole project: what was done, why, where the code is, results, file guide, and dos and don'ts.

This kit lets you rebuild our modified gem5 simulator and re-run our experiments, then check that you get the same numbers we got.

**What v9 is:** a new cache replacement rule for AI workloads. When the cache is full it decides which data to throw out, using knowledge of the AI tensors (weights, activations). It works in the L2 and L3 caches together. We compare it with LRU (the normal rule) and 8 other well-known rules. Less DRAM traffic is better.

**Time needed:** about 1–2 hours to build gem5, then about 30 minutes for the quick check.

---

## Step 0: What you need

- Linux (Ubuntu 22.04/24.04 or WSL2 on Windows), x86-64 PC, **at least 8 GB RAM**, about 40 GB free disk
- Packages (Ubuntu):
  ```bash
  sudo apt install build-essential git m4 scons zlib1g zlib1g-dev libprotobuf-dev \
       protobuf-compiler libprotoc-dev libgoogle-perftools-dev python3-dev \
       python3-pandas python3-numpy python3-matplotlib binutils \
       binutils-riscv64-linux-gnu      # RISC-V readelf/nm: needed for the unseen test's hint files
  ```

- Python environment (the unseen-test scripts use it), created once:
  ```bash
  python3 -m venv ~/gem5-project/.venv
  ~/gem5-project/.venv/bin/pip install pandas numpy pyarrow scikit-learn matplotlib protobuf
  ```

## Step 1: Put the project in the right place

The scripts expect these exact folders in your home directory:

```
~/gem5-project/arch_study/    <- this project (from the zip)
~/gem5/                       <- gem5 simulator (Step 2)
```

Unzip and move it:

```bash
mkdir -p ~/gem5-project
cp -r tensor_cache_project/arch_study ~/gem5-project/
```

## Step 2: Get gem5 and add our changes

```bash
cd ~
git clone https://github.com/gem5/gem5.git
cd gem5
git checkout cbc94c1a77          # the exact gem5 version we used (stable, after v25.1.0.1)
git checkout -b tensor-aware-rp
git apply ~/gem5-project/arch_study/testing/tensor_aware_v9.patch
git status                       # should list 9 changed files + 1 new file in src/mem/cache
```

The patch changes only cache code (`src/mem/cache/...`): the new rule `TensorAwareRP` plus small hooks.

## Step 3: Build gem5 (the long step)

```bash
cd ~/gem5
scons build/RISCV/gem5.opt -j2 --linker=gold
```

- **Use `--linker=gold`.** Without it the final linking step can run out of memory on 8 GB machines.
- `-j2` = 2 parallel jobs. With 16 GB+ RAM you can use `-j4`.
- Done when `~/gem5/build/RISCV/gem5.opt` exists.

## Step 3B: Check that v9 is inside gem5 (1 second)

```bash
~/gem5/build/RISCV/gem5.opt --quiet ~/gem5-project/arch_study/testing/check_v9_in_gem5.py
```

It must print `PASS: v9 (TensorAwareRP) is installed …`.

## Step 4: Prepare the test plans

```bash
cd ~/gem5-project/arch_study
python3 testing/setup_test.py
```

It should end with: `all needed files found - ready to run`.
It also restores the *execute* permission of the workload programs. Unzipping with Windows "Extract All" removes it, and every run would then fail with `Permission denied`.

## Step 5: Quick check (9 runs, about 30 min)

Runs LRU, DRRIP and v9 on 3 workloads:

```bash
cd ~/gem5-project/arch_study
python3 run_ai_kernels_flush14.py \
    --plan ai_kernels/experiment_plan_smoke_test.csv \
    --out datasets/smoke_test_results.csv \
    --results results/smoke_test \
    --workers 3
python3 testing/check_smoke_test.py datasets/smoke_test_results.csv
```

### Expected result

| Workload | LRU (MB) | DRRIP (MB) | **v9 (MB)** | v9 vs LRU |
|---|---|---|---|---|
| BERT_ffn1 | 19.09 | 17.17 | **16.92** | 0.886 |
| DLRM_top_mlp | 84.01 | 38.77 | **26.21** | 0.312 |
| RNNT_enc_pre1 | 83.04 | 36.60 | **25.30** | 0.305 |

The checker prints PASS for each row when your number is within 1% of ours. In our own test from a freshly unzipped kit: LRU and v9 within ±0.01%, DRRIP within −0.4%. The small differences come from the different folder path, which shifts memory addresses slightly.

## Step 6 (optional): Full test of v9 on all 72 workloads

This takes several hours. Use at most 3 workers on an 8 GB machine, because the Llama runs need about 1.8 GB each.

```bash
python3 run_ai_kernels_flush14.py \
    --plan ai_kernels/experiment_plan_v9_full_local.csv \
    --out datasets/my_v9_full.csv \
    --results results/my_v9_full \
    --workers 3
```

Compare with our results in `datasets/ai_kernels_v9_full.csv`. The `dram_bytes_read` and `dram_bytes_written` columns should match.

## Step 7 (optional, only when the team decides): unseen-workload test

4 blocks v9 has never seen (BERT-Base, GPT-2 XL, DLRM batch 128, ResNet-50 stage 3) × 10 rules. One command does everything in order: profiling runs → hint files → check that the hints exist → v9 + 8 standard rules → summary table.

```bash
cd ~/gem5-project/arch_study
bash results/launch_unseen_v9.sh BLOCK_rn50s3     # pilot on 1 workload first
bash results/launch_unseen_v9.sh                  # then all 4 (about 2-3 h)
```

Result: `datasets/unseen_v9_summary.csv` (DRAM traffic of every rule vs LRU).

---

## Where things are

| Folder / file | What it is |
|---|---|
| `PROJECT_GUIDE.md` | **Full project explanation** |
| `SETUP_GUIDE.md` | **Complete step-by-step setup** (WSL, tools, gem5 + v9, own-gem5 install, troubleshooting) |
| `testing/` | This guide, the gem5 patch, setup and checker scripts, `v9_policy.py` (v9 for your own gem5 scripts), `check_v9_in_gem5.py` |
| `results/final_charts_v2/` | Final comparison images (start with `final_scoreboard_part_differ.png`) |
| `datasets/final_scoreboard_v2_wide.csv` | Final table: 72 workloads × 10 rules |
| `datasets/ai_kernels_v9_full.csv` | Our v9 runs (to compare against) |
| `ai_kernels/` | The AI workload programs (pre-built RISC-V binaries in `bin/riscv_flush/`) and plans |
| `results/tensor_hints_v3/`, `results/tensor_hints_layers/` | Hint files v9 reads (which addresses are weights, activations, …) |
| `results/policy_final_v2/` | The trained decision-tree model |
| `run_ai_kernels_flush14.py` + `sim_config_flush14.py` | The runner and gem5 configuration used for all final results |

## Rules

- Write results to **new file names** only. The runner refuses to overwrite existing files.
- Do not edit `sim_config_flush14.py`, `run_ai_kernels_flush14.py` or the gem5 patch. Make a new copy (e.g. `flush16`) for any change, and tell the team.
- Full dos and don'ts: `PROJECT_GUIDE.md`, section 11.

## If something goes wrong

| Problem | Fix |
|---|---|
| `setup_test.py` says the project is in the wrong place | Move it to `~/gem5-project/arch_study` |
| gem5 build stops with "killed" or "ld: memory" | Use `-j1 --linker=gold` |
| A run shows `FAILED` | Look in `results/smoke_test/<run_id>/simulation.log` |
| Runs are very slow or the PC freezes | Use fewer workers (`--workers 2`) |
| Numbers differ by more than 1% | Check `git status` in `~/gem5` (patch applied?) and that you used `run_ai_kernels_flush14.py` |
