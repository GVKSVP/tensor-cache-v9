# Complete setup guide: from a fresh laptop to a finished v9 test

This guide takes you from **nothing installed** to **"PASS: results reproduced"**. Follow the parts in order and copy each command exactly.

- **What the project is and why:** see `PROJECT_GUIDE.md`.
- **Short testing summary:** see `README_TESTING.md`. This file is the long, complete version.

**Tested with:** Windows 11 + WSL 2 (version 3.0.5), Ubuntu 24.04.5 LTS, gcc 13.3, SCons 4.5.2, Python 3.12.3, git 2.43, gem5 commit `cbc94c1a77`.

| Part | What | Time |
|---|---|---|
| 0 | Check your computer | 2 min |
| 1 | Install WSL + Ubuntu (Windows only) | 15 min |
| 2 | Install the tools | 10 min |
| 3 | Put the project in place | 5 min |
| 4 | Get gem5 and add the v9 code (4B: into your own existing gem5) | 10 min |
| 5 | Build gem5, then 5B: check v9 is inside | **1–2 hours** |
| 6 | Prepare the test | 1 min |
| 7 | Quick check (9 runs) | **about 30 min** |
| 8 | (Optional) full 72-workload run | several hours |
| 9 | (Optional, when the team decides) unseen test | 2–3 hours |
| 10 | Troubleshooting | — |
| 11 | Using v9 in your own gem5 scripts | — |

---

## Part 0: Check your computer

| Need | Minimum | Why |
|---|---|---|
| RAM | **8 GB** | gem5 linking and simulations need memory |
| Free disk | **20 GB** (on C: if you use WSL) | gem5 source + build is about 7 GB |
| OS | Windows 10/11 with WSL 2, or Ubuntu 22.04/24.04 | gem5 builds on Linux |
| CPU | Any 64-bit x86 | More cores = faster build |
| Power | **Charger plugged in, lid open** | Sleep pauses or kills long runs |

Already on Linux (Ubuntu)? **Skip Part 1.**

---

## Part 1: Install WSL + Ubuntu (Windows only)

1. Open **PowerShell as Administrator** (Start → type "PowerShell" → right-click → *Run as administrator*).
2. Run:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. **Restart** the computer when asked.
4. After the restart, an Ubuntu window opens. Choose a **username** and **password** (remember the password; `sudo` asks for it).
5. Check it works. In the Ubuntu window:
   ```bash
   lsb_release -ds        # should print: Ubuntu 24.04.x LTS
   ```

From now on, **every command is typed in the Ubuntu (Linux) window**, not in PowerShell.

**Optional but nice:** install **VS Code** on Windows plus its "WSL" extension. Then `code .` in Ubuntu opens the folder, and `.md` files can be previewed with `Ctrl+Shift+V`.

---

## Part 2: Install the tools

```bash
sudo apt update
sudo apt install -y build-essential scons python3-dev git zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
    libboost-all-dev libhdf5-serial-dev libpng-dev libelf-dev libcapstone-dev \
    pkg-config m4 python3-pydot python3-venv python3-tk wget unzip \
    python3-pandas python3-numpy python3-matplotlib \
    binutils binutils-riscv64-linux-gnu
```

| Package group | Why it's needed |
|---|---|
| `build-essential scons python3-dev git zlib… protobuf… libgoogle-perftools-dev m4` | Required to build gem5 |
| `libboost… libhdf5… libpng… libelf… libcapstone… pkg-config python3-pydot` | Optional gem5 features; installing them avoids build warnings |
| `python3-pandas python3-numpy python3-matplotlib` | Our run/check scripts |
| `unzip` | Unzips the kit **keeping file permissions** |
| `binutils-riscv64-linux-gnu` | RISC-V `readelf`/`nm`, used when making hint files (unseen test) |

**Python environment.** The unseen-test scripts use it. Create it once:

```bash
mkdir -p ~/gem5-project
python3 -m venv ~/gem5-project/.venv
~/gem5-project/.venv/bin/pip install pandas numpy pyarrow scikit-learn matplotlib protobuf
```

Check:
```bash
~/gem5-project/.venv/bin/python -c "import pandas, numpy, pyarrow; print('python OK')"
```

---

## Part 3: Put the project in place

The scripts expect **exactly** these folders:

```
~/gem5-project/arch_study/    <- the project (from the zip)
~/gem5/                       <- the gem5 simulator (Part 4)
```

1. Save `tensor_cache_testing_kit_v2.zip` in your Windows **Downloads** folder.
2. In Ubuntu, copy it in and unzip it **with Linux `unzip`**. Don't use Windows "Extract All"; it removes file permissions.
   ```bash
   cd ~
   cp /mnt/c/Users/<YOUR_WINDOWS_NAME>/Downloads/tensor_cache_testing_kit_v2.zip ~/
   unzip -q tensor_cache_testing_kit_v2.zip
   cp -r tensor_cache_project/arch_study ~/gem5-project/
   ls ~/gem5-project/arch_study            # you should see PROJECT_GUIDE.md, testing, ai_kernels, ...
   ```
   Replace `<YOUR_WINDOWS_NAME>` with your Windows user folder name. If you're unsure, run `ls /mnt/c/Users/` to see it.

**Important:** keep the project **inside Linux** (`~/...`), not under `/mnt/c/...`. Running from the Windows drive is very slow, and file permissions break.

---

## Part 4: Get gem5 and add the v9 code

```bash
cd ~
git clone https://github.com/gem5/gem5.git
cd ~/gem5
git checkout cbc94c1a77                 # exact gem5 version used by the project
git checkout -b tensor-aware-rp         # our branch name
git apply ~/gem5-project/arch_study/testing/tensor_aware_v9.patch
git status
```

`git status` must show **6 modified files and 4 new (untracked) files**, all under `src/mem/cache/` (checked on a clean gem5 checkout):

```
 M src/mem/cache/base.cc
 M src/mem/cache/replacement_policies/ReplacementPolicies.py
 M src/mem/cache/replacement_policies/SConscript
 M src/mem/cache/replacement_policies/base.hh
 M src/mem/cache/tags/base.hh
 M src/mem/cache/tags/base_set_assoc.hh
?? src/mem/cache/replacement_policies/tensor_aware_rp.cc
?? src/mem/cache/replacement_policies/tensor_aware_rp.hh
?? src/mem/cache/replacement_policies/tensor_aware_tree.hh
?? src/mem/cache/replacement_policies/tensor_aware_tree_v3.hh
```

| File | What it is |
|---|---|
| `tensor_aware_rp.cc` / `.hh` (new) | **The v9 rule** (954 + 225 lines) |
| `tensor_aware_tree.hh` (new) | The learned decision tree, in C++ |
| `tensor_aware_tree_v3.hh` (new) | Experimental L2 trees (compiled in, not used by v9) |
| `ReplacementPolicies.py`, `SConscript` | Register `TensorAwareRP` and its 36 settings with gem5 |
| `base.cc`, `base.hh`, `tags/base.hh`, `tags/base_set_assoc.hh` | Small hooks (an optional bypass, off by default) |

**Optional, to keep a clean record:** `git add -A && git commit -m "Add TensorAwareRP v9 (UROP tensor-aware caching)"`

**Where the v9 rule is:** `~/gem5/src/mem/cache/replacement_policies/tensor_aware_rp.cc` (main logic) and `tensor_aware_tree.hh` (the learned tree). See `PROJECT_GUIDE.md`, section 5.


### Part 4B: You already have your own gem5? (install v9 into it)

**Recommended:** keep a **separate** gem5 folder just for this project (Part 4 above). It takes 10 more minutes, and your own gem5 stays untouched. The project scripts expect it at `~/gem5`; if your own gem5 is already there, rename yours first (`mv ~/gem5 ~/gem5_mine`).

If you **must** add v9 to an existing gem5 checkout:

```bash
cd /path/to/your/gem5
git status                               # commit or stash your own changes first!
git log --oneline -1                     # which version you have
git checkout -b tensor-aware-rp cbc94c1a77    # safest: same base version as the project
git apply --check ~/gem5-project/arch_study/testing/tensor_aware_v9.patch   # dry run: no output = OK
git apply ~/gem5-project/arch_study/testing/tensor_aware_v9.patch
```

- **Different gem5 version** (you can't switch to `cbc94c1a77`)? Try `git apply --3way …patch`. The changes only touch `src/mem/cache/`, so they usually apply to nearby versions. If parts fail, those files need manual merging. **Results are only guaranteed identical on `cbc94c1a77`.**
- Then build as in Part 5, and **always** run the check in Part 5B.

---

## Part 5: Build gem5 (the long step)

```bash
cd ~/gem5
scons build/RISCV/gem5.opt -j2 --linker=gold
```

| Part of the command | Meaning |
|---|---|
| `build/RISCV/gem5.opt` | Build gem5 for RISC-V (optimised build) |
| `-j2` | 2 parallel jobs. With 16 GB+ RAM, use `-j4`. If it fails with "killed", use `-j1`. |
| `--linker=gold` | **Must have.** The default linker runs out of memory in the last step on 8 GB machines. |

- The first time, SCons may ask questions (e.g. about git hooks). Press **Enter** to accept the defaults.
- It's done when this file exists:
  ```bash
  ls -la ~/gem5/build/RISCV/gem5.opt
  ```
- Quick test that it starts:
  ```bash
  ~/gem5/build/RISCV/gem5.opt --version | head -3
  ```


### Part 5B: Check that v9 is inside your gem5 (1 second, no simulation)

```bash
~/gem5/build/RISCV/gem5.opt --quiet ~/gem5-project/arch_study/testing/check_v9_in_gem5.py
```

| Output | Meaning |
|---|---|
| `PASS: v9 (TensorAwareRP) is installed and v9_policy.py matches the project's v9 runs` | ✅ v9 is compiled in, and its settings equal the project's real v9 runs value for value |
| `FAIL: this gem5 has no TensorAwareRP …` | The patch isn't applied or gem5 wasn't rebuilt: redo Parts 4 and 5 |
| `FAIL: N setting(s) differ` | The gem5 code differs from the project's (wrong patch or version) |

(Checked by us both ways: PASS on the project's v9 gem5, FAIL on an original gem5.)

---

## Part 6: Prepare the test

```bash
cd ~/gem5-project/arch_study
python3 testing/setup_test.py
```

Expected output:
```
made executable: NN files
written: .../ai_kernels/experiment_plan_smoke_test.csv
written: .../ai_kernels/experiment_plan_v9_full_local.csv
all needed files found - ready to run
```

What it does:
1. Restores the **execute permission** of the workload programs (in case the zip was unzipped on Windows).
2. Writes the test plans with **your** folder paths, as new files (it never overwrites).
3. Checks that gem5 and the hint files are in place.

If it says `MISSING: ...`, see Part 10.

---

## Part 7: Quick check (9 runs, about 30 min)

Runs **LRU**, **DRRIP** and **v9** on 3 workloads (BERT_ffn1, DLRM_top_mlp, RNNT_enc_pre1):

```bash
cd ~/gem5-project/arch_study
python3 run_ai_kernels_flush14.py \
    --plan ai_kernels/experiment_plan_smoke_test.csv \
    --out datasets/smoke_test_results.csv \
    --results results/smoke_test \
    --workers 3
```

While it runs you'll see one line per finished run (`SUCCESS ...`). When all 9 are done:

```bash
python3 testing/check_smoke_test.py datasets/smoke_test_results.csv
```

**Expected:** every row says `PASS`, and the last line reads `OVERALL: PASS - results reproduced`.

| Workload | LRU (MB) | DRRIP (MB) | **v9 (MB)** | v9 vs LRU |
|---|---|---|---|---|
| BERT_ffn1 | 19.09 | 17.17 | **16.92** | 0.886 |
| DLRM_top_mlp | 84.01 | 38.77 | **26.21** | 0.312 |
| RNNT_enc_pre1 | 83.04 | 36.60 | **25.30** | 0.305 |

gem5 is deterministic, so the numbers match **very closely**. In our own test from a freshly unzipped kit: LRU and v9 within ±0.01%, DRRIP within −0.4%. Tiny differences come from the project sitting in a different folder path, which shifts memory addresses slightly. The checker allows ±1%.

**Rerunning:** the runner skips runs that already succeeded. To start completely fresh, use **new** names for `--out` and `--results` (e.g. `smoke_test2`).

---

## Part 8 (optional): v9 on all 72 workloads

Several hours. Use at most 3 workers on 8 GB RAM.

```bash
cd ~/gem5-project/arch_study
nohup python3 run_ai_kernels_flush14.py \
    --plan ai_kernels/experiment_plan_v9_full_local.csv \
    --out datasets/my_v9_full.csv \
    --results results/my_v9_full \
    --workers 3 > results/my_v9_full.log 2>&1 &
```

- `nohup … &` keeps it running if you close the terminal. Watch progress with `tail -f results/my_v9_full.log`.
- Compare with our results in `datasets/ai_kernels_v9_full.csv`. The columns `dram_bytes_read` and `dram_bytes_written` should match per `layer_id`.

---

## Part 9 (optional, **only when the team decides**): unseen-workload test

4 blocks v9 has never seen (BERT-Base, GPT-2 XL, DLRM batch 128, ResNet-50 stage 3) × 10 rules. **One command** runs everything in order:

**profiling runs → hint files → check hints exist → v9 + 8 standard rules → summary**

```bash
cd ~/gem5-project/arch_study
bash results/launch_unseen_v9.sh BLOCK_rn50s3      # pilot: 1 workload first
bash results/launch_unseen_v9.sh                   # then all 4 (about 2-3 h)
cat datasets/unseen_v9_summary.csv                 # result: every rule vs LRU
```

It needs the Python environment from Part 2 (`~/gem5-project/.venv`) and `binutils-riscv64-linux-gnu`. If a hint file is missing, the script **stops** instead of running v9 without hints.

---

## Part 10: Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Permission denied` on every run | Zip unzipped on Windows, so the execute bit was lost | Run `python3 testing/setup_test.py` again (it restores it), or `chmod +x ai_kernels/bin/*/*` |
| `setup_test.py`: "Please put the project at …" | Project in the wrong folder | Move it to `~/gem5-project/arch_study` |
| `MISSING: …/gem5.opt` | gem5 not built yet, or in the wrong folder | Part 5; gem5 must be at `~/gem5` |
| `git apply` error | Wrong gem5 version | `cd ~/gem5 && git checkout cbc94c1a77`, then apply again on a clean tree (`git stash` or a fresh clone) |
| Build stops with `Killed` or `ld … memory` | Out of memory | `scons build/RISCV/gem5.opt -j1 --linker=gold` |
| Build error about a missing library | Package not installed | Re-run the Part 2 `apt install` command |
| A run says `FAILED` | See the log | `cat results/smoke_test/<run_id>_<layer>/simulation.log` |
| `cannot open hint file` | Wrong path in a plan | Use the plans written by `setup_test.py` (`*_local.csv`, `smoke_test.csv`), not the originals |
| Numbers differ by > 1% | Wrong gem5 code or runner | `git status` in `~/gem5` (patch applied?), and make sure you used `run_ai_kernels_flush14.py` |
| PC froze / runs vanished | Too many workers, or the laptop slept | `--workers 2`; keep the lid open and charger in; rerun (finished runs are skipped) |
| `python3: No module named pandas` | Python packages missing | `sudo apt install python3-pandas python3-numpy` |
| The `.md` guides look messy | Viewed as plain text | Open in VS Code and press `Ctrl+Shift+V` (preview) |

**Still stuck?** Send the team: the exact command, the last 30 lines of output, and the output of `git -C ~/gem5 status`.

---

## Part 11: Using v9 in your OWN gem5 scripts

The project's experiments use `sim_config_flush14.py`, which switches v9 on with `--repl tensor` plus about 20 `--tensor-*` flags. To use v9 in **any other** gem5 config (your own system, other cache sizes), use the helper **`testing/v9_policy.py`**. It builds the v9 policy with exactly the final settings.

```python
# inside your gem5 config script (e.g. my_config.py), after creating system.l2 and system.l3
import sys, os
sys.path.insert(0, os.path.expanduser("~/gem5-project/arch_study/testing"))
from v9_policy import apply_v9

apply_v9(system.l2, system.l3,
         hint_file="/home/<you>/gem5-project/arch_study/results/tensor_hints_layers/BERT_ffn1.txt",
         l2_size="4MiB", l2_assoc=16, l3_size="16MiB", l3_assoc=16)
```

What `apply_v9` does:

| Cache | Setting |
|---|---|
| L2 | `TensorAwareRP(level=2, …)`: working-set plan, contiguous/packed for layers ≤ L3 size, **no tree, no dueling**, recognises its own prefetch fills |
| L3 | `TensorAwareRP(level=3, …)`: packing plan + **learned tree** + L2/L3 coordination + set-aware cap 16 + dead demotion + **set dueling vs LRU (LRU default)** |
| L2 prefetcher | `StridePrefetcher` with `prefetch_on_access=True`, `prefetch_on_pf_hit=False` (the project used this for **every** rule) |
| L1 | Unchanged (LRU) |

Rules for using v9 yourself:
1. **Every workload needs its own hint file** (which addresses are weights, inputs, outputs). For our 72 workloads they're ready in `results/tensor_hints_layers/` and `results/tensor_hints_v3/`. For a new program, make one with the steps in `PROJECT_GUIDE.md` section 4.3 (profiling run → page table → `make_hints_v2.py`), or follow `results/launch_unseen_v9.sh`.
2. A hint file only fits the **exact binary and memory size** it was made with (physical addresses). Rebuilding the program or changing `--mem-size` needs new hints.
3. `hint_file=""` runs without hints: v9 then behaves almost like LRU. Don't report that as v9.
4. To compare with other rules, change **only** the replacement policy and keep everything else (including the prefetcher setting) identical.
5. For other cache sizes, pass them to `apply_v9`; the set counts and coordination sizes are computed automatically. The tuned values (alpha 0.9, cap 16, …) were chosen for 4 MiB L2 / 16 MiB L3, 16-way.

**Where everything v9-related is:**

| What | Where |
|---|---|
| v9 C++ code | `~/gem5/src/mem/cache/replacement_policies/tensor_aware_rp.cc`, `.hh`, `tensor_aware_tree.hh` |
| v9 registered in gem5 as | `TensorAwareRP` (in `ReplacementPolicies.py`) |
| v9 settings for your scripts | `~/gem5-project/arch_study/testing/v9_policy.py` |
| v9 settings in the project's runner | `sim_config_flush14.py` (`--repl tensor` + flags; see `PROJECT_GUIDE.md` §4.2) |
| Install check | `testing/check_v9_in_gem5.py` |
| All gem5 changes as one file | `testing/tensor_aware_v9.patch` |

---

## Checklist

- [ ] Part 0: 8 GB RAM, 20 GB disk, charger in
- [ ] Part 1: Ubuntu 24.04 in WSL works
- [ ] Part 2: tools and Python environment installed
- [ ] Part 3: project at `~/gem5-project/arch_study` (unzipped with Linux `unzip`)
- [ ] Part 4: gem5 at `cbc94c1a77` + patch applied (6 modified + 4 new files)
- [ ] Part 5: `~/gem5/build/RISCV/gem5.opt` exists
- [ ] Part 5B: `check_v9_in_gem5.py` says **PASS**
- [ ] Part 6: `setup_test.py` says "ready to run"
- [ ] Part 7: checker says **`OVERALL: PASS`**
- [ ] Report the PASS/FAIL table to the team
