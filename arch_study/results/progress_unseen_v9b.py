"""Progress of the testing phase: 49 runs (9 re-check + 40 unseen). Prints a line
at every 5% step, on new non-SUCCESS rows, and when nothing is running."""
import os, subprocess, time
import pandas as pd
A = os.path.expanduser("~/gem5-project/arch_study/")
F = ["datasets/smoke_recheck.csv", "datasets/ai_kernels_unseen_v9_lru.csv", "datasets/ai_kernels_unseen_v9_rules.csv"]
ST = A + "results/progress_unseen_v9b.state"
st = eval(open(ST).read()) if os.path.exists(ST) else [0, 0]
idle = 0
while True:
    ok = bad = 0
    for f in F:
        try:
            d = pd.read_csv(A + f)
            s = set(d[d.status == "SUCCESS"].run_id); ok += len(s)
            bad += len(set(d.run_id) - s)
        except Exception:
            pass
    step = int(100 * ok / 49 // 5) * 5
    if step > st[0]:
        print(f"PROGRESS {step}% ({ok} of 49 runs SUCCESS)", flush=True)
    if bad > st[1]:
        print(f"PROBLEM {bad} run(s) not SUCCESS", flush=True)
    st = [max(step, st[0]), bad]; open(ST, "w").write(repr(st))
    if ok >= 49:
        print("ALL 49 DONE", flush=True); break
    n = subprocess.run(["pgrep", "-x", "gem5.opt"], capture_output=True).stdout
    idle = 0 if n else idle + 1
    if idle == 6:
        print(f"IDLE: no simulation running ({ok} of 49 done)", flush=True); break
    time.sleep(30)
