#!/usr/bin/env python3
"""Compare the 9 test runs with the results measured on the original computer."""
import pandas as pd, sys
f = sys.argv[1] if len(sys.argv) > 1 else "datasets/smoke_test_results.csv"
d = pd.read_csv(f)
EXP = {  # DRAM MB (read + written), original computer
 ("lru","BERT_ffn1"):19.092,("lru","DLRM_top_mlp"):84.009,("lru","RNNT_enc_pre1"):83.045,
 ("drrip","BERT_ffn1"):17.168,("drrip","DLRM_top_mlp"):38.766,("drrip","RNNT_enc_pre1"):36.602,
 ("v9","BERT_ffn1"):16.923,("v9","DLRM_top_mlp"):26.207,("v9","RNNT_enc_pre1"):25.300}
ok = True; rows = []
for _, r in d.iterrows():
    rule = r.run_id.split("_")[1]; key = (rule, r.layer_id)
    got = (r.dram_bytes_read + r.dram_bytes_written) / 1e6; exp = EXP[key]
    diff = 100 * (got - exp) / exp; good = r.status == "SUCCESS" and abs(diff) <= 1.0
    ok &= good; rows.append((r.layer_id, rule, round(exp, 3), round(got, 3), f"{diff:+.2f}%", "PASS" if good else "FAIL"))
print(pd.DataFrame(rows, columns=["layer", "rule", "expected_MB", "your_MB", "diff", "result"]).sort_values(["layer","rule"]).to_string(index=False))
print("\nruns found:", len(d), "of 9")
print("OVERALL:", "PASS - results reproduced" if ok and len(d) == 9 else "NOT YET / FAIL - see rows above")
