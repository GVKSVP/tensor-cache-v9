# Step 7: v9 vs LRU + 8 standard rules, blocks and layers. Read-only; writes new CSV.
import pandas as pd, numpy as np, os
D='datasets/'
tr=lambda d:(d.dram_bytes_read.fillna(0)+d.dram_bytes_written.fillna(0))
rows=[]
# layers
lru=pd.read_csv(D+'ai_kernels_layers_v7_lru.csv'); lru['T']=tr(lru)
al=pd.read_csv(D+'ai_kernels_layers_allrules.csv'); al['T']=tr(al)
v7=pd.read_csv(D+'ai_kernels_layers_v7.csv'); v7['T']=tr(v7)
v9=pd.read_csv(D+'ai_kernels_v9_full.csv'); v9['T']=tr(v9)
rule=lambda d:d.sim_flags.str.extract(r'--repl (\S+)')[0]
for d,r in [(lru,None),(al,None),(v7,None),(v9,None)]: d['rule']=rule(d)
v9['rule']='v9'
L=pd.concat([lru,al,v7[v7.rule=='drrip'],v9[~v9.layer_id.str.match('BLOCK_|CHAIN_')]])
for _,x in L.iterrows(): rows.append((x.layer_id,'layer',x.rule,x['T'],x.sim_ticks))
# blocks
sb=pd.read_csv(D+'scoreboard_cmp_v1.csv')
for _,x in sb.iterrows():
    if x.rule.startswith('tensor') or x.rule.startswith('ours'): continue
    src=pd.read_csv(D+x.source_dataset); t=src[src.run_id==x.run_id].sim_ticks.iloc[0]
    rows.append((x.workload,'block',x.rule,x.dram_MB*1e6,t))
for _,x in v9[v9.layer_id.str.match('BLOCK_|CHAIN_')].iterrows(): rows.append((x.layer_id,'block','v9',x['T'],x.sim_ticks))
R=pd.DataFrame(rows,columns=['workload','kind','rule','dram','ticks'])
P=R.pivot_table(index=['workload','kind'],columns='rule',values='dram',aggfunc='first')
Tk=R.pivot_table(index=['workload','kind'],columns='rule',values='ticks',aggfunc='first')
std=['fifo','random','treeplru','srrip','brrip','drrip','ship','shippc']
out=[]
for idx in P.index:
    p=P.loc[idx]; l=p['lru']
    rec={'workload':idx[0],'kind':idx[1],'v9':p['v9']/l}
    for s in std: rec[s]=p.get(s,np.nan)/l
    b=min(std,key=lambda s:rec[s] if not np.isnan(rec[s]) else 9)
    rec['best_other']=b; rec['best_other_val']=rec[b]
    rec['gap_pct']=(rec['v9']/rec[b]-1)*100
    rec['time_vs_lru']=Tk.loc[idx]['v9']/Tk.loc[idx]['lru']
    out.append(rec)
O=pd.DataFrame(out).round(4)
fn=D+'final_check_v9.csv'; assert not os.path.exists(fn); O.to_csv(fn,index=False)
gm=lambda c:np.exp(np.log(O[c]).mean())
print('missing:',O[std+['v9']].isna().sum().to_dict())
for k in ['block','layer']:
    S=O[O.kind==k]; print(k,len(S),'geomean v9',round(np.exp(np.log(S.v9).mean()),4),{s:round(np.exp(np.log(S[s]).mean()),4) for s in std})
print('geomean all v9',round(gm('v9'),4),{s:round(gm(s),4) for s in std})
print('v9 >2% worse than LRU:',(O.v9>1.02).sum(),' max',O.v9.max())
print('loses >0.5%:');print(O[O.gap_pct>0.5][['workload','v9','best_other','best_other_val','gap_pct']].to_string())
print('ties 0-0.5%:');print(O[(O.gap_pct>0)&(O.gap_pct<=0.5)][['workload','v9','best_other','best_other_val','gap_pct']].to_string())
print('time >1.0 vs LRU:');print(O[O.time_vs_lru>1.0][['workload','v9','time_vs_lru']].to_string())
print('time geomean',round(np.exp(np.log(O.time_vs_lru).mean()),4),'max',O.time_vs_lru.max())
print('best-or-tied count',(O.gap_pct<=0.5).sum(),'/',len(O))
