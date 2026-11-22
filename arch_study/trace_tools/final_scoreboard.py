# Final scoreboard: v9 vs LRU + 8 standard rules, all with the same L2 prefetcher setting
# (--l2-pf-on-access 1), same config F, cold mode. Read-only; writes new files only.
import pandas as pd, numpy as np, os, sys
D='datasets/'; TAG=sys.argv[1] if len(sys.argv)>1 else 'v1'
STD=['fifo','random','treeplru','srrip','brrip','drrip','ship','shippc']
RULES=['lru']+STD+['v9']
BLK=['BLOCK_bert','BLOCK_vit','BLOCK_dlrm','BLOCK_rn50','BLOCK_mbv2','CHAIN_llama_decode','CHAIN_rnnt_encoder']
def load(f,keep=None):
    d=pd.read_csv(D+f); d=d[d.status=='SUCCESS']
    d=d[d.sim_flags.str.contains('--l2-pf-on-access 1')&(d['mode']=='cold')&(d.cache_args=='64KiB 8 4MiB 16 16MiB 16')&~d.sim_flags.str.contains('--trace')]
    d=d.copy(); d['rule']=d.sim_flags.str.extract(r'--repl (\S+)')[0]; d['src']=f
    return d if keep is None else d[d.rule.isin(keep)]
v9=load('ai_kernels_v9_full.csv'); v9['rule']='v9'
args={r.layer_id:r.args for r in v9.itertuples()}
parts=[v9,
 load('ai_kernels_layers_v7_lru.csv',['lru']), load('ai_kernels_layers_allrules.csv',STD),
 load('ai_kernels_layers_v7.csv',['drrip']),
 load('ai_kernels_step7_pfdiag2.csv',['lru']), load('ai_kernels_step7_v3safety.csv',['lru']),
 load('ai_kernels_step7_night.csv',STD)]
if os.path.exists(D+'ai_kernels_final_blocks_fair.csv'): parts.append(load('ai_kernels_final_blocks_fair.csv',STD))
if TAG!='v1' and os.path.exists(D+'ai_kernels_fit_allrules.csv'): parts.append(load('ai_kernels_fit_allrules.csv',STD))
A=pd.concat(parts)
A=A[A.apply(lambda r:args.get(r.layer_id)==r.args,axis=1)]          # same work size as v9
A=A[~(A.layer_id.isin(BLK)&(A.src=='ai_kernels_layers_v7.csv'))]
A=A.drop_duplicates(['layer_id','rule'],keep='first')
A['dram_MB']=(A.dram_bytes_read+A.dram_bytes_written)/1e6
hr=lambda h,m:h/(h+m)
A['l2_hit_rate']=hr(A.l2_hits,A.l2_misses); A['l3_hit_rate']=hr(A.l3_hits,A.l3_misses)
lru=A[A.rule=='lru'].set_index('layer_id')
A['dram_vs_lru']=A.dram_MB/A.layer_id.map(lru.dram_MB)
A['time_vs_lru']=A.sim_ticks/A.layer_id.map(lru.sim_ticks)
A['kind']=np.where(A.layer_id.isin(BLK),'block','layer')
A['model']=A.layer_id.str.replace('BLOCK_|CHAIN_','',regex=True).str.split('_').str[0].str.upper()
long=A[['layer_id','kind','model','rule','dram_MB','dram_vs_lru','l2_hit_rate','l3_hit_rate','time_vs_lru','sim_ticks','run_id','src']].sort_values(['kind','layer_id','rule'])
W=long.pivot(index='layer_id',columns='rule',values='dram_vs_lru')[RULES]
W.insert(0,'kind',long.groupby('layer_id').kind.first())
_V=W[RULES].astype(float); fits=(((_V-1).abs()<0.005)|_V.isna()).all(axis=1)&(W.kind=='layer')   # all measured rules within 0.5% of LRU
W['fits_in_cache']=fits
others=[r for r in RULES if r!='v9']
best=W[STD].astype(float).idxmin(axis=1,skipna=True) if False else W[STD].astype(float).apply(lambda r:r.idxmin() if r.notna().any() else '-',axis=1)
W['best_other']=np.where(fits,'(all equal)',best)
W['best_other_vs_lru']=np.where(fits,1.0,W[STD].astype(float).min(axis=1))
W['gap_pct']=(W.v9/W.best_other_vs_lru-1)*100
W['gap_pct']=W.gap_pct.astype(float)
W['verdict']=np.select([W.gap_pct.isna(),W.gap_pct<-0.5,W.gap_pct<=0.5],['PENDING','WIN','TIE'],'LOSS')
W['v9_time_vs_lru']=long[long.rule=='v9'].set_index('layer_id').time_vs_lru.astype(float)
W['v9_l3_hit']=long[long.rule=='v9'].set_index('layer_id').l3_hit_rate
W['lru_l3_hit']=long[long.rule=='lru'].set_index('layer_id').l3_hit_rate
for fn,d in [(f'final_scoreboard_{TAG}_long.csv',long),(f'final_scoreboard_{TAG}_wide.csv',W.round(4))]:
    assert not os.path.exists(D+fn),fn; d.to_csv(D+fn,index=('wide' in fn))
G=W.copy()
for r in STD: G.loc[G.fits_in_cache&G[r].isna(),r]=1.0      # fit-in-cache: all rules equal (LRU=DRRIP=v9 measured 1.000)
gm=lambda s:float(np.exp(np.log(s.dropna().astype(float)).mean()))
print('missing cells:',int(G[RULES].isna().sum().sum()))
for name,S in [('ALL 72',G),('7 blocks',G[G.kind=='block']),(f"{int(((G.kind=='layer')&~G.fits_in_cache).sum())} headroom layers",G[(G.kind=='layer')&~G.fits_in_cache])]:
    print(f'{name:20s}',' '.join(f'{r}={gm(S[r]):.3f}' for r in RULES))
print(W.verdict.value_counts().to_dict(),' v9>2% worse than LRU:',int((W.v9>1.02).sum()),' max v9',W.v9.max())
print('time geomean',round(gm(W.v9_time_vs_lru),4),' slower than LRU:',{k:round(float(x),4) for k,x in W[W.v9_time_vs_lru>1].v9_time_vs_lru.items()})
print(W[(W.verdict!='WIN')&~W.fits_in_cache.astype(bool)][['v9','best_other','best_other_vs_lru','gap_pct','verdict']].to_string())
print('WINS:');print(W[W.verdict=='WIN'][['v9','best_other','best_other_vs_lru','gap_pct']].to_string())
