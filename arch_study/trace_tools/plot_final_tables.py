# Table images: per workload x rule values (DRAM MB, L2/L3 hit rate, L3 misses, run time), 15 workloads per image.
import pandas as pd, numpy as np, matplotlib, sys; matplotlib.use('Agg')
TAG=sys.argv[1] if len(sys.argv)>1 else 'v1'
import matplotlib.pyplot as plt
L=pd.read_csv('datasets/final_scoreboard_'+TAG+'_long.csv'); W=pd.read_csv('datasets/final_scoreboard_'+TAG+'_wide.csv',index_col=0)
rows=[]
for src,g in L.groupby('src'):
    d=pd.read_csv('datasets/'+src).set_index('run_id')
    m=d.loc[g.run_id,['l2_hits','l2_misses','l3_hits','l3_misses','sim_seconds']].reset_index()
    m['layer_id']=g.layer_id.values; m['rule']=g.rule.values; m['dram_MB']=g.dram_MB.values; rows.append(m)
M=pd.concat(rows)
M['l2_hit']=M.l2_hits/(M.l2_hits+M.l2_misses)*100; M['l3_hit']=M.l3_hits/(M.l3_hits+M.l3_misses)*100
M['l3_miss_k']=M.l3_misses/1000; M['time_ms']=M.sim_seconds*1000
out='datasets/final_scoreboard_'+TAG+'_cache_stats.csv'; import os; assert not os.path.exists(out)
M.drop(columns='run_id').to_csv(out,index=False)
RULES=['lru','fifo','random','treeplru','srrip','brrip','drrip','ship','shippc','v9']
NAME={'lru':'LRU','fifo':'FIFO','random':'Random','treeplru':'TreePLRU','srrip':'SRRIP','brrip':'BRRIP','drrip':'DRRIP','ship':'SHiP','shippc':'SHiP-PC','v9':'Ours (v9)'}
MET=[('dram_MB','1. DRAM traffic (MB) — lower is better','{:.2f}',min),
     ('l2_hit','2. L2 hit rate (%) — higher is better','{:.1f}',max),
     ('l3_hit','3. L3 hit rate (%) — higher is better','{:.1f}',max),
     ('l3_miss_k','4. L3 misses (thousands) — lower is better','{:.1f}',min),
     ('time_ms','5. Run time (ms, simulated) — lower is better','{:.3f}',min)]
fit=W.fits_in_cache.astype(bool)
order=list(W[W.kind=='block'].sort_index().index)+list(W[(W.kind=='layer')&~fit].sort_index().index)+list(W[(W.kind=='layer')&fit].sort_index().index)
SURF,INK,INK2,HEAD,OURS,LINE='#fcfcfb','#0b0b0b','#52514e','#f0efec','#e8f0fb','#d9d8d3'
os.makedirs('results/final_charts_'+TAG+'',exist_ok=True)
for pi,i0 in enumerate(range(0,len(order),15),1):
    ws=order[i0:i0+15]; nr=len(ws)
    fig=plt.figure(figsize=(20,4.2+5*(nr+3)*0.36),dpi=130); fig.patch.set_facecolor(SURF)
    fig.text(0.02,0.995,f"Cache statistics per workload and replacement rule — workloads {i0+1}–{i0+nr} of 72",fontsize=16,color=INK,va='top')
    fig.text(0.02,0.982,"gem5 RISC-V, L2 4 MiB + L3 16 MiB, same L2 prefetcher setting for every rule. Bold = best in the row. '–' = not simulated (all rules identical). Ours highlighted.",fontsize=10.5,color=INK2,va='top')
    hh=1/len(MET)
    for k,(col,title,fmt,best) in enumerate(MET):
        ax=fig.add_axes([0.02,0.97-(k+1)*hh*0.965,0.96,hh*0.965-0.006]); ax.axis('off')
        P=M[M.layer_id.isin(ws)].pivot(index='layer_id',columns='rule',values=col).reindex(index=ws,columns=RULES)
        cells=[[w]+[fmt.format(v) if pd.notna(v) else '–' for v in P.loc[w]] for w in ws]
        t=ax.table(cellText=cells,colLabels=['Workload']+[NAME[r] for r in RULES],colWidths=[0.17]+[0.083]*10,loc='upper center',cellLoc='center',bbox=[0,0,1,0.90])
        t.auto_set_font_size(False); t.set_fontsize(9.5)
        ax.text(0,0.97,title,fontsize=12.5,color=INK,weight='bold',transform=ax.transAxes,va='top')
        for (r,c),cell in t.get_celld().items():
            cell.set_edgecolor(LINE); cell.set_linewidth(0.6); cell.get_text().set_color(INK)
            if r==0: cell.set_facecolor(HEAD); cell.get_text().set_weight('bold')
            elif c==0: cell.set_facecolor(SURF); cell.get_text().set_color(INK2); cell.get_text().set_ha('left'); cell.PAD=0.03
            else: cell.set_facecolor(OURS if RULES[c-1]=='v9' else SURF)
            if r>0 and c>=1:
                vals=P.loc[ws[r-1]].astype(float); v=vals.iloc[c-1]
                if pd.notna(v) and vals.notna().sum()>1 and np.isclose(v,best(vals.dropna()),rtol=1e-4) and not np.allclose(vals.dropna(),vals.dropna().iloc[0],rtol=1e-4):
                    cell.get_text().set_weight('bold')
    fn=f'results/final_charts_{TAG}/cache_stats_part{pi}.png'; fig.savefig(fn,facecolor=SURF); plt.close(fig); print(fn,nr)
