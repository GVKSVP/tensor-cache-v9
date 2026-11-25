#!/usr/bin/env python3
"""Comparison chart + tables for the 4 UNSEEN workloads x 10 rules (frozen v9).
Reads ai_kernels_unseen_v9_{lru,rules}.csv; writes NEW files only."""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
A=os.path.expanduser('~/gem5-project/arch_study/'); O=A+'results/unseen_charts_v9/'; LONG=A+'datasets/unseen_v9_cache_stats.csv'
for p in (O,LONG):
    if os.path.exists(p): sys.exit(f'ERROR: {p} exists')
os.makedirs(O)
STD=['fifo','random','treeplru','srrip','brrip','drrip','ship','shippc']; RULES=['lru']+STD+['v9']
NAME=dict(zip(RULES,['LRU','FIFO','Random','TreePLRU','SRRIP','BRRIP','DRRIP','SHiP','SHiP-PC','Ours (v9)']))
COL=dict(zip(STD,['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948'])); COL['v9']='#0b0b0b'
WL=dict(BLOCK_gpt2xl='GPT-2 XL block',BLOCK_dlrm128='DLRM batch 128 *',BLOCK_bertbase='BERT-Base block',BLOCK_rn50s3='ResNet-50 stage 3')
d=pd.concat([pd.read_csv(A+f'datasets/ai_kernels_unseen_v9_{x}.csv') for x in ('lru','rules')])
assert len(d)==40 and (d.status=='SUCCESS').all()
d['rule']=d.run_id.str.split('_').str[1]
d['dram_MB']=(d.dram_bytes_read+d.dram_bytes_written)/1e6
for l in ('l2','l3'): d[l+'_hit']=100*d[l+'_hits']/(d[l+'_hits']+d[l+'_misses'])
d['l3_miss_k']=d.l3_misses/1e3; d['time_ms']=d.sim_seconds*1e3
P=lambda c:d.pivot(index='layer_id',columns='rule',values=c).loc[list(WL),RULES]
M=P('dram_MB'); V=M.div(M.lru,axis=0); G=np.exp(np.log(V).mean())
d['dram_vs_lru']=d.dram_MB/d.layer_id.map(M.lru)
d[['layer_id','rule','dram_MB','dram_vs_lru','l2_hits','l2_misses','l2_hit','l3_hits','l3_misses','l3_hit','time_ms','run_id']].to_csv(LONG,index=False)
S,I,I2,GR='#fcfcfb','#0b0b0b','#52514e','#e4e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
# chart
fig,ax=plt.subplots(figsize=(15,7),dpi=150); fig.patch.set_facecolor(S); ax.set_facecolor(S)
R=STD+['v9']; x=np.arange(4); bw=.092
for j,r in enumerate(R):
    xs=x+(j-4)*bw; v=V[r].values
    ax.bar(xs,np.minimum(v,1.5),bw,color=COL[r],edgecolor=S,linewidth=1,zorder=3)
    for a,b in zip(xs,v):
        if b>1.5: ax.text(a,1.51,f'{b:.2f}↑',rotation=90,ha='center',va='bottom',fontsize=9)
        elif r=='v9': ax.text(a,b+.02,f'{b:.3f}',rotation=90,ha='center',va='bottom',fontsize=9.5,weight='bold',bbox=dict(fc=S,ec='none',pad=.5),zorder=6)
ax.axhline(1,color=I,ls=(0,(5,3)),lw=1.3,zorder=4); ax.text(3.5,1,' LRU = 1.00',va='center',fontsize=9.5,clip_on=False)
ax.set_xticks(x); ax.set_xticklabels(WL.values(),fontsize=12); ax.set_ylim(0,1.72); ax.set_xlim(-.5,3.5)
ax.set_ylabel('DRAM traffic vs LRU (lower is better)'); ax.yaxis.grid(True,color=GR,zorder=0); ax.set_axisbelow(True)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.set_title('Unseen workloads: DRAM traffic of 9 replacement rules relative to LRU',loc='left',fontsize=15,pad=26)
ax.text(0,1.02,'v9 frozen before the test; same settings for every rule. Bars above 1.5 are cut off (value printed).  * close to the DLRM used in tuning.',transform=ax.transAxes,fontsize=9.5,color=I2)
ax.legend(handles=[Patch(color=COL[r],label=NAME[r]) for r in R]+[Line2D([0],[0],color=I,ls=(0,(5,3)),label='LRU')],ncol=10,loc='upper left',bbox_to_anchor=(0,-.08),frameon=False,fontsize=10,handlelength=1.2,columnspacing=1.1)
fig.tight_layout(); fig.savefig(O+'unseen_dram_chart.png',facecolor=S); plt.close(fig)
# tables
VG=V.copy(); VG.loc['Geomean']=G
TB=[('1. DRAM traffic (MB): lower is better',M,2,1),('2. DRAM traffic vs LRU: lower is better',VG,4,1),('3. L2 hit rate (%): higher is better',P('l2_hit'),1,0),('4. L3 hit rate (%): higher is better',P('l3_hit'),1,0),('5. L3 misses (thousands): lower is better',P('l3_miss_k'),1,1),('6. Run time (ms, simulated): lower is better',P('time_ms'),3,1)]
fig,axs=plt.subplots(6,1,figsize=(17,15),dpi=140,gridspec_kw=dict(height_ratios=[5,6,5,5,5,5])); fig.patch.set_facecolor(S)
for ax,(t,X,n,low) in zip(axs,TB):
    ax.axis('off'); ax.set_title(t,loc='left',fontsize=12.5,weight='bold')
    X=X.round(n); best=X.min(axis=1) if low else X.max(axis=1)
    T=ax.table(cellText=[[WL.get(w,w)]+[f'{q:.{n}f}' for q in X.loc[w]] for w in X.index],colLabels=['Workload']+[NAME[r] for r in RULES],colWidths=[.16]+[.084]*10,cellLoc='center',bbox=[0,0,1,1])
    T.auto_set_font_size(False); T.set_fontsize(10.5)
    for (i,j),c in T.get_celld().items():
        c.set_edgecolor('#d8d7d2'); c.set_facecolor('#efeeea' if i==0 else '#e8effb' if j==10 else S)
        if i==0: c.set_text_props(weight='bold')
        elif j==0: c.set_text_props(ha='left')
        elif X.iloc[i-1,j-1]==best.iloc[i-1] and X.iloc[i-1].nunique()>1: c.set_text_props(weight='bold')
fig.suptitle('Unseen workloads: cache statistics of all 10 rules',x=.02,ha='left',fontsize=17,y=.995)
fig.text(.02,.969,'gem5 RISC-V O3, L2 4 MiB + L3 16 MiB, identical settings. Bold = best shown value in the row (none if all equal). Hit rates use gem5 plain hits/misses.',fontsize=10.5,color=I2)
fig.tight_layout(rect=(0,0,1,.965)); fig.savefig(O+'unseen_tables.png',facecolor=S)
print(V.round(4).to_string()); print(G.round(4).to_dict()); print('geomean without DLRM',np.exp(np.log(V.drop('BLOCK_dlrm128')).mean()).round(4).to_dict())
print(M.round(2).to_string()); print(P('l3_hit').round(1).to_string()); print(P('l2_hit').round(1).to_string())
