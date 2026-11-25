# Grouped-bar comparison: DRAM traffic vs LRU for 8 standard rules + ours (v9), 15 workloads per image.
import pandas as pd, numpy as np, matplotlib, sys; matplotlib.use('Agg')
TAG=sys.argv[1] if len(sys.argv)>1 else 'v1'
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
W=pd.read_csv('datasets/final_scoreboard_'+TAG+'_wide.csv',index_col=0)
STD=['fifo','random','treeplru','srrip','brrip','drrip','ship','shippc']
NAME={'fifo':'FIFO','random':'Random','treeplru':'TreePLRU','srrip':'SRRIP','brrip':'BRRIP','drrip':'DRRIP','ship':'SHiP','shippc':'SHiP-PC','v9':'Ours (v9)'}
COL=dict(zip(STD,['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']))
COL['v9']='#0b0b0b'
SURF,INK,INK2,GRID='#fcfcfb','#0b0b0b','#52514e','#e4e3df'
fit=W.fits_in_cache.astype(bool)
_F=pd.read_csv('datasets/ai_kernels_v9_full.csv').drop_duplicates('layer_id').set_index('layer_id').footprint_bytes/2**20
_C=pd.read_csv('datasets/final_scoreboard_'+TAG+'_cache_stats.csv'); _D=_C[_C.rule=='lru'].set_index('layer_id').dram_MB
def why(w):
    if not fit[w] or w not in _F: return None
    if _F[w]<=16.0: return 'fits in L3'
    if _D[w]/_F[w]>=1.0: return 'read once'
    return 'small part used'
order=pd.concat([W[W.kind=='block'].sort_index(), W[(W.kind=='layer')&~fit].sort_index(), W[(W.kind=='layer')&fit].sort_index()])
RULES=STD+['v9']; n=len(RULES); bw=0.085; CAP=1.5
pages=[order.iloc[i:i+15] for i in range(0,len(order),15)]
FOCUS=order[(order[RULES].astype(float)-1).abs().max(axis=1)>0.005]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
for pi,P in [('_differ',FOCUS)]+list(enumerate(pages,1)):
    fig,ax=plt.subplots(figsize=(max(22,1.47*len(P)),8.5),dpi=150); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    x=np.arange(len(P))
    for j,r in enumerate(RULES):
        v=P[r].astype(float).values
        xs=x+(j-(n-1)/2)*bw
        ax.bar(xs, np.minimum(np.nan_to_num(v,nan=0),CAP), bw, color=COL[r], edgecolor=SURF, linewidth=1.0, zorder=3)
        for xx,vv in zip(xs,v):
            if vv>CAP: ax.text(xx,CAP+0.01,f"{vv:.2f}↑",ha='center',va='bottom',fontsize=8,color=INK,rotation=90,zorder=6)
    for i,(w,row) in enumerate(P.iterrows()):
        xo=x[i]+(n-1)/2*bw
        ax.text(xo,float(row.v9)+0.03,f"{float(row.v9):.3f}",ha='center',va='bottom',fontsize=8.5,color=INK,rotation=90,zorder=7,
                bbox=dict(boxstyle='square,pad=0.1',fc=SURF,ec='none'))
        if fit[w] and pd.isna(W.loc[w,'fifo']):
            ax.text(x[i]-0.2,0.5,'fits in cache — other rules\nnot simulated (all equal)',ha='center',va='center',fontsize=8,color=INK2,rotation=90,zorder=5)
    ax.axhline(1.0,color=INK,linestyle=(0,(5,3)),linewidth=1.4,zorder=4)
    ax.text(len(P)-0.35,1.0,' LRU\n = 1.00',ha='left',va='center',fontsize=10,color=INK,clip_on=False)
    labels=[f"{w}  ({ {'WIN':'win ✓','TIE':'tie','LOSS':'loss ✗'}[v] if why(w) is None else 'tie: '+why(w) })" for w,v in zip(P.index,P.verdict)]
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=35,ha='right',fontsize=9.5,color=INK)
    top=CAP+0.22
    ax.set_ylim(0,top); ax.set_xlim(-0.6,len(P)-0.4); ax.text(0,1.015,'',transform=ax.transAxes)
    ax.set_ylabel('DRAM traffic vs LRU (lower is better)',color=INK)
    ax.yaxis.grid(True,color=GRID,linewidth=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    if pi=='_differ': ttl=f"The {len(P)} workloads where replacement rules make a difference  (the other {len(order)-len(P)} are identical for all rules)"
    else: ttl=f"DRAM traffic of 9 replacement rules, relative to LRU  —  workloads {15*(pi-1)+1}–{15*(pi-1)+len(P)} of {len(order)}"
    ax.set_title(ttl,loc='left',fontsize=15,color=INK,pad=34)
    ax.text(0,1.015,"gem5 RISC-V, L2 4 MiB + L3 16 MiB, same L2 prefetcher setting for every rule.  Ties = within 0.5% of the best other rule. Bars above 1.5 are cut off; their value is printed (↑)."+("" if pi=='_differ' or not fit.reindex(P.index).any() else "\n'tie: fits in L3' = all data fits in the 16 MiB L3;  'read once' = every byte used once;  'small part used' = only a small part of a big table is touched.  In all three cases the DRAM traffic is already the minimum, so every rule equals LRU."),transform=ax.transAxes,fontsize=10,color=INK2)
    h=[Patch(facecolor=COL[r],label=NAME[r]) for r in RULES]+[Line2D([0],[0],color=INK,linestyle=(0,(5,3)),label='LRU baseline')]
    ax.legend(handles=h,ncol=10,loc='upper left',bbox_to_anchor=(0,-0.30),frameon=False,fontsize=10.5,labelcolor=INK,handlelength=1.6)
    fig.tight_layout(); fn=f'results/final_charts_{TAG}/final_scoreboard_part{pi}.png'; fig.savefig(fn,facecolor=SURF); plt.close(fig); print(fn,len(P))
