"""Render participant-balanced CDFs from the aggregate human comparison JSON."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'docs/analysis/jev-grid-human-distributions-2026-10-27.json'
data=json.loads(PATH.read_text(encoding='utf-8'))
quantity_limit=1.05*max(row['value'] for side in ('BUY','SELL') for curve in data['cdf'][side]['quantity'].values() for row in curve)
styles={'human_raw':('black','-','Human raw'),'human_grid':('grey','--','Human projected'),
    'argmax':('#0072B2','-','Argmax'),'cutoff5':('#D55E00','-','Top 3 / 5%'),'cutoff3':('#009E73','-','Top 3 / 3%')}
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for i,side in enumerate(('BUY','SELL')):
    for j,stage in enumerate(('quantity','price','price')):
        ax=axes[i,j]; xmax=quantity_limit if j==0 else 1000 if j==1 else 100
        for name,(color,style,label) in styles.items():
            rows=data['cdf'][side][stage][name]
            visible=[row for row in rows if row['value']<=xmax]
            xs=[0]+[row['value'] for row in visible]+[xmax]
            ys=[0]+[row['cdf'] for row in visible]+[visible[-1]['cdf'] if visible else 0]
            ax.step(xs,ys,where='post',color=color,linestyle=style,label=label,linewidth=1.5)
        title=('Quantity, observed range' if j==0 else 'Price, full' if j==1 else 'Price, 0–100 zoom')
        ax.set_xlim(0,xmax); ax.set_ylim(0,1.02); ax.grid(alpha=.2); ax.set_title(f'{side} — {title}')
        ax.set_xlabel('units' if j==0 else 'dollars'); ax.set_ylabel('participant-balanced CDF')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=5)
fig.suptitle('Human orders and model samples: price uses common successful pairs')
fig.savefig(ROOT/'docs/analysis/jev-grid-human-distributions-2026-10-27.png',dpi=160)
