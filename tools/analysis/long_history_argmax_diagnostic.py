"""Full-coverage conditional argmax-price diagnostic; no model or source writes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from capclaims.long_history import runner,design
from tools.analysis.compare_jev_grid_human_distributions import weighted,w1
BASE=ROOT/'docs/analysis/jev-long-history-results-2026-10-27'
plan,predictions,status=runner.replay(runner.OUT)
if status['status']=='PARTIAL_TECHNICAL':raise ValueError('partial technical run')
targets=json.loads((runner.OUT/'labels-private.json').read_text(encoding='utf-8'));contexts={(c['person_id'],c['condition']):c for c in plan['contexts']};lookup={(p['person_id'],p['condition'],p['repeat']):p for p in predictions}; output={}
for side in ('BUY','SELL'):
    people=sorted({c['person_id'] for c in plan['contexts'] if c['side']==side and any(o['side']==side.lower() for o in targets[c['person_id']]['orders'])})
    human=weighted([[o['price'] for o in targets[p]['orders'] if o['side']==side.lower()] for p in people]);output[side]={}
    for condition in design.CONDITIONS:
        values=[]
        for p in people:
            repeats=[lookup[(p,condition,r)]['prices']['argmax'] for r in (0,1)]
            if any(x is None for x in repeats):raise ValueError('conditional argmax price lacks full coverage')
            values.append(repeats)
        model=weighted(values);output[side][condition]={'people':len(people),'slots':len(people)*2,'human_mean':sum(v*w for v,w in human),'model_mean':sum(v*w for v,w in model),'raw_w1':w1(human,model),'model_price_above_250_share':sum(w for v,w in model if v>250)}
result={'schema':'jev-long-history-full-conditional-argmax-price-v1','analysis_type':'post_hoc_exploratory','interpretation':'Sequential band argmax then within-selected-band argmax, conditional on the 3% selected quantity; full human-given-side eligible sample. Not a globally modal price or full argmax pipeline.','groups':output}
path=BASE.parent/(BASE.name+'-argmax-diagnostic.json');path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
