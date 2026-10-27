"""Rebuild aggregate CDFs on the frozen two-history common-success mask."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from capclaims.long_history import runner,design
from tools.analysis.compare_jev_grid_human_distributions import weighted,w1,project
BASE=ROOT/'docs/analysis/jev-long-history-results-2026-10-27'

def build(plan,predictions,targets,reference):
    contexts={(c['person_id'],c['condition']):c for c in plan['contexts']}; lookup={(p['person_id'],p['condition'],p['repeat']):p for p in predictions};people=sorted({c['person_id'] for c in plan['contexts']});output={}
    for side in ('BUY','SELL'):
        eligible={p:[o for o in targets[p]['orders'] if o['side']==side.lower()] for p in people if contexts[p,'short3']['side']==side};eligible={p:o for p,o in eligible.items() if o};output[side]={}
        for stage,field in (('quantity','qty'),('price','price')):
            key='quantities' if stage=='quantity' else 'prices';mask={p:[r for r in (0,1) if all((p,c,r) in lookup and lookup[(p,c,r)][key]['cutoff3'] is not None for c in design.CONDITIONS)] for p in eligible};selected=sorted(p for p in eligible if mask[p]);metric=reference['groups'][side]['metrics'][stage]
            if len(selected)!=metric['common_success_people'] or sum(len(mask[p]) for p in selected)!=metric['common_success_slots']:raise ValueError('common mask mismatch')
            curves={}
            if selected:
                raw=weighted([[o[field] for o in eligible[p]] for p in selected]);grid=weighted([[project(o['qty'],o['price'],side,contexts[p,'short3']['current_state']['Q'])[0 if stage=='quantity' else 1] for o in eligible[p]] for p in selected]);points={'human_raw':raw,'human_grid':grid}
                for cond in design.CONDITIONS:
                    samples=weighted([[lookup[(p,cond,r)][key]['cutoff3'] for r in mask[p]] for p in selected]);expected=metric['conditions'][cond]
                    if abs(w1(raw,samples)-expected['raw_w1'])>1e-9 or abs(w1(grid,samples)-expected['projected_w1'])>1e-9:raise ValueError('frozen W1 mismatch')
                    points[cond]=samples
                for label,values in points.items():
                    xs=sorted({v for v,w in values});curves[label]={'weighted_mean':sum(v*w for v,w in values),'cdf':[{'value':x,'cdf':sum(w for v,w in values if v<=x)} for x in xs]}
            output[side][stage]={'curves':curves,'common_success_people':len(selected),'common_success_slots':metric['common_success_slots']}
    return {'schema':'jev-long-history-cdf-v1','groups':output,'interpretation':'3% full pipeline, participant/order-or-repeat weights, identical two-history common-success masks as frozen W1.'}

def main():
    reference=json.loads(BASE.with_suffix('.json').read_text(encoding='utf-8'));plan,predictions,status=runner.replay(runner.OUT)
    if status['status']=='PARTIAL_TECHNICAL':raise ValueError('technical partial run')
    targets=json.loads((runner.OUT/'labels-private.json').read_text(encoding='utf-8'));out=build(plan,predictions,targets,reference)
    output=BASE.parent/(BASE.name+'-cdf.json');output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':'REPLAY_VALID','output':str(output)}))
if __name__=='__main__':main()
