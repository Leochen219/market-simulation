"""Regenerate aggregate CDFs for the frozen expanded-grid common-mask estimand."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from capclaims.expanded_grid import runner,design,analysis
from tools.analysis.compare_jev_grid_human_distributions import project,w1
BASE=ROOT/'docs/analysis/jev-expanded-grid-results-2026-10-27'

def curves(plan,predictions,targets,reference):
    contexts={(c['person_id'],c['formal_round']):c for c in plan['contexts']}; lookup={(p['person_id'],p['formal_round'],p['repeat']):p for p in predictions}
    eligible={}
    for key,c in contexts.items():
        same=[o for o in targets[design.coordinate(*key)]['orders'] if o['side']==c['side'].lower()]
        if same: eligible[key]=same
    groups={}
    for group,predicate in [('BUY',lambda c:c['side']=='BUY'),('SELL',lambda c:c['side']=='SELL'),('shortage',lambda c:c['stratum']=='shortage'),('surplus',lambda c:c['stratum']=='surplus')]:
        keys=[k for k in eligible if predicate(contexts[k])]; groups[group]={}
        for stage in ('quantity','price'):
            name='quantities' if stage=='quantity' else 'prices'; idx=0 if stage=='quantity' else 1
            mask={k:[r for r in (0,1) if (k[0],k[1],r) in lookup and all(lookup[(k[0],k[1],r)][name][a] is not None for a in design.ARMS)] for k in keys}
            used=[k for k in keys if mask[k]]; people=sorted({k[0] for k in used}); main=reference['groups'][group]['metrics'][stage]
            if sum(len(mask[k]) for k in keys)!=main['common_success_slots'] or len(used)!=main['common_success_contexts'] or len(people)!=main['common_success_people']: raise ValueError('common mask differs')
            if not people: groups[group][stage]={'curves':{},'common_success_slots':0}; continue
            def human(grid):
                values=[]
                for person in people:
                    person_rounds=[]
                    for k in used:
                        if k[0]!=person: continue
                        c=contexts[k]; row=[]
                        for o in eligible[k]:
                            pair=(o['qty'],o['price']); pair=project(*pair,c['side'],c['current_state']['Q']) if grid else pair; row.append(pair[idx])
                        person_rounds.append(row)
                    values.append(person_rounds)
                return analysis.balanced(values)
            raw=human(False); grid=human(True); samples={'human_raw':raw,'human_grid':grid}
            for arm in design.ARMS:
                values=analysis.balanced([[[lookup[(k[0],k[1],r)][name][arm] for r in mask[k]] for k in used if k[0]==person] for person in people])
                expected=main['arms'][arm]
                if abs(w1(raw,values)-expected['raw_w1'])>1e-9 or abs(w1(grid,values)-expected['projected_w1'])>1e-9: raise ValueError('frozen W1 mismatch')
                samples[arm]=values
            curves={}
            for name,points in samples.items():
                values=sorted({x for x,w in points}); curves[name]={'weighted_mean':sum(x*w for x,w in points),'cdf':[{'value':x,'cdf':sum(w for v,w in points if v<=x)} for x in values]}
            groups[group][stage]={'curves':curves,'common_success_slots':main['common_success_slots'],'common_success_people':main['common_success_people'],'common_success_contexts':main['common_success_contexts']}
    return {'schema':'jev-expanded-grid-cdf-v1','interpretation':'Same participant/eligible-round/order-or-repeat weighted common-success masks as replayed aggregate W1; other price arms are conditional on 3% quantity.','groups':groups}

def main():
    reference=json.loads(BASE.with_suffix('.json').read_text(encoding='utf-8'));plan,predictions,status=runner.replay(runner.OUT)
    if status['status']=='PARTIAL_TECHNICAL': raise ValueError('technical partial run')
    targets=json.loads((runner.OUT/'labels-private.json').read_text(encoding='utf-8')); result=curves(plan,predictions,targets,reference)
    path=BASE.parent/(BASE.name+'-cdf.json'); path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'REPLAY_VALID','output':str(path)}))
if __name__=='__main__': main()
