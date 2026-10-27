"""Offline audit of preserved SELL price tails and authorized development orders."""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from capclaims.behavioral_memory.source import build_authorized_cases
from capclaims.grid_sampling import decode as decode5
from capclaims.grid_sampling_v3 import decode as decode3
from tools.analysis.analyze_jev_grid_probability import _replay
from tools.probes import jev_grid_probability as v1, jev_grid_probability_v2 as v2, jev_grid_probability_v3 as v3
RUNS=ROOT/'artifacts/participant-data'; DOCS=ROOT/'docs/analysis'
NAMES=['jev-grid-probability-v1-2026-10-27','jev-grid-probability-v2-2026-10-27','jev-grid-probability-v3-2026-10-27']; ARMS=('argmax','cutoff5','cutoff3')

def interval(decoder):
    cumulative=0.0; selected=decoder['selected_id']; weights=decoder['retained_normalized_probabilities']
    for key in decoder['candidate_ids']:
        lo=cumulative; cumulative+=weights[key]
        if key==selected:
            return [lo,cumulative]
    raise ValueError('selected candidate absent')

def band_forces_high(index):
    lo,hi=v1.BANDS[index]; return lo*2.5>250

def step_summary(step):
    raw=step['raw_probabilities']; ids=step['candidate_ids']; sel=step['selected_id']; dec=step['decode']; ranked=sorted(ids,key=lambda k:(-raw[k],ids.index(k)))
    return {'selected_id':sel,'selected_value':step['selected_value'],'argmax_id':step['argmax_id'],
        'selected_raw_probability':raw[sel],'selected_rank':ranked.index(sel)+1,'retained_probability':dec['retained_normalized_probabilities'][sel],
        'retained_raw_mass':dec['retained_mass'],'cutoff_dropped_mass':dec['cutoff_dropped_mass'],'topk_dropped_mass':dec['topk_dropped_mass'],
        'retained_ids':dec['retained_ids'],'draw':dec['draw'],'selected_interval':interval(dec)}

def policy_sensitivity(rows):
    """Fixed observed response only; no conditional branches are generated."""
    result={name:{'responses':0,'decode_failures':0,'mean_high_probability':None,'_mass':[]} for name in ('full','top3','cutoff5','cutoff3')}
    for row in rows:
        step=row['step']; raw=step['raw_probabilities']; ids=step['candidate_ids']; stage=step['stage']
        high=lambda k: int(k[1:])>0 if stage=='band' else int(k[1:])*2.5>250
        for name in result:
            r=result[name]; r['responses']+=1
            try:
                if name in ('cutoff5','cutoff3'):
                    decoder=decode5 if name=='cutoff5' else decode3
                    dec=decoder(raw,ids,seed=v2.SEED,person_id=row['context'],repeat=row['repeat'],stage=stage)
                    probs=dec['retained_normalized_probabilities']
                else:
                    retained=ids if name=='full' else sorted(ids,key=lambda k:(-raw[k],ids.index(k)))[:3]
                    total=math.fsum(raw[k] for k in retained)
                    if total<=0: raise ValueError('empty support')
                    probs={k:(raw[k]/total if k in retained else 0) for k in ids}
            except ValueError: r['decode_failures']+=1; continue
            r['_mass'].append(sum(probs[k] for k in ids if high(k)))
    for r in result.values():
        mass=r.pop('_mass'); r['successful_responses']=len(mass); r['mean_high_probability']=sum(mass)/len(mass) if mass else None
    return result


def human_summary(rows):
    """Each person receives equal total weight, divided over eligible rounds and orders."""
    counts={'person_rounds':len(rows),'people':len({p for p,s,o in rows}),'no_orders':0,'only_buy':0,'sell_only':0,'mixed':0,'sell_order_n':0,'sell_price_gt250_n':0,'projected_sell_price_gt250_n':0,'sell_price_above_retail_n':0}
    per_person={}
    for person,state,orders in rows:
        sells=[x for x in orders if x['side']=='sell']; buys=[x for x in orders if x['side']=='buy']
        counts['no_orders']+=not orders; counts['only_buy']+=bool(buys) and not sells; counts['sell_only']+=bool(sells) and not buys; counts['mixed']+=bool(sells) and bool(buys)
        if not sells: continue
        counts['sell_order_n']+=len(sells); counts['sell_price_gt250_n']+=sum(o['price']>250 for o in sells)
        counts['projected_sell_price_gt250_n']+=sum(min(range(1,400),key=lambda tick:(abs(250*tick-100*o['price']),-tick))*2.5>250 for o in sells)
        counts['sell_price_above_retail_n']+=sum(o['price']>state['p'] for o in sells)
        per_person.setdefault(person,[]).append(sells)
    counts['people_with_sell']=len(per_person); counts['person_rounds_with_sell']=sum(map(len,per_person.values()))
    if per_person:
        counts['person_round_order_balanced_high_share']=sum(sum(sum(o['price']>250 for o in orders)/len(orders) for orders in rounds)/len(rounds) for rounds in per_person.values())/len(per_person)
    else: counts['person_round_order_balanced_high_share']=None
    return counts

def reference_rows(cases,labels,people,rounds):
    rows=[]
    for person in people:
        case=cases[person]; byround={x['round']:x for x in case['history']}
        byround[13]=dict(case['current_state'],orders=labels[person]['orders'])
        for r in rounds:
            x=byround[r]; rows.append((person,x,x['orders']))
    return rows


def run():
    plans=[]; records=[]; traces=[]
    for name,replay in zip(NAMES,(_replay,v2.replay,v3.replay)):
        root=RUNS/name; plan,rows=replay(root)
        if len(rows)!=60: raise ValueError('run response coverage incomplete')
        plans.append(plan); records.append(rows); traces.append([json.loads(line) for line in (root/'trace.jsonl').read_text(encoding='utf-8').splitlines()])
    cases,labels,meta=build_authorized_cases(); cases={c['person_id']:c for c in cases}; base=plans[0]; context={c['person_id']:c for c in base['contexts']}
    fields=('source_sha256','source_archive_sha256','source_policy_sha256','exclusion_manifest_sha256','exposure_registry_sha256')
    if any(p['contexts']!=base['contexts'] or any(p['metadata'][k]!=meta[k] for k in fields) for p in plans): raise ValueError('source or plan mismatch')
    if any(v1.make_state(cases[p],c['side'])!=c['base_state'] or labels[p]['round']!=13 for p,c in context.items()): raise ValueError('current state/target mismatch')
    private=[]; sell_summary={}; high=[]; all_steps=[]
    for arm,root_name,rows in zip(ARMS,NAMES,traces):
        root=RUNS/root_name; grouped={}
        for row in rows: grouped.setdefault((row['context'],row['repeat']),{})[row['step']['stage']]=row
        count={'sell_paths':0,'selected_band_differs_argmax_n':0,'raw_band_argmax_high_n':0,'high_selected_band_n':0,'high_price_n':0,'high_price_forced_by_band_n':0,'high_price_within_low_band_n':0,'price_failure_n':0}
        for (person,repeat),stages in grouped.items():
            ctx=context[person]; is_sell=ctx['side']=='SELL'; count['sell_paths']+=is_sell
            if not is_sell and not any(x.get('record_type')=='DECODE_FAILURE' for x in stages.values()): continue
            bundle={'arm':arm,'context':person,'repeat':repeat,'side':ctx['side'],'stages':{}}
            for stage,row in stages.items():
                step=row['step']; folder=root/'wire-evidence'/hashlib.sha256(step['request_id'].encode()).hexdigest()
                bundle['stages'][stage]={'trace':row,'request':json.loads((folder/'request.json').read_text(encoding='utf-8')),'response':json.loads((folder/'response.json').read_text(encoding='utf-8'))}
                if 'decode' in step: bundle['stages'][stage]['selected_interval']=interval(step['decode'])
            if not is_sell: private.append(bundle); continue
            band=stages['band']['step']; price=stages['price']; fail=price.get('record_type')=='DECODE_FAILURE'
            band_id=band.get('selected_id',band['argmax_id']); band_index=int(band_id[1:]); forced=band_forces_high(band_index)
            count['raw_band_argmax_high_n']+=band_forces_high(int(band['argmax_id'][1:])); count['selected_band_differs_argmax_n']+=band_id!=band['argmax_id']; count['high_selected_band_n']+=forced; count['price_failure_n']+=fail
            if not fail:
                is_high=price['step']['selected_value']>250; count['high_price_n']+=is_high; count['high_price_forced_by_band_n']+=is_high and forced; count['high_price_within_low_band_n']+=is_high and not forced
                if is_high:
                    alias='S'+str(len(high)+1); bundle['alias']=alias
                    if arm=='argmax': raise ValueError('unexpected v1 high price')
                    band_summary=step_summary(band); price_summary=step_summary(price['step']); quantity_summary=step_summary(stages['quantity']['step'])
                    high.append({'alias':alias,'arm':arm,'quantity':quantity_summary,'band':band_summary,'price':price_summary,'high_forced_by_band':forced})
            private.append(bundle)
        sell_summary[arm]=count
    policy={}
    for arm,rows in zip(ARMS,traces):
        sell=[row for row in rows if context[row['context']]['side']=='SELL']
        bands=[row for row in sell if row['step']['stage']=='band']
        low_prices=[row for row in sell if row['step']['stage']=='price' and row['step']['conditional_on']['band']==0]
        policy[arm]={'band':policy_sensitivity(bands),'observed_low_band_prices':policy_sensitivity(low_prices)}
    paths={b.get('alias'):b for b in private if b.get('alias') in ('S1','S2')}
    b1=paths['S1']['stages']['band']['trace']['step']; b2=paths['S2']['stages']['band']['trace']['step']
    if b1['request_sha256']!=b2['request_sha256']: raise ValueError('paired band request differs')
    same_request={'request_sha256':b1['request_sha256'],'same_request':True,
        'response_probability_l1':sum(abs(b1['raw_probabilities'][k]-b2['raw_probabilities'][k]) for k in b1['candidate_ids']),'cases':{}}
    for alias,step in (('S1',b1),('S2',b2)):
        bundle=paths[alias]; outputs={}
        for label,decoder in (('cutoff5',decode5),('cutoff3',decode3)):
            outputs[label]=decoder(step['raw_probabilities'],step['candidate_ids'],seed=v2.SEED,person_id=bundle['context'],repeat=bundle['repeat'],stage='band')['selected_id']
        same_request['cases'][alias]={'probabilities':step['raw_probabilities'],'draw':step['decode']['draw'],'same_response_selections':outputs}
    human={}; ids10=set(context)
    for label,people,rounds in [('matched10_rounds6_13',ids10,range(6,14)),('all18_rounds6_13',set(cases),range(6,14)),('all18_round13',set(cases),[13])]:
        subset=reference_rows(cases,labels,people,rounds)
        human[label]={'all':human_summary(subset),'stock_surplus_Q_gt_D':human_summary([(p,s,o) for p,s,o in subset if s['Q']>s['D']])}
    out={'schema':'jev-sell-price-tail-audit-v1','source_hashes':{k:meta[k] for k in fields},'plan_hashes':{a:hashlib.sha256((RUNS/n/'plan.json').read_bytes()).hexdigest() for a,n in zip(ARMS,NAMES)},'sell_paths':sell_summary,'same_request_band_comparison':same_request,'high_paths_distinct_contexts':len({b['context'] for b in private if 'alias' in b}),'high_cases':high,'same_response_policy_sensitivity':policy,'human_reference':human,
        'interpretation':'SELL is supplied; comparisons are descriptive. All selected prices in bands 1-3 are strictly above $250. Same-response argmax for a price remains conditional on the selected band and quantity. Unqueried branches cannot be inferred. Human historical reference is the existing development cohort, not independent held-out data.'}
    outdir=RUNS/'jev-sell-price-tail-audit-2026-10-27'; outdir.mkdir(exist_ok=True)
    (outdir/'full-inputs-private.json').write_text(json.dumps({'bundles':private},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (DOCS/'jev-sell-price-tail-audit-2026-10-27.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return out
if __name__=='__main__': print(json.dumps(run(),ensure_ascii=False))
