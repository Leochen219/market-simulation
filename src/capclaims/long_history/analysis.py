"""Offline matched-history participant-balanced numeric distributions."""
import json
from collections import Counter
from capclaims.long_history import runner,design
from tools.analysis.compare_jev_grid_human_distributions import weighted,w1,project,bounds

def baseline(ctx,method):
    current=ctx['current_state'];side=ctx['side'].lower(); candidates=[]
    for row in ctx['history']:
        orders=[o for o in row['orders'] if o['side']==side]
        if not orders or (side=='sell' and sum(o['qty'] for o in orders)>current['Q']): continue
        distance=abs(row['p']-current['p'])/49+abs(row['Q']-current['Q'])/400+abs(row['D']-current['D'])/200
        candidates.append((distance,-row['round'],orders))
    if not candidates:return None
    if method=='last': return min(candidates,key=lambda x:x[1])[2]
    if method=='nearest':return min(candidates,key=lambda x:(x[0],x[1]))[2]
    raise ValueError('unknown baseline')

def history_pool(ctx):
    current=ctx['current_state']; side=ctx['side'].lower(); rounds=[]
    for row in ctx['history']:
        orders=[o for o in row['orders'] if o['side']==side]
        if orders and (side!='sell' or sum(o['qty'] for o in orders)<=current['Q']): rounds.append(orders)
    return rounds

def pool_weighted(people,field):
    n=len(people)
    return [(o[field],1/(n*len(rounds)*len(orders))) for rounds in people for orders in rounds for o in orders]


def summarize(plan,predictions,targets):
    contexts={(c['person_id'],c['condition']):c for c in plan['contexts']}; lookup={(p['person_id'],p['condition'],p['repeat']):p for p in predictions}
    persons=sorted({c['person_id'] for c in plan['contexts']});coverage=Counter()
    for person in persons:
        c=contexts[person,'short3']; orders=targets[person]['orders'];same=[x for x in orders if x['side']==c['side'].lower()]; other=[x for x in orders if x['side']!=c['side'].lower()]
        kind='empty' if not orders else 'mixed' if same and other else 'opposite_only' if not same else 'one_given_side' if len(same)==1 else 'multiple_given_side';coverage[kind]+=1
    result={}
    for side in ('BUY','SELL'):
        eligible={p:[o for o in targets[p]['orders'] if o['side']==side.lower()] for p in persons if contexts[p,'short3']['side']==side}
        eligible={p:o for p,o in eligible.items() if o}; people=sorted(eligible); group={'eligible_people':len(people),'eligible_planned_paths_per_window':len(people)*2,'metrics':{},'baselines':{}}
        for stage in ('quantity','price'):
            value_name='quantities' if stage=='quantity' else 'prices'; item_key='qty' if stage=='quantity' else 'price'
            mask={p:[r for r in (0,1) if all((p,c,r) in lookup and lookup[(p,c,r)][value_name]['cutoff3'] is not None for c in design.CONDITIONS)] for p in people}
            selected=[p for p in people if mask[p]]; metric={'common_success_people':len(selected),'common_success_slots':sum(len(mask[p]) for p in selected),'planned_slots_per_window':len(people)*2,'conditions':{}}
            if people and stage=='price':
                raw_all=weighted([[o['price'] for o in eligible[p]] for p in people]); lo=min(2.5,min(v for v,w in raw_all));hi=max(997.5,max(v for v,w in raw_all))
                for cond in design.CONDITIONS:
                    observed=[(lookup[(p,cond,r)]['prices']['cutoff3'],1/(len(people)*2)) for p in people for r in (0,1) if (p,cond,r) in lookup and lookup[(p,cond,r)]['prices']['cutoff3'] is not None]
                    missing=max(0,1-sum(w for _,w in observed))
                    metric['conditions'][cond]={'full_price_observed_slots':len(observed),'full_price_missing_weight':missing,'full_price_w1_loose_bounds':bounds(raw_all,observed,missing,lo,hi),'raw_w1':None,'projected_w1':None}
            if selected:
                raw=weighted([[o[item_key] for o in eligible[p]] for p in selected])
                grid=weighted([[project(o['qty'],o['price'],side,contexts[p,'short3']['current_state']['Q'])[0 if stage=='quantity' else 1] for o in eligible[p]] for p in selected])
                for cond in design.CONDITIONS:
                    model=weighted([[lookup[(p,cond,r)][value_name]['cutoff3'] for r in mask[p]] for p in selected]); out={'raw_w1':w1(raw,model),'projected_w1':w1(grid,model),'human_weighted_mean':sum(x*w for x,w in raw),'model_weighted_mean':sum(x*w for x,w in model)}
                    if stage=='price':
                        out['above_250_share']=sum(w for x,w in model if x>250)
                        flags=weighted([[int(lookup[(p,cond,r)]['prices']['cutoff3']>contexts[p,cond]['current_state']['p']) for r in mask[p]] for p in selected]);out['above_retail_share']=sum(x*w for x,w in flags)
                    else:
                        totals=weighted([[sum(o['qty'] for o in eligible[p])] for p in selected]);out['human_portfolio_total_vs_model_one_order_w1']=w1(totals,model)
                    metric['conditions'].setdefault(cond,{}).update(out)
                loo={'short3':0,'long10':0,'ties':0}
                for omitted in selected if len(selected)>1 else []:
                    keep=[p for p in selected if p!=omitted]; human=weighted([[o[item_key] for o in eligible[p]] for p in keep]);scores={c:w1(human,weighted([[lookup[(p,c,r)][value_name]['cutoff3'] for r in mask[p]] for p in keep])) for c in design.CONDITIONS}
                    if abs(scores['short3']-scores['long10'])<1e-9: loo['ties']+=1
                    else:loo[min(scores,key=scores.get)]+=1
                metric['leave_one_person_out_raw']=loo
            group['metrics'][stage]=metric
            for cond in design.CONDITIONS:
                for method in ('last','nearest'):
                    valid=[p for p in selected if baseline(contexts[p,cond],method) is not None]
                    key=f'{cond}_{method}_{stage}'; entry={'coverage_people':len(valid),'eligible_people':len(people),'common_model_slots':sum(len(mask[p]) for p in valid),'baseline_scope':'all same-side historical orders from selected round; gross SELL bounded by current Q'}
                    if valid:
                        human=weighted([[o[item_key] for o in eligible[p]] for p in valid]);hist=weighted([[o[item_key] for o in baseline(contexts[p,cond],method)] for p in valid]);model=weighted([[lookup[(p,cond,r)][value_name]['cutoff3'] for r in mask[p]] for p in valid]);entry.update(historical_raw_w1=w1(human,hist),model_same_covered_people_raw_w1=w1(human,model))
                    group['baselines'][key]=entry
        group['pooled_history_baseline']={}
        for stage in ('quantity','price'):
            value_name='quantities' if stage=='quantity' else 'prices'; field='qty' if stage=='quantity' else 'price'
            mask={p:[r for r in (0,1) if all((p,c,r) in lookup and lookup[(p,c,r)][value_name]['cutoff3'] is not None for c in design.CONDITIONS)] for p in people}
            valid=[p for p in people if mask[p] and all(history_pool(contexts[p,c]) for c in design.CONDITIONS)]
            pooled={'common_covered_people':len(valid),'eligible_people':len(people),'common_model_slots':sum(len(mask[p]) for p in valid),'conditions':{},'scope':'per-person then eligible historical round then same-side order; gross SELL bounded by target inventory'}
            if valid:
                human=weighted([[o[field] for o in eligible[p]] for p in valid])
                for cond in design.CONDITIONS:
                    hist=pool_weighted([history_pool(contexts[p,cond]) for p in valid],field)
                    model=weighted([[lookup[(p,cond,r)][value_name]['cutoff3'] for r in mask[p]] for p in valid])
                    pooled['conditions'][cond]={'historical_raw_w1':w1(human,hist),'model_same_covered_people_raw_w1':w1(human,model),'historical_component_count':sum(len(orders) for p in valid for orders in history_pool(contexts[p,cond]))}
            group['pooled_history_baseline'][stage]=pooled
        six={p:[r for r in (0,1) if all((p,c,r) in lookup and all(lookup[(p,c,r)]['prices'][arm] is not None for arm in design.ARMS) for c in design.CONDITIONS)] for p in people}
        covered=[p for p in people if six[p]]
        secondary={'common_success_people':len(covered),'common_success_slots':sum(len(six[p]) for p in covered),'planned_slots_per_window':2*len(people),'conditions':{}}
        if covered:
            human=weighted([[o['price'] for o in eligible[p]] for p in covered])
            for cond in design.CONDITIONS:
                secondary['conditions'][cond]={arm:w1(human,weighted([[lookup[(p,cond,r)]['prices'][arm] for r in six[p]] for p in covered])) for arm in design.ARMS}
        group['secondary_price_six_way_common']=secondary
        result[side]=group
    stability={}
    for cond in design.CONDITIONS:
        pairs=[(lookup[(p,cond,0)],lookup[(p,cond,1)]) for p in persons if (p,cond,0) in lookup and (p,cond,1) in lookup]
        stability[cond]={}
        for node in ('quantity','band','price-b0','price-b1','price-b2','price-b3'):
            available=[(a,b) for a,b in pairs if node in a['request_hashes'] and node in b['request_hashes']]
            same=[(a,b) for a,b in available if a['request_hashes'][node]==b['request_hashes'][node]]
            l1=[sum(abs(a['probabilities'][node][k]-b['probabilities'][node][k]) for k in a['probabilities'][node]) for a,b in same]
            stability[cond][node]={'available_pairs':len(available),'same_request_body_pairs':len(same),'same_body_probability_l1_mean':sum(l1)/len(l1) if l1 else None}
    failures={cond:{arm:{'quantity_decoder_failure_n':sum(p['quantities'][arm] is None for p in predictions if p['condition']==cond),'shared_quantity_failure_n':sum(p['conditioned_quantity'] is None for p in predictions if p['condition']==cond),'band_decoder_failure_n':sum(p['price_failure_reason'][arm]=='band_decoder' for p in predictions if p['condition']==cond),'selected_price_decoder_failure_n':sum(p['price_failure_reason'][arm]=='selected_price_decoder' for p in predictions if p['condition']==cond),'planned_paths':sum(x['condition']==cond for x in plan['schedule'])} for arm in design.ARMS} for cond in design.CONDITIONS}
    return {'schema':'jev-long-history-analysis-v1','conditions':{'short3':'formal 8–10 / game 13–15','long10':'formal 1–10 / game 6–15'},'target':'formal 11 / game 16','coverage':dict(coverage),'people':len(persons),'groups':result,'failures':failures,'request_body_stability':stability,
        'interpretation':'Same people and target state, two history conditions; 3% full pipeline primary; 5% and argmax prices conditional on 3% quantity. Numeric distributions only for human given-side-positive contexts; price uses both-window common success, full legal-range missing bounds alongside. Multiple historical orders remain components, not a one-order prediction. No direction, calibration, causality or population claim.'}

def report(root,output):
    plan,pred,status=runner.replay(root)
    if status['status']=='PARTIAL_TECHNICAL': raise ValueError('technical partial run')
    targets=json.loads((root/'labels-private.json').read_text(encoding='utf-8')); result=summarize(plan,pred,targets);result['status']=status
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');return result
