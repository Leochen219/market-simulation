"""Offline participant-first descriptive distributions for replayed evidence."""
import json
from collections import defaultdict
from collections import Counter
from capclaims.expanded_grid import runner,design
from tools.analysis.compare_jev_grid_human_distributions import project,w1,bounds

def balanced(people):
    """Participant -> eligible rounds -> observations, each layer equally weighted."""
    n=len(people)
    if not n or any(not rounds or any(not values for values in rounds) for rounds in people): raise ValueError('empty layer')
    return [(x,1/(n*len(rounds)*len(values))) for rounds in people for values in rounds for x in values]

def summarize(plan,predictions,targets):
    contexts={(c['person_id'],c['formal_round']):c for c in plan['contexts']}; lookup={(x['person_id'],x['formal_round'],x['repeat']):x for x in predictions}
    coverage=Counter(); coverage_by=defaultdict(Counter); eligible={}
    for key,c in contexts.items():
        orders=targets[design.coordinate(*key)]['orders']; same=[x for x in orders if x['side']==c['side'].lower()]; other=[x for x in orders if x['side']!=c['side'].lower()]
        typ='empty' if not orders else 'mixed' if same and other else 'opposite_only' if not same else 'one_given_side' if len(same)==1 else 'multiple_given_side'
        coverage[typ]+=1; coverage_by[f"{c['side']}:round{c['formal_round']}"][typ]+=1
        if same: eligible[key]=same
    result={}
    for group,predicate in [('BUY',lambda c:c['side']=='BUY'),('SELL',lambda c:c['side']=='SELL'),('shortage',lambda c:c['stratum']=='shortage'),('surplus',lambda c:c['stratum']=='surplus')]:
        keys=[k for k in eligible if predicate(contexts[k])]; out={'eligible_contexts':len(keys),'eligible_people':len({k[0] for k in keys}),'metrics':{}}
        for stage in ('quantity','price'):
            mask={k:[r for r in (0,1) if (k[0],k[1],r) in lookup and all(lookup[(k[0],k[1],r)]['quantities' if stage=='quantity' else 'prices'][a] is not None for a in design.ARMS)] for k in keys}
            keys_ok=[k for k in keys if mask[k]]; persons=sorted({k[0] for k in keys_ok})
            out['metrics'][stage]={'common_success_slots':sum(map(len,mask.values())),'common_success_contexts':len(keys_ok),'common_success_people':len(persons),'denominator_planned_slots':len(keys)*2,'arms':{}}
            if stage=='price' and keys:
                full_people=sorted({k[0] for k in keys}); full_human=balanced([[[o['price'] for o in eligible[k]] for k in keys if k[0]==person] for person in full_people])
                lo=min(2.5,min(v for v,w in full_human)); hi=max(997.5,max(v for v,w in full_human))
                for arm in design.ARMS:
                    observed=[]
                    for person in full_people:
                        rounds=[k for k in keys if k[0]==person]
                        for k in rounds:
                            for rep in (0,1):
                                pred=lookup.get((k[0],k[1],rep)); value=pred['prices'][arm] if pred else None
                                if value is not None: observed.append((value,1/(len(full_people)*len(rounds)*2)))
                    missing=max(0.0,1-sum(w for v,w in observed))
                    out['metrics'][stage]['arms'][arm]={'raw_w1':None,'projected_w1':None,'full_eligible_price_observed_slots':len(observed),'full_eligible_price_missing_weight':missing,'full_eligible_price_w1_loose_bounds':bounds(full_human,observed,missing,lo,hi)}
            if not persons: continue
            def human(grid):
                result=[]
                for person in persons:
                    rounds=[]
                    for k in keys_ok:
                        if k[0]!=person: continue
                        c=contexts[k]; vals=[]
                        for o in eligible[k]:
                            val=(o['qty'],o['price']); val=project(*val,c['side'],c['current_state']['Q']) if grid else val
                            vals.append(val[0 if stage=='quantity' else 1])
                        rounds.append(vals)
                    result.append(rounds)
                return balanced(result)
            raw=human(False); grid=human(True)
            for arm in design.ARMS:
                nested=[]
                for person in persons:
                    nested.append([[lookup[(k[0],k[1],r)]['quantities' if stage=='quantity' else 'prices'][arm] for r in mask[k]] for k in keys_ok if k[0]==person])
                model=balanced(nested)
                armout={'raw_w1':w1(raw,model),'projected_w1':w1(grid,model),'weighted_mean':sum(v*w for v,w in model),'high_price_share':sum(w for v,w in model if v>250) if stage=='price' else None,
                    'model_value_counts':dict(Counter(str(v) for rounds in nested for values in rounds for v in values))}
                if stage=='price':
                    flags=[]
                    for person in persons:
                        rounds=[]
                        for k in keys_ok:
                            if k[0]==person: rounds.append([int(lookup[(k[0],k[1],r)]['prices'][arm]>contexts[k]['current_state']['p']) for r in mask[k]])
                        flags.append(rounds)
                    armout['price_above_retail_share']=sum(v*w for v,w in balanced(flags))
                else:
                    totals=balanced([[[sum(o['qty'] for o in eligible[k])] for k in keys_ok if k[0]==person] for person in persons])
                    armout['human_portfolio_total_vs_model_one_order_w1']=w1(totals,model)
                out['metrics'][stage]['arms'].setdefault(arm,{}).update(armout)
            loo={'omitted_people':len(persons),'unique_wins':{a:0 for a in design.ARMS},'ties':0}
            if len(persons)>1:
                for omitted in persons:
                    subset=[person for person in persons if person!=omitted]
                    h=balanced([[[o['qty' if stage=='quantity' else 'price'] for o in eligible[k]] for k in keys_ok if k[0]==person] for person in subset]); scores={}
                    for arm in design.ARMS:
                        model=balanced([[[lookup[(k[0],k[1],r)]['quantities' if stage=='quantity' else 'prices'][arm] for r in mask[k]] for k in keys_ok if k[0]==person] for person in subset]); scores[arm]=w1(h,model)
                    low=min(scores.values()); winners=[a for a in design.ARMS if abs(scores[a]-low)<1e-9]
                    if len(winners)==1: loo['unique_wins'][winners[0]]+=1
                    else: loo['ties']+=1
            out['metrics'][stage]['leave_one_person_out_raw']=loo
        result[group]=out
    stability={}
    bycontext={}
    for p in predictions: bycontext.setdefault((p['person_id'],p['formal_round']),{})[p['repeat']]=p
    for node in ('quantity','band','price-b0','price-b1','price-b2','price-b3'):
        pairs=[(x[0]['request_hashes'][node],x[1]['request_hashes'][node]) for x in bycontext.values() if 0 in x and 1 in x and node in x[0]['request_hashes'] and node in x[1]['request_hashes']]
        same=[x for x in bycontext.values() if 0 in x and 1 in x and node in x[0]['request_hashes'] and node in x[1]['request_hashes'] and x[0]['request_hashes'][node]==x[1]['request_hashes'][node]]
        l1=[sum(abs(x[0]['probabilities'][node][k]-x[1]['probabilities'][node][k]) for k in x[0]['probabilities'][node]) for x in same]
        stability[node]={'available_pairs':len(pairs),'same_request_body_sha_pairs':len(same),'same_body_raw_probability_l1_mean':sum(l1)/len(l1) if l1 else None}
    failures={arm:{'quantity_decoder_failure_n':sum(p['quantities'][arm] is None for p in predictions),'shared_3pct_quantity_failure_n':sum(p['conditioned_quantity'] is None for p in predictions),'band_decoder_failure_n':sum(p['price_failure_reason'][arm]=='band_decoder' for p in predictions),'selected_price_decoder_failure_n':sum(p['price_failure_reason'][arm]=='selected_price_decoder' for p in predictions),'unavailable_infrastructure_n':len(plan['schedule'])-len(predictions),'planned_path_denominator':len(plan['schedule'])} for arm in design.ARMS}
    return {'schema':'jev-expanded-grid-analysis-v1','failure_counts':failures,'coverage':dict(coverage),'coverage_by_side_and_round':{k:dict(v) for k,v in coverage_by.items()},'total_contexts':len(contexts),'planned_paths':len(plan['schedule']),'groups':result,'request_body_stability':stability,
        'interpretation':'One shared 3% quantity conditions all price branches. Argmax and 5% price policies are conditional on that quantity, not complete pipelines. Quantity choices are compared separately. Human given-side coverage is not a direction score. Participant then eligible round then order/repeat weighting. Repeated observations are not independent people; no calibration or causal history-length claim.'}

def report(root,output):
    plan,predictions,status=runner.replay(root)
    if status['status']=='PARTIAL_TECHNICAL': raise ValueError('technical partial run: no completed comparison')
    targets=json.loads((root/'labels-private.json').read_text(encoding='utf-8')); result=summarize(plan,predictions,targets); result['status']=status
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); return result
