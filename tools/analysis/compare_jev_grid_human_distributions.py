"""Offline participant-balanced comparison to authorized round-13 orders."""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from capclaims.behavioral_memory.source import build_authorized_cases
from tools.analysis.analyze_jev_grid_probability import _replay
from tools.probes import jev_grid_probability as v1, jev_grid_probability_v2 as v2, jev_grid_probability_v3 as v3

RUNS=ROOT/'artifacts/participant-data'; DOCS=ROOT/'docs/analysis'
NAMES=['jev-grid-probability-v1-2026-10-27','jev-grid-probability-v2-2026-10-27','jev-grid-probability-v3-2026-10-27']
ARMS=('argmax','cutoff5','cutoff3')

def weighted(people):
    if not people or any(not x for x in people): raise ValueError('empty participant group')
    return [(v,1/(len(people)*len(values))) for values in people for v in values]

def cdf(points,x): return sum(w for value,w in points if value<=x)
def w1(a,b):
    xs=sorted({x for x,w in a+b}); return sum(abs(cdf(a,x)-cdf(b,x))*(y-x) for x,y in zip(xs,xs[1:]))
def project(q,p,side,initial_q):
    grid=v1.quantity_grid(side,initial_q)
    if not grid: raise ValueError('empty quantity grid')
    quantity=min(grid,key=lambda x:(abs(x-q),-x))
    price=min(range(1,400),key=lambda tick:(abs(250*tick-100*p),-tick))*2.5
    return quantity,price

def energy(a,b):
    def dist(x,y): return math.hypot((x[0]-y[0])/990,(x[1]-y[1])/995)
    def avg(u,v): return sum(w*z*dist(x,y) for x,w in u for y,z in v)
    return 2*avg(a,b)-avg(a,a)-avg(b,b)

def bounds(h,observed,missing,lo,hi):
    xs=sorted({lo,hi}|{x for x,w in h}|{x for x,w in observed})
    lower=upper=0.0
    for x,y in zip(xs,xs[1:]):
        target=cdf(h,x); f=cdf(observed,x); l=0 if f<=target<=f+missing else min(abs(target-f),abs(target-f-missing)); u=max(abs(target-f),abs(target-f-missing))
        lower+=l*(y-x); upper+=u*(y-x)
    return [lower,upper]

def _hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _record_map(records): return {(r['context'],r['repeat'],r['stage']):r for r in records}
def common_slots(chains,ids):
    return {(p,r) for p in ids for r in (0,1) if all(ch[(p,r,'price')].get('record_type','STEP')=='STEP' for ch in chains)}

def analyze():
    plans=[]; records=[]
    for name,replay in zip(NAMES,(_replay,v2.replay,v3.replay)):
        plan,rows=replay(RUNS/name); plans.append(plan); records.append(rows)
    cases,labels,meta=build_authorized_cases(); cases={c['person_id']:c for c in cases}; plan=plans[0]; contexts=plan['contexts']; ids={x['person_id'] for x in contexts}
    fields=('source_sha256','source_archive_sha256','source_policy_sha256','exclusion_manifest_sha256','exposure_registry_sha256','source_scope','target_round')
    if any(p['contexts']!=contexts or any(p['metadata'][k]!=meta[k] for k in fields) for p in plans): raise ValueError('source or contexts differ')
    if ids!={x['person_id'] for x in v1.select_cases(list(cases.values()))}: raise ValueError('selected cohort differs')
    if any(v1.make_state(cases[c['person_id']],c['side'])!=c['base_state'] or labels[c['person_id']]['round']!=13 for c in contexts): raise ValueError('source state mismatch')
    if any(len(r)!=60 for r in records): raise ValueError('response coverage incomplete')
    chains=list(map(_record_map,records))
    human={}; projected={}; private=[]; coverage={'order_count_histogram':{},'offgrid_quantity':0,'offgrid_price':0,'below_ten_quantity':0,'portfolio_projected_sell_over_capacity_contexts':0,'sides':{}}
    for c in contexts:
        person=c['person_id']; side=c['side']; orders=labels[person]['orders']; same=[o for o in orders if o['side']==side.lower()]; opposite=[o for o in orders if o['side']!=side.lower()]
        cov=coverage['sides'].setdefault(side,{'context_n':0,'given_side_positive':0,'no_orders':0,'only_opposite':0,'mixed':0,'orders':0})
        cov['context_n']+=1; cov['given_side_positive']+=bool(same); cov['no_orders']+=not orders; cov['only_opposite']+=bool(opposite) and not same; cov['mixed']+=bool(same) and bool(opposite); cov['orders']+=len(same)
        if not same: raise ValueError('given-side nonparticipant requires explicit estimand handling')
        human[person]=[(o['qty'],o['price']) for o in same]; projected[person]=[project(o['qty'],o['price'],side,c['base_state']['own_q_units']) for o in same]
        for o,z in zip(same,projected[person]): coverage['offgrid_quantity']+=o['qty']!=z[0]; coverage['offgrid_price']+=o['price']!=z[1]; coverage['below_ten_quantity']+=o['qty']<10
        coverage['portfolio_projected_sell_over_capacity_contexts']+=side=='SELL' and sum(z[0] for z in projected[person])>c['base_state']['own_q_units']
        count=str(len(orders)); coverage['order_count_histogram'][count]=coverage['order_count_histogram'].get(count,0)+1
        private.append({'context':person,'side':side,'orders':orders,'projected_given_side_orders':projected[person],'human_weight_per_given_side_order':1/(5*len(same)),'model_quantity_weight_per_repeat':1/10})
    common=common_slots(chains,ids)
    for entry in private:
        person=entry['context']; common_repeats=[r for r in (0,1) if (person,r) in common]
        entry['common_price_repeats']=common_repeats
        entry['model_common_price_weight_per_repeat']=1/(5*len(common_repeats)) if common_repeats else None
    results={}; plotting={}
    for side in ('BUY','SELL'):
        people=[c['person_id'] for c in contexts if c['side']==side]; n=len(people)
        if n!=5 or any(not any((p,r) in common for r in (0,1)) for p in people): raise ValueError('common coverage cannot balance all participants')
        rawq=weighted([[q for q,p in human[x]] for x in people]); rawp=weighted([[p for q,p in human[x]] for x in people]); rawjoint=weighted([human[x] for x in people])
        gridq=weighted([[q for q,p in projected[x]] for x in people]); gridp=weighted([[p for q,p in projected[x]] for x in people]); gridjoint=weighted([projected[x] for x in people])
        sideout={'n_people':n,'common_price_slots':sum((p,r) in common for p in people for r in (0,1)), 'human_to_grid_w1':{'quantity':w1(rawq,gridq),'price':w1(rawp,gridp)},'arms':{}}
        plotting[side]={'quantity':{'human_raw':rawq,'human_grid':gridq},'price':{'human_raw':rawp,'human_grid':gridp}}
        for name,ch in zip(ARMS,chains):
            mq=weighted([[ch[(p,r,'quantity')]['selected_value'] for r in (0,1)] for p in people]); mp=weighted([[ch[(p,r,'price')]['selected_value'] for r in (0,1) if (p,r) in common] for p in people]); mj=weighted([[(ch[(p,r,'quantity')]['selected_value'],ch[(p,r,'price')]['selected_value']) for r in (0,1) if (p,r) in common] for p in people])
            obs=[(ch[(p,r,'price')]['selected_value'],1/(n*2)) for p in people for r in (0,1) if ch[(p,r,'price')].get('record_type','STEP')=='STEP']; missing=1-sum(w for x,w in obs)
            lo=min(2.5,min(x for x,w in rawp)); hi=max(997.5,max(x for x,w in rawp))
            sideout['arms'][name]={'raw_w1':{'quantity':w1(rawq,mq),'price':w1(rawp,mp)},'gridded_w1':{'quantity':w1(gridq,mq),'price':w1(gridp,mp)},'joint_energy_raw':energy(rawjoint,mj),'joint_energy_gridded':energy(gridjoint,mj),'price_full_cohort_w1_loose_bounds':bounds(rawp,obs,missing,lo,hi),'original_available_price_slots':len(obs),'portfolio_total_quantity_vs_one_model_order_w1':w1(weighted([[sum(q for q,_ in human[p])] for p in people]),mq)}
            plotting[side]['quantity'][name]=mq; plotting[side]['price'][name]=mp
        sideout['leave_one_out_winner_counts']={}
        for label,source in [('raw',human),('gridded',projected)]:
            sideout['leave_one_out_winner_counts'][label]={}
            for stage,idx in [('quantity',0),('price',1)]:
                winners={k:0 for k in ARMS}; ties=0
                for excluded in people:
                    subset=[p for p in people if p!=excluded]; h=weighted([[o[idx] for o in source[p]] for p in subset]); scores={}
                    for name,ch in zip(ARMS,chains):
                        m=weighted([[ch[(p,r,stage)]['selected_value'] for r in (0,1) if stage=='quantity' or (p,r) in common] for p in subset]); scores[name]=w1(h,m)
                    low=min(scores.values()); best=[name for name in ARMS if abs(scores[name]-low)<1e-9]
                    if len(best)==1: winners[best[0]]+=1
                    else: ties+=1
                sideout['leave_one_out_winner_counts'][label][stage]={'unique_wins':winners,'ties':ties,'omissions':len(people)}
        results[side]=sideout
    public={'schema':'jev-grid-human-distributions-v1','scope':'round 13; supplied side; 10 people, two model samples per person; private orders remain local','provenance':{k:meta[k] for k in fields},'plan_sha256':{a:_hash(RUNS/n/'plan.json') for a,n in zip(ARMS,NAMES)},'analysis_code_sha256':_hash(Path(__file__)),'coverage':coverage,'common_price_pairs':len(common),'common_price_contexts':len({p for p,r in common}),'common_price_contexts_with_two_repeats':sum(all((p,r) in common for r in (0,1)) for p in ids),'common_price_contexts_with_one_repeat':sum(sum((p,r) in common for r in (0,1))==1 for p in ids),'failures':{a:sum(r.get('record_type')=='DECODE_FAILURE' for r in rows) for a,rows in zip(ARMS,records)},'sides':results,'method':'Participant-balanced weighted W1, raw human primary; common-price intersection is selection-affected. Secondary joint energy uses q/990 and (price-2.5)/995, equal dimension scales. Grid sensitivity projects each order independently, nearest ties upward, without cumulative allocation. Price bounds are conservative pointwise CDF envelopes over legal model price support and raw human endpoints. Leave-one-out counts are not confidence estimates.'}
    private_dir=RUNS/'jev-grid-human-distributions-2026-10-27'; private_dir.mkdir(exist_ok=True); (private_dir/'audit-private.json').write_text(json.dumps({'records':private,'common_price_pairs':sorted(common)},indent=2),encoding='utf-8')
    public['distribution_summary']={side:{stage:{name:{'weighted_mean':sum(value*weight for value,weight in points),'above_250_share':sum(weight for value,weight in points if value>250) if stage=='price' else None} for name,points in curves.items()} for stage,curves in stages.items()} for side,stages in plotting.items()}
    public['cdf']={side:{stage:{name:[{'value':x,'cdf':cdf(points,x)} for x in sorted(set(value for value,w in points))] for name,points in curves.items()} for stage,curves in stages.items()} for side,stages in plotting.items()}
    (DOCS/'jev-grid-human-distributions-2026-10-27.json').write_text(json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return public,plotting
if __name__=='__main__':
    result,plotting=analyze(); print(json.dumps(result,ensure_ascii=False))
