"""Source validation and isolated target labels for the oracle diagnostic."""
import hashlib,json
from decimal import Decimal
from collections import defaultdict
from capclaims.conditional_behavior import source as authorized
from capclaims.personalization_xlsx import load_paper_episodes, _inner_xlsx_bytes, _workbook, _sheet_rows
from capclaims.conditional_behavior.protocol import validate_orders
from capclaims.sell_oracle import design

def split_components(ctx, orders):
    """Build identities from quantities alone; only this boundary extracts scoring prices."""
    sells=[o for o in orders if o['side']=='sell']
    cs=design.components(ctx,[o['qty'] for o in sells])
    return cs,{c['component_id']:o['price'] for c,o in zip(cs,sells)}

def validate_phases(episodes):
    """Enforce exactly one matching procurement and order phase per source row."""
    phases=defaultdict(dict)
    for e in episodes:
        if e.scope!='GA 10' or e.treatment!='GA': continue
        k=(e.participant_id,e.round)
        if e.phase not in ('procurement','orders') or e.phase in phases[k]: raise ValueError('duplicate or invalid source phase')
        phases[k][e.phase]=e
    for values in phases.values():
        if set(values)!={'procurement','orders'}: raise ValueError('missing source phase')
        proc,orders=values['procurement'],values['orders']
        if (proc.action['Q'],proc.observed['p'],proc.observed['w']) != (orders.observed['Q'],orders.observed['p'],orders.observed['w']) or orders.observed['w']!=10: raise ValueError('source phase state mismatch')
    return phases

def load():
    path=authorized.DEFAULT_SOURCE; policy_raw=authorized.POLICY.read_bytes(); policy=json.loads(policy_raw)
    archive_hash=hashlib.sha256(path.read_bytes()).hexdigest(); workbook_hash=hashlib.sha256(_inner_xlsx_bytes(path)).hexdigest()
    if policy.get('source_sha256')!=workbook_hash or not all(policy.get(k) is True for k in ('codebook_verified','intended_use_authorized','provider_transmission_permitted')): raise ValueError('source policy mismatch')
    _,episodes=load_paper_episodes(path,codebook_verified=True,intended_use_authorized=True,provider_transmission_permitted=True)
    rows={}; phases=validate_phases(episodes)
    for e in episodes:
        if e.scope!='GA 10' or e.treatment!='GA': continue
        k=(e.participant_id,e.round)
        if e.phase=='orders':
            orders=[{'side':o['type'].lower(),'qty':o['amount'],'price':o['price']} for o in e.action['orders']]
            rows[k]={'round':e.round,**e.observed,'orders':list(validate_orders(orders,initial_q=e.observed['Q']))}
    people={p for p,t in rows}
    if len(people)!=35 or len(rows)!=525 or any(set(v)!={'procurement','orders'} for v in phases.values()): raise ValueError('incomplete source')
    archive,shared,members=_workbook(path)
    try: raw=_sheet_rows(archive,members['GA 10'],shared)
    finally: archive.close()
    fields=('Player','Round','Sales Price','Order','Actual Demand','Sales (#)','Profit Projection ()','Sales New (#)','Trade Cost','Post-trade profit ()'); headers=raw[0]
    if any(headers.count(k)!=1 for k in fields): raise ValueError('accounting fields missing')
    ix={k:headers.index(k) for k in fields}; profits={}
    for row in raw[1:]:
        person=str(row[ix['Player']]).strip(); t=int(row[ix['Round']]);
        if t not in range(1,15): continue
        vals=[Decimal(str(row[ix[k]])) for k in fields[2:]]; p,q,d,sales,projection,sales_new,trade,final=vals
        if not all(v.is_finite() for v in vals) or any(v!=v.to_integral_value() for v in vals[:6]) or trade*100!=(trade*100).to_integral_value() or final*100!=(final*100).to_integral_value() or projection!=p*min(d,q)-10*q or sales!=min(d,q) or final!=p*sales_new-10*q+trade: raise ValueError('invalid historical accounting')
        k=(person,t)
        if k in profits or (int(p),int(q),int(d)) != (rows[k]['p'],rows[k]['Q'],rows[k]['D']): raise ValueError('accounting state mismatch')
        profits[k]=int(final*100)
    if len(profits)!=490: raise ValueError('accounting coverage mismatch')
    contexts=[]; components=[]; labels={}
    for p in sorted(people):
        alias='dev-'+hashlib.sha256(p.encode()).hexdigest()[:24]
        for t in range(11,16):
            current=rows[(p,t)]
            if current['Q']<=current['D']: continue
            history=[{**{k:rows[(p,r)][k] for k in ('round','p','Q','D','orders')},'profit_cents':profits[(p,r)]} for r in range(t-10,t)]
            ctx={'person_id':alias,'formal_round':t,'current_state':{k:current[k] for k in ('round','p','w','Q','D')},'history':history}
            contexts.append(ctx)
            cs,scoring=split_components(ctx,current['orders']);components.extend(cs);labels.update(scoring)
    metadata={'source_sha256':workbook_hash,'archive_sha256':archive_hash,'policy_sha256':hashlib.sha256(policy_raw).hexdigest(),'people':35,'candidate_person_rounds':175,'state_contexts':len(contexts),'state_people':len({c['person_id'] for c in contexts}),'priced_events':len({(c['person_id'],c['formal_round']) for c in components}),'priced_people':len({c['person_id'] for c in components}),'components':len(components),'no_sell_events':len(contexts)-len({(c['person_id'],c['formal_round']) for c in components})}
    event_counts=defaultdict(int)
    for c in components: event_counts[(c['person_id'],c['formal_round'])]+=1
    metadata.update(single_sell_events=sum(n==1 for n in event_counts.values()),multiple_sell_events=sum(n>1 for n in event_counts.values()),off_grid_component_quantities=sum(c['quantity']%10!=0 for c in components))
    if (metadata['single_sell_events'],metadata['multiple_sell_events'],metadata['off_grid_component_quantities'])!=(49,22,58): raise ValueError('component census mismatch')
    if (metadata['state_contexts'],metadata['state_people'],metadata['priced_events'],metadata['priced_people'],metadata['components'],metadata['no_sell_events'])!=(76,30,71,28,106,5): raise ValueError('source census mismatch')
    return sorted('dev-'+hashlib.sha256(p.encode()).hexdigest()[:24] for p in people),contexts,components,labels,metadata
