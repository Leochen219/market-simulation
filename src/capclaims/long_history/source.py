"""Authorized development-only formal 1..11 source, with exact profit checks."""
from __future__ import annotations
import hashlib
from decimal import Decimal,InvalidOperation
from capclaims.behavioral_memory.source import build_authorized_cases
from capclaims.conditional_behavior.source import DEFAULT_SOURCE,_money_cents
from capclaims.conditional_behavior.protocol import validate_orders,validated_history_row
from capclaims.personalization_xlsx import load_paper_episodes,_workbook,_sheet_rows


def load():
    existing,_,metadata=build_authorized_cases(); aliases={c['person_id'] for c in existing}
    manifest,episodes=load_paper_episodes(DEFAULT_SOURCE,codebook_verified=True,intended_use_authorized=True,provider_transmission_permitted=True)
    def alias(person): return 'dev-'+hashlib.sha256(person.encode('utf-8')).hexdigest()[:24]
    selected={e.participant_id for e in episodes if e.scope=='GA 10' and e.treatment=='GA' and alias(e.participant_id) in aliases}
    if len(selected)!=18 or {alias(p) for p in selected}!=aliases or manifest.source_sha256!=metadata['source_sha256']: raise ValueError('authorized cohort or source mismatch')
    parts={}
    for ep in episodes:
        if ep.participant_id not in selected or ep.scope!='GA 10' or ep.treatment!='GA' or ep.round not in range(1,12): continue
        phases=parts.setdefault((ep.participant_id,ep.round),{})
        if ep.phase not in ('procurement','orders') or ep.phase in phases: raise ValueError('unexpected duplicate episode')
        phases[ep.phase]=ep
    if len(parts)!=198 or any(set(value)!={'procurement','orders'} for value in parts.values()): raise ValueError('incomplete source episode coverage')
    archive,shared,members=_workbook(DEFAULT_SOURCE)
    try: rows=_sheet_rows(archive,members['GA 10'],shared)
    finally: archive.close()
    fields=('Player','Round','Sales Price','Order','Actual Demand','Sales (#)','Profit Projection ()','Sales New (#)','Trade Cost','Post-trade profit ()')
    headers=[str(x) for x in rows[0]]
    if any(headers.count(x)!=1 for x in fields): raise ValueError('accounting columns not unique')
    ix={name:headers.index(name) for name in fields}; profits={}
    for row in rows[1:]:
        person=str(row[ix['Player']]).strip()
        if person not in selected: continue
        t=int(row[ix['Round']])
        if t not in range(1,12): continue
        def number(name):
            try:
                v=Decimal(str(row[ix[name]]))
                if not v.is_finite() or v!=v.to_integral_value(): raise ValueError
                return int(v)
            except (InvalidOperation,ValueError): raise ValueError('invalid cached accounting number') from None
        p,q,d,sales,projection,sales_new=(number(k) for k in ('Sales Price','Order','Actual Demand','Sales (#)','Profit Projection ()','Sales New (#)'))
        trade=_money_cents(row[ix['Trade Cost']],'Trade Cost'); final=_money_cents(row[ix['Post-trade profit ()']],'Post-trade profit')
        if projection!=p*min(d,q)-10*q or sales!=min(d,q) or final!=(p*sales_new-10*q)*100+trade: raise ValueError('accounting identity mismatch')
        k=(person,t); ep=parts[k]['orders']; proc=parts[k]['procurement']; observed=ep.observed
        if (observed['p'],observed['Q'],observed['D'],observed['w'])!=(p,q,d,10) or proc.action['Q']!=q or proc.observed['p']!=p or proc.observed['w']!=10 or k in profits: raise ValueError('source state mismatch or duplicate')
        profits[k]=final
    if len(profits)!=198: raise ValueError('accounting coverage incomplete')
    cases=[]; targets={}
    for person in sorted(selected):
        history=[]
        for t in range(1,12):
            ep=parts[(person,t)]['orders']; obs=ep.observed
            orders=[{'side':o['type'].lower(),'qty':o['amount'],'price':o['price']} for o in ep.action['orders']]
            orders=list(validate_orders(orders,initial_q=obs['Q']))
            if t<11:
                history.append(validated_history_row({'round':t,'p':obs['p'],'Q':obs['Q'],'D':obs['D'],'orders':orders,'profit_cents':profits[(person,t)]},allow_profit=True))
            else:
                current={'round':11,'p':obs['p'],'w':10,'Q':obs['Q'],'D':obs['D']}
                targets[alias(person)]={'round':11,'orders':orders,'profit_cents':profits[(person,t)]}
        cases.append({'person_id':alias(person),'history':history,'current_state':current})
    metadata=dict(metadata,history_n=180,history_rounds=list(range(1,11)),target_round=11,history_window='development-source-extension-v1')
    return cases,targets,metadata
