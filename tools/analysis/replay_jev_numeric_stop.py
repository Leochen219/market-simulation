"""Offline replay of the unfrozen numeric client-stop policy; never calls Jev."""
import json
from collections import Counter
from pathlib import Path
from capclaims.long_history import design, runner
from capclaims.numeric_stop_policy import decode_or_stop, discard_draft_on_stop, POLICY_VERSION

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/participant-data/jev-long-history-v1-2026-10-27'
PRIVATE=ROOT/'artifacts/participant-data/jev-numeric-stop-replay-2026-10-27'
PUBLIC=ROOT/'docs/analysis/jev-numeric-stop-replay-2026-10-27.json'


def replay():
    plan,predictions,status=runner.replay(SOURCE)
    if status['status']=='PARTIAL_TECHNICAL':
        raise ValueError('technical failure cannot be mapped to stop')
    labels=json.loads((SOURCE/'labels-private.json').read_text(encoding='utf-8'))
    lookup={(p['person_id'],p['condition'],p['repeat']):p for p in predictions}
    chains={}
    for line in (SOURCE/'trace.jsonl').read_text(encoding='utf-8').splitlines():
        t=json.loads(line); p=t['path']; s=t['step']; key=(p['person_id'],p['condition'],p['repeat'])
        node='price-b'+str(s['band']) if s['stage']=='price' else s['stage']
        chains.setdefault(key,{})[node]=s
    rows=[]
    for key,chain in chains.items():
        old=lookup[key]; selected={}; stopped=None
        for stage in ('quantity','band','price'):
            node='price-'+selected['band'] if stage=='price' else stage
            step=chain[node]
            result=decode_or_stop(step['raw_probabilities'],step['candidate_ids'],seed=design.SEED,person_id=design.coordinate(key[0],old['formal_round']),repeat=key[2],stage=stage)
            if result['status']=='client_stop':
                stopped=stage; break
            selected[stage]=result['selected_id']
            assert result['decision']==step['decisions']['cutoff3']
        price=float(int(selected['price'][1:])*2.5) if stopped is None else None
        if stopped is None:
            if price!=old['prices']['cutoff3'] or int(selected['quantity'][1:])!=old['quantities']['cutoff3']:
                raise ValueError('successful old prediction differs')
        elif old['prices']['cutoff3'] is not None or old['price_failure_reason']['cutoff3'] not in ('band_decoder','selected_price_decoder','upstream_quantity'):
            raise ValueError('stop does not match old decoder failure')
        # This diagnostic specifies one order and starts with no submitted orders.
        quantity=int(selected['quantity'][1:]) if 'quantity' in selected else None
        if stopped:
            applied=discard_draft_on_stop([],{'side':old['side'],'quantity':quantity,'price':None})
            if applied['submitted_orders']!=[] or applied['pending_draft'] is not None:
                raise ValueError('stop committed an incomplete draft')
            committed=applied['submitted_orders']
        else:
            committed=[{'side':old['side'],'quantity':quantity,'price':price}]
        orders=labels[key[0]]['orders']
        rows.append({'person_id':key[0],'condition':key[1],'repeat':key[2],'side':old['side'],'outcome':'client_stop' if stopped else 'selected','stop_stage':stopped,'selected_draft_quantity':quantity,'draft_discarded':bool(stopped),'committed_orders':committed,'price':price,'target_orders_empty':not bool(orders)})
    if len(rows)!=len(plan['schedule']):raise ValueError('incomplete replay')
    counts=Counter(row['outcome'] for row in rows); stages=Counter(row['stop_stage'] for row in rows if row['stop_stage'])
    summary={'schema':'jev-numeric-stop-offline-replay-v1','policy_version':POLICY_VERSION,'scope':'Offline alternative policy, not the frozen result and not integrated into a live market. Only selected price branch examined. This one-order probe starts with no prior submitted orders. A client stop discards the unfinished draft and leaves previously submitted orders intact; it is not a model STOP choice. Invalid responses and technical failures remain errors.','frozen_status':status['status'],'paths':len(rows),'selected_paths':counts['selected'],'client_stops':counts['client_stop'],'stop_stages':dict(stages),'successful_prices_above_250':sum(r['price'] is not None and r['price']>250 for r in rows),'target_empty_people':sum(not bool(x['orders']) for x in labels.values()),'stopped_paths_with_empty_target_orders':sum(r['outcome']=='client_stop' and r['target_orders_empty'] for r in rows),'note':'Target emptiness is descriptive only; target orders are not inputs, and a stop does not establish improvement. Missing prices remain null, never zero.'}
    return summary,rows


def main():
    summary,rows=replay(); PRIVATE.mkdir(parents=True,exist_ok=True)
    (PRIVATE/'paths-private.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    PUBLIC.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
