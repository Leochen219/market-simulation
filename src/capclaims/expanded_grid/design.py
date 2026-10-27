"""Construct the frozen rolling-window cohort without target leakage."""
import hashlib,random
from copy import deepcopy
from capclaims.behavioral_memory.memory import compile_behavioral_memory
from capclaims.behavioral_memory.projection import project_behavioral_memory
from capclaims.conditional_behavior.protocol import build_observation
from tools.probes import jev_grid_probability as v1

ARMS=('argmax','cutoff5','cutoff3'); SEED='expanded-grid-20261027'; ROUNDS=range(9,14)

def cohort(cases,labels):
    if len(cases)!=18 or len({c['person_id'] for c in cases})!=18: raise ValueError('need 18 distinct development people')
    contexts=[]; targets={}
    for case in cases:
        person=case['person_id']; episodes={r['round']:r for r in case['history']}
        if set(episodes)!=set(range(6,13)): raise ValueError('source history incomplete')
        for t in ROUNDS:
            if t==13: current=deepcopy(case['current_state'])
            else:
                row=episodes[t]; current={'round':t,'p':row['p'],'w':10,'Q':row['Q'],'D':row['D']}
            history=[deepcopy(episodes[r]) for r in range(t-3,t)]
            relation='shortage' if current['D']>current['Q'] else 'surplus' if current['Q']>current['D'] else 'equal'
            if relation=='equal': continue
            if t==13:
                if labels[person]['round']!=13: raise ValueError('target round mismatch')
                target=deepcopy(labels[person])
            else: target={'round':t,'orders':deepcopy(row['orders']),'profit_cents':row['profit_cents']}
            side='SELL' if relation=='surplus' else 'BUY'
            if not v1.quantity_grid(side,current['Q']): raise ValueError('empty legal quantity grid')
            ctx={'person_id':person,'formal_round':t,'stratum':relation,'side':side,'current_state':current,'history':history}
            ctx['base_state']=make_state(current,history,side); contexts.append(ctx); targets[f'{person}:formal{t}']=target
    return contexts,targets

def make_state(current,history,side):
    t=current['round']
    if [r['round'] for r in history]!=list(range(t-3,t)): raise ValueError('history must be exactly three preceding rounds')
    memory,_=compile_behavioral_memory(history,current,t-1)
    state=build_observation(state=current,history=history,arm='E')
    state.update(formal_round=t,round_mapping='formal_round + 5 = wire_round',history_window_n=3,behavioral_memory=project_behavioral_memory(memory,'M1'))
    for key in ('numeric_prefix','decimal_place','numeric_prefixes','decimal_places','decision_interface','protocol_interface'): state.pop(key,None)
    state['public_instructions']['trading']=('Diagnostic grid only: quantity choices are complete multiples of 10; price choices are complete $2.50 ticks. The model predicts numeric choices conditional on the supplied side; this probe does not predict direction.')
    state['submitted_order_ledger'].update({'submitted_order_count':0,'remaining_order_slots':1,'submitted_buy_quantity_units':0,'submitted_sell_quantity_units':0,'remaining_sell_submission_capacity_units':state['own_q_units']})
    state.update(selected_order_count=1,remaining_planned_order_slots=1,slot=0,partial_orders=[],own_side=side)
    return state

def schedule(contexts):
    rows=[]
    for repeat in (0,1):
        rng=random.Random(int(hashlib.sha256(f'{SEED}:{repeat}'.encode()).hexdigest(),16))
        groups=[[{'person_id':c['person_id'],'formal_round':c['formal_round'],'repeat':repeat} for c in contexts if c['stratum']==stratum] for stratum in ('shortage','surplus')]
        for group in groups: rng.shuffle(group)
        block=[item for i in range(max(map(len,groups))) for group in groups for item in group[i:i+1]]
        for row in block:
            offset=int(hashlib.sha256(f"{SEED}:{row['person_id']}:{row['formal_round']}:{repeat}".encode()).hexdigest(),16)%4
            row['band_order']=[(offset+i)%4 for i in range(4)]
        rows.extend(block)
    return rows

def coordinate(person,t): return f'{person}:formal{t}'
