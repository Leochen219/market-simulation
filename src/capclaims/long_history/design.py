"""Create matched short/long pre-target inputs without target labels."""
import hashlib,random
from copy import deepcopy
from capclaims.behavioral_memory.memory import compile_behavioral_memory
from capclaims.behavioral_memory.projection import project_behavioral_memory
from capclaims.conditional_behavior.protocol import build_observation
from tools.probes import jev_grid_probability as v1

ARMS=('argmax','cutoff5','cutoff3'); CONDITIONS=('short3','long10'); SEED='long-history-20261027-v1'

def make_state(current,history,side):
    if current.get('round')!=11 or len(history) not in (3,10) or [r['round'] for r in history]!=list(range(11-len(history),11)): raise ValueError('history must precede formal round 11 exactly')
    memory,_=compile_behavioral_memory(history,current,10)
    state=build_observation(state=current,history=history,arm='E')
    state.update(formal_round=11,round_mapping='formal_round + 5 = wire_round',history_window_n=len(history),behavioral_memory=project_behavioral_memory(memory,'M1'))
    for key in ('numeric_prefix','decimal_place','numeric_prefixes','decimal_places','decision_interface','protocol_interface'): state.pop(key,None)
    state['public_instructions']['trading']=('Diagnostic grid only: quantity choices are complete multiples of 10; price choices are complete $2.50 ticks. The model predicts numeric choices conditional on the supplied side; this probe does not predict direction.')
    state['submitted_order_ledger'].update({'submitted_order_count':0,'remaining_order_slots':1,'submitted_buy_quantity_units':0,'submitted_sell_quantity_units':0,'remaining_sell_submission_capacity_units':state['own_q_units']})
    state.update(selected_order_count=1,remaining_planned_order_slots=1,slot=0,partial_orders=[],own_side=side)
    return state

def cohort(cases):
    if len(cases)!=18 or len({c['person_id'] for c in cases})!=18: raise ValueError('need 18 distinct development people')
    contexts=[];excluded=0
    for case in cases:
        person=case['person_id']; current=case['current_state']; t=current['round']
        if t!=11 or set(current)!={'round','p','w','Q','D'} or [r['round'] for r in case['history']]!=list(range(1,11)): raise ValueError('source state/history mismatch')
        relation='shortage' if current['D']>current['Q'] else 'surplus' if current['Q']>current['D'] else 'equal'
        if relation=='equal': excluded+=1;continue
        side='SELL' if relation=='surplus' else 'BUY'
        if not v1.quantity_grid(side,current['Q']): raise ValueError('empty quantity grid')
        for condition,n in (('short3',3),('long10',10)):
            hist=deepcopy(case['history'][-n:]); cur=deepcopy(current)
            contexts.append({'person_id':person,'formal_round':11,'condition':condition,'stratum':relation,'side':side,'current_state':cur,'history':hist,'base_state':make_state(cur,hist,side)})
    return contexts,excluded

def schedule(contexts):
    rows=[]
    for repeat in (0,1):
        block=[{'person_id':c['person_id'],'formal_round':11,'condition':c['condition'],'repeat':repeat} for c in contexts]
        random.Random(int(hashlib.sha256(f'{SEED}:{repeat}'.encode()).hexdigest(),16)).shuffle(block)
        for row in block:
            offset=int(hashlib.sha256(f"{SEED}:{row['person_id']}:{row['formal_round']}:{repeat}".encode()).hexdigest(),16)%4
            row['band_order']=[(offset+i)%4 for i in range(4)]
        rows.extend(block)
    return rows

def coordinate(person,t): return f'{person}:formal{t}'
