"""Pure component identities and price-only model input."""
from copy import deepcopy
from decimal import Decimal
from collections import Counter
import hashlib,json,random
from capclaims.behavioral_memory.memory import compile_behavioral_memory
from capclaims.behavioral_memory.projection import project_behavioral_memory
from capclaims.flat_price_grid import PRICE_VALUES, EMPTY
from capclaims.grid_sampling_v3 import decode
from capclaims.backends.jev import normalize_jev_probabilities
from capclaims.sell_semantics.design import DEFINITIONS
SEED='jev-sell-oracle-20261028-v1'
PROMPT=('You are an observer predicting this participant from the supplied history and current state. Forecast behavior, not advice or profit maximization. Predict the submitted SELL limit price for one supplied order component at the given exact quantity. Other current orders are not provided. Choose exactly one complete dollar price from the listed $5 grid.')
OPTIONS={k:f'{v:.2f}' for k,v in PRICE_VALUES.items()}

def _id(kind,person,t,q,occ=None):
    raw=json.dumps([SEED,kind,person,t,q,occ],separators=(',',':')).encode(); return kind+'-'+hashlib.sha256(raw).hexdigest()
def components(ctx,quantities):
    seen=Counter();out=[]
    for q in quantities:
        if type(q) is not int or not 1<=q<=ctx['current_state']['Q']: raise ValueError('invalid oracle quantity')
        occurrence=seen[q];seen[q]+=1;person=ctx['person_id'];t=ctx['formal_round']
        out.append({'component_id':_id('component',person,t,q,occurrence),'query_id':_id('query',person,t,q),'person_id':person,'formal_round':t,'quantity':q,'occurrence':occurrence})
    return out

def queries(components):
    result={}
    for c in components:
        q={k:c[k] for k in ('query_id','person_id','formal_round','quantity')}
        if c['query_id'] in result and result[c['query_id']]!=q: raise ValueError('query collision')
        result[c['query_id']]=q
    rows=sorted(result.values(),key=lambda r:r['query_id']);random.Random(int(hashlib.sha256(SEED.encode()).hexdigest(),16)).shuffle(rows);return rows

def state(ctx,q):
    cur=ctx['current_state'];hist=ctx['history'];t=ctx['formal_round']
    if type(q) is not int or not 1<=q<=cur['Q'] or cur['round']!=t or [h['round'] for h in hist]!=list(range(t-10,t)): raise ValueError('invalid state or oracle quantity')
    memory,_=compile_behavioral_memory(hist,cur,t-1)
    raw=[{'round':h['round'],'p':h['p'],'Q':h['Q'],'D':h['D'],'orders':deepcopy(h['orders']),'profit_dollars':f"{Decimal(h['profit_cents'])/Decimal(100):.2f}"} for h in hist]
    return {'formal_round':t,'own_p':cur['p'],'wholesale_price_dollars':cur['w'],'own_q_units':cur['Q'],'own_d_units':cur['D'],'own_side':'SELL','supplied_quantity_units':q,'history':raw,'behavioral_memory':project_behavioral_memory(memory,'M1'),'public_instructions':{'economic_definitions':DEFINITIONS,'trading':'The quantity is externally supplied exactly; no quantity is being selected. Price choices are complete $5 ticks from $5 through $995.'}}

def decide(raw,c):
    normalized,meta=normalize_jev_probabilities(raw,list(OPTIONS)); coordinate=f"{c['person_id']}:formal{c['formal_round']}:qty{c['quantity']}:occurrence{c['occurrence']}"
    try: d=decode(raw,list(OPTIONS),seed=SEED,person_id=coordinate,repeat=0,stage='price')
    except ValueError as e:
        if str(e)!=EMPTY: raise
        return {'status':'client_numeric_abstain','selected_id':None,'normalized':normalized,'normalization':meta}
    return {'status':'selected','selected_id':d['selected_id'],'decision':d}
