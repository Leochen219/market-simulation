"""Pure matched-state and deterministic sampling contract for flat-grid study."""
from copy import deepcopy
from capclaims import flat_price_grid as grid
from capclaims.backends.jev import normalize_jev_probabilities
from capclaims.long_history import design as old_design
from capclaims.grid_sampling_v3 import decode
from tools.probes.jev_grid_probability import PROMPT

PROMPT_PRICE = PROMPT + ' Choose exactly one complete dollar price on the full $5 grid; this is conditional on the chosen side and quantity.'
EMPTY = 'top3_cutoff03 has no surviving positive-mass candidate; no fallback is allowed'

def price_spec(context, quantity):
    base=context['base_state']
    # stage_spec validates legal quantities, order ledger and builds candidates.
    _, options = grid.stage_spec(base, context['side'], 'price', quantity)
    state=deepcopy(base)
    state.pop('model_chosen_band', None)
    state['own_qty_units']=quantity
    state['model_chosen_quantity']=quantity
    instructions=state.get('public_instructions')
    if not isinstance(instructions,dict) or not isinstance(instructions.get('trading'),str):
        raise ValueError('missing original trading instructions')
    original=instructions['trading']
    if original.count('$2.50') != 1: raise ValueError('unexpected original trading instructions')
    instructions['trading']=original.replace('$2.50','$5.00')
    return state, options

def key(row): return (row['person_id'],row['formal_round'],row['condition'],row['repeat'])

def request_id(row):
    return f"fg5-{row['condition']}-r{row['repeat']}-{row['person_id']}-t{row['formal_round']}-price"

def schedule(old_schedule, predictions):
    index={key(p):p for p in predictions}
    if len(index)!=len(predictions) or len(old_schedule)!=len(predictions): raise ValueError('baseline coverage mismatch')
    out=[]
    for row in old_schedule:
        p=index.pop(key(row),None)
        if p is None or type(p.get('conditioned_quantity')) is not int: raise ValueError('missing baseline quantity')
        hashes=p.get('request_hashes',{})
        if not hashes.get('quantity'): raise ValueError('missing baseline quantity provenance')
        out.append({**deepcopy(row),'conditioned_quantity':p['conditioned_quantity'], 'baseline_request_hashes':deepcopy(hashes)})
    if index or len({request_id(r) for r in out})!=len(out): raise ValueError('duplicate or extra identity')
    return out

def decide(raw, ids, row):
    normalized, metadata=normalize_jev_probabilities(raw, ids)
    try:
        decision=decode(raw,ids,seed=old_design.SEED,person_id=old_design.coordinate(row['person_id'],row['formal_round']),repeat=row['repeat'],stage='price')
    except ValueError as exc:
        if str(exc)!=EMPTY: raise
        return {'status':'client_numeric_abstain','selected_id':None,'normalized':normalized,'normalization':metadata}
    return {'status':'selected','selected_id':decision['selected_id'],'decision':decision}
