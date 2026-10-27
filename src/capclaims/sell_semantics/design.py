"""Deterministic paired intervention, preserving the frozen parent request."""
from copy import deepcopy
import hashlib, random
from capclaims.flat_grid_study import design as parent

DEFINITIONS = ("own_p is revenue per unit of this participant's own demand D served, not an order limit or market price. Other participants have different private p. The wholesale cost of Q was incurred before D was revealed. Units neither used to serve own demand nor sold expire at round end with no salvage value. Orders are blind limits: a BUY limit is the maximum acceptable unit price; a SELL limit is the minimum. Bids rank highest first and asks lowest first. Fills may be partial or zero. All fills in a market trade at one uniform price: the midpoint of the last matched bid and ask. An order's fill price can differ from its limit; a limit can affect the clearing price when marginal. Historical order prices are submitted limits, not realized prices. These definitions describe the rules, not a recommended action.")
SCHEDULE_SEED = 'jev-sell-semantics-schedule-20261027-v1'
PROMPT_PRICE=parent.PROMPT_PRICE
old_design=parent.old_design
normalize_jev_probabilities=parent.normalize_jev_probabilities
key=parent.key

def price_spec(context, quantity, arm):
    state, options = parent.price_spec(context, quantity)
    if arm == 'rules':
        if 'economic_definitions' in state['public_instructions']: raise ValueError('intervention field already exists')
        state['public_instructions']['economic_definitions'] = DEFINITIONS
    elif arm != 'original': raise ValueError('invalid arm')
    return state, options

def request_id(row):
    return parent.request_id(row) + '-' + row['arm']

def decide(raw, ids, row): return parent.decide(raw, ids, row)

def schedule(rows):
    output=[]
    for repeat in (0,1):
        paths=[deepcopy(r) for r in rows if r['repeat']==repeat]
        if len(paths)!=16 or len({key(r) for r in paths})!=16: raise ValueError('invalid paired paths')
        paths.sort(key=key)
        rng=random.Random(int(hashlib.sha256(f'{SCHEDULE_SEED}:{repeat}'.encode()).hexdigest(),16))
        rng.shuffle(paths)
        precedence={key(row): ('original' if i<8 else 'rules') for i,row in enumerate(paths)}
        block=[{**row,'arm':arm} for row in paths for arm in ('original','rules')]
        rng.shuffle(block)
        positions={}
        for i,row in enumerate(block): positions.setdefault(key(row),[]).append(i)
        for k,(a,b) in positions.items():
            if block[a]['arm'] != precedence[k]: block[a],block[b]=block[b],block[a]
        output.extend(block)
    if len({request_id(r) for r in output})!=64: raise ValueError('duplicate request')
    return output
