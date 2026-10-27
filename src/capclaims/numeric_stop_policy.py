"""Unfrozen client abstention for numeric grids, not a model STOP choice."""
from copy import deepcopy
import re
from capclaims.grid_sampling_v3 import decode

POLICY_VERSION='numeric_empty_support_stop_v1'


def decode_or_stop(raw_probabilities, candidate_ids, *, seed, person_id, repeat, stage):
    """Run frozen v3 decoding; stop only on its validated empty-support error."""
    if stage not in ('quantity', 'band', 'price'):
        raise ValueError('invalid numeric stage')
    if not isinstance(candidate_ids, (list, tuple)) or not candidate_ids:
        raise ValueError('candidate IDs must be a nonempty list or tuple')
    pattern={'quantity':r'q(?:[1-9][0-9]*)', 'band':r'b[0-3]', 'price':r'p(?:[1-9][0-9]*)'}[stage]
    if any(not isinstance(x,str) or re.fullmatch(pattern,x) is None for x in candidate_ids):
        raise ValueError('candidate ID does not match stage')
    if stage=='price' and any(int(x[1:])>399 for x in candidate_ids):
        raise ValueError('price tick out of range')
    if stage=='quantity' and any(int(x[1:])>990 or int(x[1:])%10 for x in candidate_ids):
        raise ValueError('quantity out of range')
    try:
        result=decode(raw_probabilities,candidate_ids,seed=seed,person_id=person_id,repeat=repeat,stage=stage)
    except ValueError as exc:
        if str(exc)!='top3_cutoff03 has no surviving positive-mass candidate; no fallback is allowed':
            raise
        return {'policy_version':POLICY_VERSION,'stage':stage,'candidate_count':len(candidate_ids),
                'raw_max_probability':max(float(raw_probabilities[k]) for k in candidate_ids),
                'stop_origin':'client_empty_support','status':'client_stop','selected_id':None,
                'reason':'empty_valid_numeric_support'}
    return {'policy_version':POLICY_VERSION,'stage':stage,'candidate_count':len(candidate_ids),
            'raw_max_probability':max(float(raw_probabilities[k]) for k in candidate_ids),
            'stop_origin':None,'status':'selected','selected_id':result['selected_id'],'decision':result}


def discard_draft_on_stop(submitted_orders, pending_draft):
    """Return existing submitted orders and discard the unfinished draft."""
    if not isinstance(submitted_orders,list):
        raise ValueError('submitted orders must be a list')
    return {'submitted_orders':deepcopy(submitted_orders),'pending_draft':None}
